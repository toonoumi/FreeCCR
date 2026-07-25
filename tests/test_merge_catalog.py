#!/usr/bin/env python3
"""
Tests for cross-session persistence of 3-way-merged (trichrome) image edits:
the composite merge key + signature, serialize routing (a merged image is
cataloged under its three sources, never under any one source RAW's file key),
staleness against all three sources, and _restore_image's merged construction.

The full re-merge+restore round trip needs an aligned RAW triplet the repo does
not ship (the same CI boundary the merge feature itself documents), so these
tests exercise the catalog/JSON layer with real files + a normal CCRImage marked
merged, plus a monkeypatched constructor for _restore_image. See
spec/merge-catalog-persistence.md.
"""

import os
import sys
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

import cv2  # noqa: E402
from core import catalog  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402


def _png(tmp_path, name, w=64, h=48, seed=1):
    """A small real image file (content irrelevant — only its bytes/mtime and,
    for the red source, a decodable image matter)."""
    rng = np.random.default_rng(seed)
    img = (rng.integers(2000, 60000, size=(h, w, 3))).astype(np.uint16)
    path = str(tmp_path / name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def _triplet(tmp_path):
    r = _png(tmp_path, "shot_R.png", seed=1)
    g = _png(tmp_path, "shot_G.png", seed=2)
    b = _png(tmp_path, "shot_B.png", seed=3)
    return [r, g, b]


def _merged_image(sources, demosaic=True):
    """A real CCRImage decoded from the red source, then marked as a merge of
    the three sources — enough for the catalog/JSON layer without rawpy."""
    img = CCRImage(sources[0])
    img.is_merged = True
    img.merge_sources = list(sources)
    img.merge_demosaic = demosaic
    img._catalog_signature = catalog._merge_signature(sources)
    return img


class TestMergeKey:
    def test_prefix_and_order_sensitive(self, tmp_path):
        r, g, b = _triplet(tmp_path)
        k = catalog._merge_key([r, g, b])
        assert catalog._is_merge_key(k)
        assert k.startswith("merge:")
        # Order matters (R/G/B): swapping two sources is a different image.
        assert catalog._merge_key([g, r, b]) != k

    def test_disjoint_from_file_key(self, tmp_path):
        r, g, b = _triplet(tmp_path)
        assert not catalog._is_merge_key(catalog._file_key(r))
        assert catalog._merge_key([r, g, b]) != catalog._file_key(r)

    def test_catalog_key_for_image_routes(self, tmp_path):
        r, g, b = _triplet(tmp_path)
        merged = _merged_image([r, g, b])
        assert catalog._catalog_key_for_image(merged) == catalog._merge_key([r, g, b])
        normal = CCRImage(r)
        assert catalog._catalog_key_for_image(normal) == catalog._file_key(r)


class TestSerializeMergeFields:
    def test_merged_fields(self, tmp_path):
        r, g, b = _triplet(tmp_path)
        merged = _merged_image([r, g, b], demosaic=False)
        state = catalog.serialize_image(merged)
        assert state["is_merged"] is True
        assert state["merge_sources"] == [r, g, b]
        assert state["merge_demosaic"] is False

    def test_normal_fields_default_safe(self, tmp_path):
        r, _, _ = _triplet(tmp_path)
        state = catalog.serialize_image(CCRImage(r))
        assert state["is_merged"] is False
        assert state["merge_sources"] is None
        assert state["merge_demosaic"] is True


class TestUpdateRouting:
    def test_stored_under_merge_key_not_red_file(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        r, g, b = _triplet(tmp_path)
        merged = _merged_image([r, g, b])
        merged.adjustment_settings = {"temperature": 7}
        catalog.update_for_images([merged], path=cat)

        files = catalog.load_catalog(cat)["files"]
        assert catalog._merge_key([r, g, b]) in files
        # The red source keeps NO per-file record — a later plain open of it is
        # unaffected (the corruption the composite key exists to prevent).
        assert catalog._file_key(r) not in files
        assert catalog.entries_for_path(r, path=cat) is None

    def test_entries_for_merge_round_trip(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        r, g, b = _triplet(tmp_path)
        merged = _merged_image([r, g, b])
        merged.adjustment_settings = {"contrast": 15, "saturation": -3}
        merged.crop_rect = (0.1, 0.1, 0.9, 0.9)
        catalog.update_for_images([merged], path=cat)

        entries = catalog.entries_for_merge([r, g, b], path=cat)
        assert entries is not None and len(entries) == 1
        state = entries[0]
        assert state["is_merged"] is True
        assert state["adjustment_settings"] == {"contrast": 15, "saturation": -3}
        assert state["crop_rect"] == [0.1, 0.1, 0.9, 0.9]

    def test_mixed_batch_keys_both(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        r, g, b = _triplet(tmp_path)
        a = _png(tmp_path, "plain.png", seed=9)
        normal = CCRImage(a)
        normal.adjustment_settings = {"tint": 4}
        merged = _merged_image([r, g, b])
        catalog.update_for_images([normal, merged], path=cat)

        files = catalog.load_catalog(cat)["files"]
        assert catalog._file_key(a) in files
        assert catalog._merge_key([r, g, b]) in files


class TestMergeStaleness:
    def test_changed_source_invalidates(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        r, g, b = _triplet(tmp_path)
        merged = _merged_image([r, g, b])
        merged.adjustment_settings = {"temperature": 20}
        catalog.update_for_images([merged], path=cat)
        assert catalog.entries_for_merge([r, g, b], path=cat) is not None
        # Rewrite the GREEN source (size + mtime change) — a different capture.
        time.sleep(0.05)
        _png(tmp_path, "shot_G.png", w=80, h=60, seed=99)
        assert catalog.entries_for_merge([r, g, b], path=cat) is None

    def test_unknown_triplet_is_none(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        r, g, b = _triplet(tmp_path)
        assert catalog.entries_for_merge([r, g, b], path=cat) is None

    def test_missing_source_is_none(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        r, g, b = _triplet(tmp_path)
        catalog.update_for_images([_merged_image([r, g, b])], path=cat)
        os.remove(b)  # a source the user deleted since editing
        assert catalog.entries_for_merge([r, g, b], path=cat) is None


class TestRemovalPreservation:
    def test_preserved_merge_survives_save_when_unloaded(self, tmp_path):
        """A merged image removed from the list (no longer loaded) keeps its
        edits: update_for_images with a merge-keyed preserved record must write
        it under the merge key (not mangle the key via _file_key)."""
        cat = str(tmp_path / "catalog.json")
        r, g, b = _triplet(tmp_path)
        merged = _merged_image([r, g, b])
        merged.adjustment_settings = {"exposure": 6}
        state = catalog.serialize_image(merged)
        preserved = {
            catalog._merge_key([r, g, b]): {
                "signature": catalog._merge_signature([r, g, b]),
                "entries": {merged.display_name: state},
            }
        }
        catalog.update_for_images([], path=cat, preserved=preserved)

        entries = catalog.entries_for_merge([r, g, b], path=cat)
        assert entries is not None and len(entries) == 1
        assert entries[0]["adjustment_settings"] == {"exposure": 6}


class TestRestoreConstruction:
    def test_restore_image_builds_merged(self, tmp_path, monkeypatch):
        """_restore_image threads is_merged/merge_sources/merge_demosaic into the
        CCRImage it constructs, preferring the LIVE sources + demosaic."""
        r, g, b = _triplet(tmp_path)
        merged = _merged_image([r, g, b], demosaic=True)
        state = catalog.serialize_image(merged)

        recorded = {}

        class _RecordingImage:
            def __init__(self, file_path, **kwargs):
                recorded["file_path"] = file_path
                recorded.update(kwargs)
                self.file_path = file_path
                self.contrast_base = 0
                self.temperature_base = 0
                self.brightness_base = 0
                self.exposure_base = 0.0
                self.tint_balance_factor = 1.0
                self.dust_feather = 0.25

            def update_thumbnail_and_preview(self):
                pass

        monkeypatch.setattr("core.ccr_image.CCRImage", _RecordingImage)
        out = catalog._restore_image(r, state, live_merge_sources=[r, g, b],
                                     live_merge_demosaic=False)
        assert isinstance(out, _RecordingImage)
        assert recorded["is_merged"] is True
        assert recorded["merge_sources"] == [r, g, b]
        # Live demosaic wins over the stored value (True).
        assert recorded["merge_demosaic"] is False

    def test_restore_image_normal_state_not_merged(self, tmp_path, monkeypatch):
        r, _, _ = _triplet(tmp_path)
        state = catalog.serialize_image(CCRImage(r))

        recorded = {}

        class _RecordingImage:
            def __init__(self, file_path, **kwargs):
                recorded.update(kwargs)
                self.file_path = file_path
                self.contrast_base = 0
                self.temperature_base = 0
                self.brightness_base = 0
                self.exposure_base = 0.0
                self.tint_balance_factor = 1.0
                self.dust_feather = 0.25

            def update_thumbnail_and_preview(self):
                pass

        monkeypatch.setattr("core.ccr_image.CCRImage", _RecordingImage)
        catalog._restore_image(r, state)
        assert recorded["is_merged"] is False
        assert recorded["merge_sources"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
