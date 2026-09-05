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


class TestRekeyToReplacement:
    """Replacing the originals with a linear TIFF must carry the edits over.

    The "merge:" record is validated source-by-source, so deleting the three
    RAWs makes it permanently unmatchable. Without a re-key every edit the user
    made would be stranded on a dead key and the reloaded TIFF would come back
    untouched. See spec/merge-linear-tiff-replace.md.
    """

    def _replaced(self, tmp_path, cat, **attrs):
        """A merged image with edits, cataloged, plus its replacement file."""
        r, g, b = _triplet(tmp_path)
        merged = _merged_image([r, g, b])
        merged.adjustment_settings = {"contrast": 15, "balance_g": 22}
        merged.crop_rect = (0.1, 0.2, 0.8, 0.9)
        merged.crop_angle = 3.0
        merged.rotation_angle = 90
        merged.dust_spots = [{"x": 0.5, "y": 0.5, "r": 0.01}]
        for k, v in attrs.items():
            setattr(merged, k, v)
        catalog.update_for_images([merged], path=cat)
        tiff = _png(tmp_path, "shot_RGB.tiff", seed=9)
        return merged, [r, g, b], tiff

    def test_edits_move_to_the_replacement_file(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        merged, sources, tiff = self._replaced(tmp_path, cat)
        assert catalog.rekey_merges_to_files([(merged, tiff)], path=cat) == 1

        entries = catalog.entries_for_path(tiff, path=cat)
        assert entries and len(entries) == 1
        state = entries[0]
        assert state["adjustment_settings"] == {"contrast": 15, "balance_g": 22}
        assert state["rotation_angle"] == 90
        assert state["dust_spots"] == [{"x": 0.5, "y": 0.5, "r": 0.01}]

    def test_the_crop_comes_with_it(self, tmp_path):
        """The bake deliberately does NOT apply the crop, so it must survive as
        a live, reversible setting on the replacement."""
        cat = str(tmp_path / "catalog.json")
        merged, _sources, tiff = self._replaced(tmp_path, cat)
        catalog.rekey_merges_to_files([(merged, tiff)], path=cat)
        state = catalog.entries_for_path(tiff, path=cat)[0]
        assert state["crop_rect"] == [0.1, 0.2, 0.8, 0.9]
        assert state["crop_angle"] == 3.0

    def test_baked_geometry_and_merge_identity_are_stripped(self, tmp_path):
        """Everything the replacement file already CONTAINS must not be replayed
        on top of it — the slice chain above all, which is in the pixels."""
        cat = str(tmp_path / "catalog.json")
        merged, _sources, tiff = self._replaced(
            tmp_path, cat, source_ops=[(0, (0.0, 0.0, 0.5, 1.0))],
            slice_group="grp-1", is_duplicate=True)
        catalog.rekey_merges_to_files([(merged, tiff)], path=cat)
        state = catalog.entries_for_path(tiff, path=cat)[0]
        assert state["source_ops"] == []
        assert state["is_merged"] is False
        assert state["merge_sources"] is None
        assert state["slice_group"] is None
        assert state["slice_parent"] is None
        assert state["is_duplicate"] is False
        assert state["display_name"] == os.path.basename(tiff)

    def test_the_dead_merge_record_is_dropped(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        merged, sources, tiff = self._replaced(tmp_path, cat)
        assert catalog._merge_key(sources) in catalog.load_catalog(cat)["files"]
        catalog.rekey_merges_to_files([(merged, tiff)], path=cat)
        assert catalog._merge_key(sources) not in catalog.load_catalog(cat)["files"]

    def test_a_missing_replacement_is_skipped_not_raised(self, tmp_path):
        """Bookkeeping must never be able to break the destructive path that
        calls it — the files are already written and deleted by then."""
        cat = str(tmp_path / "catalog.json")
        merged, sources, _tiff = self._replaced(tmp_path, cat)
        gone = str(tmp_path / "never_written.tiff")
        assert catalog.rekey_merges_to_files([(merged, gone)], path=cat) == 0
        # ...and the merge record is left alone, so nothing is lost either.
        assert catalog._merge_key(sources) in catalog.load_catalog(cat)["files"]

    def test_empty_input_is_a_noop(self, tmp_path):
        assert catalog.rekey_merges_to_files([], path=str(tmp_path / "c.json")) == 0
        assert not os.path.exists(str(tmp_path / "c.json"))


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
