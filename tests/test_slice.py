#!/usr/bin/env python3
"""
Tests for slice mode's model layer: source_ops reading, single-decode
child construction, backend grid slicing, region composition for nested
slices, and cut cleaning.
"""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

import cv2  # noqa: E402
from core.ccr_backend import ccr_backend, CCRBackend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402


def _coordinate_png(tmp_path, w=600, h=400, name="scan.png"):
    """16-bit PNG whose channel 0 encodes x and channel 1 encodes y, so any
    region read can be verified pixel-exactly."""
    img = np.zeros((h, w, 3), dtype=np.uint16)
    img[..., 0] = (np.arange(w, dtype=np.uint16) * 100)[None, :]
    img[..., 1] = (np.arange(h, dtype=np.uint16) * 150)[:, None]
    path = str(tmp_path / name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path, img


class TestSourceOps:
    def test_read_image_crops_to_region(self, tmp_path):
        path, img = _coordinate_png(tmp_path)
        obj = CCRImage(path, source_ops=[(0, (0.25, 0.5, 0.75, 1.0))])
        # Region of a 600x400 image: x 150..450, y 200..400
        expected = img[200:400, 150:450]
        np.testing.assert_array_equal(obj.resized_raw, expected)
        assert obj.original_full_size == (200, 300)

    def test_no_ops_reads_whole_file(self, tmp_path):
        path, img = _coordinate_png(tmp_path)
        obj = CCRImage(path)
        np.testing.assert_array_equal(obj.resized_raw, img)
        assert obj.original_full_size == (400, 600)

    def test_rotation_op_matches_reference_warp(self, tmp_path):
        import cv2 as _cv2
        path, img = _coordinate_png(tmp_path, w=400, h=400)
        obj = CCRImage(path, source_ops=[(1500, (0.0, 0.0, 0.5, 1.0))])
        m = _cv2.getRotationMatrix2D((200, 200), -15.0, 1.0)
        rotated = _cv2.warpAffine(img, m, (400, 400), flags=_cv2.INTER_LINEAR,
                                  borderMode=_cv2.BORDER_CONSTANT, borderValue=0)
        np.testing.assert_array_equal(obj.resized_raw, rotated[:, 0:200])

    def test_preloaded_constructor_skips_file_read(self, tmp_path):
        path, img = _coordinate_png(tmp_path)
        crop = img[0:200, 0:300]
        obj = CCRImage(path, source_ops=[(0, (0.0, 0.0, 0.5, 0.5))],
                       preloaded_img=crop, preloaded_full_size=(200, 300),
                       display_name="scan_s1.png")
        np.testing.assert_array_equal(obj.resized_raw, crop)
        assert obj.original_full_size == (200, 300)
        assert obj.display_name == "scan_s1.png"
        # Must own its pixels, not view the shared parent decode
        assert obj.resized_raw.base is None

    def test_reload_respects_ops(self, tmp_path):
        path, img = _coordinate_png(tmp_path)
        obj = CCRImage(path, source_ops=[(0, (0.5, 0.0, 1.0, 0.5))])
        obj.reload_image()
        np.testing.assert_array_equal(obj.resized_raw, img[0:200, 300:600])


class TestCleanSliceCuts:
    def test_basic(self):
        assert CCRBackend._clean_slice_cuts([0.5]) == [0.0, 0.5, 1.0]

    def test_sorted_and_deduped(self):
        out = CCRBackend._clean_slice_cuts([0.7, 0.3, 0.302, 0.7005])
        assert out == [0.0, 0.3, 0.7, 1.0]

    def test_edge_cuts_dropped(self):
        assert CCRBackend._clean_slice_cuts([0.001, 0.999]) == [0.0, 1.0]

    def test_empty(self):
        assert CCRBackend._clean_slice_cuts([]) == [0.0, 1.0]


class TestSliceImage:
    def _load(self, tmp_path, **png_kwargs):
        path, img = _coordinate_png(tmp_path, **png_kwargs)
        ccr_backend.images = [CCRImage(path)]
        ccr_backend.file_paths = [path]
        return path, img

    def test_vertical_cuts_make_three_children(self, tmp_path):
        path, img = self._load(tmp_path)
        n = ccr_backend.slice_image_by_index(0, [1.0 / 3.0, 2.0 / 3.0], [])
        assert n == 3
        assert ccr_backend.get_image_count() == 3
        regions = [im.source_ops[-1][1] for im in ccr_backend.images]
        assert regions[0] == pytest.approx((0.0, 0.0, 1.0 / 3.0, 1.0))
        assert regions[1] == pytest.approx((1.0 / 3.0, 0.0, 2.0 / 3.0, 1.0))
        assert regions[2] == pytest.approx((2.0 / 3.0, 0.0, 1.0, 1.0))
        assert all(len(im.source_ops) == 1 and im.source_ops[0][0] == 0
                   for im in ccr_backend.images)
        # Children carry the correct pixels (reading order, left to right)
        np.testing.assert_array_equal(ccr_backend.images[1].resized_raw,
                                      img[:, 200:400])
        # Distinct display names
        names = [im.display_name for im in ccr_backend.images]
        assert names == ["scan_s1.png", "scan_s2.png", "scan_s3.png"]
        # All un-converted, fresh state
        assert all(not im.converted for im in ccr_backend.images)

    def test_grid_two_by_two(self, tmp_path):
        self._load(tmp_path)
        n = ccr_backend.slice_image_by_index(0, [0.5], [0.5])
        assert n == 4
        # Reading order: TL, TR, BL, BR
        regions = [im.source_ops[-1][1] for im in ccr_backend.images]
        assert regions[0] == pytest.approx((0.0, 0.0, 0.5, 0.5))
        assert regions[1] == pytest.approx((0.5, 0.0, 1.0, 0.5))
        assert regions[2] == pytest.approx((0.0, 0.5, 0.5, 1.0))
        assert regions[3] == pytest.approx((0.5, 0.5, 1.0, 1.0))

    def test_nested_slice_composes_into_original_coords(self, tmp_path):
        path, img = self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])     # two halves
        # Slice the RIGHT half again at its own middle
        n = ccr_backend.slice_image_by_index(1, [0.5], [])
        assert n == 2
        # The nested children's chains extend the parent's
        assert ccr_backend.images[1].source_ops == [
            (0, (0.5, 0.0, 1.0, 1.0)), (0, (0.0, 0.0, 0.5, 1.0))]
        assert ccr_backend.images[2].source_ops == [
            (0, (0.5, 0.0, 1.0, 1.0)), (0, (0.5, 0.0, 1.0, 1.0))]
        # The nested child's pixels come from the right place in the ORIGINAL
        np.testing.assert_array_equal(ccr_backend.images[2].resized_raw,
                                      img[:, 450:600])
        # ...and a fresh re-read from the file reproduces them
        reread = ccr_backend.images[2].read_image(path, preview=True)
        np.testing.assert_array_equal(reread, img[:, 450:600])
        # Nested names extend the parent's name — no collision with cousins
        names = [im.display_name for im in ccr_backend.images]
        assert names == ["scan_s1.png", "scan_s2_s1.png", "scan_s2_s2.png"]
        assert len(set(names)) == len(names)

    def test_no_effective_cuts_is_noop(self, tmp_path):
        self._load(tmp_path)
        n = ccr_backend.slice_image_by_index(0, [0.001], [])
        assert n == 0
        assert ccr_backend.get_image_count() == 1

    def test_replaces_parent_in_place(self, tmp_path):
        path1, _ = _coordinate_png(tmp_path, name="a.png")
        path2, _ = _coordinate_png(tmp_path, name="b.png")
        ccr_backend.images = [CCRImage(path1), CCRImage(path2)]
        ccr_backend.file_paths = [path1, path2]
        n = ccr_backend.slice_image_by_index(0, [0.5], [])
        assert n == 2
        assert ccr_backend.get_image_count() == 3
        # b.png stays last; slices of a.png take positions 0 and 1
        assert os.path.basename(ccr_backend.images[2].file_path) == "b.png"
        assert ccr_backend.file_paths == [im.file_path for im in ccr_backend.images]

    def test_full_res_export_read_uses_region(self, tmp_path):
        """The export path (full-res read) must return only the slice."""
        path, img = self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])
        child = ccr_backend.images[1]
        full = child.read_image(child.file_path, preview=False)
        np.testing.assert_array_equal(full, img[:, 300:600])

    def test_fine_rotation_baked_into_slices(self, tmp_path):
        """Cuts are made on the rotated frame the user saw; children consume
        the rotation (their own fine_rotation_angle starts at 0) and re-reads
        from the file reproduce the same rotated pixels."""
        path, img = self._load(tmp_path, w=400, h=400)
        ccr_backend.images[0].fine_rotation_angle = 1500  # 15 degrees
        n = ccr_backend.slice_image_by_index(0, [0.5], [])
        assert n == 2
        m = cv2.getRotationMatrix2D((200, 200), -15.0, 1.0)
        rotated = cv2.warpAffine(img, m, (400, 400), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        np.testing.assert_array_equal(ccr_backend.images[0].resized_raw,
                                      rotated[:, 0:200])
        assert ccr_backend.images[0].source_ops == [(1500, (0.0, 0.0, 0.5, 1.0))]
        assert all(im.fine_rotation_angle == 0 for im in ccr_backend.images)
        # A fresh file read (export/hi-res path) reproduces the same pixels
        reread = ccr_backend.images[1].read_image(path, preview=True)
        np.testing.assert_array_equal(reread, rotated[:, 200:400])


class TestSliceCroppedImage:
    """A confirmed crop is baked into the slices: cuts are placed on the
    cropped frame the user sees, so the pieces tile the KEPT region.
    See spec/slice-crop-and-slice-all.md."""

    def _load_cropped(self, tmp_path, crop, angle=0.0, fine=0, **png_kwargs):
        path, img = _coordinate_png(tmp_path, **png_kwargs)
        obj = CCRImage(path)
        # The display-level crop is only shown (and so only baked) for a
        # converted image — mark it converted without touching the pixels.
        obj.converted = True
        obj.crop_rect = crop
        obj.crop_angle = angle
        obj.fine_rotation_angle = fine
        ccr_backend.images = [obj]
        ccr_backend.file_paths = [path]
        return path, img, obj

    def test_children_tile_the_crop(self, tmp_path):
        path, img, _ = self._load_cropped(tmp_path, (0.25, 0.25, 0.75, 0.75))
        n = ccr_backend.slice_image_by_index(0, [0.5], [])
        assert n == 2
        # Two ops: the crop, then the (un-rotated) tile
        for child in ccr_backend.images:
            assert len(child.source_ops) == 2
            assert child.source_ops[0] == (0, (0.25, 0.25, 0.75, 0.75))
            assert child.crop_rect is None
            assert child.crop_angle == 0.0
        assert ccr_backend.images[0].source_ops[1] == (0, (0.0, 0.0, 0.5, 1.0))
        assert ccr_backend.images[1].source_ops[1] == (0, (0.5, 0.0, 1.0, 1.0))
        # Pixels: the crop of a 600x400 frame is x 150..450, y 100..300;
        # its halves are x 150..300 and x 300..450.
        np.testing.assert_array_equal(ccr_backend.images[0].resized_raw,
                                      img[100:300, 150:300])
        np.testing.assert_array_equal(ccr_backend.images[1].resized_raw,
                                      img[100:300, 300:450])
        # original_full_size matches the region actually held
        assert ccr_backend.images[0].original_full_size == (200, 150)

    def test_export_read_reproduces_the_crop_tile(self, tmp_path):
        """The full-res path replays the chain from the file — it must land on
        the same pixels the preview shows."""
        path, img, _ = self._load_cropped(tmp_path, (0.25, 0.25, 0.75, 0.75))
        ccr_backend.slice_image_by_index(0, [0.5], [])
        child = ccr_backend.images[1]
        full = child.read_image(child.file_path, preview=False)
        np.testing.assert_array_equal(full, img[100:300, 300:450])

    def test_unconverted_crop_is_not_baked(self, tmp_path):
        """The crop isn't displayed on an un-converted negative, so slicing
        one must still cut the whole frame — otherwise the cuts the user
        placed and the pieces produced disagree."""
        path, img = _coordinate_png(tmp_path)
        obj = CCRImage(path)
        obj.crop_rect = (0.25, 0.25, 0.75, 0.75)
        ccr_backend.images = [obj]
        ccr_backend.file_paths = [path]
        n = ccr_backend.slice_image_by_index(0, [0.5], [])
        assert n == 2
        assert all(len(im.source_ops) == 1 for im in ccr_backend.images)
        np.testing.assert_array_equal(ccr_backend.images[0].resized_raw,
                                      img[:, 0:300])

    def test_straightened_crop_matches_reference_extraction(self, tmp_path):
        """With a crop_angle the bake is a frame rotation plus an axis-aligned
        box; it must reproduce apply_crop_to_image's rotated branch."""
        from core.ccr_processor import apply_crop_to_image
        crop = (0.2, 0.2, 0.8, 0.8)
        path, img, _ = self._load_cropped(tmp_path, crop, angle=7.0,
                                          w=400, h=400)
        n = ccr_backend.slice_image_by_index(0, [0.5], [])
        assert n == 2
        rot = ccr_backend.images[0].source_ops[0][0]
        assert rot == -700           # -angle, in 1/100 degree
        reference = apply_crop_to_image(img, crop, 7.0)
        half = reference.shape[1] // 2
        # Both do exactly one bilinear resample, on grids that agree to the
        # pixel; compare the interior (the rims differ by sub-pixel sampling
        # of the black border). An un-rotated crop scores ~570 here, so this
        # really does pin the rotation down.
        got = ccr_backend.images[0].resized_raw
        assert got.shape[1] == pytest.approx(half, abs=1)
        a = got[20:-20, 20:half - 20].astype(np.int64)
        b = reference[20:-20, 20:half - 20].astype(np.int64)
        diff = np.abs(a - b)
        assert diff.mean() < 1.0       # the ramp steps by 100/150 per pixel
        assert diff.max() <= 100

    def test_crop_and_fine_rotation_bake_in_display_order(self, tmp_path):
        """The display crops first and micro-rotates the cropped canvas; the
        slice chain must carry the two ops in that order."""
        path, img, _ = self._load_cropped(tmp_path, (0.2, 0.2, 0.8, 0.8),
                                          fine=300, w=400, h=400)
        n = ccr_backend.slice_image_by_index(0, [0.5], [])
        assert n == 2
        ops = ccr_backend.images[0].source_ops
        assert len(ops) == 2
        assert ops[0] == (0, (0.2, 0.2, 0.8, 0.8))   # crop first
        assert ops[1][0] == 300                      # then the micro-rotation
        # The children consumed the rotation
        assert all(im.fine_rotation_angle == 0 for im in ccr_backend.images)
        # Independent replay of crop-then-rotate-then-cut
        cropped = img[80:320, 80:320]
        m = cv2.getRotationMatrix2D((120, 120), -3.0, 1.0)
        rotated = cv2.warpAffine(cropped, m, (240, 240), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        np.testing.assert_array_equal(ccr_backend.images[0].resized_raw,
                                      rotated[:, 0:120])

    def test_reset_restores_the_parent_crop(self, tmp_path, monkeypatch):
        from core import catalog
        monkeypatch.setattr(catalog, "default_catalog_path",
                            lambda: str(tmp_path / "catalog.json"))
        ccr_backend._catalog_preserved = {}
        crop = (0.25, 0.25, 0.75, 0.75)
        path, img, _ = self._load_cropped(tmp_path, crop, angle=4.0)
        ccr_backend.slice_image_by_index(0, [0.5], [])
        assert ccr_backend.get_image_count() == 2
        assert ccr_backend.reset_slice_by_indices([0]) == 0
        assert ccr_backend.get_image_count() == 1
        parent = ccr_backend.images[0]
        assert parent.source_ops == []
        assert parent.crop_rect == pytest.approx(crop)
        assert parent.crop_angle == pytest.approx(4.0)
        assert parent.fine_rotation_angle == 0

    def test_reset_of_legacy_round_without_ops_added(self, tmp_path, monkeypatch):
        """Rounds sliced before this feature have no `ops_added` key; reset
        must still strip exactly the one op they appended."""
        from core import catalog
        monkeypatch.setattr(catalog, "default_catalog_path",
                            lambda: str(tmp_path / "catalog.json"))
        ccr_backend._catalog_preserved = {}
        path, img = _coordinate_png(tmp_path)
        ccr_backend.images = [CCRImage(path)]
        ccr_backend.file_paths = [path]
        ccr_backend.slice_image_by_index(0, [0.5], [])
        for im in ccr_backend.images:
            im.slice_parent.pop("ops_added", None)
            im.slice_parent.pop("crop_rect", None)
            im.slice_parent.pop("crop_angle", None)
        assert ccr_backend.reset_slice_by_indices([0]) == 0
        assert ccr_backend.get_image_count() == 1
        assert ccr_backend.images[0].source_ops == []
        assert ccr_backend.images[0].crop_rect is None
        np.testing.assert_array_equal(ccr_backend.images[0].resized_raw, img)

    def test_nested_slice_of_a_cropped_slice(self, tmp_path):
        """A slice has no crop of its own, so re-slicing it appends one op —
        on top of the two the cropped round left."""
        path, img, _ = self._load_cropped(tmp_path, (0.25, 0.25, 0.75, 0.75))
        ccr_backend.slice_image_by_index(0, [0.5], [])
        ccr_backend.slice_image_by_index(1, [], [0.5])
        assert ccr_backend.get_image_count() == 3
        child = ccr_backend.images[2]
        assert len(child.source_ops) == 3
        # Bottom-right quadrant of the crop: x 300..450, y 200..300
        np.testing.assert_array_equal(child.resized_raw, img[200:300, 300:450])
        np.testing.assert_array_equal(
            child.read_image(path, preview=True), img[200:300, 300:450])


class TestSliceAllImages:
    def _load_many(self, tmp_path, n=3):
        paths = []
        for i in range(n):
            path, _ = _coordinate_png(tmp_path, name=f"scan{i}.png")
            paths.append(path)
        ccr_backend.images = [CCRImage(p) for p in paths]
        ccr_backend.file_paths = list(paths)
        return paths

    def test_every_image_sliced_at_the_same_cuts(self, tmp_path):
        self._load_many(tmp_path, 3)
        sliced, pieces, skipped = ccr_backend.slice_all_images([1 / 3, 2 / 3], [])
        assert (sliced, pieces, skipped) == (3, 9, 0)
        assert ccr_backend.get_image_count() == 9
        # Order preserved: each scan's three slices stay together, in order
        names = [im.display_name for im in ccr_backend.images]
        assert names == [f"scan{i}_s{j}.png" for i in range(3) for j in (1, 2, 3)]
        # Every child carries its own parent's region, not the first parent's
        for im in ccr_backend.images:
            assert len(im.source_ops) == 1
        assert ccr_backend.file_paths == [im.file_path for im in ccr_backend.images]

    def test_unsliceable_image_is_skipped_not_mangled(self, tmp_path):
        paths = self._load_many(tmp_path, 3)
        # A B/W conversion with a baked fine rotation: cuts placed on the
        # display would land offset in the source, so it must be left alone.
        ccr_backend.images[1].converted = True
        ccr_backend.images[1].conversion_inputs = {
            "mode": "bw", "bw": ((5.0, 5.0, 5.0), None), "fine_rot": 250}
        sliced, pieces, skipped = ccr_backend.slice_all_images([0.5], [])
        assert (sliced, skipped) == (2, 1)
        assert ccr_backend.get_image_count() == 5
        untouched = [im for im in ccr_backend.images if not im.source_ops]
        assert len(untouched) == 1
        assert untouched[0].file_path == paths[1]

    def test_respects_each_image_own_crop(self, tmp_path):
        paths = self._load_many(tmp_path, 2)
        _, img = _coordinate_png(tmp_path, name="probe.png")
        ccr_backend.images[0].converted = True
        ccr_backend.images[0].crop_rect = (0.0, 0.0, 0.5, 1.0)
        sliced, _pieces, skipped = ccr_backend.slice_all_images([0.5], [])
        assert (sliced, skipped) == (2, 0)
        # The cropped image's halves tile its crop (x 0..300), the plain
        # image's halves tile the whole frame (x 0..600)
        np.testing.assert_array_equal(ccr_backend.images[1].resized_raw,
                                      img[:, 150:300])
        np.testing.assert_array_equal(ccr_backend.images[3].resized_raw,
                                      img[:, 300:600])

    def test_no_images_is_a_noop(self, tmp_path):
        ccr_backend.images = []
        ccr_backend.file_paths = []
        assert ccr_backend.slice_all_images([0.5], []) == (0, 0, 0)


class TestEditInheritance:
    def _scan_png(self, tmp_path, w=600, h=400):
        rng = np.random.default_rng(21)
        yy, xx = np.mgrid[0:h, 0:w]
        base = 20000 + 25000 * (xx / w) + 10000 * (yy / h)
        img = np.stack([base * 1.2, base, base * 0.7], axis=-1)
        img += rng.normal(0, 1500, img.shape)
        img = np.clip(img, 1000, 64000).astype(np.uint16)
        path = str(tmp_path / "negative.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return path, img

    def test_conversion_and_adjustments_inherited(self, tmp_path):
        path, img = self._scan_png(tmp_path)
        parent = CCRImage(path)
        parent.reference_frame = (20, 20, 580, 380)
        ccr_backend.images = [parent]
        ccr_backend.file_paths = [path]
        ccr_backend.convert_negative_by_index(0)
        assert parent.converted
        parent.adjustment_settings = {"temperature": 25, "contrast": 10}
        parent_converted = parent.resized_raw.copy()

        n = ccr_backend.slice_image_by_index(0, [0.5], [])
        assert n == 2
        for child in ccr_backend.images:
            assert child.converted
            assert child.conversion_inputs["mode"] == "ref_params"
            assert child.adjustment_settings == {"temperature": 25, "contrast": 10}
            # Inherited dicts must not be shared between siblings
        assert (ccr_backend.images[0].adjustment_settings
                is not ccr_backend.images[1].adjustment_settings)
        # Colors match the parent's conversion (same constants replayed)
        left = parent_converted[:, 0:300]
        np.testing.assert_allclose(
            ccr_backend.images[0].resized_raw.astype(np.int64),
            left.astype(np.int64), atol=3)

    def test_ref_converted_child_exports(self, tmp_path):
        """ref_params children must export through the full-res path (this
        used to raise 'reference_frame is None' and fail every export)."""
        path, img = self._scan_png(tmp_path)
        parent = CCRImage(path)
        parent.reference_frame = (20, 20, 580, 380)
        ccr_backend.images = [parent]
        ccr_backend.file_paths = [path]
        ccr_backend.black_point_bgr = None
        ccr_backend.white_point_bgr = None
        ccr_backend.convert_negative_by_index(0)
        ccr_backend.slice_image_by_index(0, [0.5], [])
        out = str(tmp_path / "slice_export.tiff")
        ok = ccr_backend.export_image_by_index(0, out, jpg_output=False)
        assert ok
        import tifffile
        exported = tifffile.imread(out)
        # Full-res export of the left half of a 600x400 source
        assert exported.shape == (400, 300, 3)

    def test_tint_balance_factor_inherited(self, tmp_path):
        path, img = self._scan_png(tmp_path)
        parent = CCRImage(path)
        parent.reference_frame = (20, 20, 580, 380)
        ccr_backend.images = [parent]
        ccr_backend.file_paths = [path]
        ccr_backend.convert_negative_by_index(0)
        parent_factor = parent.tint_balance_factor
        ccr_backend.slice_image_by_index(0, [0.5], [])
        for child in ccr_backend.images:
            assert child.tint_balance_factor == pytest.approx(parent_factor)

    def test_bases_inherited(self, tmp_path):
        path, img = self._scan_png(tmp_path)
        parent = CCRImage(path)
        parent.converted = True
        parent.conversion_inputs = {"mode": "bw",
                                    "bw": ((52000.0, 48000.0, 39000.0),
                                           (9000.0, 9500.0, 7000.0)),
                                    "fine_rot": 0}
        parent.contrast_base = 60
        parent.temperature_base = 10
        ccr_backend.images = [parent]
        ccr_backend.file_paths = [path]
        n = ccr_backend.slice_image_by_index(0, [0.5], [])
        assert n == 2
        for child in ccr_backend.images:
            assert child.converted
            assert child.conversion_inputs["mode"] == "bw"
            assert child.contrast_base == 60
            assert child.temperature_base == 10


class TestResetSlice:
    @pytest.fixture(autouse=True)
    def _catalog_to_tmp(self, tmp_path, monkeypatch):
        from core import catalog
        monkeypatch.setattr(catalog, "default_catalog_path",
                            lambda: str(tmp_path / "catalog.json"))
        ccr_backend._catalog_preserved = {}

    def _load(self, tmp_path, **png_kwargs):
        path, img = _coordinate_png(tmp_path, **png_kwargs)
        ccr_backend.images = [CCRImage(path)]
        ccr_backend.file_paths = [path]
        return path, img

    def test_reset_restores_original(self, tmp_path):
        path, img = self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])
        assert ccr_backend.get_image_count() == 2
        restored = ccr_backend.reset_slice_by_indices([0])
        assert restored == 0
        assert ccr_backend.get_image_count() == 1
        parent = ccr_backend.images[0]
        assert parent.source_ops == []
        assert parent.display_name is None  # back to the file basename
        assert not parent.converted
        np.testing.assert_array_equal(parent.resized_raw, img)
        assert ccr_backend.file_paths == [path]

    def test_reset_inherits_template_adjustments(self, tmp_path):
        self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])
        ccr_backend.images[1].adjustment_settings = {"temperature": 25}
        restored = ccr_backend.reset_slice_by_indices([1])
        assert restored == 0
        assert ccr_backend.images[0].adjustment_settings == {"temperature": 25}

    def test_reset_selecting_all_siblings_resets_once(self, tmp_path):
        self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [1.0 / 3.0, 2.0 / 3.0], [])
        restored = ccr_backend.reset_slice_by_indices([0, 1, 2])
        assert restored == 0
        assert ccr_backend.get_image_count() == 1

    def test_reset_nested_restores_one_level(self, tmp_path):
        path, img = self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])     # scan_s1, scan_s2
        ccr_backend.slice_image_by_index(0, [], [0.5])     # scan_s1 -> _s1,_s2
        assert ccr_backend.get_image_count() == 3
        restored = ccr_backend.reset_slice_by_indices([0])
        assert restored == 0
        assert ccr_backend.get_image_count() == 2
        # The middle level comes back; its top-level sibling is untouched
        assert ccr_backend.images[0].display_name == "scan_s1.png"
        assert len(ccr_backend.images[0].source_ops) == 1
        np.testing.assert_array_equal(ccr_backend.images[0].resized_raw,
                                      img[:, 0:300])
        assert ccr_backend.images[1].display_name == "scan_s2.png"

    def test_reset_top_round_removes_nested_descendants(self, tmp_path):
        path, img = self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])
        ccr_backend.slice_image_by_index(0, [], [0.5])     # re-slice the left
        # Resetting from the top-level sibling folds EVERYTHING back
        restored = ccr_backend.reset_slice_by_indices([2])
        assert restored == 0
        assert ccr_backend.get_image_count() == 1
        assert ccr_backend.images[0].source_ops == []
        np.testing.assert_array_equal(ccr_backend.images[0].resized_raw, img)

    def test_reset_restores_baked_fine_rotation(self, tmp_path):
        path, img = self._load(tmp_path, w=400, h=400)
        ccr_backend.images[0].fine_rotation_angle = 300   # 3°, baked by slice
        ccr_backend.slice_image_by_index(0, [0.5], [])
        assert ccr_backend.images[0].source_ops[-1][0] == 300
        restored = ccr_backend.reset_slice_by_indices([0])
        assert restored == 0
        parent = ccr_backend.images[0]
        assert parent.fine_rotation_angle == 300  # live again, not baked
        assert parent.source_ops == []
        np.testing.assert_array_equal(parent.resized_raw, img)

    def test_reset_converted_slices_restores_conversion(self, tmp_path):
        rng = np.random.default_rng(21)
        yy, xx = np.mgrid[0:400, 0:600]
        base = 20000 + 25000 * (xx / 600) + 10000 * (yy / 400)
        img = np.stack([base * 1.2, base, base * 0.7], axis=-1)
        img += rng.normal(0, 1500, img.shape)
        img = np.clip(img, 1000, 64000).astype(np.uint16)
        path = str(tmp_path / "negative.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        parent = CCRImage(path)
        parent.reference_frame = (20, 20, 580, 380)
        ccr_backend.images = [parent]
        ccr_backend.file_paths = [path]
        ccr_backend.convert_negative_by_index(0)
        parent_converted = parent.resized_raw.copy()

        ccr_backend.slice_image_by_index(0, [0.5], [])
        restored = ccr_backend.reset_slice_by_indices([0])
        assert restored == 0
        back = ccr_backend.images[0]
        assert back.converted
        assert back.conversion_inputs["mode"] == "ref_params"
        # Same conversion constants replayed -> colors match the original
        np.testing.assert_allclose(back.resized_raw.astype(np.int64),
                                   parent_converted.astype(np.int64), atol=3)

    def test_reset_on_unsliced_returns_none(self, tmp_path):
        self._load(tmp_path)
        assert ccr_backend.reset_slice_by_indices([0]) is None
        assert ccr_backend.get_image_count() == 1

    def test_reset_updates_catalog(self, tmp_path):
        from core import catalog
        path, _ = self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])
        ccr_backend.save_catalog()
        assert len(catalog.entries_for_path(path)) == 2
        ccr_backend.reset_slice_by_indices([0])
        entries = catalog.entries_for_path(path)
        assert len(entries) == 1
        assert entries[0]["source_ops"] == []

    @staticmethod
    def _is_copy_slice(im):
        return bool(im.display_name and im.display_name.startswith("scan_copy1_s"))

    def test_reset_does_not_destroy_independent_duplicate_round(self, tmp_path):
        """Reset must collapse only the selected slice's lineage. A duplicate
        of the original, sliced separately, is an independent round and must
        survive — including any edits on its slices."""
        self._load(tmp_path)
        ccr_backend.duplicate_images_by_indices([0])       # scan + scan_copy1
        ccr_backend.slice_image_by_index(0, [0.5], [])     # slice the original
        # The copy is now at the end; slice it into 3 and edit one slice
        copy_idx = next(i for i, im in enumerate(ccr_backend.images)
                        if im.is_duplicate)
        ccr_backend.slice_image_by_index(copy_idx, [1 / 3, 2 / 3], [])
        copy_slices = [im for im in ccr_backend.images if self._is_copy_slice(im)]
        assert len(copy_slices) == 3
        copy_slices[0].adjustment_settings = {"temperature": 17}
        assert ccr_backend.get_image_count() == 5  # 2 original + 3 copy

        # Reset the ORIGINAL's round (one of its slices)
        orig_slice = next(i for i, im in enumerate(ccr_backend.images)
                          if im.source_ops and not self._is_copy_slice(im))
        ccr_backend.reset_slice_by_indices([orig_slice])
        # Original collapses to 1; the copy's 3 slices and their edits remain
        survivors = ccr_backend.images
        assert len([im for im in survivors if not im.source_ops]) == 1
        remaining_copy = [im for im in survivors if self._is_copy_slice(im)]
        assert len(remaining_copy) == 3
        assert remaining_copy[0].adjustment_settings == {"temperature": 17}

    def test_reset_duplicate_round_leaves_original_round(self, tmp_path):
        """The mirror of the above: resetting the copy's round must not
        touch the original's slices."""
        self._load(tmp_path)
        ccr_backend.duplicate_images_by_indices([0])
        ccr_backend.slice_image_by_index(0, [0.5], [])
        copy_idx = next(i for i, im in enumerate(ccr_backend.images)
                        if im.is_duplicate)
        ccr_backend.slice_image_by_index(copy_idx, [1 / 3, 2 / 3], [])
        copy_slice = next(i for i, im in enumerate(ccr_backend.images)
                          if self._is_copy_slice(im))
        ccr_backend.reset_slice_by_indices([copy_slice])
        # Copy collapses back to 1; original's 2 slices remain untouched
        assert len([im for im in ccr_backend.images
                    if self._is_copy_slice(im)]) == 0
        original_slices = [im for im in ccr_backend.images
                           if im.display_name in ("scan_s1.png", "scan_s2.png")]
        assert len(original_slices) == 2

    def test_reset_missing_file_fails_gracefully(self, tmp_path):
        path, _ = self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])
        os.remove(path)  # source gone before reset re-decodes it
        # Must not raise; nothing changes
        result = ccr_backend.reset_slice_by_indices([0])
        assert result is None
        assert ccr_backend.get_image_count() == 2

    def test_reset_from_duplicate_of_nested_slice_keeps_parent_name(self, tmp_path):
        self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])     # scan_s1, scan_s2
        # Re-slice scan_s2 -> scan_s2_s1, scan_s2_s2
        s2_idx = next(i for i, im in enumerate(ccr_backend.images)
                      if im.display_name == "scan_s2.png")
        ccr_backend.slice_image_by_index(s2_idx, [0.5], [])
        nested_idx = next(i for i, im in enumerate(ccr_backend.images)
                          if im.display_name == "scan_s2_s1.png")
        ccr_backend.duplicate_images_by_indices([nested_idx])
        dup_idx = next(i for i, im in enumerate(ccr_backend.images)
                       if im.display_name == "scan_s2_s1_copy1.png")
        # Resetting the duplicate restores its parent canvas (scan_s2),
        # NOT the whole file — the name must come from the stored snapshot.
        ccr_backend.reset_slice_by_indices([dup_idx])
        # The restored duplicate-lineage parent represents scan_s2
        restored = [im for im in ccr_backend.images
                    if im.is_duplicate and len(im.source_ops) == 1]
        assert len(restored) == 1
        assert restored[0].display_name == "scan_s2.png"

    def test_reset_purges_preserved_removed_sibling(self, tmp_path):
        """A sibling slice removed as an actual image is held in
        _catalog_preserved; resetting the round must drop it so it doesn't
        resurrect as a ghost on the next open."""
        from core import catalog
        path, _ = self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [1 / 3, 2 / 3], [])  # s1, s2, s3
        s2_idx = next(i for i, im in enumerate(ccr_backend.images)
                      if im.display_name == "scan_s2.png")
        ccr_backend.remove_images_by_indices([s2_idx])  # preserved
        # _catalog_preserved is keyed by the final catalog key (the normalized
        # file key for a normal image; a "merge:" composite for a merged one).
        assert catalog._file_key(path) in ccr_backend._catalog_preserved
        # Reset via a surviving sibling
        s1_idx = next(i for i, im in enumerate(ccr_backend.images)
                      if im.display_name == "scan_s1.png")
        ccr_backend.reset_slice_by_indices([s1_idx])
        # The preserved ghost is purged and the catalog holds only the parent
        entries = catalog.entries_for_path(path)
        assert len(entries) == 1
        assert entries[0]["source_ops"] == []

    def test_reset_then_reset_again_collapses_nested_stack(self, tmp_path):
        path, img = self._load(tmp_path)
        ccr_backend.slice_image_by_index(0, [0.5], [])     # scan_s1, scan_s2
        ccr_backend.slice_image_by_index(0, [], [0.5])     # scan_s1 nested
        # First reset: one level (scan_s1 restored)
        ccr_backend.reset_slice_by_indices([0])
        assert ccr_backend.get_image_count() == 2
        s1 = next(im for im in ccr_backend.images
                  if im.display_name == "scan_s1.png")
        # The restored mid-level parent must re-join its own round so a
        # second reset collapses to the original
        idx = ccr_backend.images.index(s1)
        ccr_backend.reset_slice_by_indices([idx])
        assert ccr_backend.get_image_count() == 1
        assert ccr_backend.images[0].source_ops == []
        np.testing.assert_array_equal(ccr_backend.images[0].resized_raw, img)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
