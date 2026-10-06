# Slice: respect the crop, and slice every image at once

Two related changes to Slice mode:

1. **Slicing a cropped image slices the CROP.** Today the confirmed crop is
   silently dropped: the children tile the whole scan, so a user who framed a
   strip and then cut it in two gets two pieces of the *original*, film rebate
   and all. After this change the children tile the kept region — the result is
   "the cropped image, cut in two".
2. **Long-pressing the Slice button opens a scope menu**, whose second entry
   slices *every* loaded image at the same cut positions. A roll scanned at a
   fixed pitch (two or four frames per scan, always in the same place) becomes
   one gesture instead of one per scan.

## 1. Goals / non-goals

Goals

- Cuts are placed on the frame the user is looking at, which for a cropped
  image is the cropped frame.
- Each slice's pixels — preview, hi-res zoom detail and full-res export — come
  from the correct region of the ORIGINAL file at full quality, as they already
  do for an uncropped slice.
- The parent's straighten (`crop_angle`) and any canvas micro-rotation
  (`fine_rotation_angle`) are honoured, in the order the display applies them.
- `Reset Slice` restores the parent exactly, crop included.
- Slice-all applies the *same fractional* cuts to every image, each image
  resolving those fractions against its own (cropped) frame.

Non-goals

- The children's crop is **baked**, not inherited as an editable `crop_rect`.
  A slice is a new frame; its own crop starts empty. `Reset Slice` is the way
  back to the parent's crop.
- Slice-all does not detect per-image content. It is a geometric repeat, not
  auto-framing — Auto Frame already exists for that.
- No new per-image state. Scope is a transient UI arming, not saved.

## 2. UX

### 2.1 Cropped slicing

Entering Slice on a cropped image now shows the **cropped** frame (dimmed, as
today) instead of the whole scan. Cut lines are armed from the rims of that
frame, so the reachable cut range is the kept region. Fine rotation stays
visible, as it does today.

On Enter the slices tile the kept region. Each child's own `crop_rect` is
`None`: what it holds *is* the crop.

The crop is only baked when it is actually on screen — that is, when the image
is converted or Positive mode is on, the same gate `update_preview` uses to
display a crop. An unconverted image shows (and therefore slices) the whole
negative, as before.

### 2.2 Long-press scope menu

Press and hold the **Slice** button for ~450 ms (or right-click it) to open a
small menu:

```
  Slice this image           ✓
  Slice all images at the same cuts
```

- A checkmark marks the armed scope.
- Choosing an entry **enters Slice mode** if it isn't active, and otherwise
  only re-arms the scope — placed lines survive, so a user who has already
  positioned the cuts can switch to "all" without redoing them.
- A plain click on Slice is unchanged: toggle Slice mode with the scope reset
  to "this image".
- The scope resets to "this image" after a slice completes and on cancel/Esc,
  so "all" can never fire by surprise on the next slice.
- The hint line names the armed scope while "all" is active.

Enter with "all" armed slices every loaded image at the placed fractions. The
progress dialog counts images, not pieces (`3 / 12 images`).

Images that cannot be sliced accurately are **skipped**, not mangled: the same
condition `enter_slice_mode` refuses on, a `bw` conversion with a baked
`fine_rot`, where cuts would land offset in the source. The completion hint
reports how many were skipped.

## 3. Data model

No new persisted image fields. Two additions inside the existing
`slice_parent` snapshot dict (already serialized verbatim by
`catalog._slice_parent_to_json`):

| key | meaning |
|---|---|
| `ops_added` | how many `source_ops` entries this round appended (1, or 2 when a crop was baked). Absent in old catalogs ⇒ 1. |
| `crop_rect` / `crop_angle` | the parent's crop at slice time, so `Reset Slice` can put it back. |

## 4. Processing / math

### 4.1 The display chain, and therefore the bake order

A cropped image is displayed as

```
decode → source_ops → [rotate −crop_angle, crop to the box] → rotate fine_rotation
```

`apply_crop_to_image`'s rotated branch samples output pixel `(u,v)` at
`C + R(A)·(u − bw/2, v − bh/2)`, i.e. the content rotated by `−A` about the box
centre `C`. A `source_ops` entry is *rotate about the frame centre, then crop a
fractional rect* — so the crop is expressible as one entry:

Rotating the whole frame by `−A` about the frame centre `F` maps a source point
`p` to `F + R(−A)(p − F)`. For `p` inside the box,

```
F + R(−A)(p − F) = C' + R(−A)(p − C),    C' = F + R(−A)(C − F)
```

which is exactly the axis-aligned `bw × bh` box centred at `C'`. So:

```
op_crop = (rot, region)
rot      = round(−crop_angle · 100)            # fine-rotation units, Qt CW-positive
region   = ((cx' − bw/2)/w, (cy' − bh/2)/h,
            (cx' + bw/2)/w, (cy' + bh/2)/h)
cx' = F.x + dx·cos A + dy·sin A
cy' = F.y − dx·sin A + dy·cos A                 (dx, dy) = C − F
```

With `rot == 0` this degenerates to `region = crop_rect`, which is the common
case: the Crop panel folds any `fine_rotation_angle` into `crop_angle` on entry
and clears it on confirm, so a freshly cropped image normally has
`crop_angle == 0` *or* `fine_rotation_angle == 0`, never a stacked pair.

The round therefore appends `op_crop` (when there is a crop) **followed by** the
existing `(fine_rotation, child_tile)` entry — crop first, rotation second,
matching the display. Two rotations about two different centres cannot collapse
into one entry, which is why this is two ops and why `ops_added` has to be
recorded.

The backend applies `op_crop` to its one shared decode through the *same*
helper `_apply_source_ops` uses (`CCRImage.apply_source_op`), so the
preview pixels and the export's independent replay clamp and round identically.
A rotated box that overhangs the frame is clamped by that helper, as every
`source_ops` region already is.

### 4.2 Full-resolution bookkeeping

`original_full_size` for a child is the parent's full size scaled by the region
fractions. With a crop baked, the crop's own fractions enter the product first —
computed with `_ops_full_size`'s exact `max(1, round(f·n))` form so the child's
reported size matches what a fresh file read produces.

### 4.3 Conversion replay

Unchanged, with one ordering constraint: for `mode == "ref"` the reference rect
lives in the parent's **un-cropped** frame, so `compute_reference_norm_params`
must run on the decode *before* the crop is applied. `mode == "bw"` anchors are
absolute colours and are order-independent.

### 4.4 Slice-all

`CCRBackend.slice_all_images(x_cuts, y_cuts, progress_callback)` walks the list
**back to front** so replacing image *i* with *n* children never invalidates an
index still to be visited. Each image is sliced by the existing
`slice_image_by_index`, i.e. each resolves the same fractions against its own
cropped frame, bakes its own rotation and replays its own conversion. Returns
`(images_sliced, pieces_created, skipped)`.

`CCRBackend.can_slice_image(img)` is the shared accuracy guard (the
`bw` + baked `fine_rot` case), used by `enter_slice_mode` for its hint and by
`slice_all_images` to skip.

## 5. Integration points

| file | change |
|---|---|
| `core/ccr_image.py` | extract `apply_source_op(img, rotation, region)` as a static method; `_apply_source_ops` loops over it. |
| `core/ccr_backend.py` | `slice_image_by_index`: bake the display crop, record `ops_added`/crop in `slice_parent`; `_reset_one_slice_round`: strip `ops_added` ops and restore the crop; new `slice_all_images`, `can_slice_image`, `_display_crop_for`. |
| `widgets/image_preview.py` | show the crop in slice mode (drop `not self.slice_mode` from the display-crop gate); `slice_scope` state + `confirm_slices` dispatch; `SliceWorker` gains an all-images mode; `SliceProgressDialog` labels images vs pieces. |
| `widgets/long_press_button.py` | new: `LongPressButton`, a `QPushButton` that emits `longPressed` after a hold (and on right-click). |
| `widgets/sliders_panel.py` | Slice button becomes a `LongPressButton`; `_on_slice_long_press` builds the scope menu. |

## 6. Test plan

`tests/test_slice.py` additions:

- **crop baked, no angle**: a cropped parent sliced in two yields children
  whose `source_ops` are `[(0, crop), (0, tile)]`, whose pixels equal the
  corresponding halves of the cropped coordinate image, whose `crop_rect` is
  `None`, and whose `original_full_size` matches a fresh full-res read.
- **re-read parity**: `read_image(preview=False)` on a child returns the same
  region (the export path), for both the angled and un-angled crop.
- **crop + straighten**: with `crop_angle != 0` the child pixels match a
  reference `apply_crop_to_image` + tile computed independently.
- **crop + fine rotation**: both ops present, in order crop-then-rotate.
- **gate**: an *unconverted* image with a `crop_rect` set slices the whole
  frame (one op), since the crop isn't displayed.
- **reset**: `Reset Slice` on a cropped round restores one image with the
  parent's `crop_rect`/`crop_angle` and empty `source_ops`; an old catalog
  entry without `ops_added` still resets a one-op round.
- **slice-all**: three loaded images, two cuts ⇒ 9 images in the right order,
  each child's `source_ops` carrying its own parent's chain; back-to-front
  iteration verified by display names; an unsliceable image is skipped and
  counted.

`tests/test_long_press_button.py`:

- the hold timer emits `longPressed` once and suppresses the subsequent
  `clicked`;
- a short press emits `clicked` and not `longPressed`;
- right-click emits `longPressed`.
