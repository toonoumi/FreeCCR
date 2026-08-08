# Orientation as a Copy/Sync Setting Group

Add **Orientation** — the coarse 90° rotation and the horizontal/vertical
mirror flags — to the group list both "apply to many" popups offer, so a roll
scanned upside-down or emulsion-side-up can be righted once and pushed onto the
other frames instead of clicking `[` / `]` on every one.

## Goals

- A new `SYNC_GROUPS` entry, `orientation`, covering `rotation_angle`,
  `horizontal_mirrored` and `vertical_mirrored`.
- It appears — same layout, same wording rules — in **both** dialogs that are
  driven by `SYNC_GROUPS`: **Sync to All** and `Ctrl/Cmd+C`'s **Copy Settings**.
  Checking it makes **Sync to All** push the source frame's orientation onto
  every image, and makes `Ctrl/Cmd+V` apply it to the frame(s) the user pastes
  onto.
- Orientation participates in undo exactly like crop: one snapshot per changed
  image, restored by `Ctrl+Z`.
- A frame whose orientation is *not* being synced/pasted keeps its own,
  untouched — same per-group isolation the other entries have.

## Non-goals

- **`fine_rotation_angle` is not included.** The micro-rotation is measured per
  frame from *that frame's* own film edge (`auto_fine_angle`), so copying one
  frame's value onto another mis-straightens it. It also folds into
  `crop_angle` (`folded_crop_angle`), which the existing **Crop** group already
  carries — the straighten a user sets in the Crop panel therefore still
  travels, under Crop.
- No new UI surface: no toolbar button, no thumbnail context-menu entry. The
  two existing popups are the whole delivery.
- No re-processing. Rotation and mirroring are display-level transforms applied
  at paint time (`ImagePreview.apply_transformations`,
  `ThumbnailList.apply_frontend_transformations`) and at export
  (`ccr_processor`). They do not touch `resized_raw`, the histogram, or any
  conversion input, so syncing them must **not** trigger
  `update_thumbnail_and_preview()`.

## UX / interaction

The group is inserted **directly after `crop`**, keeping the two geometry
entries adjacent:

```
Color Profile (Color / B&W)
White Balance / Tint
Tone (gain, brightness, contrast, ...)
Saturation
Crop
Orientation (rotate 90° / mirror)      ← new
Channel Levels
Subtractive Saturations (per color)
Curves
```

Everything else about the dialogs is unchanged: checked by default on first
open, remembered per-dialog while the app is open, Select All / Deselect All,
and the `Copied N of M setting groups.` hint — whose `M` simply becomes 9.

Because a remembered selection is read with `.get(gid, True)` in the dialog and
`.get(gid)` at apply time, an in-flight session that predates the group is not
a concern (the selections are session-only), and a *stale* clipboard selection
that lacks `orientation` reads as unchecked.

## Data model

Nothing new is persisted. `rotation_angle` / `horizontal_mirrored` /
`vertical_mirrored` already live on `CCRImage`, are already captured by
`capture_undo_state` / `pop_undo_state`, and are already serialised by
`core/catalog.py`. This feature only adds new *writers*.

New `SlidersPanel` state, alongside `copied_profile` / `copied_crop`:

| Attribute | Meaning |
| --- | --- |
| `copied_orientation` | `(rotation_angle, horizontal_mirrored, vertical_mirrored)`, or `None` when the orientation group was not copied. |

Orientation is a whole-image property with no per-area meaning, so — like
Color Profile and Crop — it is read off the `CCRImage`, never off the live UI,
regardless of which layer is active.

## Behaviour

### Sync to All (`_perform_sync_to_all`)

1. Read `src_orientation = (rot, hflip, vflip)` from the source image.
2. Per target: `orientation_changes = sync_orientation and target triple !=
   src_orientation`. Fold it into the existing "nothing to change → skip"
   guard so an all-identical batch still pushes no dead undo entries.
3. After `push_undo_state()`, assign the three attributes.
4. **Do not** add `orientation_changes` to the condition that calls
   `update_thumbnail_and_preview()` — see Non-goals. The trailing
   `update_all_thumbnails()` re-runs `apply_frontend_transformations`, and
   `update_preview()` re-reads the orientation for the canvas.
5. Zoom needs no reset: the currently displayed image *is* the sync source, so
   its own orientation cannot change here.

### Copy (`copy_adjustment_settings`)

`copied_orientation = (img.rotation_angle, img.horizontal_mirrored,
img.vertical_mirrored)` when the group is checked, else `None` — mirroring how
`copied_crop` is built (`getattr` defaults keep a stub image safe).

### Paste (`paste_adjustment_settings`)

Inside the existing `if img is not None:` block, alongside profile and crop:
assign the three attributes from `copied_orientation`.

Unlike Sync to All, the paste target **is** the displayed image, so a changed
orientation moves the displayed content under a zoomed viewport. Reset the zoom
first — the same reason `rotate_left`/`rotate_right` and `undo_last_action`
call `_reset_zoom()`. Guard it with `hasattr`, since tests drive the panel with
a stub preview.

The refresh at the tail of paste (`update_thumbnail(idx)` +
`update_preview(idx)`) already picks the new orientation up; no extra call.

## Integration points

| Location | Change |
| --- | --- |
| `SYNC_GROUPS` (`src/widgets/sliders_panel.py:27`) | New `("orientation", "Orientation (rotate 90° / mirror)", ())` entry after `crop`. |
| `SlidersPanel.__init__` (`:313`) | New `copied_orientation = None`, next to `copied_crop`. |
| `_perform_sync_to_all` (`:1437`) | Read the source triple; per-target change detection, undo, assignment. |
| `copy_adjustment_settings` (`:1886`) | Populate `copied_orientation`. |
| `paste_adjustment_settings` (`:1933`) | Apply `copied_orientation`; reset zoom when it lands. |
| `tests/test_sub_saturation_crop_undo.py::TestSyncGroups::test_expected_group_ids` | Expected id list gains `"orientation"`. |
| `user_guide/get_started.md:38` | Add orientation to the enumerated copyable groups. |

The `°` in the label is safe on the macOS ASCII-stdout path
(`memory: macos-ascii-stdout-crash`): group **labels** are only ever rendered
into `QCheckBox` text, while the one `print` in `_perform_sync_to_all` logs
group **ids**. The same dialog already carries `☑`/`☐` in its button text.

Deliberately unchanged: `_tether_template` / `_apply_tether_template` already
carry orientation to a new capture, `core/catalog.py` already persists it, and
`SlidersPanel.ADJUSTMENT_KEYS` gains nothing (the group has an empty key tuple,
like `crop`, `profile` and `curves`), so
`TestSyncGroups::test_groups_partition_adjustment_keys` keeps passing as-is.

## Resolved edge cases

- **Default orientation is a real value.** Pasting/syncing a frame that is
  upright and unmirrored *straightens* targets that were rotated, exactly as a
  copied `crop_rect is None` clears a target's crop.
- **Crop + orientation together.** `crop_rect` is stored in un-rotated,
  un-mirrored image coordinates, so the two groups are independent and can be
  synced in either combination without interacting.
- **Reference frame.** Also stored in un-rotated full-image coordinates —
  unaffected.
- **Crop mode open during a paste.** `update_preview` → `apply_transformations`
  rebuilds the crop overlay from the new base transform, the same path a manual
  rotate takes while cropping.
- **Unconverted target.** Not gated. Orientation is display-level and applies
  whether or not the frame has been converted.

## Test plan

New `tests/test_orientation_sync_group.py`, reusing the offscreen-Qt harness
and stubbed-dialog pattern of `tests/test_copy_settings_dialog.py`:

1. **Group registration** — `orientation` is present in `SYNC_GROUPS`, sits
   immediately after `crop`, and carries an empty key tuple (so it cannot
   perturb the ADJUSTMENT_KEYS partition).
2. **Copy stores the triple** — copying with only `orientation` checked leaves
   `copied_orientation == (90, True, False)` and an empty `copied_adjustment`.
3. **Copy without the group** — `copied_orientation is None`.
4. **Paste applies it** — a target at `(0, False, False)` becomes the copied
   triple, and one undo state is pushed.
5. **Paste without the group leaves the target's orientation alone** — copy WB
   only, paste onto a rotated frame: rotation/mirrors unchanged.
6. **Paste restores the default orientation** — copying an upright source onto
   a rotated target un-rotates it.
7. **Sync to All** — orientation lands on every image; images that already
   matched push no undo state.
8. **Sync isolation** — syncing orientation only does not disturb targets'
   adjustment values, crop, or profile.
9. **No reprocess** — syncing orientation alone leaves each target's
   `histogram_data` byte-identical (guards the "display-level only" contract).
10. **`fine_rotation_angle` is never copied or synced** — a source with a
    non-zero fine rotation leaves targets' fine rotation untouched.

Existing suites that must keep passing unchanged:
`tests/test_copy_settings_dialog.py`, `tests/test_sub_saturation_crop_undo.py`
(`TestSyncGroups` after its id-list update), `tests/test_cineon_log.py`.
