# Adjustment Profiles (user looks in the Color Profile dropdown)

## Goal

Let the user save the current image's adjustments as a named **profile** and
re-apply it to any image from the same **Color Profile** dropdown that today
holds only *Color* and *Black & White* — the Lightroom "profile/preset" gesture.
A roll-wide look (a Channel Levels placement, a tone curve, a B&W conversion
with its contrast) becomes one click instead of copy/paste across sessions.

## Non-goals

- No geometry. Crop, orientation and anamorphic de-squeeze are per-frame /
  per-lens facts, not a look; they are not offered when saving.
- No area layers, dust spots, conversion inputs (B/W points, reference frame,
  film stock) — a profile is a grade, not a conversion.
- No catalog/schema change. The image does not remember "which profile was
  applied"; the dropdown *derives* it (see Display). Profiles are app-wide
  (QSettings), like film stocks.
- No import/export of profile files (could come later).

## UX

Color Profile row: `Color Profile [ combo ▾ ] [＋] [−]`

Combo items, in order:

1. `Color` 2. `Black & White` — unchanged, indices 0/1 keep mapping to
   `CCRImage.color_profile` (`COLOR_PROFILES`).
3. a separator (only when at least one user profile exists)
4. user profiles, sorted case-insensitively by name.

The combo is an `ElidingComboBox` (user-named entries, see
`test_panel_width.py`). Every row carries string item data — `"builtin:color"`,
`"builtin:bw"`, `"user:<name>"`; the separator has none — so handlers
never do index arithmetic around the separator.

- **＋ Save profile**: opens the existing `SyncSettingsDialog` restricted to
  the profile-eligible groups (all `SYNC_GROUPS` except `crop`,
  `orientation`, `desqueeze`), titled "Save Profile", with a **name field** at
  the top. Save is refused for an empty name or no groups. A name that already
  exists (case-insensitive) asks to replace. The group choice is remembered
  for the session. The saved profile becomes the combo's selection
  (it matches by construction).
- **Selecting a user profile** applies it to the current image's **global**
  layer: one undo step; only the profile's groups are overwritten, everything
  else keeps the image's values. If an area layer is active it is left alone
  (the dropdown is a whole-image control, like the Cineon checkbox).
- **− Delete**: enabled only while a user profile is the displayed item;
  confirms, removes it from the store. The image keeps its settings; the combo
  falls back to Color/B&W.
- Selecting `Color` / `Black & White` behaves exactly as today (changes only
  `color_profile`).

### Display (derived, never stored)

On every `_sync_color_profile_combo` the combo shows the **first (sorted) user
profile whose saved values all equal the image's current global values**; if
none matches it shows the built-in `Color` / `Black & White` row for
`img.color_profile`. So touching a slider covered by the profile makes the
combo fall back honestly to Color/B&W, and an image that was graded
identically by hand shows the profile name. No catalog field, no stale label.

## Data model

`src/core/adjustment_profiles.py` — pure JSON helpers, no Qt (same shape as
`core/film_stocks.py`). Stored in QSettings key `adjust/profiles` as a JSON
list of records:

```json
{
  "name": "Roll look",
  "groups": ["profile", "tone", "curves", "channels"],
  "color_profile": "bw",            // only meaningful when "profile" in groups
  "settings": {"brightness": 12, "...": 0, "cineon_log": true},
  "curves": {"rgb": [[0,0],[0.5,0.6],[1,1]]},   // or null
  "created": "2026-11-07"
}
```

- `settings` holds every key of every selected adjustment-key group (missing
  values filled with the slider default at save time), plus `cineon_log`
  (bool) when `channels` is selected — the same carriage rule Sync/Paste use.
- Validation on decode is defensive: bad JSON / non-list → `[]`; records
  without a name or with no groups are dropped; non-numeric settings values
  dropped; `color_profile` coerced to `"color"`/`"bw"`; `curves` must be a
  dict or is treated as `null`. Duplicate names (case-insensitive) keep the
  first. Helpers: `decode_profiles`, `encode_profiles`, `find_profile`,
  `upsert_profile`, `remove_profile`, plus:
- `merge_profile_settings(current, profile, group_keys, adjustment_keys,
  default_for) -> dict` — complete dict (every adjustment key), the profile's
  group keys overwritten, `curves` replaced/removed when `curves` is a group
  else carried over, `cineon_log` from the profile when `channels` is a group
  else carried over. Same merge contract as `paste_adjustment_settings`.
- `profile_matches(current, color_profile, profile, group_keys, default_for)
  -> bool` — the Display test above. Curves compare with identity ≡ `None`.

`group_keys` is a `{gid: keys}` mapping supplied by the panel from
`SYNC_GROUPS`, so the core module never imports Qt and new groups need no
change there. Unknown group ids in a stored profile are ignored.

## Integration points

- `SyncSettingsDialog`: new optional `groups` (subset of ids to show) and
  `name_prompt` (adds a `QLineEdit`; `name()` accessor). Existing callers
  unchanged.
- `SlidersPanel._create_color_profile_row`: eliding combo + ＋/− buttons.
- `_reload_adjustment_profiles()` reads `self._settings` (tests swap it for a
  temp ini, as the film-stock tests do).
- `_sync_color_profile_combo` → derived selection; `on_color_profile_changed`
  maps index → built-in or user profile.
- `set_sliders_enabled` gates the two buttons with the combo.
- Undo: `img.push_undo_state()` once before applying; `end_undo_burst()` first.
- Sync to All / Copy Settings: untouched; the `profile` group still means the
  Color/B&W flag only.

## Test plan

`tests/test_adjustment_profiles.py`:

- store: decode garbage → `[]`; round trip; bad records dropped; duplicate
  names; upsert replaces case-insensitively; remove.
- merge: only group keys overwritten; complete dict; curves replace/remove/
  carry; cineon carry vs override.
- matches: equal → True; one slider off → False; ungrouped keys ignored;
  curves identity ≡ None; color_profile only checked with `profile` group.
- UI (offscreen, temp QSettings, stub image in backend): combo lists
  built-ins + separator + profiles; save flow (dialog monkeypatched) persists
  and selects; selecting applies to global settings with one undo step and
  leaves other groups alone; slider change → combo falls back to Color/B&W;
  delete removes and falls back; built-in indices still map to
  `COLOR_PROFILES`.
