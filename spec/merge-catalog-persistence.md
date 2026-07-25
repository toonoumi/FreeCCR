# Cross-session persistence of 3-way-merged image edits

> Status: draft. This is the parked follow-up called out in
> `spec/three-way-rgb-merge.md` ("Follow-up: key the catalog on a composite of
> the 3 source signatures"). It makes a trichrome-merged image save and restore
> its edits **just like a normal image**.

## Summary

Today a 3-way RGB-merged image is a **session-only** artifact: it is excluded
from the per-file edit catalog on every serialization path, so its conversion,
sliders, crop, reference frame and dust are lost when the app closes and the
same three RAWs are re-imported (they re-merge fresh, un-edited).

This change persists a merged image's edits under a **composite catalog key**
derived from its three source RAWs (in `(R, G, B)` order) plus a **composite
freshness signature** (each source's size + mtime). When the same three files
are merged again and none has changed on disk, the merged image comes back with
all its edits — identical to how a normal single-file image is restored.

## Goals

- A merged image's full edit state (conversion inputs + replay, adjustment
  sliders, crop, orientation bases, reference frame, dust spots/plan, area
  layers, colour profile, slice lineage, duplicate flag) is **saved to the
  catalog** and **restored on the next merge of the same three frames**.
- The catalog entry is keyed on a **composite of the three sources**, so it can
  never collide with, or corrupt, the per-file record of any single source RAW
  (e.g. a later *normal* open of the red exposure).
- Staleness is validated against **all three** sources: if any source file
  changed (size/mtime) since the edits were cataloged, the stored edits are
  ignored and the triplet re-merges fresh (matching the single-file rule).
- Duplicates and slices of a merged image persist and restore too (the same
  all-or-nothing per-key restore the single-file path already uses).
- Parity on removal: removing a merged **actual** image from the list keeps its
  stored edits (preserved for the next save); removing a merged **duplicate**
  drops its entry (a discarded copy must not resurrect).

## Non-Goals

- No change to the merge maths, the decode, or the `is_merged`/`merge_sources`
  data model beyond what persistence needs.
- No relink UI: "the same three images" means the same absolute paths with
  unchanged content, exactly as the single-file catalog already defines
  identity. Moving/renaming a source yields a fresh merge (new key).
- The parked **density-inversion** experiment is untouched.
- Merged-image **slice reset** already rebuilds a non-merged parent (a
  pre-existing limitation noted in the merge spec); this change does not fix
  that, it only keeps the preserved-catalog bookkeeping keyed consistently.

## Data model

### Composite key (`catalog.py`)

```
MERGE_KEY_PREFIX = "merge:"
_merge_key(sources)         -> "merge:" + "|".join(_file_key(p) for p in sources)
_is_merge_key(key)          -> key.startswith("merge:")
_catalog_key_for_image(img) -> _merge_key(img.merge_sources) if img.is_merged
                                and img.merge_sources else _file_key(img.file_path)
```

- `_file_key` already normcases + abspaths each path, so the composite is stable
  per machine. Order is significant (R/G/B); the merge always sorts its inputs
  by filename, so the same three files always produce the same key.
- The `merge:` prefix guarantees a merge key is **disjoint** from every
  single-file key (a real path never starts with `merge:`), so a merge and a
  plain open of the red RAW live under separate records.

### Composite signature

```
_merge_signature(sources) -> {"sources": [ {"size", "mtime"}, … ]}   # R,G,B order
```

`entries_for_merge(sources)` returns the stored states only when the record
exists **and** every stored per-source signature matches the current file
(size equal, mtime within 1.0 s — the same tolerance as `entries_for_path`).
Any mismatch or missing/oversized list → `None` (re-merge fresh).

### Serialized state (`serialize_image`)

Three fields are added to every serialized image (harmless for non-merged
images, which write `is_merged=False`, `merge_sources=None`):

```
"is_merged":      bool
"merge_sources":  [red, green, blue] | None   # only when is_merged
"merge_demosaic": bool                         # metadata / restore fallback
```

`merge_sources` in the record confirms identity; on restore the **live** paths
selected in the current import are used to build the image (they equal the
stored ones because the key matched). `merge_demosaic` is stored as a fallback,
but restore prefers the **current global** `rgb_merge_mode` demosaic setting so
the re-merge honours the user's current decode preference (the conversion replay
adapts — `ref`/`bw` recompute; only `ref_params` replays fixed anchors, and the
two demosaic modes differ only marginally in normalized percentile terms).

## Processing / flow

### Restore (import with merge mode ON)

`ccr_backend._load_merged_triplets` loads the catalog once for the batch, then
for each triplet calls the new
`catalog.create_images_for_merge(sources, merge_demosaic, catalog_data)`, which
mirrors `create_images_for_path`:

- No cataloged entry → build one fresh merged `CCRImage` (a plain merge), tag
  its `_catalog_signature` with the composite signature.
- Cataloged entries present → restore each via `_restore_image(sources[0],
  state, live_merge_sources=sources, live_merge_demosaic=merge_demosaic)`,
  which now constructs a **merged** `CCRImage` (passes `is_merged`/
  `merge_sources`/`merge_demosaic`) so every re-read re-merges then applies the
  restored `source_ops`/conversion. All-or-nothing: any entry failure falls
  back to a fresh merge, flagged `_catalog_restore_failed` so the next save does
  not erase the stored record.

Because `_restore_image` already replays the conversion, restores dust plans,
crop, bases, etc., merged images inherit the entire single-file restore path for
free; the only merge-specific work is constructing a merged image and keying by
the composite.

### Save

`ccr_backend.save_catalog` → `catalog.update_for_images(self.images,
preserved=self._catalog_preserved)`:

- Grouping keys every image by `_catalog_key_for_image` (merge key for merged,
  file key otherwise). A merged image **with no `merge_sources`** is skipped
  (never keyed under a source file's record — the original corruption guard).
- The signature fallback branches on the key: `_merge_signature(merge_sources)`
  for a merge key, `_file_signature(file_path)` otherwise. Normally the
  signature captured at load (`_catalog_signature`) is used, as for files.

### Removal parity

`_catalog_preserved` is now keyed by the **final catalog key**
(`_catalog_key_for_image`) uniformly (was the raw file path). Consequently:

- `remove_images_by_indices`: a removed merged **actual** image is preserved
  under its merge key; a removed merged **duplicate** is dropped via
  `remove_duplicate_entries` under its merge key.
- `remove_duplicate_entries` and the preserved-merge inside `update_for_images`
  treat their dict keys as **already-final** catalog keys (apply `_file_key`
  only to non-merge keys / not at all).
- `_purge_preserved_slice_round` is looked up by `_catalog_key_for_image(template)`
  so it stays consistent for both kinds.

## Integration points

| Where | Change |
|---|---|
| `catalog.py` | `MERGE_KEY_PREFIX`, `_merge_key`, `_is_merge_key`, `_catalog_key_for_image`, `_merge_signature`; `entries_for_merge`; `create_images_for_merge`. |
| `catalog.serialize_image` | add `is_merged`/`merge_sources`/`merge_demosaic`. |
| `catalog._restore_image` | accept `live_merge_sources`/`live_merge_demosaic`; construct a merged `CCRImage` when the state is merged. |
| `catalog.update_for_images` | group by `_catalog_key_for_image`; skip an unkeyable merge; merge-aware signature fallback; preserved keys used as-is. |
| `catalog.remove_duplicate_entries` | keys are final catalog keys (`_file_key` only for non-merge keys). |
| `ccr_backend._load_merged_triplets` | load catalog once; per-triplet `create_images_for_merge` (returns a list — supports restored slices/dups); extend results; unchanged sort/dedup/commit. |
| `ccr_backend.remove_images_by_indices` | remove the `is_merged: continue`; preserve/drop merged under `_catalog_key_for_image`. |
| `ccr_backend._purge_preserved_slice_round` | key by `_catalog_key_for_image(template)`. |

`CATALOG_VERSION` stays **2**: the new fields are additive and default safely on
old records (`state.get("is_merged", False)` → normal restore), and old builds
ignore unknown keys, so no version bump is needed.

## Edge cases

- **A source changed on disk** → `entries_for_merge` returns `None`; fresh merge.
- **Only some of the three present / one missing at import** → `merge_raw_channels`
  raises during the fresh/restore build; the loader records `last_merge_error`
  and skips the triplet, exactly as today.
- **Merged image never edited** → a pristine state is written (as for normal
  images) and restores to pristine; harmless, bounded by catalog pruning.
- **Merge key vs single-file key** → disjoint by the `merge:` prefix; a normal
  open of the red RAW is unaffected and vice-versa.
- **Duplicate of a merged image** → restored as a merged duplicate (carries
  `is_merged`/`merge_sources`); removing it drops its entry under the merge key.
- **Global demosaic toggled between sessions** → the re-merge uses the current
  setting; stored conversion replays on top (see Data model rationale).

## Test plan

`tests/test_merge_catalog.py` — pure/JSON-layer unit tests (no rawpy):

1. `_merge_key`/`_is_merge_key`: order-sensitive, prefix present, disjoint from a
   single-file key; `_catalog_key_for_image` routes merged vs normal.
2. `_merge_signature` + `entries_for_merge`: match returns the states; a changed
   source (size/mtime) → `None`; a missing record → `None`; wrong-length stored
   list → `None`.
3. `serialize_image` includes the three merge fields with correct values for a
   merged stub and safe defaults for a normal image.
4. `update_for_images` routing: a merged stub image is stored **under the merge
   key**, **never** under `_file_key(red)`; a mixed batch (one normal, one
   merged) writes both keys; the stored merge state round-trips through
   `entries_for_merge`.
5. Removal parity via `update_for_images(preserved=…)` with a merge-keyed
   preserved record: the preserved state survives a save where the image is not
   loaded.

The full re-merge + restore round trip (constructing a real merged `CCRImage`)
needs an aligned RAW triplet, which the repo does not ship — the same CI
boundary the merge feature itself documented. It is covered by launch
verification: merge three RAWs, convert + adjust, restart, re-import the same
three → edits restored.

Existing suites `tests/test_catalog.py`, `tests/test_duplicate_remove.py`,
`tests/test_slice.py`, `tests/test_three_way_merge.py` must stay green (the
key-uniformity refactor of `_catalog_preserved` is internal).
