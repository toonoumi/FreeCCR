"""Named adjustment profiles — user looks offered in the Color Profile dropdown.

A profile is a snapshot of chosen setting groups of an image's GLOBAL layer
(the same groups "Sync to All" / "Copy Settings" use, minus geometry), saved
under a user-chosen name and re-applied from the Color Profile combo next to
Color / Black & White. See spec/adjustment-profiles.md.

Pure JSON helpers only (no Qt): the sliders panel persists the encoded string
in QSettings under adjust/profiles. Group membership is passed in by the
caller as a {group_id: keys} mapping, so this module never needs to know the
panel's group list. Records are validated defensively on decode so a
hand-edited or corrupt settings value can never crash the panel. Names are
unique case-insensitively; lists returned by the helpers are always new lists
sorted case-insensitively by name.
"""
import copy
import json
import math

COLOR_PROFILE_VALUES = ("color", "bw")


def _clean_settings(value):
    """Keep only str -> finite number / bool entries."""
    if not isinstance(value, dict):
        return {}
    out = {}
    for k, v in value.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = int(v) if float(v).is_integer() else float(v)
    return out


def _valid_profile(record):
    """Validate/normalise one record into the canonical dict shape; None if
    unusable (no name, or no groups)."""
    if not isinstance(record, dict):
        return None
    name = str(record.get("name") or "").strip()
    groups = record.get("groups")
    if not name or not isinstance(groups, (list, tuple)):
        return None
    groups = [g for g in dict.fromkeys(groups) if isinstance(g, str) and g]
    if not groups:
        return None
    color_profile = record.get("color_profile")
    if color_profile not in COLOR_PROFILE_VALUES:
        color_profile = "color"
    curves = record.get("curves")
    return {
        "name": name,
        "groups": groups,
        "color_profile": color_profile,
        "settings": _clean_settings(record.get("settings")),
        "curves": copy.deepcopy(curves) if isinstance(curves, dict) and curves else None,
        "created": str(record.get("created") or ""),
    }


def _sorted_profiles(profiles):
    return sorted(profiles, key=lambda p: p["name"].casefold())


def decode_profiles(text):
    """JSON string → validated, name-sorted list of profile dicts. Corrupt
    JSON or a non-list top level → []; bad records are dropped individually
    and case-insensitive duplicate names keep the first occurrence."""
    try:
        data = json.loads(text) if text else []
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out, seen = [], set()
    for record in data:
        profile = _valid_profile(record)
        if profile is None or profile["name"].casefold() in seen:
            continue
        seen.add(profile["name"].casefold())
        out.append(profile)
    return _sorted_profiles(out)


def encode_profiles(profiles):
    """Inverse of decode_profiles (records re-validated on the way out)."""
    clean = [p for p in map(_valid_profile, profiles or []) if p is not None]
    return json.dumps(clean)


def find_profile(profiles, name):
    """Case-insensitive lookup by name; None when absent."""
    key = str(name or "").strip().casefold()
    if not key:
        return None
    for profile in profiles or []:
        if profile["name"].casefold() == key:
            return profile
    return None


def upsert_profile(profiles, profile):
    """Insert `profile`, replacing any same-named record (case-insensitive).
    Returns a NEW sorted list; raises ValueError on an invalid record."""
    profile = _valid_profile(profile)
    if profile is None:
        raise ValueError("invalid adjustment profile record")
    key = profile["name"].casefold()
    out = [p for p in (profiles or []) if p["name"].casefold() != key]
    out.append(profile)
    return _sorted_profiles(out)


def remove_profile(profiles, name):
    """Drop the record matching `name` (case-insensitive); returns a NEW list."""
    key = str(name or "").strip().casefold()
    return [p for p in (profiles or []) if p["name"].casefold() != key]


def _profile_keys(profile, group_keys):
    """Adjustment keys covered by the profile's (known) groups."""
    return [k for gid in profile["groups"] for k in group_keys.get(gid, ())]


def build_profile(name, groups, settings, color_profile, group_keys,
                  default_for, created=""):
    """Snapshot `settings` (an image's global adjustment dict) for `groups`.
    Every key of every selected group is recorded, defaults filled in, so the
    profile is complete regardless of which keys the source dict carried."""
    record = {"name": name, "groups": list(groups),
              "color_profile": color_profile, "created": created}
    keys = [k for gid in groups for k in group_keys.get(gid, ())]
    snap = {}
    for k in keys:
        if k == "cineon_log":
            snap[k] = bool(settings.get("cineon_log"))
        else:
            snap[k] = settings.get(k, default_for(k))
    record["settings"] = snap
    record["curves"] = (copy.deepcopy(settings.get("curves"))
                        if "curves" in groups and settings.get("curves") else None)
    return _valid_profile(record)


def merge_profile_settings(current, profile, group_keys, adjustment_keys,
                           default_for):
    """The image's new global dict after applying `profile`.

    Same contract as paste_adjustment_settings: a COMPLETE dict (every
    adjustment key, missing ones filled with defaults), the profile's group
    keys overwritten, everything else kept. Curves and the Cineon flag live
    outside adjustment_keys and follow the profile only when their group
    ("curves" / "channels") is in it; otherwise the target's own value is
    carried across."""
    current = current or {}
    settings = profile["settings"]
    merged = {k: current.get(k, default_for(k)) for k in adjustment_keys}
    for k in _profile_keys(profile, group_keys):
        if k in merged:
            merged[k] = settings.get(k, default_for(k))
    if "curves" in profile["groups"]:
        if profile.get("curves"):
            merged["curves"] = copy.deepcopy(profile["curves"])
    elif current.get("curves") is not None:
        merged["curves"] = current["curves"]
    cineon = (settings.get("cineon_log") if "channels" in profile["groups"]
              else current.get("cineon_log"))
    if cineon:
        merged["cineon_log"] = True
    return merged


def profile_matches(current, color_profile, profile, group_keys, default_for):
    """True when every value the profile sets already equals the image's.
    Drives the dropdown's derived selection — nothing records which profile
    was applied, so this is the only source of truth."""
    current = current or {}
    settings = profile["settings"]
    groups = profile["groups"]
    if "profile" in groups and color_profile != profile["color_profile"]:
        return False
    for k in _profile_keys(profile, group_keys):
        if k == "cineon_log":
            if bool(current.get(k)) != bool(settings.get(k)):
                return False
        elif current.get(k, default_for(k)) != settings.get(k, default_for(k)):
            return False
    if "curves" in groups and (current.get("curves") or None) != (profile.get("curves") or None):
        return False
    return True
