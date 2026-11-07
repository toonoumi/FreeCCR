#!/usr/bin/env python3
"""Guard against decoding with a LibRaw whose Fujifilm X-T3/X-T4 family
colour matrix has the wrong sign.

LibRaw 0.21 (rawpy <= 0.25) ships the blue row of that matrix as
(+580, 1303, 5980)/10000, where Adobe's DNG matrix (and RawTherapee's
camconst.json) is (-580, 1303, 5980). The flipped coefficient pulls blue out
of warm colours, so every X-T3/X-T4/X-T30/X100V/X-S10 RAW rendered skin tones
yellow (skin hue 33.5 deg vs 26.7 deg in the camera JPEG; 27.4 deg with the
sign corrected). LibRaw 0.22 (rawpy 0.28) fixes it. requirements.txt pins
rawpy>=0.28.0; Intel macOS is exempt because no newer wheel exists there.
"""

import platform
import sys

import pytest
import rawpy

INTEL_MAC = sys.platform == "darwin" and platform.machine() == "x86_64"


@pytest.mark.skipif(INTEL_MAC, reason="no rawpy >= 0.28 wheel for x86_64 macOS")
def test_libraw_has_the_fixed_fujifilm_matrix():
    assert tuple(rawpy.libraw_version) >= (0, 22), (
        f"LibRaw {rawpy.libraw_version} has the sign-flipped X-T3/X-T4 colour "
        f"matrix (yellow skin tones); install rawpy>=0.28.0")
