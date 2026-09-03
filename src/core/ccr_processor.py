import numpy as np
import cv2
import os
import threading
import time
import tifffile
import gc

from core.color_management import apply_export_colorspace, inject_jpeg_icc

# Try to import PyOpenCL, but handle gracefully if not available
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    print("PyOpenCL not available. GPU acceleration will be disabled.")
    OPENCL_AVAILABLE = False
    cl = None
    cl_array = None

# The hi-res zoom worker can call adjust_image_opencl concurrently with the
# GUI thread; pyopencl command queues are not safe for concurrent submission.
_opencl_lock = threading.Lock()

# Global OpenCL cache
_opencl_cache = {
    'ctx': None,
    'queue': None,
    'program': None,
    'kernel': None,
    'device_name': None
}

def _initialize_opencl():
    """
    Initialize OpenCL environment and compile kernel once. Cache the results.
    Returns True if successful, False otherwise.
    """
    global _opencl_cache
    
    # Check if PyOpenCL is available
    if not OPENCL_AVAILABLE:
        return False
    
    # Check if already initialized
    if _opencl_cache['program'] is not None:
        return True
    
    try:
        # Setup OpenCL context and queue automatically
        platforms = cl.get_platforms()
        if not platforms:
            print("No OpenCL platforms found")
            return False
        
        # Use the first available platform and device
        platform = platforms[0]
        devices = platform.get_devices()
        if not devices:
            print("No OpenCL devices found")
            return False
        
        device = devices[0]
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)
        
        # OpenCL kernel that exactly matches the CPU version logic
        kernel_code = """
        __kernel void adjust(
            __global float *img,
            __global float *params,
            __global float *band_lut,
            int n_pixels
        ) {
            int gid = get_global_id(0);
            if (gid >= n_pixels) return;

            float kelvin_shift = params[0];
            float tint_shift = params[1];
            float exposure = params[2];
            float brightness = params[3];
            float blackpoint = params[4];
            float whitepoint = params[5];
            float contrast = params[6];
            float saturation = params[7];
            float balance_factor = params[8];  // Global balance factor calculated on CPU
            float highlights = params[9];
            float shadows = params[10];
            float ch_input_gain   = params[11];
            float ch_master_shift = params[12];
            float ch_master_gain  = params[13];
            float ch_r_shift      = params[14];
            float ch_r_gain       = params[15];
            float ch_r_blackpoint = params[16];
            float ch_g_shift      = params[17];
            float ch_g_gain       = params[18];
            float ch_g_blackpoint = params[19];
            float ch_b_shift      = params[20];
            float ch_b_gain       = params[21];
            float ch_b_blackpoint = params[22];
            float sub_saturation  = params[23];

            int idx = gid * 3;
            float r = img[idx];
            float g = img[idx+1];
            float b = img[idx+2];

            // Channel Levels — the FIRST adjustment stage, ahead of White
            // Balance and the whole look domain. Three ordered sub-stages:
            // Input Gain (uniform) -> per-channel Shift/Gain/Blackpoint ->
            // Master Shift/Gain (uniform, its own stage — NOT summed into the
            // per-channel values as it used to be).
            //
            // Only a NON-windowed base reaches the kernel with these non-zero:
            // on a windowed base the numpy pre-stage runs them un-clamped before
            // the window clamp and zeroes them here, so parity is automatic.
            // Non-windowed paths carry no sub-black data, so this clamps to [0,1].
            // Divisors mirror CH_SLIDER_DIV / CH_INPUT_GAIN_DIV / CH_MIN_RANGE,
            // textually substituted at build time so they cannot drift.
            if (ch_input_gain != 0.0f || ch_master_shift != 0.0f || ch_master_gain != 0.0f ||
                ch_r_shift != 0.0f || ch_r_gain != 0.0f || ch_r_blackpoint != 0.0f ||
                ch_g_shift != 0.0f || ch_g_gain != 0.0f || ch_g_blackpoint != 0.0f ||
                ch_b_shift != 0.0f || ch_b_gain != 0.0f || ch_b_blackpoint != 0.0f) {

                float ig = pow(2.0f, clamp(ch_input_gain, -100.0f, 100.0f) / CH_INPUT_GAIN_DIV);
                float rs  = clamp(ch_r_shift,      -100.0f, 100.0f) / CH_SLIDER_DIV;
                float rg  = clamp(ch_r_gain,       -100.0f, 100.0f) / CH_SLIDER_DIV;
                float rbp = clamp(ch_r_blackpoint, -100.0f, 100.0f) / CH_SLIDER_DIV;
                float gs  = clamp(ch_g_shift,      -100.0f, 100.0f) / CH_SLIDER_DIV;
                float gg  = clamp(ch_g_gain,       -100.0f, 100.0f) / CH_SLIDER_DIV;
                float gbp = clamp(ch_g_blackpoint, -100.0f, 100.0f) / CH_SLIDER_DIV;
                float bs  = clamp(ch_b_shift,      -100.0f, 100.0f) / CH_SLIDER_DIV;
                float bg  = clamp(ch_b_gain,       -100.0f, 100.0f) / CH_SLIDER_DIV;
                float bbp = clamp(ch_b_blackpoint, -100.0f, 100.0f) / CH_SLIDER_DIV;
                float ms  = clamp(ch_master_shift, -100.0f, 100.0f) / CH_SLIDER_DIV;
                float mg  = clamp(ch_master_gain,  -100.0f, 100.0f) / CH_SLIDER_DIV;

                float xr = r / 65535.0f;
                float xg = g / 65535.0f;
                float xb = b / 65535.0f;

                // 1. Input Gain (uniform, before the per-channel work)
                if (ig != 1.0f) { xr *= ig; xg *= ig; xb *= ig; }

                // 2. Per channel. Untouched channels are skipped entirely so the
                //    round-trip can't drift them by 1 LSB.
                if (rs != 0.0f || rg != 0.0f || rbp != 0.0f) {
                    float den = fmax((1.0f - rg) - rbp, CH_MIN_RANGE);
                    if (rs != rbp) { xr += (rs - rbp); }
                    if (den != 1.0f) { xr /= den; }
                }
                if (gs != 0.0f || gg != 0.0f || gbp != 0.0f) {
                    float den = fmax((1.0f - gg) - gbp, CH_MIN_RANGE);
                    if (gs != gbp) { xg += (gs - gbp); }
                    if (den != 1.0f) { xg /= den; }
                }
                if (bs != 0.0f || bg != 0.0f || bbp != 0.0f) {
                    float den = fmax((1.0f - bg) - bbp, CH_MIN_RANGE);
                    if (bs != bbp) { xb += (bs - bbp); }
                    if (den != 1.0f) { xb /= den; }
                }

                // 3. Master shift / gain (uniform, after the per-channel work)
                if (ms != 0.0f) { xr += ms; xg += ms; xb += ms; }
                float mden = fmax(1.0f - mg, CH_MIN_RANGE);
                if (mden != 1.0f) { xr /= mden; xg /= mden; xb /= mden; }

                r = clamp(xr, 0.0f, 1.0f) * 65535.0f;
                g = clamp(xg, 0.0f, 1.0f) * 65535.0f;
                b = clamp(xb, 0.0f, 1.0f) * 65535.0f;
            }

            // Temperature and Tint (Lightroom-like perceptual adjustments)
            if (kelvin_shift != 0.0f || tint_shift != 0.0f) {
                // Calculate luminance for tone-aware masking
                float img_norm_r = r / 65535.0f;
                float img_norm_g = g / 65535.0f;
                float img_norm_b = b / 65535.0f;
                float luminance = img_norm_r * 0.299f + img_norm_g * 0.587f + img_norm_b * 0.114f;
                
                // Create smooth asymmetric tone-aware strength curve (Lightroom-like)
                // Define strength levels for different tonal regions
                float shadow_strength = 0.8f;      // 80% strength in shadows (0-30% luminance)
                float midtone_strength = 1.0f;     // 100% strength in midtones (30-60% luminance)  
                float highlight_strength = 0.25f;  // 25% strength in highlights (60-100% luminance)
                
                // Transition points
                float shadow_to_mid = 0.3f;       // Shadows to midtones transition at 30% luminance
                float mid_to_highlight = 0.6f;    // Midtones to highlights transition at 60% luminance
                
                // Create smooth asymmetric curve using sigmoid blending
                float tone_curve;
                
                if (luminance <= shadow_to_mid) {
                    // Shadow region (0-30%): smooth transition from 80% to 100%
                    float shadow_progress = clamp(luminance / shadow_to_mid, 0.0f, 1.0f);
                    tone_curve = shadow_strength + (midtone_strength - shadow_strength) * shadow_progress;
                } else if (luminance <= mid_to_highlight) {
                    // Midtone region (30-60%): stay at 100% strength
                    tone_curve = midtone_strength;
                } else {
                    // Highlight region (60-100%): smooth sigmoid transition from 100% to 25%
                    float highlight_progress = (luminance - mid_to_highlight) / (1.0f - mid_to_highlight);
                    // Use sigmoid for smooth natural rolloff
                    float sigmoid_factor = 1.0f / (1.0f + exp(-8.0f * (highlight_progress - 0.5f)));
                    tone_curve = midtone_strength - (midtone_strength - highlight_strength) * sigmoid_factor;
                }
                
                // Temperature (R/B scaling with logarithmic perceptual response)
                if (kelvin_shift != 0.0f) {
                    // Map slider values [-100, 100] to Kelvin temperatures [2000K, 8000K]
                    // Neutral point (slider 0) = 5000K
                    float neutral_kelvin = 5000.0f;
                    float current_kelvin = neutral_kelvin + (kelvin_shift / 100.0f) * 3000.0f;
                    
                    // Calculate Kelvin delta from neutral
                    float kelvin_delta = current_kelvin - neutral_kelvin;
                    
                    // Logarithmic scaling for Kelvin - stronger impact at low end
                    float kelvin_abs = fabs(kelvin_delta);
                    
                    // Create logarithmic response curve based on actual Kelvin values
                    // Linear scale: full 3000K swing = 40% R/B shift
                    float perceptual_scale = (kelvin_delta / 3000.0f) * 0.40f;

                    // tone_curve already covers spatial (shadow/highlight) weighting
                    float effective_scale = perceptual_scale * tone_curve;
                    
                    float r_scale = 1.0f + effective_scale;
                    float b_scale = 1.0f - effective_scale;
                    
                    r *= r_scale;  // R
                    b *= b_scale;  // B
                }

                // Tint (G-M scaling with perceptual mapping and enhanced midtone sensitivity)
                if (tint_shift != 0.0f) {
                    // Use the global balance factor calculated on CPU for exact matching
                    // This ensures identical results between CPU and OpenCL versions
                    
                    // Enhanced midtone and skin tone sensitivity for tint
                    // Tint is most visible in skin tones and neutral areas
                    float skin_tone_sensitivity = 1.0f + 0.5f * exp(-12.0f * pow(luminance - 0.35f, 2.0f));  // Peak at 35% luminance
                    
                    // Create perceptual tint curve - stronger response in certain ranges
                    float tint_abs = fabs(tint_shift);
                    float perceptual_tint = 0.0f;
                    if (tint_abs > 0.0f) {
                        // Sigmoid-like curve for tint perception
                        perceptual_tint = tanh(tint_abs * 0.02f) * sign(tint_shift) * 0.18f;
                    }
                    
                    // Apply perceptual tint with tone awareness, balance factor, and skin tone sensitivity
                    float effective_tint = perceptual_tint * tone_curve * balance_factor * skin_tone_sensitivity;
                    
                    // Tint primarily affects green, with complementary adjustments to R/B
                    float g_scale = 1.0f - effective_tint;  // Green channel (inverse of tint shift)
                    float r_tint_scale = 1.0f + (0.3f * effective_tint);  // Slight red compensation
                    float b_tint_scale = 1.0f + (0.3f * effective_tint);  // Slight blue compensation
                    
                    g *= g_scale;  // G
                    r *= r_tint_scale;  // R  
                    b *= b_tint_scale;  // B
                }
            }

            // Gain — shares Channel Levels "Master Gain"'s /300 curve (uniform
            // linear gain out = in / (1 - v/300), hard-clipped to [0,1]), but the
            // Gain slider spans +-200 (3x .. 0.6x) while Master Gain stays +-100.
            if (exposure != 0.0f) {
                float gm = clamp(exposure, -200.0f, 200.0f) / 300.0f;
                float wv = 1.0f - gm;
                r = clamp((r / 65535.0f) / wv, 0.0f, 1.0f) * 65535.0f;
                g = clamp((g / 65535.0f) / wv, 0.0f, 1.0f) * 65535.0f;
                b = clamp((b / 65535.0f) / wv, 0.0f, 1.0f) * 65535.0f;
            }
            
            // Brightness (Adobe-like: lift lower midtones, preserve highlights)
            if (brightness != 0.0f) {
                float img_norm_r = r / 65535.0f;
                float img_norm_g = g / 65535.0f;
                float img_norm_b = b / 65535.0f;
                
                float brightness_scale = brightness / 8.0f;   // -10.0 to +10.0 for -100 to +100
                // The curve parameter: positive = lift, negative = compress
                float curve = 1.0f - 0.3f * brightness_scale;  // 2.2 to 0.8 for -100 to +100

                // Use pow directly like CPU version, without fmax protection
                img_norm_r = pow(img_norm_r, curve);
                img_norm_g = pow(img_norm_g, curve);
                img_norm_b = pow(img_norm_b, curve);
                
                r = img_norm_r * 65535.0f;
                g = img_norm_g * 65535.0f;
                b = img_norm_b * 65535.0f;
            }

            // Highlights / Shadows (anchored per-channel tone-region roll-off)
            // Region bumps are zero at both endpoints so pure black and pure
            // white stay anchored; highlights roll off smoothly below white.
            if (highlights != 0.0f || shadows != 0.0f) {
                float hs_peak = 0.10546875f;  // peak of x^3*(1-x); normalizes bumps to 1.0
                float hs_strength = 0.30f;    // max channel offset at the bump peak
                float h_amt = highlights / 100.0f;
                float s_amt = shadows / 100.0f;

                float xr = r / 65535.0f;
                float omr = 1.0f - xr;
                xr = xr + h_amt * hs_strength * (xr*xr*xr) * omr / hs_peak
                        + s_amt * hs_strength * xr * (omr*omr*omr) / hs_peak;

                float xg = g / 65535.0f;
                float omg = 1.0f - xg;
                xg = xg + h_amt * hs_strength * (xg*xg*xg) * omg / hs_peak
                        + s_amt * hs_strength * xg * (omg*omg*omg) / hs_peak;

                float xb = b / 65535.0f;
                float omb = 1.0f - xb;
                xb = xb + h_amt * hs_strength * (xb*xb*xb) * omb / hs_peak
                        + s_amt * hs_strength * xb * (omb*omb*omb) / hs_peak;

                r = clamp(xr, 0.0f, 1.0f) * 65535.0f;
                g = clamp(xg, 0.0f, 1.0f) * 65535.0f;
                b = clamp(xb, 0.0f, 1.0f) * 65535.0f;
            }

            // Black/White point (Adobe-like: remap input range)
            if (blackpoint != 0.0f || whitepoint != 0.0f) {
                float img_norm_r = r / 65535.0f;
                float img_norm_g = g / 65535.0f;
                float img_norm_b = b / 65535.0f;
                
                // Map [-100, 100] to [0, 0.2] for black, [1, 0.8] for white
                float black_clip = clamp(blackpoint, -100.0f, 100.0f) / 300.0f;
                float white_clip = clamp(whitepoint, -100.0f, 100.0f) / 300.0f;  // -0.2 to +0.2
                float black_val = 0.0f + black_clip;
                float white_val = 1.0f - white_clip;
                float range = white_val - black_val;
                
                // Piecewise linear remap
                if (range > 1e-6f) {
                    img_norm_r = clamp((img_norm_r - black_val) / range, 0.0f, 1.0f);
                    img_norm_g = clamp((img_norm_g - black_val) / range, 0.0f, 1.0f);
                    img_norm_b = clamp((img_norm_b - black_val) / range, 0.0f, 1.0f);
                }
                
                r = img_norm_r * 65535.0f;
                g = img_norm_g * 65535.0f;
                b = img_norm_b * 65535.0f;
            }
            
            // Contrast (continuous S-curve for both positive and negative)
            if (contrast != 0.0f) {
                float img_norm_r = r / 65535.0f;
                float img_norm_g = g / 65535.0f;
                float img_norm_b = b / 65535.0f;
                
                float midpoint = 0.5f;
                // Map contrast [-100, 100] to k [-0.95, 0.95]
                float k = clamp(contrast / 105.0f, -0.95f, 0.95f);
                
                // S-curve: compress for negative, expand for positive, fixed endpoints
                img_norm_r = ((1.0f + k) * (img_norm_r - midpoint)) / (1.0f + k * fabs(img_norm_r - midpoint) * 2.0f) + midpoint;
                img_norm_g = ((1.0f + k) * (img_norm_g - midpoint)) / (1.0f + k * fabs(img_norm_g - midpoint) * 2.0f) + midpoint;
                img_norm_b = ((1.0f + k) * (img_norm_b - midpoint)) / (1.0f + k * fabs(img_norm_b - midpoint) * 2.0f) + midpoint;
                
                r = img_norm_r * 65535.0f;
                g = img_norm_g * 65535.0f;
                b = img_norm_b * 65535.0f;
            }

            // Mid-high tone weighted saturation adjustment
            if (saturation != 0.0f) {
                float img_norm_r = r / 65535.0f;
                float img_norm_g = g / 65535.0f;
                float img_norm_b = b / 65535.0f;
                
                // Convert RGB to grayscale using luminance weights
                float gray = img_norm_r * 0.299f + img_norm_g * 0.587f + img_norm_b * 0.114f;
                
                float luminance_offset = gray - 0.50f;
                float mid_high_weight = exp(-(luminance_offset * luminance_offset) / (0.35f * 0.35f));
                
                // Create dynamic saturation factor based on mid-high tone weighting
                // Maximum effect at 65% luminance, minimal effect in deep shadows/highlights
                float min_saturation_factor = 0.2f;  // 20% of full saturation in extremes
                float saturation_curve = min_saturation_factor + (1.0f - min_saturation_factor) * mid_high_weight;
                
                // Apply the mid-high tone weighted saturation scaling
                float saturation_scale = 1.0f + (saturation / 100.0f);  // Base saturation scale
                float dynamic_saturation_scale = 1.0f + (saturation_scale - 1.0f) * saturation_curve;
                
                // Blend between grayscale and original based on mid-high tone weighted saturation
                img_norm_r = gray + dynamic_saturation_scale * (img_norm_r - gray);
                img_norm_g = gray + dynamic_saturation_scale * (img_norm_g - gray);
                img_norm_b = gray + dynamic_saturation_scale * (img_norm_b - gray);
                
                img_norm_r = clamp(img_norm_r, 0.0f, 1.0f);
                img_norm_g = clamp(img_norm_g, 0.0f, 1.0f);
                img_norm_b = clamp(img_norm_b, 0.0f, 1.0f);
                
                r = img_norm_r * 65535.0f;
                g = img_norm_g * 65535.0f;
                b = img_norm_b * 65535.0f;
            }

            // Subtractive (film-density) saturation: scale each pixel's
            // chromaticity ratios by a power while pinning the dominant
            // channel, so saturation is gained by absorbing light in the
            // other channels (darker, denser colors) instead of adding it.
            if (sub_saturation != 0.0f) {
                float sr = clamp(r / 65535.0f, 0.0f, 1.0f);
                float sg = clamp(g / 65535.0f, 0.0f, 1.0f);
                float sb = clamp(b / 65535.0f, 0.0f, 1.0f);
                float mx = fmax(sr, fmax(sg, sb));
                if (mx > 1e-6f) {
                    float gamma_s = pow(2.0f, sub_saturation / 100.0f);
                    sr = mx * pow(sr / mx, gamma_s);
                    sg = mx * pow(sg / mx, gamma_s);
                    sb = mx * pow(sb / mx, gamma_s);
                }
                r = sr * 65535.0f;
                g = sg * 65535.0f;
                b = sb * 65535.0f;
            }

            // (Channel Levels used to run here. It is now the FIRST stage, at
            // the top of this kernel — see spec/channel-levels-pre-clamp.md.)

            // Per-color-band "Subtractive Saturations". Parameter deltas
            // come from a 720-bin hue LUT computed on the CPU with the
            // exact same blend math as the numpy path, so both paths agree
            // by construction. band_lut layout: [bin*4 + (subsat,sat,
            // bright,hue)]. params[24] != 0 enables the block. Pixels with
            // HSV saturation below the gate floor are skipped untouched —
            // the numpy path's gate zeroes every delta there too.
            if (params[24] != 0.0f) {
                float nr = clamp(r / 65535.0f, 0.0f, 1.0f);
                float ng = clamp(g / 65535.0f, 0.0f, 1.0f);
                float nb = clamp(b / 65535.0f, 0.0f, 1.0f);
                float mx = fmax(nr, fmax(ng, nb));
                float mn = fmin(nr, fmin(ng, nb));
                float ch_delta = mx - mn;
                float hsv_s = (mx > 1e-9f) ? (ch_delta / mx) : 0.0f;
                if (hsv_s > 0.06f) {
                    float hue;
                    if (mx == nr) {
                        hue = 60.0f * fmod((ng - nb) / ch_delta + 6.0f, 6.0f);
                    } else if (mx == ng) {
                        hue = 60.0f * ((nb - nr) / ch_delta + 2.0f);
                    } else {
                        hue = 60.0f * ((nr - ng) / ch_delta + 4.0f);
                    }
                    // Linear interpolation between bins (matching the
                    // numpy path); the last bin wraps back to red.
                    float pos = hue * 2.0f;
                    int b0 = min((int)pos, 719);
                    float frac = pos - (float)b0;
                    int b1 = (b0 == 719) ? 0 : b0 + 1;
                    float d_subsat = mix(band_lut[b0 * 4 + 0],
                                         band_lut[b1 * 4 + 0], frac);
                    float d_sat    = mix(band_lut[b0 * 4 + 1],
                                         band_lut[b1 * 4 + 1], frac);
                    float d_bright = mix(band_lut[b0 * 4 + 2],
                                         band_lut[b1 * 4 + 2], frac);
                    float d_hue    = mix(band_lut[b0 * 4 + 3],
                                         band_lut[b1 * 4 + 3], frac);
                    float gt = clamp((hsv_s - 0.06f) / 0.14f, 0.0f, 1.0f);
                    float gate = gt * gt * (3.0f - 2.0f * gt);

                    float hsv_v = mx;
                    // 0.30f = _BAND_HUE_FULL_SCALE/100 — keep in sync with
                    // the numpy path's constants.
                    hue = fmod(hue + d_hue * 0.30f * gate + 360.0f, 360.0f);
                    hsv_s = clamp(hsv_s * fmax(1.0f + d_sat / 100.0f * gate,
                                               0.0f), 0.0f, 1.0f);
                    hsv_v = clamp(hsv_v * exp2(d_bright / 100.0f * gate),
                                  0.0f, 1.0f);

                    // HSV -> RGB
                    float cc = hsv_v * hsv_s;
                    float hp = hue / 60.0f;
                    float xx = cc * (1.0f - fabs(fmod(hp, 2.0f) - 1.0f));
                    float mm = hsv_v - cc;
                    if (hp < 1.0f)      { nr = cc; ng = xx; nb = 0.0f; }
                    else if (hp < 2.0f) { nr = xx; ng = cc; nb = 0.0f; }
                    else if (hp < 3.0f) { nr = 0.0f; ng = cc; nb = xx; }
                    else if (hp < 4.0f) { nr = 0.0f; ng = xx; nb = cc; }
                    else if (hp < 5.0f) { nr = xx; ng = 0.0f; nb = cc; }
                    else                { nr = cc; ng = 0.0f; nb = xx; }
                    nr += mm; ng += mm; nb += mm;

                    // Film-density subsat with per-pixel strength: pin the
                    // dominant channel, power down the others (the global
                    // sub_saturation model; pow(0, gamma) stays 0).
                    float strength = d_subsat * gate;
                    if (strength != 0.0f) {
                        float m2 = fmax(nr, fmax(ng, nb));
                        if (m2 > 1e-6f) {
                            float gam = exp2(strength / 100.0f);
                            // Snap sub-1e-6 ratios to 0 before the pow, like
                            // the CPU path: HSV round-trip noise on an
                            // exactly-dark channel must stay dark (gam<1 has
                            // unbounded slope at 0), so pow(0,gam)=0 instead
                            // of lifting it tens of counts.
                            float rr = nr / m2, rg = ng / m2, rb = nb / m2;
                            nr = m2 * pow(rr < 1e-6f ? 0.0f : rr, gam);
                            ng = m2 * pow(rg < 1e-6f ? 0.0f : rg, gam);
                            nb = m2 * pow(rb < 1e-6f ? 0.0f : rb, gam);
                        }
                    }
                    r = clamp(nr, 0.0f, 1.0f) * 65535.0f;
                    g = clamp(ng, 0.0f, 1.0f) * 65535.0f;
                    b = clamp(nb, 0.0f, 1.0f) * 65535.0f;
                }
            }

            // Final clamp and store results
            r = clamp(r, 0.0f, 65535.0f);
            g = clamp(g, 0.0f, 65535.0f);
            b = clamp(b, 0.0f, 65535.0f);

            img[idx] = r;
            img[idx+1] = g;
            img[idx+2] = b;
        }
        """

        # Substitute the Channel Levels constants textually so the kernel and the
        # numpy path cannot drift apart (spec/channel-levels-pre-clamp.md §3.3).
        kernel_code = (kernel_code
                       .replace("CH_INPUT_GAIN_DIV", f"{CH_INPUT_GAIN_DIV:.6f}f")
                       .replace("CH_SLIDER_DIV", f"{CH_SLIDER_DIV:.6f}f")
                       .replace("CH_MIN_RANGE", f"{CH_MIN_RANGE:.6f}f"))

        # Compile the program
        program = cl.Program(ctx, kernel_code).build()
        
        # Create and cache the kernel
        kernel = cl.Kernel(program, "adjust")
        
        # Cache everything
        _opencl_cache['ctx'] = ctx
        _opencl_cache['queue'] = queue
        _opencl_cache['program'] = program
        _opencl_cache['kernel'] = kernel
        _opencl_cache['device_name'] = device.name
        
        print(f"OpenCL initialized successfully - Device: {device.name} on platform: {platform.name}")
        return True
        
    except Exception as e:
        print(f"OpenCL initialization failed: {e}")
        return False


def safe_unicode_path(file_path: str) -> str:
    """
    Ensure file path is properly encoded for Unicode support across different systems.
    """
    return os.path.normpath(file_path)


def safe_cv2_imwrite(output_path: str, image: np.ndarray, params=None) -> bool:
    """
    Safe image writing that handles Unicode file paths.
    params: optional cv2 encode parameters, e.g. [cv2.IMWRITE_JPEG_QUALITY, 92]
    """
    output_path = safe_unicode_path(output_path)
    params = params or []
    try:
        # Try normal cv2.imwrite first
        success = cv2.imwrite(output_path, image, params)
        if success:
            return True

        # If that fails, try encoding to bytes and using alternative method
        try:
            # Get file extension
            _, ext = os.path.splitext(output_path)
            # Encode image to memory buffer
            success, buffer = cv2.imencode(ext, image, params)
            if success:
                # Write buffer to file
                with open(output_path, 'wb') as f:
                    f.write(buffer.tobytes())
                return True
        except Exception as e:
            print(f"Failed to write image with Unicode path handling: {output_path}, error: {e}")
            return False
            
    except Exception as e:
        print(f"Failed to write image: {output_path}, error: {e}")
        return False
    
    return False


def safe_tifffile_imwrite(output_path: str, image: np.ndarray, **kwargs) -> bool:
    """
    Safe TIFF writing that handles Unicode file paths.
    """
    output_path = safe_unicode_path(output_path)
    try:
        tifffile.imwrite(output_path, image, **kwargs)
        return True
    except Exception as e:
        print(f"Failed to write TIFF image: {output_path}, error: {e}")
        return False




def _load_export_source(ccr_image, output_path, max_long_side):
    """Load the working image for an export pass.

    For a DOWNSIZED export with no user crop, decode small and resize to the
    export target up front, so the whole normalize + look + adjustments + warp
    pipeline runs at the OUTPUT resolution instead of at full resolution
    followed by a throwaway downscale at the very end. This is the single
    biggest export-speed lever for non-full-size exports.

    - RAW: decode at half size (preview=True) only when the half-size long edge
      is still >= max_long_side, so we never under-deliver resolution; the
      reader then resizes to max_long_side.
    - Non-RAW: preview is ignored by the decoder, but passing max_long_side
      resizes during read all the same.
    - A user crop is excluded: it changes which region maps to max_long_side
      (the late resize handles that), so cropped exports keep decoding full.
    - Full-size exports (max_long_side is None) and the in-app processing path
      (output_path is None) are unchanged.

    Output dimensions are identical to the old path (the trailing
    resize_image_to_max_pixel is kept as a no-op / fine-rotation-expansion
    catch); only the resolution work happens at is reduced.
    """
    if output_path is None:
        return ccr_image.resized_raw
    crop = getattr(ccr_image, "crop_rect", None)
    if max_long_side is not None and crop is None:
        full = getattr(ccr_image, "original_full_size", None)
        half_ok = full is not None and (max(full) // 2) >= int(max_long_side)
        return ccr_image.read_image(ccr_image.file_path, preview=half_ok,
                                    max_long_side=int(max_long_side))
    return ccr_image.read_image(ccr_image.file_path, preview=False)


def ccr_normalize_with_reference(ccr_image,output_path=None,jpg_out=False,jpg_quality=95,max_long_side=None,output_colorspace="srgb",reference_rect=None,fine_rot=None) -> np.ndarray:
    """
    Normalize and align the image using the CCR algorithm, using a reference rectangle
    for percentile calculations instead of a crop factor.

    Args:
        ccr_image: CCRImage object
        reference_rect: (x1, y1, x2, y2) in resized_raw coordinates. When given
            (the conversion_inputs snapshot from convert time) it is used instead
            of the live ccr_image.reference_frame, so a replay reproduces the
            conversion even after the on-canvas frame was cleared or redrawn.
            Coordinates only — the pixel data is always re-read from the file.
        fine_rot: fine rotation angle (hundredths of a degree) to apply to the
            reference image before the percentile crop, as at convert time.
            None falls back to the live angle. The final output orientation
            warp always uses the live angle (it must match the preview).

    Returns:
        np.ndarray: CCR-normalized and inverted image, dtype uint16
    """
    print("Starting CCR normalization...")
    total_start_time = time.time()
    
    # Get the working image
    step_start = time.time()
    img = _load_export_source(ccr_image, output_path, max_long_side)
    if img is None:
        raise ValueError("CCRImage.resized_raw is None")
    print(f"Image loading: {time.time() - step_start:.3f}s")

    # Apply fine rotation rotation
    step_start = time.time()
    fine_angle = ccr_image.fine_rotation_angle / 100.0
    # The reference-crop warp reproduces convert time; the final output warp
    # (fine_angle) tracks the live angle so the export matches the preview.
    ref_fine_angle = fine_angle if fine_rot is None else fine_rot / 100.0
    h_flip = ccr_image.horizontal_mirrored
    v_flip = ccr_image.vertical_mirrored

    # Center of rotation
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    
    if output_path is not None: # this is for output
        img_ref = ccr_image.resize_image_to_max_pixel(img, 1080)
    else:  # this is for processing
        img_ref = ccr_image.resized_raw
    
    # fine Rotation
    if ref_fine_angle != 0:
        center_ref = (img_ref.shape[1] // 2, img_ref.shape[0] // 2)
        w_ref, h_ref = img_ref.shape[1], img_ref.shape[0]
        rot_mat = cv2.getRotationMatrix2D(center_ref, -ref_fine_angle, 1.0)
        img_ref = cv2.warpAffine(img_ref, rot_mat, (w_ref, h_ref), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        # Clean up rotation matrix as it's no longer needed
        del rot_mat
        # print(f"Rotated image by {angle} degrees")
        # if output_path is not None: # this is for output
        #     # when outputting rotate original image as well
        #     rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        #     img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    print(f"Image setup and rotation: {time.time() - step_start:.3f}s")

    # Reference frame — the caller's snapshot wins over the live frame (a
    # right-click clears the live frame without touching the conversion).
    step_start = time.time()
    if reference_rect is None:
        reference_rect = ccr_image.reference_frame
    if reference_rect is None:
        raise ValueError("CCRImage.reference_frame is None")

    mapped_rect = map_rect_to_original(
        ccr_image.resized_raw.shape,
        img_ref.shape,
        reference_rect
    )

    x1, y1, x2, y2 = mapped_rect

    img = img.astype(np.float32, copy=False)
    ref_crop = img_ref[y1:y2, x1:x2]
    print(f"Reference frame setup: {time.time() - step_start:.3f}s")

    # # Show the image using matplotlib for debugging
    # plt.figure(figsize=(8, 8))
    # plt.imshow(to_8bit(ref_crop))
    # plt.title("ref_crop")
    # plt.axis('off')
    # plt.show()

    # Black/white point normalization per channel with three-segment linear compression
    step_start = time.time()
    norm = np.empty_like(img, dtype=np.float32)
    norm_ref = np.empty_like(img_ref, dtype=np.float32)
    for c in range(3):
        ch_crop = ref_crop[..., c]
        # Get percentiles for linear mapping with compressed extremes
        p10 = np.percentile(ch_crop, 1)    # 1st percentile
        p90 = np.percentile(ch_crop, 99)    # 99th percentile  
        
        ch_full = img[..., c]
        ch_full_ref = img_ref[..., c]
        
        # Linear mapping: p10->6086, p90->43882:
        # Formula: output = (input - p10) / (p90 - p10) * (43882 - 6086) + 6086
        np.subtract(ch_full, p10, out=norm[..., c])
        np.divide(norm[..., c], (p90 - p10), out=norm[..., c])
        np.multiply(norm[..., c], (65535 - 8192), out=norm[..., c])
        np.add(norm[..., c], 8192, out=norm[..., c])
        np.clip(norm[..., c], 0, 65535, out=norm[..., c])

        np.subtract(ch_full_ref, p10, out=norm_ref[..., c])
        np.divide(norm_ref[..., c], (p90 - p10), out=norm_ref[..., c])
        np.multiply(norm_ref[..., c], (65535 - 8192), out=norm_ref[..., c])
        np.add(norm_ref[..., c], 8192, out=norm_ref[..., c])
        np.clip(norm_ref[..., c], 0, 65535, out=norm_ref[..., c])
    
    # Clean up intermediate arrays
    del ref_crop
    print(f"BWPN: {time.time() - step_start:.3f}s")
      # Optical density alignment (conservative optimization)
    step_start = time.time()
    ref_norm_crop = norm_ref[y1:y2, x1:x2]
    np.add(ref_norm_crop, 1e-6, out=ref_norm_crop)
    od_crop = -np.log10(ref_norm_crop / 65535.0)
    mean_od_crop = np.mean(od_crop, axis=(0, 1))
    target_mean_od = np.mean(mean_od_crop)
    scaling_factors = target_mean_od / (mean_od_crop + 1e-12)  # Only add division by zero protection

    # Apply scaling to full image (keep original approach)
    norm_full = norm
    np.add(norm_full, 1e-6, out=norm_full)
    od_full = -np.log10(norm_full / 65535.0)
    od_aligned_full = od_full * scaling_factors
      # Clean up intermediate arrays
    del ref_norm_crop, od_crop, mean_od_crop, scaling_factors, od_full
    
    np.power(10, -od_aligned_full, out=od_aligned_full)
    od_aligned_full *= 65535.0
    np.clip(od_aligned_full, 0, 65535, out=od_aligned_full)
    rgb_aligned_full = od_aligned_full.astype(np.uint16, copy=False)

    # Invert
    rgb_inverted_full = 65535 - rgb_aligned_full
    
    # Clean up more intermediate arrays
    del od_aligned_full, rgb_aligned_full, norm, norm_ref
    print(f"ODAI: {time.time() - step_start:.3f}s")

    # # --- Brightness normalization using grayscale ---
    # # Convert to grayscale using standard luminance weights
    # gray = np.dot(rgb_inverted_full[..., :3], [0.299, 0.587, 0.114])

    # # Compute histogram and find the peak (mode)
    # hist, bin_edges = np.histogram(gray, bins=256, range=(0, 65535))
    # peak_bin = np.argmax(hist)
    # peak_value = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0

    # # Target: map histogram peak to 55% of 65535
    # target_peak = 0.55 * 65535

    # # Compute scaling factor to map peak to target
    # brightness_scale = target_peak / (peak_value + 1e-6)
    # rgb_scaled = rgb_inverted_full * brightness_scale

    # # Stretch so the brightest point reaches 65535
    # max_scaled = np.max(rgb_scaled)
    # if max_scaled > 0:
    #     stretch_scale = 65535.0 / max_scaled
    # else:
    #     stretch_scale = 1.0    # rgb_brightness_normalized = np.clip(rgb_scaled * stretch_scale, 0, 65535).astype(np.uint16)    # Apply inverted gamma correction for inverted linear data
    # Since we have inverted linear data, apply inverted gamma 1.5
    rgb_norm = rgb_inverted_full.astype(np.float32) / 65535.0
      # Apply inverted gamma 2.2 (use gamma = 2.2 for inverted image)
    # gamma == 1.0 here, so np.power was a no-op — just clip (saves a full-res
    # transcendental pass per export).
    gamma_corrected = np.clip(rgb_norm, 0.0, 1.0)
    del rgb_norm

    # Convert to LAB-like processing for saturation
    # Calculate luminance using standard weights
    luminance = np.dot(gamma_corrected[..., :3], [0.299, 0.587, 0.114])
    luminance_expanded = np.expand_dims(luminance, axis=-1)
    
    # Create saturation curve that has minimal effect in shadows and stronger effect in midtones/highlights
    # Using power curve: luminance^0.8 gives gentle increase from shadows to highlights
    saturation_curve = np.power(luminance, 0.8)  # Smooth curve from 0 to 1
    del luminance  # Clean up luminance as it's no longer needed
    base_saturation = 1.15  # 15% maximum saturation increase

    # Calculate dynamic saturation factor: minimal in shadows (1.02), full in highlights (1.12)
    min_saturation = 1.00  # 2% minimum saturation in pure shadows
    saturation_range = base_saturation - min_saturation  # 0.10 range
    dynamic_saturation = min_saturation + saturation_range * saturation_curve
    del saturation_curve
    dynamic_saturation = np.expand_dims(dynamic_saturation, axis=-1)
    
    # Apply luminance-aware saturation by blending between grayscale and color
    gamma_corrected = luminance_expanded + dynamic_saturation * (gamma_corrected - luminance_expanded)
    del luminance_expanded, dynamic_saturation
    gamma_corrected = np.clip(gamma_corrected, 0.0, 1.0)
    
    # Convert back to 16-bit and assign to rgb_brightness_normalized
    # rgb_brightness_normalized = np.clip(gamma_corrected * 65535.0, 0, 65535).astype(np.uint16)

        # Shadow-specific color correction: add warmth and green to dark shadows only
    # Convert back to normalized for shadow correction
    shadow_corrected = gamma_corrected
    # Calculate luminance for curve-based shadow correction
    shadow_luminance = np.dot(shadow_corrected[..., :3], [0.299, 0.587, 0.114])
    
    # Create smooth exponential curves that naturally target shadows
    # These curves provide maximum effect in deep shadows and fade smoothly to highlights
    
    # Warmth curve: exponential decay from shadows (stronger effect in darker areas)
    warmth_curve = np.exp(-shadow_luminance * 4.0)  # Exponential decay, strong in shadows
    warmth_strength = 0.35 * warmth_curve  # 30% max correction in pure black
    del warmth_curve  # Clean up as it's no longer needed
    # Green tint curve: similar but with different decay rate for natural look
    green_curve = np.exp(-shadow_luminance * 3.5)  # Slightly different curve shape
    del shadow_luminance  # Clean up as it's no longer needed
    green_strength = 0.15 * green_curve  # 12% max correction in pure black
    del green_curve  # Clean up as it's no longer needed

    # Apply corrections using smooth curves (no masks or conditionals)
    shadow_corrected[..., 0] *= (1.0 + warmth_strength * 0.8)  # Red: moderate warmth boost
    shadow_corrected[..., 1] *= (1.0 + green_strength)  # Green: boost to counter magenta
    shadow_corrected[..., 2] *= (1.0 - warmth_strength)  # Blue: reduce to counter blue cast
    del warmth_strength, green_strength  # Clean up as they're no longer needed
    
    # Convert back to 16-bit
    rgb_brightness_normalized = np.clip(shadow_corrected * 65535.0, 0, 65535).astype(np.uint16)

    del shadow_corrected, gamma_corrected  # Clean up as they're no longer needed

    # Clean up rgb_inverted_full as it's no longer needed
    del rgb_inverted_full
    gc.collect()
    # --- End of brightness normalization ---

    # Reference-frame conversion does not (yet) use the windowed working space —
    # its base is full-range, so clear the flag (handles a bw→ref re-convert).
    ccr_image._ws_windowed = False

    # --- apply user adjustments --- only when outputting
    step_start = time.time()
    if output_path is not None:  # this is for processing
        rgb_brightness_normalized=ccr_image.apply_adjustments(rgb_brightness_normalized)
    print(f"User adjustments: {time.time() - step_start:.3f}s")

    # --- End of user adjustments ---

    if output_path is not None:  # this is for output
        # User crop (normalized rect in un-rotated/un-flipped space) — applied
        # before flips/rotation so it matches the cropped preview orientation.
        rgb_brightness_normalized = apply_crop_to_image(
            rgb_brightness_normalized, getattr(ccr_image, 'crop_rect', None),
            getattr(ccr_image, 'crop_angle', 0.0))
        step_start = time.time()
        # Apply flips and rotation to rgb_brightness_normalized before export
        if h_flip and v_flip:
            rgb_brightness_normalized = cv2.flip(rgb_brightness_normalized, -1)
        elif h_flip:
            rgb_brightness_normalized = cv2.flip(rgb_brightness_normalized, 1)
        elif v_flip:
            rgb_brightness_normalized = cv2.flip(rgb_brightness_normalized, 0)

        # --- ADD THIS BLOCK: rotate pixels for 90/180/270 degree rotation ---
        angle = ccr_image.rotation_angle % 360
        if angle == 90:
            # Rotate 90 degrees clockwise
            rgb_brightness_normalized = np.rot90(rgb_brightness_normalized, k=3)
        elif angle == 180:
            # Rotate 180 degrees
            rgb_brightness_normalized = np.rot90(rgb_brightness_normalized, k=2)
        elif angle == 270:
            # Rotate 270 degrees clockwise (or 90 degrees CCW)
            rgb_brightness_normalized = np.rot90(rgb_brightness_normalized, k=1)
        # --- END BLOCK ---
        print(rgb_brightness_normalized.shape)
        print(f"Flips and rotation transforms: {time.time() - step_start:.3f}s")

        print(f"Rotated image by {angle} degrees (no crop)")
        step_start = time.time()
        if output_path is not None: # this is for output
            # when outputting rotate original image as well
            h, w = rgb_brightness_normalized.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, -fine_angle, 1.0)
            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])
            new_w = int(w * abs_cos + h * abs_sin)
            new_h = int(h * abs_cos + w * abs_sin)
            rot_mat[0, 2] += (new_w - w) / 2
            rot_mat[1, 2] += (new_h - h) / 2
            try:
                rgb_brightness_normalized = cv2.warpAffine(
                    rgb_brightness_normalized, rot_mat, (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
            except Exception as e:
                print(f"Warning: warpAffine failed due to image size or memory error: {e}")
            # Clean up rotation variables
            del rot_mat, center, abs_cos, abs_sin
        print(f"Final rotation: {time.time() - step_start:.3f}s")

            # angle = ccr_image.rotation_angle
            # if angle != 0:
            #     rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
            #     rgb_brightness_normalized = cv2.warpAffine(
            #         rgb_brightness_normalized,
            #         rot_mat,
            #         (w, h),
            #         flags=cv2.INTER_LINEAR,
            #         borderMode=cv2.BORDER_CONSTANT,
            #         borderValue=0
            #     )

        # Ensure output_path has proper extension and handle Unicode
        step_start = time.time()
        write_export_image(ccr_image, rgb_brightness_normalized, output_path,
                           jpg_out, jpg_quality, max_long_side, output_colorspace)
        print(f"File saving: {time.time() - step_start:.3f}s")
        #debug ----------------------v

        # img_disp = to_8bit(rgb_brightness_normalized)
        # # Draw the mapped reference rectangle on the display
        
        # cv2.rectangle(img_disp, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        # if img_disp.ndim == 2:
        #     img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)
        # plt.figure(figsize=(8, 8))
        # plt.imshow(img_disp)
        # plt.title("CCR Normalized Image")
        # plt.axis('off')
        # plt.show()
        total_elapsed = time.time() - total_start_time
        print(f"TOTAL CCR normalization time: {total_elapsed:.3f}s")
        gc.collect()
        return None  # Return None for output processing

    total_elapsed = time.time() - total_start_time
    print(f"TOTAL CCR normalization time: {total_elapsed:.3f}s")
    return rgb_brightness_normalized

# Calibrated DEFAULT SLOPE for the default-slope mode (used when only a black
# point is sampled — no white point). The conversion runs in optical-DENSITY
# (log) space so that re-sampling the black point under a new light source
# cancels that light's per-channel scaling: log10(base/img) is invariant to a
# per-channel factor k applied to both base and img. The slope is a SINGLE
# SCALAR (contrast only) so no per-channel colour balance is baked in — all
# balance comes from the per-light black-point divide. Calibrated as the mean
# of the per-channel "DENSITY" values that log_bwpoint_slopes reports across
# typical rolls / light sources (~0.69–1.10 observed → ~0.8). Adjust contrast
# by changing this scalar (or with the contrast slider); white balance / the
# Auto WB picker cleans up any residual per-image cast.
DEFAULT_DENSITY_SLOPE = 0.8
# Optional display gamma applied after the density map (1.0 = off / pure
# density-linear). Raise toward ~2.2 only if the default render is too dark.
DEFAULT_DENSITY_GAMMA = 1.0
# Scan floor so log10(base/img) stays finite for near-black (very dense) pixels.
_DENSITY_FLOOR = 1.0


# --- Sprocket-hole / clear-film white mask (spec/sprocket-hole-mask.md) ---
# On a 135 negative the punched sprocket holes and clear rebate scan BRIGHTER than
# the sampled film base, so after inversion they clamp to pure black. This optional
# reversal-film look paints those clear-film regions WHITE as the last render step.
# The mask is derived from the RAW scan ("clearer than the sampled black point plus
# a padding") and cleaned so the holes stay SHARP: noise specks are dropped by
# connected-component area (no morphological erosion that would round corners),
# interior gaps (dust/markings inside a hole) are filled by a flood-fill imfill
# (no dilation that would round/expand the outer edge), and only a light 1 px
# feather is applied for anti-aliasing. The threshold is resolution-independent;
# the area cutoff and feather scale with the buffer's long side so the preview,
# hi-res zoom, and export agree geometrically.
def _sprocket_cfg():
    """Read the (env-overridable) mask parameters. Fraction/abs padding are per-
    channel; min-area/feather are quoted at SPROCKET_REF_LONG and scaled per buffer."""
    def _f(name, default):
        try:
            return float(os.environ.get(name, "").strip() or default)
        except ValueError:
            return default
    return {
        "pad_frac": _f("FREECCR_SPROCKET_PAD_FRAC", 0.20),
        "pad_abs":  _f("FREECCR_SPROCKET_PAD_ABS", 0.02),      # fraction of full scale
        "min_area_px": _f("FREECCR_SPROCKET_MIN_AREA_PX", 24.0),  # speckle cutoff @1080
        "feather_px": _f("FREECCR_SPROCKET_FEATHER_PX", 1.0),  # anti-alias only @1080
    }


SPROCKET_REF_LONG = 1080   # long side the px params are quoted at


def _odd_ksize(radius: float) -> int:
    """Odd cv2 kernel size (>= 3) covering `radius` px each side of the centre."""
    return max(3, int(round(radius)) * 2 + 1)


def _fill_interior_holes(mask_u8):
    """Fill regions of 0 fully enclosed by 1 (interior holes) WITHOUT touching the
    outer boundary — classic imfill via a border flood fill. Unlike a
    morphological close it never dilates or rounds the true edge, so the sprocket
    holes stay sharp. A 1 px background pad guarantees the flood seed is outside
    any hole even when a hole touches the frame edge."""
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    ffmask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 255)          # reachable-from-border background
    holes = cv2.bitwise_not(ff)                      # everything NOT reachable = holes
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def compute_sprocket_alpha(raw_bgr, black_point_bgr):
    """A white-mask alpha (uint8 H×W, 0..255) marking clear-film (sprocket /
    rebate) regions — pixels clearer than the sampled film base by a padding — or
    None if the black point is unset or nothing qualifies. The holes are kept
    sharp (area-based speckle removal + interior-hole fill, only a light feather).

    `raw_bgr` is the working scan (BGR, the same array the invert helpers consume);
    `black_point_bgr` is the sampled clear/film-base anchor (BGR, HIGH values).
    Pure/thread-safe. See spec/sprocket-hole-mask.md §4.1."""
    if black_point_bgr is None or raw_bgr is None:
        return None
    cfg = _sprocket_cfg()
    d = raw_bgr.astype(np.float32, copy=False)
    if d.ndim != 3 or d.shape[2] < 3:
        return None
    bp = np.asarray(black_point_bgr, dtype=np.float32)[:3]
    head = np.maximum(65535.0 - bp, 1.0)                 # per-channel headroom above base
    pad = np.maximum(cfg["pad_frac"] * head, cfg["pad_abs"] * 65535.0)
    thr = bp + pad                                       # (3,) BGR
    # AND across channels: clear film is bright in EVERY channel; exposed scene
    # content (even deep shadow) is denser than the zero-exposure base, so it is
    # lower than base in every channel and structurally excluded.
    mask = np.all(d[..., :3] > thr[None, None, :], axis=2)
    if not mask.any():
        return None
    m = (mask.astype(np.uint8)) * 255
    scale = max(d.shape[0], d.shape[1]) / float(SPROCKET_REF_LONG)
    # Drop tiny speckles by connected-component AREA — no morphological erosion,
    # so hole edges/corners keep their true (sharp) shape.
    min_area = max(1, int(round(cfg["min_area_px"] * scale * scale)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num <= 1:
        return None
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0                                         # label 0 is the background
    keep = np.flatnonzero(areas >= min_area)
    if keep.size == 0:
        return None
    m = np.isin(labels, keep).astype(np.uint8) * 255
    # Fill interior holes (dust/markings inside a hole) without rounding the outer
    # boundary — a flood-fill imfill, NOT a morphological close.
    m = _fill_interior_holes(m)
    # Light feather for anti-aliasing only (keeps the holes crisp).
    feather = cfg["feather_px"] * scale
    if feather >= 0.5:
        ksz = _odd_ksize(feather)
        m = cv2.GaussianBlur(m, (ksz, ksz), feather / 2.0)
    return m


def apply_sprocket_mask(rgb_u16, alpha_u8):
    """Composite the clear-film regions to white (65535): alpha 0 keeps the image,
    255 is full white, in-between is the feathered blend. Returns the input
    unchanged when `alpha_u8` is None or shapes don't match. Channel-order-agnostic
    (paints neutral white). See spec/sprocket-hole-mask.md §4.1."""
    if alpha_u8 is None or rgb_u16 is None:
        return rgb_u16
    if alpha_u8.shape[:2] != rgb_u16.shape[:2]:
        return rgb_u16
    a = alpha_u8.astype(np.float32)[..., None] / 255.0
    out = rgb_u16.astype(np.float32) * (1.0 - a) + 65535.0 * a
    return np.clip(out, 0, 65535).astype(np.uint16)


# --- Working space with headroom (spec/working-space-headroom.md) ---
# The inverted "base" buffer reserves a narrow DISPLAY WINDOW inside the 16-bit
# container and keeps out-of-window data as recoverable headroom instead of
# hard-clipping at white. Display/export render only the window; the White Point
# slider recovers headroom (it runs un-clamped, BEFORE the window clamp). ON by
# default on a normal launch; set FREECCR_WORKING_SPACE=0 to force legacy full-range
# (byte-identical to before). A neutral conversion looks the same either way (only
# 10-bit-quantized in the window); the difference is recoverable headroom.
def _ws_enabled() -> bool:
    return os.environ.get("FREECCR_WORKING_SPACE", "1").strip().lower() not in (
        "0", "false", "no", "off", "")


# Window geometry: a `_WS_BITS`-wide window (default 10-bit = 1024 codes) placed
# low in the container, leaving `_WS_LO` display units of shadow margin below
# black and the remainder (~6 stops at 10-bit) as highlight headroom above white.
_WS_BITS = int(os.environ.get("FREECCR_WS_BITS", "10"))
# 1.0 display unit of shadow margin (~4 stops of sub-base density at
# DEFAULT_DENSITY_SLOPE=0.8). Channel Levels' shift spans +-0.667, so the older
# 0.5 margin was narrower than the control that reaches into it. Widening it is
# nearly free: the window WIDTH is unchanged (1024 codes), so display precision
# is identical and highlight headroom only drops 5.989 -> 5.977 stops.
# See spec/channel-levels-pre-clamp.md §3.5.
_WS_LO = float(os.environ.get("FREECCR_WS_LO", "1.0"))
_WS_WIDTH = float(1 << _WS_BITS)                   # window width in codes (1024)
WS_B = _WS_LO * _WS_WIDTH                           # code for display-black (d=0)
WS_W = (1.0 + _WS_LO) * _WS_WIDTH                   # code for display-white (d=1)
_WS_INV_WIDTH = 1.0 / (WS_W - WS_B)                 # == 1/_WS_WIDTH
# Highlight headroom, in stops, between display-white and the container ceiling
# (~6 at the 10-bit default). The White Point recovery slider spans exactly this
# range, so WP=-100 maps the ceiling back to white (recovers ALL headroom).
_WS_HEADROOM_STOPS = float(np.log2((65535.0 - WS_B) / (WS_W - WS_B)))


def encode_window(d: np.ndarray) -> np.ndarray:
    """Map display values `d` (0=black, 1=white, may overshoot either end) into
    windowed uint16 container codes, clamped to [0,65535]. Only data beyond the
    container (≈+6 stops / −0.5 below black at the 10-bit default) is discarded;
    everything between white and the container top survives as recoverable
    headroom. `d` may be modified in place."""
    code = d
    code *= np.float32(WS_W - WS_B)
    code += np.float32(WS_B)
    np.clip(code, 0.0, 65535.0, out=code)
    return code.astype(np.uint16)


_WB_TEMP_STRENGTH = 0.40   # full-slider R/B scale: s = (slider/100)*0.40 (matches
                           #   the previous tone-aware WB midtone strength)
_WB_TINT_STRENGTH = 0.26   # = 0.18 * skin(0.45): the previous tint midtone strength


def _white_balance_gains(kelvin_shift: float, tint_shift: float,
                         tint_balance_factor: float = 1.0):
    """Flat per-channel white-balance gains (gr, gg, gb) for Temperature/Tint.
    Temperature: s=(kelvin/100)*0.40 -> R*(1+s), B*(1-s). Tint:
    t=tanh(tint*0.02)*0.26*balance -> G*(1-t), R*(1+0.3t), B*(1+0.3t). Neutral
    (0,0) -> (1,1,1). See spec/working-space-white-balance.md."""
    gr = gg = gb = 1.0
    if kelvin_shift != 0.0:
        s = (kelvin_shift / 100.0) * _WB_TEMP_STRENGTH
        gr *= (1.0 + s)
        gb *= (1.0 - s)
    if tint_shift != 0.0:
        t = float(np.tanh(tint_shift * 0.02)) * _WB_TINT_STRENGTH * tint_balance_factor
        gg *= (1.0 - t)
        gr *= (1.0 + 0.3 * t)
        gb *= (1.0 + 0.3 * t)
    return float(gr), float(gg), float(gb)


# --- Channel Levels: the FIRST adjustment stage (spec/channel-levels-pre-clamp.md)
# Slider -> parameter mappings. All three sliders share one divisor so the
# controls stay predictable relative to each other.
CH_SLIDER_DIV = 150.0       # shift/gain/blackpoint: slider +-100 -> +-0.667
CH_INPUT_GAIN_DIV = 25.0    # input gain: slider +-100 -> 2^+-4 (+-4 stops)
# Floor on the per-channel range (1-gain)-black. At CH_SLIDER_DIV=150 the pair
# (gain=+100, blackpoint=+100) gives (1-0.667)-0.667 = -0.333, and a NEGATIVE
# denominator inverts the channel. The old /300 scale bottomed out at +0.333, so
# it never needed this guard.
CH_MIN_RANGE = 0.1          # => at most a 10x per-channel gain


def _apply_channel_levels(d: np.ndarray,
                          ch_input_gain: float, ch_master_shift: float,
                          ch_master_gain: float,
                          ch_r_shift: float, ch_r_gain: float, ch_r_blackpoint: float,
                          ch_g_shift: float, ch_g_gain: float, ch_g_blackpoint: float,
                          ch_b_shift: float, ch_b_gain: float, ch_b_blackpoint: float,
                          clamp: bool) -> np.ndarray:
    """Channel Levels, applied to float DISPLAY values `d` (0 = display black,
    1 = display white). Index 0=R, 1=G, 2=B. Modifies `d` in place.

    Three ordered sub-stages:
      1. Input Gain  — a uniform multiplier BEFORE the per-channel work.
      2. Per channel — out = (in + shift - black) / max((1 - gain) - black, MIN).
      3. Master      — a uniform shift + gain AFTER the per-channel work. Master
                       is its own stage, NOT summed into the per-channel values.

    `clamp` is the pipeline-position switch:
      False (windowed base) — no clipping at all, so `d` stays un-clamped for the
        White Balance / White Point / Gain stages and the single window clamp
        that follows. This is what lets a shift TRANSLATE the histogram: content
        below display-black rises into the window instead of being unreachable,
        and content pushed out the bottom lands in the shadow margin rather than
        being destroyed.
      True (full-range base: reference mode, positive mode, area layers) — clip to
        [0,1] on the way out, as before. There is no sub-black data on those paths
        and the later pow() stages must not see negatives."""
    ig = 2.0 ** (float(np.clip(ch_input_gain, -100.0, 100.0)) / CH_INPUT_GAIN_DIV)
    shifts = (float(np.clip(ch_r_shift, -100.0, 100.0)) / CH_SLIDER_DIV,
              float(np.clip(ch_g_shift, -100.0, 100.0)) / CH_SLIDER_DIV,
              float(np.clip(ch_b_shift, -100.0, 100.0)) / CH_SLIDER_DIV)
    gains = (float(np.clip(ch_r_gain, -100.0, 100.0)) / CH_SLIDER_DIV,
             float(np.clip(ch_g_gain, -100.0, 100.0)) / CH_SLIDER_DIV,
             float(np.clip(ch_b_gain, -100.0, 100.0)) / CH_SLIDER_DIV)
    blacks = (float(np.clip(ch_r_blackpoint, -100.0, 100.0)) / CH_SLIDER_DIV,
              float(np.clip(ch_g_blackpoint, -100.0, 100.0)) / CH_SLIDER_DIV,
              float(np.clip(ch_b_blackpoint, -100.0, 100.0)) / CH_SLIDER_DIV)
    ms = float(np.clip(ch_master_shift, -100.0, 100.0)) / CH_SLIDER_DIV
    mg = float(np.clip(ch_master_gain, -100.0, 100.0)) / CH_SLIDER_DIV

    # 1. Input Gain (uniform, before everything else)
    if ig != 1.0:
        d *= np.float32(ig)
    # 2. Per-channel shift / gain / blackpoint. Untouched channels are skipped
    #    entirely so the round-trip can't drift them by 1 LSB.
    for c in range(3):
        s, g, bp = shifts[c], gains[c], blacks[c]
        if s == 0.0 and g == 0.0 and bp == 0.0:
            continue
        den = max((1.0 - g) - bp, CH_MIN_RANGE)
        if s != bp:
            d[..., c] += np.float32(s - bp)
        if den != 1.0:
            d[..., c] /= np.float32(den)
    # 3. Master shift / gain (uniform, after the per-channel work)
    if ms != 0.0:
        d += np.float32(ms)
    mden = max(1.0 - mg, CH_MIN_RANGE)
    if mden != 1.0:
        d /= np.float32(mden)

    if clamp:
        np.clip(d, 0.0, 1.0, out=d)
    return d


def _channel_levels_active(ch_input_gain, ch_master_shift, ch_master_gain,
                           ch_r_shift, ch_r_gain, ch_r_blackpoint,
                           ch_g_shift, ch_g_gain, ch_g_blackpoint,
                           ch_b_shift, ch_b_gain, ch_b_blackpoint) -> bool:
    """True when any Channel Levels slider is off zero (the stage is a no-op
    otherwise, and skipping it keeps a neutral render bit-exact)."""
    return any(p != 0.0 for p in (
        ch_input_gain, ch_master_shift, ch_master_gain,
        ch_r_shift, ch_r_gain, ch_r_blackpoint,
        ch_g_shift, ch_g_gain, ch_g_blackpoint,
        ch_b_shift, ch_b_gain, ch_b_blackpoint))


def _apply_working_space_recovery(img16: np.ndarray, exposure: float,
                                  white_point: float = 0.0,
                                  kelvin_shift: float = 0.0, tint_shift: float = 0.0,
                                  tint_balance_factor: float = 1.0,
                                  ch_input_gain: float = 0.0,
                                  ch_master_shift: float = 0.0,
                                  ch_master_gain: float = 0.0,
                                  ch_r_shift: float = 0.0, ch_r_gain: float = 0.0,
                                  ch_r_blackpoint: float = 0.0,
                                  ch_g_shift: float = 0.0, ch_g_gain: float = 0.0,
                                  ch_g_blackpoint: float = 0.0,
                                  ch_b_shift: float = 0.0, ch_b_gain: float = 0.0,
                                  ch_b_blackpoint: float = 0.0) -> np.ndarray:
    """De-window a windowed base, apply the highlight-recovery controls UN-clamped
    (so headroom can be pulled back below white), then clamp to the display window
    and return a normal full-range [0,65535] positive for the look chain. This
    single numpy pre-stage is shared by the CPU and GPU adjustment paths, so
    GPU/CPU parity is automatic — the kernel never sees headroom.

    Channel Levels runs FIRST here (ahead of White Balance and the recovery
    controls) and un-clamped — see spec/channel-levels-pre-clamp.md.

    Two recovery controls, both neutral at 0 (so a neutral edit is byte-identical):
      White Point — the WIDE-range recovery, stops-based across the FULL headroom:
        WP=-100 maps the container ceiling exactly to white (recovers everything),
        WP=0 neutral, WP>0 pushes highlights up. Perceptual feel: typical blown
        highlights come back within the first chunk of negative travel.
      Gain/Exposure — the existing linear `1/(1−v/300)` gain (±~0.74 stops), kept
        for fine tone control on top of the White Point recovery."""
    d = img16.astype(np.float32)
    d -= np.float32(WS_B)
    d *= np.float32(_WS_INV_WIDTH)
    # Channel Levels - the FIRST adjustment stage, un-clamped so a shift moves the
    # whole histogram: sub-black content (film base) rises into the window instead
    # of being unreachable, and content pushed out the bottom lands in the shadow
    # margin instead of clipping. See spec/channel-levels-pre-clamp.md.
    if _channel_levels_active(ch_input_gain, ch_master_shift, ch_master_gain,
                              ch_r_shift, ch_r_gain, ch_r_blackpoint,
                              ch_g_shift, ch_g_gain, ch_g_blackpoint,
                              ch_b_shift, ch_b_gain, ch_b_blackpoint):
        _apply_channel_levels(d, ch_input_gain, ch_master_shift, ch_master_gain,
                              ch_r_shift, ch_r_gain, ch_r_blackpoint,
                              ch_g_shift, ch_g_gain, ch_g_blackpoint,
                              ch_b_shift, ch_b_gain, ch_b_blackpoint,
                              clamp=False)
    # White balance - flat per-channel gain in the scene-linear working domain
    # (before the window clamp) so a warm/cool shift lands in headroom (recoverable)
    # instead of clipping, and cooling can pull highlights back. Index 0=R,1=G,2=B.
    if kelvin_shift != 0.0 or tint_shift != 0.0:
        gr, gg, gb = _white_balance_gains(kelvin_shift, tint_shift, tint_balance_factor)
        if gr != 1.0:
            d[..., 0] *= np.float32(gr)
        if gg != 1.0:
            d[..., 1] *= np.float32(gg)
        if gb != 1.0:
            d[..., 2] *= np.float32(gb)
    if white_point != 0.0:
        wp = float(np.clip(white_point, -100.0, 100.0))
        d *= np.float32(2.0 ** (_WS_HEADROOM_STOPS * wp / 100.0))   # un-clamped
    if exposure != 0.0:
        white_val = 1.0 - np.clip(exposure, -200.0, 200.0) / 300.0
        d *= np.float32(1.0 / white_val)        # un-clamped: recovers overshoot
    np.clip(d, 0.0, 1.0, out=d)                  # window clamp: enter display range
    d *= np.float32(65535.0)
    return d.astype(np.uint16)


# --- No-anchor inversion: a faithful port of NamiColor's negative transform ---
# Source: github.com/Wavechaser/NamiColor -> NamiColor_dev/NamiColor_dev.c (3.1,
# GPL-3.0), the `neg` branch of transform():
#     inputScale = 16.0 ; invScale = -1.0
#     init = invScale * log10(inputScale * p)      // transmission -> density
#     init = init * inputGain + 1.0
# inputGain is NamiColor's own slider (default 1.0); FreeCCR leaves it at the
# default here because Channel Levels carries its own Input Gain downstream.
# Everything NamiColor does after `init` (channel/master shift, gain, blackpoint)
# is what Channel Levels already does, so the conversion stops here.
NAMI_INPUT_SCALE = 16.0    # DCTL `inputScale` for negatives
NAMI_BASE_OFFSET = 1.0     # DCTL `+ 1.0f` for negatives (reversals use 0.8)


def _namicolor_invert(img_f: np.ndarray) -> np.ndarray:
    """No-anchor inversion in DENSITY (log) space, per NamiColor's negative
    transform: `d = -log10(16 * p) + 1.0`, with `p = v/65535` the LINEAR scene
    value.

    Used when NEITHER a black point nor a white point was sampled. Nothing is
    measured off the frame: the `16` and `+1.0` are NamiColor's fixed constants,
    which place a film base sitting near half scale at roughly Cineon black
    (93/1023). The frame's own cast and placement survive, and the user grades
    them with Channel Levels — which, on this base, is operating in the same
    density space NamiColor's sliders do.

    The negative decode is linear Adobe RGB (`gamma=(1,1)`,
    `output_color=Adobe`, ccr_image.py:538-551), which is exactly the input the
    DCTL expects, so no linearisation is needed here.

    Output leaves [0,1] at both ends, by design: clear film (brighter than the
    assumed base) goes NEGATIVE into the shadow margin, dense areas well above
    white into the highlight headroom. Both are recoverable through Channel
    Levels. See spec/no-anchor-convert.md.

    NOT ported: NamiColor's Adobe->Rec.2020 matrix (it is applied in-place
    sequentially in the DCTL, so the G and B rows consume already-overwritten
    values — a genuine bug that is baked into that look; FreeCCR is colour
    managed and must not reintroduce it), and the optional "Fit to Cineon Base"
    lift (`postLift`, default OFF upstream)."""
    d = np.maximum(img_f, _DENSITY_FLOOR)                    # copy; avoids log10(0)
    d *= np.float32(NAMI_INPUT_SCALE / 65535.0)              # 16 * p
    np.log10(d, out=d)
    np.negative(d, out=d)                                    # density above 1/16 scale
    d += np.float32(NAMI_BASE_OFFSET)
    if _ws_enabled():
        return encode_window(d)
    np.clip(d, 0.0, 1.0, out=d)
    d *= np.float32(65535.0)
    return d.astype(np.uint16)


def _default_slope_invert(img_f: np.ndarray, black_point_bgr,
                          slopes_bgr=None) -> np.ndarray:
    """Density-space inversion for the black-point-only mode. `img_f` is a
    float32 (H,W,3) BGR array.

    Per channel: out = clip(slope[c] * max(log10(base/img), 0), 0, 1),
    then an optional display gamma, scaled to uint16. `slopes_bgr` is a saved
    film-stock preset's per-channel density slopes (spec/film-stock-slopes.md);
    None uses the baked scalar DEFAULT_DENSITY_SLOPE for every channel —
    byte-identical to the pre-preset behaviour. The per-channel base divide
    carries the light source's colour balance (log10(base/img) cancels any
    per-channel light scaling, so re-sampling the black point under a new
    light keeps colour consistent without a white point); a film stock's
    slopes add that stock's per-channel contrast character on top."""
    out = np.empty_like(img_f)
    for c in range(3):
        base = max(float(black_point_bgr[c]), 1.0)
        slope = (DEFAULT_DENSITY_SLOPE if slopes_bgr is None
                 else float(slopes_bgr[c]))
        ch = np.maximum(img_f[..., c], _DENSITY_FLOOR)   # copy; avoids /0 & log(0)
        np.divide(base, ch, out=ch)                      # base / img
        np.log10(ch, out=ch)                             # optical density above base
        ch *= np.float32(slope)                          # slope = contrast (per channel)
        out[..., c] = ch
    if _ws_enabled():
        # Keep highlight overshoot (density above the 1/SLOPE ceiling) as headroom
        # AND sub-base density (img brighter than the black point — film base,
        # rebate, clear film) as shadow margin. Flooring the latter at 0 used to
        # make the film base numerically identical to a true image shadow, so a
        # per-channel shift could not lift one without tinting the other.
        # See spec/channel-levels-pre-clamp.md §3.4.
        if DEFAULT_DENSITY_GAMMA != 1.0:
            # A display gamma is only defined on [0,1]; a fractional power of a
            # negative is NaN. Leave the sub-black region linear (continuous at 0).
            np.power(out, np.float32(1.0 / DEFAULT_DENSITY_GAMMA), out=out,
                     where=(out > 0.0))
        return encode_window(out)
    np.maximum(out, 0.0, out=out)                        # legacy: img brighter than base -> 0
    np.clip(out, 0.0, 1.0, out=out)
    if DEFAULT_DENSITY_GAMMA != 1.0:
        np.power(out, np.float32(1.0 / DEFAULT_DENSITY_GAMMA), out=out)
    out *= np.float32(65535.0)
    return out.astype(np.uint16)


# --- Auto-exposure for default-slope mode (spec/auto-exposure-default-slope.md) ---
AUTO_EXPOSURE_PERCENTILE = 98.0     # top-2% highlight
AUTO_EXPOSURE_TARGET = 0.98         # nominal placement at 98% of full scale
WHITE_EXCLUDE_FRACTION = 0.99       # luminance >= this is treated as holder/clear → excluded
MIN_CONTENT_FRACTION = 0.005        # need >=0.5% non-white pixels to trust the estimate
EXPOSURE_BASE_MIN = -100.0          # exposure-base clamp (-2 EV nominal)
EXPOSURE_BASE_MAX = 100.0           # exposure-base clamp (+2 EV nominal)


def compute_auto_exposure_gain(img_bgr: np.ndarray, ws_windowed: bool = False) -> float:
    """Auto-exposure for default-slope mode: return the EXPOSURE-base value that
    would place the (film-holder-excluded) AUTO_EXPOSURE_PERCENTILE luminance at
    AUTO_EXPOSURE_TARGET of full scale.

    The value rides the Gain/Exposure stage (a uniform `1/(1−v/300)` gain). With
    the working space ON it is applied UN-clamped before the window clamp (see
    _apply_working_space_recovery), so the lifted top lands in highlight headroom
    instead of hard-clipping. Pure-white pixels (luminance >= WHITE_EXCLUDE_FRACTION·
    65535 in display scale) are the film holder / clear surround and are excluded
    so they don't peg the estimate. Returns 0.0 when there isn't enough non-white
    content. When `ws_windowed`, `img_bgr` is a windowed base, so it is decoded to
    the display [0,65535] scale (clamped) first so the percentile target matches
    the legacy full-range behaviour."""
    if ws_windowed:
        img = img_bgr.astype(np.float32)
        img -= np.float32(WS_B)
        img *= np.float32(_WS_INV_WIDTH)
        np.clip(img, 0.0, 1.0, out=img)
        img *= np.float32(65535.0)
    else:
        img = img_bgr.astype(np.float32, copy=False)
    # BGR → luminance.
    lum = 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]
    keep = lum < (WHITE_EXCLUDE_FRACTION * 65535.0)
    vals = lum[keep]
    if vals.size < MIN_CONTENT_FRACTION * lum.size:
        return 0.0
    v98 = float(np.percentile(vals, AUTO_EXPOSURE_PERCENTILE))
    if v98 <= 1.0:
        return 0.0
    g = (AUTO_EXPOSURE_TARGET * 65535.0) / v98
    exposure_base = 50.0 * float(np.log2(g))
    return float(np.clip(exposure_base, EXPOSURE_BASE_MIN, EXPOSURE_BASE_MAX))


# --- Auto Gain (spec/auto-gain.md) -------------------------------------------
# Secretly offset the Gain stage so the top-2% in-bound highlight lands at 95%
# of the working-space window — without moving the Gain slider. "In-bound" = a
# de-windowed display value in [0, 1]: between the sampled clear (→0) and dense
# (→1) conversion points. Over-range / specular pixels (>1, denser than the dense
# sample) and sub-black pixels (<0, clearer than the clear sample / film holder)
# are discarded so they don't drive the gain. Toggleable in Settings → General.
AG_PERCENTILE = 98.0       # top 2% highlight
AG_TARGET = 0.95           # placed at 95% of the window (display white = 1.0)
AG_HI = 1.0                # in-bound ceiling = sampled dense point / SLOPE ceiling
# Master Gain range: v=-100..+100 → g = 1/(1 - v/CH_SLIDER_DIV) = 0.6..3.0. The
# slider spans EXACTLY this gain range, so no achievable gain is clipped away.
AG_GMIN = 0.6
AG_GMAX = 3.0
AG_EPS = 1e-4


def compute_auto_gain_offset(base_u16: np.ndarray, ws_windowed: bool = False) -> float:
    """Return the invisible MASTER GAIN offset that places the AG_PERCENTILE
    in-bound highlight luminance at AG_TARGET of the working-space window.

    The offset rides Channel Levels' Master Gain (g = 1/(1 - v/CH_SLIDER_DIV))
    and is ADDED to the user's Master Gain value; with that slider at 0 the
    realized gain is exactly the clamped g, so the measured highlight lands at
    AG_TARGET. Master Gain is the app's one gain control — the old general-
    adjustments "Gain" slider was the same math at a different scale and is gone.

    Depends only on the base pixels + window geometry (NOT on any slider), so it
    is constant across a slider drag. Index 0=R, 1=G, 2=B. Returns 0.0 when there
    isn't enough in-bound content (mostly headroom/holder). See spec/auto-gain.md."""
    d = base_u16.astype(np.float32)
    if ws_windowed:
        d -= np.float32(WS_B)
        d *= np.float32(_WS_INV_WIDTH)        # de-window; headroom kept (d may exceed 1)
    else:
        d *= np.float32(1.0 / 65535.0)        # legacy full-range base
    lum = 0.299 * d[..., 0] + 0.587 * d[..., 1] + 0.114 * d[..., 2]   # RGB
    keep = (lum >= 0.0) & (lum <= AG_HI)       # in-bound: between clear(0)/dense(1)
    vals = lum[keep]
    if vals.size < MIN_CONTENT_FRACTION * lum.size:
        return 0.0
    p = float(np.percentile(vals, AG_PERCENTILE))
    if p <= AG_EPS:
        return 0.0
    g = float(np.clip(AG_TARGET / p, AG_GMIN, AG_GMAX))
    # Inverse of the Master Gain curve g = 1/(1 - v/CH_SLIDER_DIV). AG_GMIN/GMAX
    # are exactly the slider's endpoints, so this never needs clipping — the
    # clamp is belt-and-braces against a future retune of either constant.
    v = CH_SLIDER_DIV * (1.0 - 1.0 / g)
    return float(np.clip(v, -100.0, 100.0))


def _twopoint_invert(img_f: np.ndarray, black_point_bgr, white_point_bgr,
                     density: bool) -> np.ndarray:
    """Two-point B/W-point inversion → positive uint16, BGR (H,W,3).

    `img_f` is a float32 scan; `black_point_bgr` is the clear/film-base sample
    (HIGH scan value), `white_point_bgr` the dense/exposed sample (LOW value).
    Both modes map the SAME endpoints — clear → black (0), dense → white
    (65535) — and differ only in the curve between them:

    density=True  (opt-in, physically correct): per channel recover optical
      density D = log10(base/img) and normalise by Dmax = log10(base/dense).
      Because the raw scan is linear in transmittance (V ∝ T), this is the true
      density recovery; the normalised density is ALREADY the positive (clear→0,
      dense→1), so there is no separate 65535−x inversion. A per-channel divide
      (the base ratio) carries colour balance with no tone-dependent cast.

    density=False (legacy): an affine stretch in transmittance,
      (img − dense)/(base − dense), then a linear 65535−x invert. Bit-identical
      to the prior two-point behaviour. See spec/density-bwpoint-toggle.md.
    """
    ws = _ws_enabled()
    if ws:
        # Working-space path: build the per-channel DISPLAY positive `d` (0=clear,
        # 1=dense) WITHOUT clipping the top, so density beyond the sampled dense
        # point survives as headroom, then encode into the window.
        d = np.empty_like(img_f)
        for c in range(3):
            base = max(float(black_point_bgr[c]), 1.0)
            dense = max(float(white_point_bgr[c]), 1.0)
            if density:
                dmax = float(np.log10(base / dense)) if base > dense else 0.0
                if dmax <= 1e-6:                        # no usable density range
                    d[..., c] = 0.0
                    continue
                ch = np.maximum(img_f[..., c], _DENSITY_FLOOR)
                np.divide(base, ch, out=ch)
                np.log10(ch, out=ch)
                np.divide(ch, dmax, out=ch)             # clear→0, dense→1 (overshoot kept)
                d[..., c] = ch
            else:
                denom = base - dense
                if abs(denom) < 1.0:
                    d[..., c] = 1.0                      # degenerate → white (matches legacy)
                    continue
                # positive d = 1 − transmittance-stretch (overshoot kept)
                t = (img_f[..., c] - dense) / denom
                np.subtract(1.0, t, out=d[..., c])
        return encode_window(d)

    norm = np.empty_like(img_f)
    for c in range(3):
        base = max(float(black_point_bgr[c]), 1.0)     # clear / film base (HIGH)
        dense = max(float(white_point_bgr[c]), 1.0)    # dense / exposed (LOW)
        if density:
            dmax = float(np.log10(base / dense)) if base > dense else 0.0
            if dmax <= 1e-6:                            # no usable density range
                norm[..., c] = 0.0
                continue
            ch = np.maximum(img_f[..., c], _DENSITY_FLOOR)   # copy; avoids /0 & log(0)
            np.divide(base, ch, out=ch)                # base / img
            np.log10(ch, out=ch)                       # optical density above base
            np.divide(ch, dmax, out=ch)                # normalise: clear→0, dense→1
            np.clip(ch, 0.0, 1.0, out=ch)              # img brighter than base → 0
            np.multiply(ch, 65535.0, out=norm[..., c])  # positive (no extra invert)
        else:
            denom = base - dense
            if abs(denom) < 1.0:
                norm[..., c] = 0.0
                continue
            np.subtract(img_f[..., c], dense, out=norm[..., c])
            np.divide(norm[..., c], denom, out=norm[..., c])
            np.multiply(norm[..., c], 65535.0, out=norm[..., c])
            np.clip(norm[..., c], 0, 65535, out=norm[..., c])
    if density:
        # already oriented as the positive — clip and return.
        return np.clip(norm, 0, 65535).astype(np.uint16)
    return (65535.0 - norm).clip(0, 65535).astype(np.uint16)


def ccr_normalize_with_bwpoint(ccr_image, black_point_bgr=None, white_point_bgr=None,
                               output_path=None, jpg_out=False,
                               jpg_quality=95, max_long_side=None,
                               output_colorspace="srgb", density=False,
                               slopes_bgr=None):
    """
    Film negative conversion using the same pipeline as ccr_normalize_with_reference
    but with explicit per-channel B/W points instead of auto-detected percentiles.

    black_point_bgr: (B,G,R) scan values of transparent/clear film area (HIGH values).
                     Transparent areas → output black (0) after inversion.
                     If None (and white_point_bgr is None too) NO anchor was
                     sampled: the conversion is a plain per-channel flip and the
                     user grades it with Channel Levels. See
                     spec/no-anchor-convert.md.
    white_point_bgr: (B,G,R) scan values of dense/exposed film area (LOW values).
                     Dense areas → output white (65535) after inversion.
                     If None, the default-slope mode is used: a density-space
                     inversion with the baked scalar DEFAULT_DENSITY_SLOPE is
                     applied to the black point alone (see _default_slope_invert).
    density:         two-point mode only — True recovers optical density in log
                     space; False (default) uses the legacy linear transmittance
                     stretch. Density is opt-in; callers pass the live setting.
                     Ignored when white_point_bgr is None (that path is always
                     density). See spec/density-bwpoint-toggle.md.
    slopes_bgr:      black-point-only mode only — a saved film-stock preset's
                     per-channel density slopes replacing the scalar
                     DEFAULT_DENSITY_SLOPE. None = default slope. Ignored when
                     a white point is given (the sampled pair wins). See
                     spec/film-stock-slopes.md.

    Pipeline: BWPN (user B/W points) → inversion → saturation boost → shadow correction
    ODAI is skipped because per-channel B/W point mapping already normalises channels.
    """
    total_start_time = time.time()

    # --- Load working image ---
    img = _load_export_source(ccr_image, output_path, max_long_side)
    if img is None:
        raise ValueError("CCRImage: could not load image data for B/W point conversion")

    # Sprocket-hole / clear-film white mask (reversal look), derived from the RAW
    # scan relative to the sampled black point while the raw is still in hand.
    # Computed regardless of the toggle: the preview caches it (so turning the
    # toggle on later needs no re-conversion) and the export composites it only
    # when enabled. See spec/sprocket-hole-mask.md.
    sprocket_alpha = compute_sprocket_alpha(img, black_point_bgr)

    # Fine rotation uses DISPLAY semantics, like the reference pipeline: the
    # per-pixel B/W-point mapping is rotation-independent, so the preview
    # result stays un-rotated (the canvas item applies the live angle) and an
    # export warps ONCE at the end, on an expanded canvas. Warping here as
    # well used to (a) show the rotation twice on screen — the baked warp
    # plus the display transform — and (b) double-warp exports, with the
    # first warp cropping the corners on the un-expanded canvas.
    fine_angle = ccr_image.fine_rotation_angle / 100.0
    h_flip = ccr_image.horizontal_mirrored
    v_flip = ccr_image.vertical_mirrored

    # --- BWPN: B/W point values are absolute anchors, constant across the whole roll ---
    #
    # With no_auto_bright=True in rawpy, film base and dense-area values are identical in every
    # frame. The sampled B/W points are applied directly as fixed per-channel anchors.
    img_f = img.astype(np.float32)
    if black_point_bgr is None:
        # --- No-anchor mode: NamiColor's density inversion, nothing measured ---
        rgb_inverted = _namicolor_invert(img_f)
        del img_f
        print(f"BWPN (no anchor, NamiColor density invert): "
              f"{time.time() - total_start_time:.3f}s")
    elif white_point_bgr is None:
        # --- Default-slope mode (black point only): density-space inversion ---
        rgb_inverted = _default_slope_invert(img_f, black_point_bgr, slopes_bgr)
        del img_f
        print(f"BWPN ({'film-stock slope' if slopes_bgr is not None else 'default slope'}): "
              f"{time.time() - total_start_time:.3f}s")
    else:
        # --- Two-point mode (ODAI skipped: per-channel B/W points already
        # equalise channels). density=True recovers optical density (log space);
        # density=False is the legacy linear transmittance stretch. ---
        rgb_inverted = _twopoint_invert(img_f, black_point_bgr, white_point_bgr, density)
        del img_f
        print(f"BWPN (user points, {'density' if density else 'linear'}): "
              f"{time.time() - total_start_time:.3f}s")
    gc.collect()

    # --- Post-invert "look" DISABLED (saturation boost + shadow warmth) ---
    # Commented out to produce a neutral linear inversion: no luminance-weighted
    # saturation curve, no shadow warmth/green shift. rgb_result is the raw
    # inverted image. Re-enable by uncommenting the block below.
    rgb_result = rgb_inverted
    # # --- Saturation boost (identical to main pipeline) ---
    # rgb_norm = rgb_inverted.astype(np.float32) / 65535.0
    # # gamma == 1.0 here, so np.power was a no-op — just clip (saves a full-res
    # # transcendental pass per export).
    # gamma_corrected = np.clip(rgb_norm, 0.0, 1.0)
    # del rgb_norm
    #
    # luminance = np.dot(gamma_corrected[..., :3], [0.299, 0.587, 0.114])
    # luminance_expanded = np.expand_dims(luminance, axis=-1)
    # saturation_curve = np.power(luminance, 0.8)
    # del luminance
    # base_saturation = 1.15
    # min_saturation = 1.00
    # dynamic_saturation = min_saturation + (base_saturation - min_saturation) * saturation_curve
    # del saturation_curve
    # dynamic_saturation = np.expand_dims(dynamic_saturation, axis=-1)
    # gamma_corrected = luminance_expanded + dynamic_saturation * (gamma_corrected - luminance_expanded)
    # del luminance_expanded, dynamic_saturation
    # gamma_corrected = np.clip(gamma_corrected, 0.0, 1.0)
    #
    # # --- Shadow warmth correction (identical to main pipeline) ---
    # shadow_corrected = gamma_corrected
    # shadow_luminance = np.dot(shadow_corrected[..., :3], [0.299, 0.587, 0.114])
    # warmth_curve = np.exp(-shadow_luminance * 4.0)
    # warmth_strength = 0.35 * warmth_curve
    # del warmth_curve
    # green_curve = np.exp(-shadow_luminance * 3.5)
    # del shadow_luminance
    # green_strength = 0.15 * green_curve
    # del green_curve
    # shadow_corrected[..., 0] *= (1.0 + warmth_strength * 0.8)
    # shadow_corrected[..., 1] *= (1.0 + green_strength)
    # shadow_corrected[..., 2] *= (1.0 - warmth_strength)
    # del warmth_strength, green_strength
    #
    # rgb_result = np.clip(shadow_corrected * 65535.0, 0, 65535).astype(np.uint16)
    # del shadow_corrected, gamma_corrected, rgb_inverted
    gc.collect()

    # Working space: the B/W-point inversions emit a WINDOWED base when enabled
    # (highlight headroom preserved). False when the feature is off → legacy
    # full-range, byte-identical.
    ws = _ws_enabled()
    if output_path is None:
        # Preview conversion: the result becomes the live resized_raw, so the
        # live state is updated to describe it. Non-destructive base offsets
        # applied through the adjustment pipeline (UI shows 0); contrast_base
        # 0 disables the baked-in contrast S-curve (was 40). An EXPORT must
        # not touch any of this — flagging a never-converted image's live
        # full-range resized_raw as windowed blows its preview to near-white.
        ccr_image.contrast_base = 0
        ccr_image.temperature_base = 0
        ccr_image._ws_windowed = ws
        # Cache the sprocket alpha for the live preview overlay (applied last, and
        # gated by the toggle, in update_thumbnail_and_preview).
        ccr_image.sprocket_alpha = sprocket_alpha
    if ws:
        # ASCII only: this prints during EVERY B/W-point conversion, and an
        # em dash here crashed converts on macOS (ASCII stdout, issue #86).
        print(f"[working-space] windowed base: ~{_WS_HEADROOM_STOPS:.1f} stops "
              f"highlight headroom - recover with White Point (drag negative)")

    # Auto-exposure (default-slope mode only): compute once from the preview and
    # store as a non-destructive uniform-gain base; export/zoom reuse it. With a
    # white point the two-point map already sets the cut, so force it to 0.
    if output_path is None:
        ccr_image.exposure_base = (
            compute_auto_exposure_gain(rgb_result, ws_windowed=ws)
            if white_point_bgr is None else 0.0)

    # --- User adjustments (export only) ---
    if output_path is not None:
        # Describe the buffer this conversion just produced via overrides
        # (same values a preview conversion would have baked into live state)
        # so a never-converted image's live state stays untouched.
        rgb_result = ccr_image.apply_adjustments(rgb_result, contrast_base=0,
                                                 temperature_base=0,
                                                 ws_windowed=ws)
        # White the sprocket holes / clear film as the last look step — after all
        # adjustments, before the geometric block (same un-rotated space as the
        # preview overlay, so it is WYSIWYG). Gated on the live toggle; deferred
        # import avoids a module cycle. See spec/sprocket-hole-mask.md §4.2.
        from core.ccr_backend import ccr_backend as _bk
        if getattr(_bk, "sprocket_mask_white", False):
            rgb_result = apply_sprocket_mask(rgb_result, sprocket_alpha)

    # --- Export path: flips, rotation, file write ---
    if output_path is not None:
        # User crop (normalized rect in un-rotated/un-flipped space) — applied
        # before flips/rotation so it matches the cropped preview orientation.
        rgb_result = apply_crop_to_image(rgb_result, getattr(ccr_image, 'crop_rect', None),
                                         getattr(ccr_image, 'crop_angle', 0.0))
        # Flips
        if h_flip and v_flip:
            rgb_result = cv2.flip(rgb_result, -1)
        elif h_flip:
            rgb_result = cv2.flip(rgb_result, 1)
        elif v_flip:
            rgb_result = cv2.flip(rgb_result, 0)

        # 90-degree rotation
        angle = ccr_image.rotation_angle % 360
        if angle == 90:
            rgb_result = np.rot90(rgb_result, k=3)
        elif angle == 180:
            rgb_result = np.rot90(rgb_result, k=2)
        elif angle == 270:
            rgb_result = np.rot90(rgb_result, k=1)

        # Fine rotation at full resolution
        if fine_angle != 0:
            h_r, w_r = rgb_result.shape[:2]
            center_r = (w_r // 2, h_r // 2)
            rot_mat = cv2.getRotationMatrix2D(center_r, -fine_angle, 1.0)
            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])
            new_w = int(w_r * abs_cos + h_r * abs_sin)
            new_h = int(h_r * abs_cos + w_r * abs_sin)
            rot_mat[0, 2] += (new_w - w_r) / 2
            rot_mat[1, 2] += (new_h - h_r) / 2
            try:
                rgb_result = cv2.warpAffine(rgb_result, rot_mat, (new_w, new_h),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            except Exception as e:
                print(f"Warning: warpAffine failed: {e}")

        # Write file
        write_export_image(ccr_image, rgb_result, output_path, jpg_out,
                           jpg_quality, max_long_side, output_colorspace)
        print(f"TOTAL bwpoint normalization time: {time.time() - total_start_time:.3f}s")
        gc.collect()
        return None

    print(f"TOTAL bwpoint normalization time: {time.time() - total_start_time:.3f}s")
    return rgb_result


def ccr_export_positive(ccr_image, output_path=None, jpg_out=False,
                        jpg_quality=95, max_long_side=None, output_colorspace="srgb"):
    """Positive-mode processing/export: NO negative inversion.

    The image is already a normal positive (decoded in sRGB; see
    spec/positive-mode.md), so this just applies the user adjustments, crop,
    orientation and output colour space — mirroring the
    bwpoint export tail without the normalization step.

    output_path is None returns the adjusted in-memory array (parity with the
    other normalize functions); otherwise the file is written and None returned.
    """
    total_start_time = time.time()
    img = _load_export_source(ccr_image, output_path, max_long_side)
    if img is None:
        raise ValueError("CCRImage.resized_raw is None")

    fine_angle = ccr_image.fine_rotation_angle / 100.0
    h_flip = ccr_image.horizontal_mirrored
    v_flip = ccr_image.vertical_mirrored

    # In-app processing path: just the adjusted positive (preview/thumbnail use
    # update_thumbnail_and_preview, which also applies adjustments — this mirrors
    # the other normalize functions' output_path-is-None contract).
    if output_path is None:
        return ccr_image.apply_adjustments(img)

    rgb_result = ccr_image.apply_adjustments(img)

    # User crop (normalized rect in un-rotated/un-flipped space) — applied
    # before flips/rotation so it matches the cropped preview orientation.
    rgb_result = apply_crop_to_image(rgb_result, getattr(ccr_image, 'crop_rect', None),
                                     getattr(ccr_image, 'crop_angle', 0.0))
    # Flips
    if h_flip and v_flip:
        rgb_result = cv2.flip(rgb_result, -1)
    elif h_flip:
        rgb_result = cv2.flip(rgb_result, 1)
    elif v_flip:
        rgb_result = cv2.flip(rgb_result, 0)

    # 90-degree rotation
    angle = ccr_image.rotation_angle % 360
    if angle == 90:
        rgb_result = np.rot90(rgb_result, k=3)
    elif angle == 180:
        rgb_result = np.rot90(rgb_result, k=2)
    elif angle == 270:
        rgb_result = np.rot90(rgb_result, k=1)

    # Fine rotation at full resolution (canvas-expanding, like the other paths)
    if fine_angle != 0:
        h_r, w_r = rgb_result.shape[:2]
        center_r = (w_r // 2, h_r // 2)
        rot_mat = cv2.getRotationMatrix2D(center_r, -fine_angle, 1.0)
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])
        new_w = int(w_r * abs_cos + h_r * abs_sin)
        new_h = int(h_r * abs_cos + w_r * abs_sin)
        rot_mat[0, 2] += (new_w - w_r) / 2
        rot_mat[1, 2] += (new_h - h_r) / 2
        try:
            rgb_result = cv2.warpAffine(rgb_result, rot_mat, (new_w, new_h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        except Exception as e:
            print(f"Warning: warpAffine failed: {e}")

    write_export_image(ccr_image, rgb_result, output_path, jpg_out,
                       jpg_quality, max_long_side, output_colorspace)
    print(f"TOTAL positive export time: {time.time() - total_start_time:.3f}s")
    gc.collect()
    return None


def ccr_normalize_with_refparams(ccr_image, p_lo, p_hi, od_factors,
                                 output_path=None, jpg_out=False,
                                 jpg_quality=95, max_long_side=None,
                                 output_colorspace="srgb"):
    """
    Conversion/export pipeline for sliced children carrying precomputed
    reference-conversion constants (conversion_inputs mode "ref_params").
    The constants were derived from the parent's reference frame at slice
    time; read_image applies the child's source_ops chain, so the identical
    conversion replays at any resolution. Fine rotation set AFTER slicing is
    applied at the end like the reference pipeline (display semantics).
    """
    total_start_time = time.time()
    img = _load_export_source(ccr_image, output_path, max_long_side)
    if img is None:
        raise ValueError("CCRImage: could not load image data for ref-params conversion")

    rgb_result = apply_reference_normalization(img, p_lo, p_hi, od_factors)
    ccr_image._ws_windowed = False   # reference path is full-range, not windowed

    if output_path is None:
        print(f"TOTAL ref-params normalization time: {time.time() - total_start_time:.3f}s")
        return rgb_result

    # --- Export path: adjustments, crop, flips, rotation, write ---
    rgb_result = ccr_image.apply_adjustments(rgb_result)
    rgb_result = apply_crop_to_image(rgb_result, getattr(ccr_image, 'crop_rect', None),
                                     getattr(ccr_image, 'crop_angle', 0.0))

    h_flip = ccr_image.horizontal_mirrored
    v_flip = ccr_image.vertical_mirrored
    if h_flip and v_flip:
        rgb_result = cv2.flip(rgb_result, -1)
    elif h_flip:
        rgb_result = cv2.flip(rgb_result, 1)
    elif v_flip:
        rgb_result = cv2.flip(rgb_result, 0)

    angle = ccr_image.rotation_angle % 360
    if angle == 90:
        rgb_result = np.rot90(rgb_result, k=3)
    elif angle == 180:
        rgb_result = np.rot90(rgb_result, k=2)
    elif angle == 270:
        rgb_result = np.rot90(rgb_result, k=1)

    fine_angle = ccr_image.fine_rotation_angle / 100.0
    if fine_angle != 0:
        h_r, w_r = rgb_result.shape[:2]
        center_r = (w_r // 2, h_r // 2)
        rot_mat = cv2.getRotationMatrix2D(center_r, -fine_angle, 1.0)
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])
        new_w = int(w_r * abs_cos + h_r * abs_sin)
        new_h = int(h_r * abs_cos + w_r * abs_sin)
        rot_mat[0, 2] += (new_w - w_r) / 2
        rot_mat[1, 2] += (new_h - h_r) / 2
        try:
            rgb_result = cv2.warpAffine(rgb_result, rot_mat, (new_w, new_h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        except Exception as e:
            print(f"Warning: warpAffine failed: {e}")

    write_export_image(ccr_image, rgb_result, output_path, jpg_out,
                       jpg_quality, max_long_side, output_colorspace)
    print(f"TOTAL ref-params normalization time: {time.time() - total_start_time:.3f}s")
    gc.collect()
    return None


def to_8bit(img16: np.ndarray) -> np.ndarray:
    # Clip to 16-bit range, then scale to 8-bit
    img16 = np.clip(img16, 0, 65535)
    img8 = (img16 / 257).astype(np.uint8)
    return img8


def write_export_image(ccr_image, rgb_u16, output_path, jpg_out, jpg_quality,
                       max_long_side, output_colorspace="srgb"):
    """Single export write chokepoint shared by all three conversion pipelines.

    Resizes to the requested long edge, maps the (sRGB-encoded) result into the
    chosen output colour space, then writes a 16-bit TIFF (deflate) or 8-bit
    JPEG with the matching ICC profile embedded so a colour-managed viewer
    interprets the file correctly. Returns the resolved output path.
    """
    if max_long_side:
        rgb_u16 = ccr_image.resize_image_to_max_pixel(rgb_u16, max_long_side)
    # Re-encode to the target colour space and get the ICC bytes to embed.
    rgb_u16, icc = apply_export_colorspace(rgb_u16, output_colorspace)
    output_path = safe_unicode_path(output_path)
    if jpg_out:
        output_path = os.path.splitext(output_path)[0] + ".jpg"
        img_8 = cv2.cvtColor(to_8bit(rgb_u16), cv2.COLOR_RGB2BGR)  # RGB -> BGR for cv2
        ok, buf = cv2.imencode(".jpg", img_8,
                               [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])
        if not ok:
            raise IOError(f"Failed to encode JPEG for {output_path}")
        # cv2 can't embed ICC, so inject an APP2 ICC_PROFILE segment and write
        # the bytes ourselves (also the Unicode-safe path, like safe_cv2_imwrite).
        data = inject_jpeg_icc(buf.tobytes(), icc)
        try:
            with open(output_path, "wb") as f:
                f.write(data)
        except Exception as e:
            raise IOError(f"Failed to save image to {output_path}: {e}")
    else:
        output_path = os.path.splitext(output_path)[0] + ".tiff"
        if not safe_tifffile_imwrite(output_path, rgb_u16, photometric="rgb",
                                     compression="deflate", iccprofile=icc):
            raise IOError(f"Failed to save image to {output_path}")
    print(f"Normalized image saved to {output_path}")
    return output_path


def apply_crop_to_image(img: np.ndarray, crop_rect_norm, crop_angle: float = 0.0) -> np.ndarray:
    """
    Crop an image using a box of normalized (x1, y1, x2, y2) fractions
    defined in un-rotated/un-flipped image space (the same space as
    resized_raw), optionally rotated by crop_angle degrees about the box
    center (positive = clockwise on screen, matching Qt's rotate()).
    Returns the input unchanged when the rect is missing or degenerate,
    so callers can pass it unconditionally.
    """
    if crop_rect_norm is None:
        return img
    h, w = img.shape[:2]
    fx1, fy1, fx2, fy2 = crop_rect_norm
    if crop_angle:
        # Rotated crop box: output pixel (u, v) samples the source at
        # C + R(angle) . (u - bw/2, v - bh/2). R matches Qt's
        # clockwise-positive rotate() in y-down coords, so the exported
        # content equals the on-screen selection. Areas outside the source
        # are filled black. The box rect is not clamped to the image here —
        # a rotated box may legitimately overhang the image bounds.
        bw = int(round((fx2 - fx1) * w))
        bh = int(round((fy2 - fy1) * h))
        if bw < 2 or bh < 2:
            return img
        cx = (fx1 + fx2) / 2.0 * w
        cy = (fy1 + fy2) / 2.0 * h
        theta = np.deg2rad(crop_angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        offset = np.array([cx, cy]) - rot @ np.array([bw / 2.0, bh / 2.0])
        affine = np.hstack([rot, offset[:, None]]).astype(np.float32)
        return cv2.warpAffine(img, affine, (bw, bh),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x1 = max(0, min(w - 1, int(round(fx1 * w))))
    y1 = max(0, min(h - 1, int(round(fy1 * h))))
    x2 = max(x1 + 1, min(w, int(round(fx2 * w))))
    y2 = max(y1 + 1, min(h, int(round(fy2 * h))))
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return img
    return img[y1:y2, x1:x2]


# --- Resolution-independent conversion helpers (zoom hi-res detail) --------
# These reproduce ccr_normalize_with_reference / ccr_normalize_with_bwpoint's
# preview-path math with the normalization constants split out, so the same
# conversion can be re-applied to a higher-resolution decode and come out
# color-matched to the 1080px preview.

_LUM_WEIGHTS = np.array([[0.299, 0.587, 0.114]], dtype=np.float32)


def apply_postinvert_look(rgb_inverted: np.ndarray) -> np.ndarray:
    """Shared saturation-boost + shadow-warmth styling applied after
    inversion — identical math to both conversion pipelines, computed with
    cv2 SIMD primitives (5-10x faster than numpy at hi-res sizes) and
    in-place float32 ops."""
    x = rgb_inverted.astype(np.float32)
    x *= np.float32(1.0 / 65535.0)
    np.clip(x, 0.0, 1.0, out=x)

    luminance = cv2.transform(x, _LUM_WEIGHTS)          # (h, w) float32
    sat_curve = cv2.pow(luminance, 0.8)
    dynamic = np.float32(0.15) * sat_curve
    dynamic += np.float32(1.0)                          # 1.0 + 0.15 * lum^0.8
    lum3 = luminance[..., None]
    x -= lum3
    x *= dynamic[..., None]
    x += lum3
    np.clip(x, 0.0, 1.0, out=x)

    shadow_lum = cv2.transform(x, _LUM_WEIGHTS)
    warmth = cv2.exp(shadow_lum * np.float32(-4.0))
    warmth *= np.float32(0.35)
    green = cv2.exp(shadow_lum * np.float32(-3.5))
    green *= np.float32(0.15)
    x[..., 0] *= (np.float32(1.0) + warmth * np.float32(0.8))
    x[..., 1] *= (np.float32(1.0) + green)
    x[..., 2] *= (np.float32(1.0) - warmth)
    x *= np.float32(65535.0)
    np.clip(x, 0, 65535, out=x)
    return x.astype(np.uint16)


def compute_reference_norm_params(ref_img: np.ndarray, reference_rect,
                                  fine_rotation_angle: int):
    """Derive the per-channel percentile anchors and OD alignment factors
    that ccr_normalize_with_reference computes from the reference frame of
    ref_img (the 1080px scan), so the conversion can be replayed at any
    resolution. Returns (p_lo[3], p_hi[3], od_factors[3])."""
    img_ref = ref_img
    fine_angle = fine_rotation_angle / 100.0
    if fine_angle != 0:
        h, w = img_ref.shape[:2]
        rot = cv2.getRotationMatrix2D((w // 2, h // 2), -fine_angle, 1.0)
        img_ref = cv2.warpAffine(img_ref, rot, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x1, y1, x2, y2 = map_rect_to_original(ref_img.shape, img_ref.shape, reference_rect)
    ref_crop = img_ref[y1:y2, x1:x2].astype(np.float32)

    p_lo = np.empty(3, dtype=np.float64)
    p_hi = np.empty(3, dtype=np.float64)
    norm_crop = np.empty_like(ref_crop)
    for c in range(3):
        p_lo[c] = np.percentile(ref_crop[..., c], 1)
        p_hi[c] = np.percentile(ref_crop[..., c], 99)
        norm_crop[..., c] = np.clip(
            (ref_crop[..., c] - p_lo[c]) / (p_hi[c] - p_lo[c])
            * (65535 - 8192) + 8192, 0, 65535)
    od_crop = -np.log10((norm_crop + 1e-6) / 65535.0)
    mean_od = np.mean(od_crop, axis=(0, 1))
    target = np.mean(mean_od)
    od_factors = target / (mean_od + 1e-12)
    return p_lo, p_hi, od_factors


def apply_reference_normalization(img: np.ndarray, p_lo, p_hi, od_factors) -> np.ndarray:
    """Apply the reference-frame normalization + inversion + standard look to
    an image of any resolution, using precomputed constants.

    The pipeline's OD alignment (od = -log10(v); od *= f; out = 10^-od) is
    algebraically just out = v^f — computed that way here, which halves the
    transcendental work on large hi-res frames. The per-channel linear map
    is folded into a single cv2.transform and the power uses cv2.pow (SIMD)."""
    # v*s + (8192 - p_lo*s), pre-divided by 65535 into the unit domain
    affine = np.zeros((3, 4), dtype=np.float64)
    for c in range(3):
        s = (65535.0 - 8192.0) / (p_hi[c] - p_lo[c])
        affine[c, c] = s / 65535.0
        affine[c, 3] = (8192.0 - p_lo[c] * s) / 65535.0
    norm = cv2.transform(img.astype(np.float32), affine)
    np.clip(norm, 0.0, 1.0, out=norm)
    norm += np.float32(1e-6 / 65535.0)
    channels = list(cv2.split(norm))
    for c in range(3):
        cv2.pow(channels[c], float(od_factors[c]), channels[c])
    norm = cv2.merge(channels)
    norm *= np.float32(65535.0)
    np.clip(norm, 0, 65535, out=norm)
    inverted = 65535 - norm.astype(np.uint16)
    return apply_postinvert_look(inverted)


def apply_bwpoint_normalization(img: np.ndarray, black_point_bgr, white_point_bgr=None,
                                density: bool = False,
                                slopes_bgr=None) -> np.ndarray:
    """B/W-point conversion at any resolution: absolute per-channel anchors +
    inversion — mirrors ccr_normalize_with_bwpoint's preview path (the anchors
    are global constants, so no rescaling is needed). Used for the zoom hi-res
    replay and slice/reset, so it MUST match the entry pipeline exactly.

    When black_point_bgr is ALSO None, no anchor was sampled at all and the
    conversion is NamiColor's density inversion with its fixed constants (see
    _namicolor_invert); Channel Levels does the grading, in the same density
    space NamiColor's own sliders work in. See spec/no-anchor-convert.md.

    When white_point_bgr is None, the default-slope mode is used: a density-space
    inversion applied to the black point alone (see _default_slope_invert), with
    `slopes_bgr` (a film-stock preset's per-channel density slopes, snapshotted
    in conversion_inputs["slopes"]) or the baked scalar DEFAULT_DENSITY_SLOPE
    when None. Otherwise the two-point math runs, in optical-density space when
    density=True or the legacy linear transmittance stretch when density=False
    (default; bit-identical to the prior behaviour). Replay callers pass
    ci.get("density", False) / ci.get("slopes") so legacy conversions stay
    as converted. See spec/density-bwpoint-toggle.md, spec/film-stock-slopes.md.

    The post-invert "look" stays DISABLED so the hi-res replay matches the
    look-less preview/export path."""
    img_f = img.astype(np.float32)
    if black_point_bgr is None:
        # No anchors at all: NamiColor's density inversion (spec/no-anchor-convert.md).
        return _namicolor_invert(img_f)
    if white_point_bgr is None:
        # Default-slope mode (black point only): density-space inversion.
        return _default_slope_invert(img_f, black_point_bgr, slopes_bgr)
    return _twopoint_invert(img_f, black_point_bgr, white_point_bgr, density)


def log_bwpoint_slopes(black_point_bgr, white_point_bgr):
    """Diagnostic: report BOTH the LINEAR slope and the DENSITY-SPACE slope
    implied by a user's B/W-point pick, so a fixed default slope can be
    CALIBRATED from a typical roll and baked into the code. ONLY logs.

    LINEAR slope  S_lin[c] = 65535 / (base[c] - white[c])  is exactly what the
    CURRENT two-point method uses (inverted = S_lin * (base - img)). Bake it to
    run the default mode directly in the existing linear pipeline (seamless
    toggle, no new code). It is tied to absolute scan brightness, so it is valid
    only when scans are exposed consistently (which the bw path already assumes).

    DENSITY slope S_den[c] = 1 / log10(base[c] / white[c])  multiplies optical
    density d = log10(base/img). Bake it for a brightness-INVARIANT default
    (the scan exposure cancels in the log; cf. OpenEnlarge's FAITHFUL_SCALE), at
    the cost of a log-space inversion whose look differs from the linear path.

    Run on a representative frame, read the MEAN / per-channel values, bake one.
    """
    labels = ("B", "G", "R")
    print("=== SLOPE CALIBRATION (B/W points) ===")
    print(f"  black point (clear/base, BGR): {tuple(black_point_bgr)}")
    print(f"  white point (dense,      BGR): {tuple(white_point_bgr)}")
    lin, den = [], []
    for c in range(3):
        base = float(black_point_bgr[c])
        white = float(white_point_bgr[c])
        if base <= 0 or white <= 0 or base <= white:
            print(f"  [{labels[c]}] invalid (base={base}, white={white}); skipped")
            continue
        s_lin = 65535.0 / (base - white)        # linear gain (current method)
        d = float(np.log10(base / white))       # optical density base -> white
        s_den = 1.0 / d                         # density slope: output 1.0 at D
        lin.append(s_lin)
        den.append(s_den)
        print(f"  [{labels[c]}] LINEAR 65535/(base-white)={s_lin:.4f}    "
              f"DENSITY 1/log10(base/white)={s_den:.4f}  (D={d:.4f})")
    if lin:
        print(f"  MEAN  linear = {sum(lin)/len(lin):.4f}   density = {sum(den)/len(den):.4f}")
        print(f"  per-channel LINEAR  (B,G,R) = {[round(s, 4) for s in lin]}")
        print(f"  per-channel DENSITY (B,G,R) = {[round(s, 4) for s in den]}")
    print("=== bake the LINEAR vector (current pipeline) OR the DENSITY vector (log) ===")


def compute_density_slopes(black_point_bgr, white_point_bgr):
    """Per-channel DENSITY slopes implied by a sampled B/W-point pair:
    S[c] = 1 / log10(base[c] / dense[c]) — the value that maps the pair's dense
    sample to display white in _default_slope_invert. This is what a film-stock
    preset stores (spec/film-stock-slopes.md): a property of the stock (its
    per-dye-layer characteristic-curve gamma), invariant to per-channel light
    scaling — unlike the linear slope (see log_bwpoint_slopes above).

    Returns a (B, G, R) tuple of floats, or None when ANY channel is unusable
    (non-positive sample, base not above dense, or ~zero density range): a pair
    that can't characterise all three channels is rejected whole."""
    slopes = []
    for c in range(3):
        base = float(black_point_bgr[c])
        dense = float(white_point_bgr[c])
        if base <= 0.0 or dense <= 0.0 or base <= dense:
            return None
        d = float(np.log10(base / dense))
        if d <= 1e-6:
            return None
        slopes.append(1.0 / d)
    return tuple(slopes)


def auto_fine_angle(img16: np.ndarray, debug: bool = False) -> float:
    """
    Analyze a 16-bit image and estimate the rotation angle (in degrees)
    needed to make the dominant horizontal lines horizontal.

    Args:
        img16 (np.ndarray): 16-bit input image (H, W) or (H, W, 3)
        debug (bool): If True, show the most significant line on the image.

    Returns:
        float: Estimated rotation angle in degrees (positive = counterclockwise)
    """
    # Convert to 8-bit grayscale
    if img16.ndim == 3:
        gray = cv2.cvtColor((img16 / 257).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor((img16 / 257).astype(np.uint8), cv2.COLOR_BGR2RGB)
    else:
        gray = (img16 / 257).astype(np.uint8)
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Edge detection
    edges = cv2.Canny(gray, 160, 255, apertureSize=3)

    # if debug:
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(edges, cmap='gray')
    #     plt.title("Edge Detection")
    #     plt.axis('off')
    #     plt.show()

    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 4, maxLineGap=20)
    if lines is None:
        if debug:
            print("No lines detected.")
        return 0.0

    # Find the longest nearly-horizontal line
    max_len = 0
    best_angle = 0.0
    best_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        angle = np.degrees(np.arctan2(dy, dx))
        # Consider lines within +/- 30 degrees of horizontal
        if (abs(angle) < 30 or abs(angle) > 150) and length > max_len:
            max_len = length
            best_angle = angle
            best_line = (x1, y1, x2, y2)

    # if debug and best_line is not None:

    #     img_debug = img_rgb.copy()
    #     x1, y1, x2, y2 = best_line
    #     # Draw the best line in red
    #     cv2.line(img_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(img_debug)
    #     plt.title(f"Longest horizontal-like line (angle={best_angle:.2f}°)")
    #     plt.axis('off')
    #     plt.show()

    # The angle is relative to the x-axis; positive = counterclockwise
    # Return negative to indicate the rotation needed to deskew
    return -best_angle if best_line is not None else 0.0


def auto_frame_v2(img16: np.ndarray, fine_rotation_angle: int, debug: bool = False) -> tuple:
    """
    Optimized auto_frame using white/black area masking and largest rectangle detection.
    Based on the methodology from the POC notebook with improved workflow.
    
    Args:
        img16 (np.ndarray): 16-bit input image (H, W) or (H, W, 3)
        fine_rotation_angle (int): Fine rotation angle in hundredths of a degree
        debug (bool): If True, show the detected frame on the image.
        
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the reference frame rectangle.
    """
    original_shape = img16.shape
    
    # Step 1: Apply fine rotation
    angle = fine_rotation_angle / 100.0
    h, w = img16.shape[:2]
    center = (w // 2, h // 2)
    if angle != 0:
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        img16 = cv2.warpAffine(img16, rot_mat, (w, h), flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    print(f"Rotated image by {angle} degrees, original shape: {original_shape}, new shape: {img16.shape}")

    # Step 2: Create white area mask
    def create_white_area_mask(image, threshold_percentile=95, min_brightness=0.9):
        """Create a binary mask for white/bright areas in the image"""
        if image.ndim == 3:
            # Convert RGB to grayscale using luminance formula
            gray = 0.2 * image[..., 0] + 0.3 * image[..., 1] + 0.5 * image[..., 2]
        else:
            gray = image.copy()
        
        # Normalize to 0-1 range
        gray_norm = (gray - gray.min()) / (np.ptp(gray) + 1e-8)
        
        # Create binary mask for white areas
        white_mask = gray_norm > min_brightness
        return white_mask

    # Step 3: Create black area mask  
    def create_black_area_mask(image, red_weight=0.5, blue_weight=0.2, threshold=0.02):
        """Create a binary mask for black/dark areas using red and blue channel weighted grayscale"""
        if image.ndim == 3:
            # Calculate green weight to ensure weights sum to 1
            green_weight = 1.0 - red_weight - blue_weight
            green_weight = max(0.0, green_weight)  # Ensure non-negative
            
            # Weighted grayscale conversion with all three channels
            gray = (red_weight * image[..., 0] + 
                   green_weight * image[..., 1] + 
                   blue_weight * image[..., 2])
        else:
            gray = image.copy()
        
        # Normalize to 0-1 range
        gray_norm = (gray - gray.min()) / (np.ptp(gray) + 1e-8)
        
        # Create binary mask for black areas
        black_mask = gray_norm < threshold
        return black_mask

    # Step 4: Create combined mask
    white_mask = create_white_area_mask(img16, min_brightness=0.9)
    black_mask = create_black_area_mask(img16, red_weight=0.6, blue_weight=0.2, threshold=0.02)
    
    # Combine white and black masks using OR operation
    combined_mask = np.logical_or(white_mask, black_mask)
    
    # Apply morphological operations to clean up the combined mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask_cleaned = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    combined_mask = combined_mask_cleaned.astype(bool)

    # Step 5: Find largest non-masked rectangle using optimized histogram method
    def find_largest_non_masked_rectangle(mask):
        """Find the largest rectangle that contains only False values in a binary mask"""
        # Invert mask so we're looking for areas with True values (non-masked areas)
        inverted_mask = ~mask
        
        rows, cols = inverted_mask.shape
        heights = np.zeros(cols, dtype=int)
        max_area = 0
        best_rect = (0, 0, 0, 0, 0)  # (top, left, height, width, area)
        
        for row in range(rows):
            # Update heights histogram
            for col in range(cols):
                if inverted_mask[row, col]:
                    heights[col] += 1
                else:
                    heights[col] = 0
            
            # Find largest rectangle in current histogram
            stack = []
            for col in range(cols + 1):
                h = heights[col] if col < cols else 0
                
                while stack and heights[stack[-1]] > h:
                    height = heights[stack.pop()]
                    width = col if not stack else col - stack[-1] - 1
                    area = height * width
                    
                    if area > max_area:
                        max_area = area
                        left = 0 if not stack else stack[-1] + 1
                        top = row - height + 1
                        best_rect = (top, left, height, width, area)
                
                stack.append(col)
        
        # Shrink the rectangle's long side by 2%, shrink both ends
        # Also shrink the height side by 0.5%
        top, left, height, width, area = best_rect
        if height > width:
            # Height is the long side
            shrink_amount = int(height * 0.02)
            height = max(1, height - 2 * shrink_amount)
            top += shrink_amount
            # Also shrink width by 0.5%
            width_shrink = int(width * 0.01)
            width = max(1, width - 2 * width_shrink)
            left += width_shrink
        else:
            # Width is the long side  
            shrink_amount = int(width * 0.02)
            width = max(1, width - 2 * shrink_amount)
            left += shrink_amount
            # Also shrink height by 0.5%
            height_shrink = int(height * 0.01)
            height = max(1, height - 2 * height_shrink)
            top += height_shrink
        
        area = height * width
        best_rect = (top, left, height, width, area)
        
        return best_rect

    # Find the largest valid rectangle
    top, left, rect_height, rect_width, area = find_largest_non_masked_rectangle(combined_mask)
    
    if rect_height == 0 or rect_width == 0:
        # fallback: whole image
        print("Warning: No valid rectangle found, using whole image")
        return map_rect_to_original(img16.shape, original_shape, (0, 0, w, h))
    
    # Convert to x1, y1, x2, y2 format
    x1, y1 = left, top
    x2, y2 = left + rect_width, top + rect_height
    
    # Add small padding if possible
    padding = 2
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Map the rectangle back to original image size
    final_rect = map_rect_to_original(img16.shape, original_shape, (x1, y1, x2, y2))
    
    print(f"Detected rectangle: {rect_width}x{rect_height} (area: {area} pixels, "
          f"{area/(h*w)*100:.2f}% of image)")
    
    # Debug visualization (if enabled)
    # if debug:
    #     img_disp = to_8bit(img16)
    #     if img_disp.ndim == 2:
    #         img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)
    #     cv2.rectangle(img_disp, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(img_disp)
    #     plt.title("auto_frame_v2: Final Cropping Box")
    #     plt.axis('off')
    #     plt.show()
    
    return final_rect

def auto_frame(img16: np.ndarray, fine_rotation_angle: int, debug: bool = False) -> tuple:
    """
    Automatically determine the reference frame for an image by detecting the largest rectangle,
    excluding pure black and pure white areas.

    Args:
        img16 (np.ndarray): 16-bit input image (H, W) or (H, W, 3)
        fine_rotation_angle (int): Fine rotation angle in hundredths of a degree
        debug (bool): If True, show the detected frame on the image.

    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the reference frame rectangle.
    """
    # Apply fine rotation
    angle = fine_rotation_angle / 100.0
    h, w = img16.shape[:2]
    center = (w // 2, h // 2)
    if angle != 0:
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        img16 = cv2.warpAffine(img16, rot_mat, (w, h), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_CONSTANT,borderValue=0)

    img8 = to_8bit(img16)

    # Select only black and white pixels
    if img8.ndim == 3:
        img_hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
        v = img_hsv[..., 2]
        s = img_hsv[..., 1]
        black_mask = v < 40
        white_mask = (v > 240) & (s < 30)
        bw_mask = (black_mask | white_mask).astype(np.uint8)
    else:
        black_mask = img8 < 50
        white_mask = img8 > 240
        bw_mask = (black_mask | white_mask).astype(np.uint8)


    # if debug:
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(bw_mask, cmap='gray')
    #     plt.title("Inverted BW Mask (Content Region)")
    #     plt.axis('off')
    #     plt.show()
    # Morphological closing to connect black and white regions
    kernel = np.ones((301, 301), np.uint8)
    bw_mask_closed = cv2.morphologyEx(bw_mask, cv2.MORPH_CLOSE, kernel)

    # Invert the mask to get the content region
    content_mask = 1 - bw_mask_closed



    # Find largest connected component in the content mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(content_mask, connectivity=8)
    if num_labels <= 1:
        # Only background found
        return (0, 0, img8.shape[1], img8.shape[0])
    # Ignore label 0 (background), find largest
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    x1, y1, x2, y2 = x, y, x + w, y + h

    # if debug:
    #     img_debug = img8.copy()
    #     if img_debug.ndim == 2:
    #         img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2RGB)
    #     cv2.rectangle(img_debug, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 2)
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(img_debug)
    #     plt.title("Detected Largest Valid Rectangle")
    #     plt.axis('off')
    #     plt.show()

    return (x1, y1, x2, y2)

def map_rect_to_original(resized_shape, original_shape, rect):
    """
    Map rectangle coordinates from resized image to original image size.

    Args:
        resized_shape (tuple): (height, width) of resized image
        original_shape (tuple): (height, width) of original image
        rect (tuple): (x1, y1, x2, y2) in resized image

    Returns:
        tuple: (x1, y1, x2, y2) mapped to original image coordinates (rounded to int)
    """
    rh, rw = resized_shape[:2]
    oh, ow = original_shape[:2]
    x1, y1, x2, y2 = rect

    scale_x = ow / rw
    scale_y = oh / rh

    x1o = int(round(x1 * scale_x))
    y1o = int(round(y1 * scale_y))
    x2o = int(round(x2 * scale_x))
    y2o = int(round(y2 * scale_y))
    return (x1o, y1o, x2o, y2o)


def compute_neutral_temp_tint(r: float, g: float, b: float,
                              tint_balance_factor: float = 1.0) -> tuple:
    """
    Given the mean RGB (0–65535) of a user-picked neutral reference point,
    compute the temperature and tint slider values [-100, 100] that make that
    point neutral (R == G == B) after adjust_image's temperature/tint stage.

    Inverts the FLAT per-channel WB that _white_balance_gains applies (no tone or
    skin weighting; see spec/working-space-white-balance.md):
        temperature:  r *= (1 + s),  b *= (1 - s),   s = (slider/100) * 0.40
        tint:         g *= (1 - t),  r *= (1 + 0.3t),  b *= (1 + 0.3t),
                      t = tanh(slider * 0.02) * 0.26 * balance_factor
    Solving r(1+s) = b(1-s) gives s; then m(1+0.3t) = g(1-t) gives t, where m is
    the common R/B value after the temperature step.
    """
    eps = 1e-6
    r = max(float(r), eps)
    g = max(float(g), eps)
    b = max(float(b), eps)

    # Temperature: choose s so the R and B channels meet.
    s = (b - r) / (b + r)
    temp_slider = float(np.clip(s * 100.0 / _WB_TEMP_STRENGTH, -100.0, 100.0))

    # Use the achieved (possibly clamped) scale for the tint step.
    s_eff = (temp_slider / 100.0) * _WB_TEMP_STRENGTH
    m = (r * (1.0 + s_eff) + b * (1.0 - s_eff)) / 2.0

    # Tint: choose t so G meets the common R/B level m.
    t = (g - m) / (g + 0.3 * m)
    denom = _WB_TINT_STRENGTH * tint_balance_factor
    x = float(np.clip(t / max(denom, eps), -0.999, 0.999))
    tint_slider = float(np.clip(np.arctanh(x) / 0.02, -100.0, 100.0))

    return int(round(temp_slider)), int(round(tint_slider))


def adjust_image(
    img16: np.ndarray,
    kelvin_shift: float = 0.0,
    tint_shift: float = 0.0,
    exposure: float = 0.0,
    brightness: float = 0.0,
    blackpoint: float = 0.0,
    whitepoint: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    tint_balance_factor: float = 1.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    ch_input_gain: float = 0.0,
    ch_master_shift: float = 0.0,
    ch_master_gain: float = 0.0,
    ch_r_shift: float = 0.0,
    ch_r_gain: float = 0.0,
    ch_r_blackpoint: float = 0.0,
    ch_g_shift: float = 0.0,
    ch_g_gain: float = 0.0,
    ch_g_blackpoint: float = 0.0,
    ch_b_shift: float = 0.0,
    ch_b_gain: float = 0.0,
    ch_b_blackpoint: float = 0.0,
    sub_saturation: float = 0.0,
    band_settings: dict = None,
    ws_windowed: bool = False,
) -> np.ndarray:
    """
    Apply temperature, tint, exposure, brightness, blackpoint, whitepoint, highlights, shadows,
    contrast, and saturation adjustments to a 16-bit image.
    All input factors are in range [-100, 100], 0 = no change.
    band_settings optionally carries the per-color-band sliders
    (band_<color>_<param> keys), applied as the final step.
    ws_windowed: when True, img16 is a windowed working-space base — de-window it,
    apply the Gain/Exposure recovery un-clamped, clamp to the display window, and
    consume exposure here (the rest of the chain runs on a normal full-range image).
    Returns a 16-bit image.
    """
    if ws_windowed:
        img16 = _apply_working_space_recovery(img16, exposure, whitepoint,
                                              kelvin_shift, tint_shift, tint_balance_factor,
                                              ch_input_gain, ch_master_shift, ch_master_gain,
                                              ch_r_shift, ch_r_gain, ch_r_blackpoint,
                                              ch_g_shift, ch_g_gain, ch_g_blackpoint,
                                              ch_b_shift, ch_b_gain, ch_b_blackpoint)
        exposure = 0.0       # consumed by the recovery pre-stage
        whitepoint = 0.0     # White Point is the headroom-recovery control here
        kelvin_shift = 0.0   # White Balance consumed (flat per-channel gain, pre-clamp)
        tint_shift = 0.0
        # Channel Levels consumed too — it ran FIRST, un-clamped, inside the
        # recovery (spec/channel-levels-pre-clamp.md). Zero it so the block
        # below can't apply it a second time.
        ch_input_gain = ch_master_shift = ch_master_gain = 0.0
        ch_r_shift = ch_r_gain = ch_r_blackpoint = 0.0
        ch_g_shift = ch_g_gain = ch_g_blackpoint = 0.0
        ch_b_shift = ch_b_gain = ch_b_blackpoint = 0.0
    img = img16.astype(np.float32)

    # Channel Levels - the FIRST adjustment stage, ahead of White Balance and the
    # whole look domain. Only reached on a NON-windowed base (reference mode,
    # positive mode, area layers); a windowed base consumed it above, un-clamped.
    # Those paths carry no sub-black data, so this runs clamped, as before.
    if _channel_levels_active(ch_input_gain, ch_master_shift, ch_master_gain,
                              ch_r_shift, ch_r_gain, ch_r_blackpoint,
                              ch_g_shift, ch_g_gain, ch_g_blackpoint,
                              ch_b_shift, ch_b_gain, ch_b_blackpoint):
        img /= np.float32(65535.0)
        _apply_channel_levels(img, ch_input_gain, ch_master_shift, ch_master_gain,
                              ch_r_shift, ch_r_gain, ch_r_blackpoint,
                              ch_g_shift, ch_g_gain, ch_g_blackpoint,
                              ch_b_shift, ch_b_gain, ch_b_blackpoint,
                              clamp=True)
        img *= np.float32(65535.0)

    # White balance - flat per-channel gain (spec/working-space-white-balance.md).
    # ws_windowed consumed it above (pre-clamp, headroom-safe); this covers the
    # non-windowed paths. Index 0=R, 1=G, 2=B.
    if kelvin_shift != 0.0 or tint_shift != 0.0:
        _gr, _gg, _gb = _white_balance_gains(kelvin_shift, tint_shift, tint_balance_factor)
        img[..., 0] *= np.float32(_gr)
        img[..., 1] *= np.float32(_gg)
        img[..., 2] *= np.float32(_gb)
        kelvin_shift = 0.0
        tint_shift = 0.0

    # Map input factors from [-100, 100] to useful ranges
    exposure_scale = exposure * 2.0 / 100.0  # -2.0 to +2.0 stops
    brightness_scale = brightness / 8.0
    blackpoint_scale = 1.0 - (blackpoint / 100.0)    # -1.0 to +1.0 (fraction of 65535)
    whitepoint_scale = 1.0 + (whitepoint / 100.0)  # 0.0 to 2.0 (scaling factor)
    contrast_scale = 1.0 + (contrast / 100.0)      # 0.0 to 2.0
    saturation_scale = 1.0 + (saturation / 100.0)  # 0.0 to 2.0


    # Gain — shares Channel Levels "Master Gain"'s /300 curve: a uniform linear
    # gain out = in / (1 - v/300), hard-clipped to [0,1] (v=200 -> 3x, v=100 ->
    # 1.5x, v=-200 -> 0.6x). The Gain slider spans +-200 while Master Gain stays
    # +-100.
    if exposure != 0.0:
        gm = np.clip(exposure, -200.0, 200.0) / 300.0   # /300 like ch_master_gain
        white_val = 1.0 - gm                            # black_val is 0
        img = np.clip(img / 65535.0 / white_val, 0.0, 1.0) * 65535.0
    
    # Brightness
    if brightness != 0.0:
        img_norm = img / 65535.0
        brightness_scale = brightness / 8.0
        curve = 1.0 - 0.3 * brightness_scale
        img_norm = np.power(img_norm, curve)
        img_norm = np.clip(img_norm, 0.0, 1.0)
        img = img_norm * 65535.0

    # Highlights / Shadows (anchored per-channel tone-region roll-off)
    # Region "bumps" are zero at both endpoints (0 and 1) so pure black and
    # pure white stay anchored — highlights roll off smoothly below white
    # rather than the white point itself being scaled.
    if highlights != 0.0 or shadows != 0.0:
        HS_PEAK = 0.10546875   # peak of x^3*(1-x), normalizes bumps to peak 1.0
        HS_STRENGTH = 0.30     # max channel offset at the bump peak for full slider
        x = img / 65535.0
        one_minus = 1.0 - x
        # Highlight bump peaks at x=0.75; shadow bump peaks at x=0.25
        wh = (x ** 3) * one_minus / HS_PEAK
        ws = x * (one_minus ** 3) / HS_PEAK
        x = x + (highlights / 100.0) * HS_STRENGTH * wh + (shadows / 100.0) * HS_STRENGTH * ws
        img = np.clip(x, 0.0, 1.0) * 65535.0

    # Black/White point (Adobe-like: remap input range)
    if blackpoint != 0.0 or whitepoint != 0.0:
        img_norm = img / 65535.0
        # Map [-100, 100] to [0, 0.2] for black, [1, 0.8] for white
        black_clip = np.clip(blackpoint, -100, 100) / 300.0  # -0.333 to +0.333 (matches white point scale)
        white_clip = np.clip(whitepoint, -100, 100) / 300.0  # -0.2 to +0.2
        black_val = 0.0 + black_clip
        white_val = 1.0 - white_clip
        # Piecewise linear remap
        img_norm = (img_norm - black_val) / (white_val - black_val)
        img_norm = np.clip(img_norm, 0, 1)
        img = img_norm * 65535.0
    # Contrast (continuous S-curve for both positive and negative)
    if contrast != 0.0:
        img_norm = img / 65535.0
        midpoint = 0.5
        # Map contrast [-100, 100] to k [-0.95, 0.95]
        k = np.clip(contrast / 105.0, -0.95, 0.95)
        # S-curve: compress for negative, expand for positive, fixed endpoints
        def s_curve(x, k):
            return ((1 + k) * (x - midpoint)) / (1 + k * np.abs(x - midpoint) * 2) + midpoint
        img_norm = s_curve(img_norm, k)
        img = img_norm * 65535.0

    # Mid-high tone weighted saturation adjustment
    if saturation != 0.0:
        img_norm = img / 65535.0
        # Convert RGB to grayscale using luminance weights
        gray = np.dot(img_norm[..., :3], [0.299, 0.587, 0.114])
        gray_expanded = np.expand_dims(gray, axis=-1)
        
        # Create mid-high tone weighted curve: bell curve peaked at 65% luminance
        # Using Gaussian-like curve: exp(-((luminance - 0.65) / 0.25)^2)
        mid_high_weight = np.exp(-((gray - 0.50) / 0.35) ** 2)
        
        # Create dynamic saturation factor based on mid-high tone weighting
        # Maximum effect at 65% luminance, minimal effect in deep shadows/highlights
        min_saturation_factor = 0.2  # 20% of full saturation in extremes
        saturation_curve = min_saturation_factor + (1.0 - min_saturation_factor) * mid_high_weight
        
        # Apply the mid-high tone weighted saturation scaling
        dynamic_saturation_scale = 1.0 + (saturation_scale - 1.0) * saturation_curve
        dynamic_saturation_scale = np.expand_dims(dynamic_saturation_scale, axis=-1)
        
        # Blend between grayscale and original based on mid-high tone weighted saturation
        img_norm = gray_expanded + dynamic_saturation_scale * (img_norm - gray_expanded)
        img_norm = np.clip(img_norm, 0, 1)
        img = img_norm * 65535.0

    # Subtractive (film-density) saturation: scale each pixel's chromaticity
    # ratios by a power while pinning the dominant channel, so saturation is
    # gained by absorbing light in the other channels (darker, denser colors)
    # instead of adding it. Mirrors the OpenCL kernel block exactly.
    if sub_saturation != 0.0:
        img_norm = np.clip(img / 65535.0, 0.0, 1.0)
        mx = np.max(img_norm, axis=-1, keepdims=True)
        gamma_s = 2.0 ** (sub_saturation / 100.0)
        safe_mx = np.maximum(mx, 1e-6)
        img_norm = np.where(mx > 1e-6,
                            mx * (img_norm / safe_mx) ** gamma_s,
                            img_norm)
        img = img_norm * 65535.0

    # (Channel Levels used to run here, post-clamp. It is now the FIRST stage —
    # see the top of this function and spec/channel-levels-pre-clamp.md.)

    if band_settings:
        # Run the band step in float, pre-quantization — the same staging
        # as the OpenCL kernel, so CPU and GPU renders stay aligned.
        bin_deltas = _band_bin_lut(band_settings)
        if bin_deltas is not None:
            feather = float(band_settings.get('band_feather', _BAND_FEATHER_DEFAULT) or 0)
            img_norm = np.clip(img / 65535.0, 0.0, 1.0).astype(np.float32)
            img = _apply_color_bands_float(img_norm, bin_deltas, feather) * 65535.0

    img = np.clip(img, 0, 65535)
    return img.astype(np.uint16)


# --- Per-color-band "Subtractive Saturations" (Resolve-style color vectors) ---
# Each band is a hue sector with four sliders in [-100, 100]:
#   subsat — film-density saturation (same model as the global slider),
#   sat    — additive saturation,  bright — luminance gain,
#   hue    — rotation toward the neighboring sectors (±30° at full scale).
COLOR_BANDS = ("red", "skin", "yellow", "green", "cyan", "blue", "purple")
BAND_PARAMS = ("subsat", "sat", "bright", "hue")
BAND_ADJUSTMENT_KEYS = tuple(f"band_{color}_{param}"
                             for color in COLOR_BANDS for param in BAND_PARAMS)

# Sector centers in degrees on the HSV wheel (red=0, green=120, blue=240),
# in wheel order. "skin" sits in the orange range between red and yellow,
# like Resolve's dedicated skin-tone vector. "cyan" centers the wide
# green->blue span (true cyan = 180), splitting it into two even sectors.
# MUST stay sorted ascending by center — _band_param_deltas searchsorts it.
BAND_HUE_CENTERS = (
    ("red", 0.0), ("skin", 28.0), ("yellow", 58.0),
    ("green", 120.0), ("cyan", 180.0), ("blue", 240.0), ("purple", 300.0),
)

# Near-neutral pixels carry hue noise, not color: fade every band effect in
# over this HSV-saturation range so grays stay untouched.
_BAND_SAT_GATE_LO = 0.06
_BAND_SAT_GATE_HI = 0.20

_BAND_HUE_FULL_SCALE = 30.0   # hue slider ±100 → ±30°

_BAND_LUT_BINS = 720          # 0.5° hue resolution for the per-pixel lookup


_BAND_CENTERS_ARR = np.array([center for _name, center in BAND_HUE_CENTERS],
                             dtype=np.float32)


def _band_param_deltas(hue_deg: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Blend per-band parameter values over the hue wheel.

    `lut` is (num_bands, num_params) in BAND_HUE_CENTERS order. Each pixel
    sits in the sector between two adjacent band centers; the two bands'
    values are bridged with a smoothstep ramp, so a pixel is influenced by
    at most two bands and the blend weights always sum to 1. Returns
    hue_deg.shape + (num_params,) float32.
    """
    flat_hue = hue_deg.ravel()
    idx = np.searchsorted(_BAND_CENTERS_ARR, flat_hue, side="right") - 1
    last = len(_BAND_CENTERS_ARR) - 1
    wrap = idx == last                       # purple→red sector crosses 360
    nxt = np.where(wrap, 0, idx + 1)
    c0 = _BAND_CENTERS_ARR[idx]
    c1 = np.where(wrap, np.float32(360.0), _BAND_CENTERS_ARR[nxt])
    t = np.clip((flat_hue - c0) / (c1 - c0), 0.0, 1.0)
    ramp = (t * t * (3.0 - 2.0 * t)).astype(np.float32)[:, np.newaxis]
    out = lut[idx] * (1.0 - ramp) + lut[nxt] * ramp
    return out.reshape(hue_deg.shape + (lut.shape[1],))


def _band_bin_lut(settings: dict):
    """The (_BAND_LUT_BINS, 4) float32 table of blended per-param deltas by
    hue bin, or None when every band slider is 0. Shared by the numpy path
    and the OpenCL kernel so both apply identical curves."""
    names = [name for name, _center in BAND_HUE_CENTERS]
    lut = np.zeros((len(names), len(BAND_PARAMS)), dtype=np.float32)
    for color in COLOR_BANDS:
        lut[names.index(color)] = [
            float(settings.get(f"band_{color}_{param}", 0) or 0)
            for param in BAND_PARAMS]
    if not lut.any():
        return None
    bin_hues = np.arange(_BAND_LUT_BINS, dtype=np.float32) \
        * (360.0 / _BAND_LUT_BINS)
    return _band_param_deltas(bin_hues, lut)


def _band_weights(hue_deg: np.ndarray, needed) -> dict:
    """Per-band membership weight from the pixel hue (degrees, [0, 360)).
    Thin wrapper over _band_param_deltas with one-hot columns; mainly for
    tests — the pipeline blends parameters directly."""
    names = [name for name, _center in BAND_HUE_CENTERS]
    lut = np.zeros((len(names), len(needed)), dtype=np.float32)
    order = list(needed)
    for col, name in enumerate(order):
        lut[names.index(name), col] = 1.0
    blended = _band_param_deltas(hue_deg, lut)
    return {name: blended[..., col] for col, name in enumerate(order)}


def apply_color_band_adjustments(img16: np.ndarray, settings: dict) -> np.ndarray:
    """Apply the per-color-band sliders to a 16-bit RGB image.
    Returns img16 unchanged when every band slider is 0.

    Quantizes by TRUNCATION to match exactly the in-pipeline band path
    (adjust_image / the OpenCL kernel both truncate), so every entry point
    produces identical output."""
    bin_deltas = _band_bin_lut(settings)
    if bin_deltas is None:
        return img16
    feather = float(settings.get('band_feather', _BAND_FEATHER_DEFAULT) or 0)
    rgb = np.clip(img16.astype(np.float32) / 65535.0, 0.0, 1.0)
    rgb = _apply_color_bands_float(rgb, bin_deltas, feather)
    return np.clip(rgb * 65535.0, 0.0, 65535.0).astype(np.uint16)


# Spatial feathering: at feather=100 the band correction is low-passed with a
# Gaussian of this fraction of the image's long edge. Scaling by image size
# keeps the softness visually consistent between the 1080px preview and the
# full-resolution export.
_BAND_FEATHER_MAX_FRAC = 0.012

# Default feather amount when a settings dict carries no explicit band_feather
# (new edits and pre-feathering catalogs): a gentle edge-softening baseline.
# Mirrors SlidersPanel.SLIDER_DEFAULTS["band_feather"] so UI and render agree.
_BAND_FEATHER_DEFAULT = 10.0


def _feather_band_effect(orig: np.ndarray, adjusted: np.ndarray,
                         feather: float) -> np.ndarray:
    """Soften the band effect's hard spatial edges (the DaVinci-qualifier
    "blur radius" idea): the bands key per pixel from hue/sat, so the effect
    can jump abruptly across a colour boundary. Rather than blur the image
    (which would lose detail), low-pass only the *correction* the bands
    introduced and re-add it to the untouched original — detail is preserved,
    only the effect's transition is feathered.

    orig / adjusted: float32 RGB in [0, 1]. feather in 0..100 (0 = off).
    Like any matte blur this can bleed the correction slightly across
    high-contrast colour edges (a faint halo), so keep the amount modest.
    """
    if feather <= 0:
        return adjusted
    h, w = orig.shape[:2]
    sigma = (min(feather, 100.0) / 100.0) * _BAND_FEATHER_MAX_FRAC * max(h, w)
    if sigma < 0.5:
        return adjusted
    delta = adjusted - orig
    cv2.GaussianBlur(delta, (0, 0), sigmaX=sigma, sigmaY=sigma, dst=delta)
    out = orig + delta
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _apply_color_bands_float(rgb: np.ndarray, bin_deltas: np.ndarray,
                             feather: float = 0.0) -> np.ndarray:
    """Per-color-band core on a float32 RGB image in [0, 1].

    Band selection happens on the ORIGINAL hue, so a band's own hue
    rotation can't move pixels out of (or into) its influence. sat/bright/
    hue act in HSV; subsat then reuses the global subtractive-saturation
    (pin the dominant channel, power down the others) with a per-pixel
    strength. Operating in float (callers quantize once at the end) keeps
    the CPU path's staging identical to the OpenCL kernel's.

    The exact band blend is evaluated once on a small hue table
    (bin_deltas, from _band_bin_lut), then pixels look it up by hue bin —
    one int index + one take per active parameter instead of full-
    resolution interpolation math. 0.5° bins are far below visible
    precision for curves this smooth. May modify rgb in place; use the
    return value. When feather > 0 the band correction is spatially
    low-passed (see _feather_band_effect) before being returned.
    """
    orig = rgb.copy() if feather > 0 else None
    param_active = [bool(bin_deltas[:, p].any())
                    for p in range(len(BAND_PARAMS))]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)   # H in [0,360), S/V in [0,1]
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Linear interpolation between bins keeps the response continuous in
    # hue — nearest-bin steps would turn sub-LSB float drift into visible
    # delta jumps at bin edges (and break CPU/GPU agreement).
    pos = hue * (_BAND_LUT_BINS / 360.0)
    bin0 = np.minimum(pos.astype(np.int32), _BAND_LUT_BINS - 1)
    frac = pos - bin0
    bin1 = np.where(bin0 == _BAND_LUT_BINS - 1, 0, bin0 + 1)  # wrap to red
    del pos

    def _lookup(p):
        col = np.ascontiguousarray(bin_deltas[:, p])
        return np.take(col, bin0) * (1.0 - frac) + np.take(col, bin1) * frac

    subsat_d, sat_d, bright_d, hue_d = (
        _lookup(p) if param_active[p] else None
        for p in range(len(BAND_PARAMS)))
    del bin0, bin1, frac

    g = np.clip((sat - _BAND_SAT_GATE_LO) /
                (_BAND_SAT_GATE_HI - _BAND_SAT_GATE_LO), 0.0, 1.0)
    gate = g * g * (3.0 - 2.0 * g)
    del g

    if hue_d is not None:
        hsv[..., 0] = np.mod(
            hue + hue_d * (_BAND_HUE_FULL_SCALE / 100.0) * gate, 360.0)
        del hue_d
    if sat_d is not None:
        hsv[..., 1] = np.clip(sat * np.maximum(1.0 + sat_d / 100.0 * gate, 0.0),
                              0.0, 1.0)
        del sat_d
    if bright_d is not None:
        hsv[..., 2] = np.clip(val * np.exp2(bright_d / 100.0 * gate), 0.0, 1.0)
        del bright_d
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    strength = subsat_d * gate if subsat_d is not None else None
    # Hi-res frames make these intermediates big — drop them before the pow
    del hsv, hue, sat, val, gate, subsat_d

    if strength is not None:
        # Touch only affected pixels: the gather is linear in coverage, so
        # a band that owns 1/6 of the image costs ~1/6 of a full-image pow.
        active = np.flatnonzero(strength)
        if active.size:
            flat = rgb.reshape(-1, 3)
            sub = flat[active]
            gamma = np.exp2(strength.ravel()[active] / 100.0)[:, np.newaxis]
            mx = np.max(sub, axis=1, keepdims=True)
            ratio = np.clip(sub / np.maximum(mx, 1e-6), 1e-20, 1.0)
            # Ratios below 1e-6 are HSV round-trip noise (a real 1-count
            # channel is 1.5e-5): snap them to the 1e-20 floor, which is far
            # enough down that even gamma 0.5 maps it below half an LSB —
            # exactly-dark channels stay dark, like the global model's
            # pow(0, gamma) == 0 and the OpenCL kernel.
            ratio = np.where(ratio < 1e-6, np.float32(1e-20), ratio)
            flat[active] = np.where(mx > 1e-6,
                                    mx * np.exp(np.log(ratio) * gamma), sub)
            rgb = flat.reshape(rgb.shape)

    if orig is not None:
        rgb = _feather_band_effect(orig, rgb, feather)
    return rgb


def adjust_image_opencl(
    img16: np.ndarray,
    kelvin_shift: float = 0.0,
    tint_shift: float = 0.0,
    exposure: float = 0.0,
    brightness: float = 0.0,
    blackpoint: float = 0.0,
    whitepoint: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    tint_balance_factor: float = 1.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    ch_input_gain: float = 0.0,
    ch_master_shift: float = 0.0,
    ch_master_gain: float = 0.0,
    ch_r_shift: float = 0.0,
    ch_r_gain: float = 0.0,
    ch_r_blackpoint: float = 0.0,
    ch_g_shift: float = 0.0,
    ch_g_gain: float = 0.0,
    ch_g_blackpoint: float = 0.0,
    ch_b_shift: float = 0.0,
    ch_b_gain: float = 0.0,
    ch_b_blackpoint: float = 0.0,
    sub_saturation: float = 0.0,
    band_settings: dict = None,
    ws_windowed: bool = False,
) -> np.ndarray:
    """
    GPU-accelerated (OpenCL) version of adjust_image.
    Uses cached OpenCL context and compiled program for better performance.
    """
    global _opencl_cache

    # Working space: do the de-window + Gain/Exposure recovery + window clamp in
    # numpy here (cheap on the ≤1080px preview), then run the kernel on the
    # resulting normal full-range image with exposure consumed. Sharing this
    # pre-stage with the CPU path makes GPU/CPU parity automatic — the kernel
    # never sees headroom.
    if ws_windowed:
        img16 = _apply_working_space_recovery(img16, exposure, whitepoint,
                                              kelvin_shift, tint_shift, tint_balance_factor,
                                              ch_input_gain, ch_master_shift, ch_master_gain,
                                              ch_r_shift, ch_r_gain, ch_r_blackpoint,
                                              ch_g_shift, ch_g_gain, ch_g_blackpoint,
                                              ch_b_shift, ch_b_gain, ch_b_blackpoint)
        exposure = 0.0
        whitepoint = 0.0
        ws_windowed = False
        kelvin_shift = 0.0
        tint_shift = 0.0
        # Channel Levels ran FIRST, un-clamped, inside the pre-stage — zero it so
        # the kernel (and the CPU fallback below) can't re-apply it. This is what
        # makes GPU/CPU parity free on the windowed path: the kernel never sees
        # headroom OR Channel Levels. See spec/channel-levels-pre-clamp.md.
        ch_input_gain = ch_master_shift = ch_master_gain = 0.0
        ch_r_shift = ch_r_gain = ch_r_blackpoint = 0.0
        ch_g_shift = ch_g_gain = ch_g_blackpoint = 0.0
        ch_b_shift = ch_b_gain = ch_b_blackpoint = 0.0

    # White balance - flat per-channel gain done in numpy so the kernel (and the
    # CPU fallback) never apply WB -> automatic CPU/GPU parity. ws consumed above.
    if kelvin_shift != 0.0 or tint_shift != 0.0:
        _gr, _gg, _gb = _white_balance_gains(kelvin_shift, tint_shift, tint_balance_factor)
        img16 = img16.astype(np.float32)
        img16[..., 0] *= np.float32(_gr)
        img16[..., 1] *= np.float32(_gg)
        img16[..., 2] *= np.float32(_gb)
        kelvin_shift = 0.0
        tint_shift = 0.0

    # Spatial band feathering is a CPU post-blur the per-pixel kernel can't
    # do, so run the whole adjustment on the CPU when it is active (keeps the
    # band + feather staging identical). This is also the normal fallback when
    # OpenCL is unavailable.
    feather_active = bool(band_settings) and \
        (float(band_settings.get('band_feather', _BAND_FEATHER_DEFAULT) or 0) > 0) and \
        any(band_settings.get(k, 0) for k in BAND_ADJUSTMENT_KEYS)
    if feather_active or not _initialize_opencl():
        return adjust_image(img16, kelvin_shift, tint_shift, exposure, brightness,
                          blackpoint, whitepoint, contrast, saturation, tint_balance_factor,
                          highlights, shadows,
                          ch_input_gain, ch_master_shift, ch_master_gain,
                          ch_r_shift, ch_r_gain, ch_r_blackpoint,
                          ch_g_shift, ch_g_gain, ch_g_blackpoint,
                          ch_b_shift, ch_b_gain, ch_b_blackpoint,
                          sub_saturation=sub_saturation,
                          band_settings=band_settings)

    try:
      # Serialize GPU submissions: the hi-res zoom worker may run this
      # concurrently with the GUI thread's preview refresh.
      with _opencl_lock:
        # Use cached OpenCL objects
        ctx = _opencl_cache['ctx']
        queue = _opencl_cache['queue']
        kernel = _opencl_cache['kernel']

        img = img16.astype(np.float32)

        # Use the pre-calculated balance factor instead of calculating it
        balance_factor = tint_balance_factor

        img_flat = img.reshape(-1, 3)
        img_buf = cl_array.to_device(queue, img_flat)

        # The per-color-band deltas ship as a hue-binned LUT (the same
        # table the numpy path uses); a 4-float dummy keeps the kernel
        # argument valid when the bands are inactive.
        band_lut = _band_bin_lut(band_settings) if band_settings else None
        band_active = 1.0 if band_lut is not None else 0.0
        lut_flat = (np.ascontiguousarray(band_lut.ravel())
                    if band_lut is not None
                    else np.zeros(4, dtype=np.float32))
        lut_buf = cl_array.to_device(queue, lut_flat)

        # Prepare parameters as numpy array (params[0..10] existing,
        # params[11..22] channel levels, params[23] subtractive saturation,
        # params[24] per-color-band enable flag)
        params = np.array([
            kelvin_shift, tint_shift, exposure, brightness,
            blackpoint, whitepoint, contrast, saturation, balance_factor,
            highlights, shadows,
            ch_input_gain, ch_master_shift, ch_master_gain,
            ch_r_shift, ch_r_gain, ch_r_blackpoint,
            ch_g_shift, ch_g_gain, ch_g_blackpoint,
            ch_b_shift, ch_b_gain, ch_b_blackpoint,
            sub_saturation,
            band_active,
        ], dtype=np.float32)

        params_buf = cl_array.to_device(queue, params)

        # Execute the pre-compiled kernel
        n_pixels = img_flat.shape[0]
        kernel(queue, (n_pixels,), None, img_buf.data, params_buf.data,
               lut_buf.data, np.int32(n_pixels))

        # Get results and reshape
        result = img_buf.get().reshape(img.shape)
        return np.clip(result, 0, 65535).astype(np.uint16)

    except Exception as e:
        print(f"OpenCL processing failed: {e}")
        # Fallback to CPU version
        return adjust_image(img16, kelvin_shift, tint_shift, exposure, brightness,
                          blackpoint, whitepoint, contrast, saturation, tint_balance_factor,
                          highlights, shadows,
                          ch_input_gain, ch_master_shift, ch_master_gain,
                          ch_r_shift, ch_r_gain, ch_r_blackpoint,
                          ch_g_shift, ch_g_gain, ch_g_blackpoint,
                          ch_b_shift, ch_b_gain, ch_b_blackpoint,
                          sub_saturation=sub_saturation,
                          band_settings=band_settings)



def cleanup_opencl():
    """
    Clean up OpenCL resources. Call this when shutting down the application.
    """
    global _opencl_cache
    
    if not OPENCL_AVAILABLE:
        return
    
    try:
        if _opencl_cache['queue'] is not None:
            _opencl_cache['queue'].finish()
        
        # Reset all cached objects
        _opencl_cache['ctx'] = None
        _opencl_cache['queue'] = None
        _opencl_cache['program'] = None
        _opencl_cache['kernel'] = None
        _opencl_cache['device_name'] = None
        
        print("OpenCL resources cleaned up")
    except Exception as e:
        print(f"Error cleaning up OpenCL resources: {e}")


# ---------------------------------------------------------------------------
# Tone curves (Photoshop-style Curves control)
#
# Curve state is a dict of channel -> list of [x, y] control points in the
# 0..255 display domain, e.g.
#   {"rgb": [[0,0],[255,255]], "r": [...], "g": [...], "b": [...]}
# "rgb" is the composite ("All") curve applied to every channel before the
# per-channel r/g/b curves (matching Photoshop's compose order). Missing or
# malformed channels are treated as identity. See spec/curves-tone-control.md.
# ---------------------------------------------------------------------------

_CURVE_CHANNELS = ("rgb", "r", "g", "b")
_IDENTITY_POINTS = [[0.0, 0.0], [255.0, 255.0]]


def _normalize_points(points):
    """Sanitize a control-point list: keep finite [x,y] pairs, clamp to
    [0,255], sort by x, drop duplicate-x points. Returns None if the result is
    effectively the 2-point identity (so callers can skip it)."""
    if not points:
        return None
    cleaned = []
    for p in points:
        try:
            x = float(p[0]); y = float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cleaned.append([min(255.0, max(0.0, x)), min(255.0, max(0.0, y))])
    if len(cleaned) < 2:
        return None
    cleaned.sort(key=lambda q: q[0])
    # Drop points sharing an x with the previous one (keep the first).
    dedup = [cleaned[0]]
    for q in cleaned[1:]:
        if q[0] > dedup[-1][0]:
            dedup.append(q)
    if len(dedup) < 2:
        return None
    # Exactly the identity diagonal? Treat as no-op.
    if dedup == _IDENTITY_POINTS:
        return None
    return dedup


def _is_identity_curves(curves) -> bool:
    """True when curves is None/empty or every channel is the identity."""
    if not curves:
        return True
    for ch in _CURVE_CHANNELS:
        if _normalize_points(curves.get(ch)) is not None:
            return False
    return True


def _monotone_cubic(xs, ys, xq):
    """Fritsch-Carlson monotone cubic interpolation of (xs, ys) at xq.
    Prevents overshoot so the tone curve never inverts local contrast.
    xs must be strictly increasing. Returns a float array shaped like xq."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)
    if n == 2:
        # Straight line — exact, and the common (identity / single move) case.
        return np.interp(xq, xs, ys)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    # Tangents
    m = np.empty(n, dtype=np.float64)
    m[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    m[0] = delta[0]
    m[-1] = delta[-1]
    # Enforce monotonicity (Fritsch-Carlson)
    for i in range(n - 1):
        if delta[i] == 0.0:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            a = m[i] / delta[i]
            b = m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9.0:
                t = 3.0 / np.sqrt(s)
                m[i] = t * a * delta[i]
                m[i + 1] = t * b * delta[i]
    xq = np.asarray(xq, dtype=np.float64)
    # Segment index for each query point
    idx = np.clip(np.searchsorted(xs, xq) - 1, 0, n - 2)
    x0 = xs[idx]; x1 = xs[idx + 1]
    y0 = ys[idx]; y1 = ys[idx + 1]
    m0 = m[idx]; m1 = m[idx + 1]
    hh = (x1 - x0)
    t = (xq - x0) / hh
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * hh * m0 + h01 * y1 + h11 * hh * m1


def build_channel_lut(points) -> np.ndarray:
    """Build a 256-entry float curve (output 0..255) from control points in the
    0..255 domain using monotone cubic interpolation. Returns the identity ramp
    for an identity/invalid point set."""
    pts = _normalize_points(points)
    x = np.arange(256, dtype=np.float64)
    if pts is None:
        return x.astype(np.float32)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    y = _monotone_cubic(xs, ys, x)
    return np.clip(y, 0.0, 255.0).astype(np.float32)


# Shared LUT expansion: turn a 256-entry (0..255) curve into a full
# 65536 -> uint16 LUT. Used by both apply_curves (per-channel) and the Gamma
# luminance LUT, so the two stay bit-for-bit identical. The sample/source ramps
# are pure constants — built once at import, not per call.
_LUT16_SAMPLE = np.arange(65536, dtype=np.float32)
_LUT16_SRC_IDX = np.arange(256, dtype=np.float32) * (65535.0 / 255.0)


def _expand_curve256_to_lut16(curve256: np.ndarray) -> np.ndarray:
    """Expand a 256-entry (output 0..255) curve to a 16-bit input->output LUT."""
    return np.interp(_LUT16_SAMPLE, _LUT16_SRC_IDX,
                     curve256 * (65535.0 / 255.0)).astype(np.uint16)


def apply_curves(img16: np.ndarray, curves) -> np.ndarray:
    """Apply Photoshop-style tone curves to a 16-bit RGB image.

    curves: dict of channel ("rgb"/"r"/"g"/"b") -> list of [x,y] points in the
    0..255 domain. The composite "rgb" curve is applied first, then the
    per-channel curve. Identity channels are skipped. Returns a uint16 array;
    the input is returned unchanged when every channel is identity.
    """
    if _is_identity_curves(curves):
        return img16

    rgb_lut = build_channel_lut(curves.get("rgb"))     # 256 -> 0..255 float
    rgb_idx = np.clip(np.rint(rgb_lut), 0, 255).astype(np.intp)

    out = np.empty_like(img16)
    for c, key in enumerate(("r", "g", "b")):
        # Compose the per-channel curve over the composite curve in 0..255 space,
        # then expand to a full 16-bit LUT (shared with the Gamma luminance path).
        ch_lut = build_channel_lut(curves.get(key))
        lut16 = _expand_curve256_to_lut16(ch_lut[rgb_idx])
        out[..., c] = lut16[img16[..., c]]
    return out


# --- Gamma slider (center-point tone curve) --------------------------------
# The Gamma slider is a single control point on the composite ("rgb") tone
# curve that moves diagonally, perpendicular to the identity line, from the
# center (127.5, 127.5): toward the top-left to brighten (+) or the bottom-right
# to darken (-). It reuses the exact monotone-cubic interpolation of the Curves
# editor (build_channel_lut / _monotone_cubic via apply_curves), so the result
# is identical to dragging that midpoint by hand. Endpoints stay pinned, so the
# effect is confined to the visible [0,1] window. See spec/gamma-slider.md.
_GAMMA_MAX_OFFSET = 63.75   # perpendicular offset (0..255 domain) at slider +-100


def gamma_curve_points(gamma: float):
    """The 3-point 'rgb' control-point list for a Gamma slider value, in the
    0..255 domain used by the Curves editor. gamma=0 -> identity diagonal."""
    offset = (float(gamma) / 100.0) * _GAMMA_MAX_OFFSET
    return [[0.0, 0.0], [127.5 - offset, 127.5 + offset], [255.0, 255.0]]


def _gamma_lut16(gamma: float) -> np.ndarray:
    """Full 16-bit LUT (65536 -> uint16) for the Gamma tone curve, expanded the
    same way apply_curves builds the composite 'rgb' LUT — including its 256-level
    rounding (apply_curves composes through an identity per-channel ramp with
    np.rint) — so a neutral pixel renders bit-for-bit identically in both the
    per-channel and luminance paths."""
    curve256 = np.rint(build_channel_lut(gamma_curve_points(gamma)))   # 256 levels
    return _expand_curve256_to_lut16(curve256)


# Rec.601 luma weights (R, G, B) — the same weights the saturation / shadow
# stages in this module already use.
_GAMMA_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _apply_gamma_luminance(img16: np.ndarray, gamma: float) -> np.ndarray:
    """Hue-preserving Gamma: apply the tone curve to luminance only, then scale
    the RGB channels by the ratio L'/L. Uniform scaling leaves chromaticity (hue
    + HSV saturation) untouched except where a channel clips. Neutrals render
    bit-for-bit identically to the per-channel path (out = L'/L * v = L' when
    R=G=B=v). See spec/gamma-luminance-mode.md."""
    lut16 = _gamma_lut16(gamma)
    rgbf = img16.astype(np.float32)
    lum = rgbf @ _GAMMA_LUMA                                       # (H, W)
    idx = np.clip(np.rint(lum), 0, 65535).astype(np.uint16)
    lum_out = lut16[idx].astype(np.float32)
    # Floor the denominator at one count: pure black (lum==0 -> lum_out==0) stays
    # black, and near-black pixels can't blow up k (which would amplify shadow
    # chroma noise).
    k = lum_out / np.maximum(lum, 1.0)
    out = rgbf * k[..., np.newaxis]
    # Round (not truncate) so a neutral pixel recovers lut16[v] exactly: for
    # R=G=B=v, out = v * lut16[v] / v differs from lut16[v] only by float error.
    return np.clip(np.rint(out), 0.0, 65535.0).astype(np.uint16)


def apply_gamma_curve(img16: np.ndarray, gamma: float,
                      luminance: bool = False) -> np.ndarray:
    """Apply the Gamma slider as a center-point tone curve. No-op at 0.

    luminance=False (default): apply the curve PER CHANNEL (via apply_curves) —
    the standard filmic look, which shifts hue/saturation of non-neutral colours.
    luminance=True: apply the curve to luminance and scale RGB together, so hue
    and HSV saturation are preserved (except where a channel clips)."""
    if not gamma:
        return img16
    if luminance:
        return _apply_gamma_luminance(img16, gamma)
    return apply_curves(img16, {"rgb": gamma_curve_points(gamma)})


# --- Cineon film log → Rec.709 (γ 2.2) display conversion -------------------
# Optional FINAL pipeline stage (the "Cineon Log → Rec.709" checkbox in the
# Channel Levels section; settings key "cineon_log"): interpret the fully-
# adjusted image as Cineon printing-density log and convert to Rec.709 video
# with a plain 2.2 gamma. Standard Kodak constants: 10-bit code 95 = black
# (Dmin), 685 = 90% white — the same codes the Scopes parade marks — with
# 0.002 log10 density per code over a 0.6 negative gamma. Values above 685
# clip to white (the classic video-range conversion, no soft shoulder). See
# spec/cineon-display-transform.md.
CINEON_BLACK_CODE = 95.0
CINEON_WHITE_CODE = 685.0
_CINEON_NEG_GAMMA = 0.6
_CINEON_DENSITY_PER_CODE = 0.002

_cineon_lut16_cache = None


def _cineon_rec709_lut16() -> np.ndarray:
    """65536-entry uint16 LUT: 16-bit input (≡ 10-bit code · 1023/65535) →
    Rec.709 gamma-2.2 output. Built once and cached."""
    global _cineon_lut16_cache
    if _cineon_lut16_cache is None:
        code = np.linspace(0.0, 1023.0, 65536, dtype=np.float64)
        gain = _CINEON_DENSITY_PER_CODE / _CINEON_NEG_GAMMA
        off = 10.0 ** ((CINEON_BLACK_CODE - CINEON_WHITE_CODE) * gain)
        lin = (10.0 ** ((code - CINEON_WHITE_CODE) * gain) - off) / (1.0 - off)
        lin = np.clip(lin, 0.0, 1.0)
        _cineon_lut16_cache = np.round(
            np.power(lin, 1.0 / 2.2) * 65535.0).astype(np.uint16)
    return _cineon_lut16_cache


def apply_cineon_to_rec709(img16: np.ndarray) -> np.ndarray:
    """Cineon log → Rec.709 (γ 2.2), per channel. Applied after ALL other
    adjustments so preview, hi-res zoom and export transform identically."""
    return _cineon_rec709_lut16()[img16]


# --- Dust removal (spot inpainting) ----------------------------------------
# Dust edits are stored as NORMALIZED spots (fractions of width/height) on the
# image, so the same definition reproduces at the 1080px preview, the hi-res
# zoom detail and the full-res export — the resolution-independent contract the
# area masks and reference-norm replay already honor. See spec/dust-removal.md.

def rasterize_dust_mask(spots, h: int, w: int) -> np.ndarray:
    """Rasterize dust spots into a uint8 {0,255} mask of shape (h, w).

    spots: list of {"kind", "pts": [[x,y],...], "r"} where x,y are normalized
    over width/height and r is a fraction of WIDTH. Each spot is the union of
    equal-radius circular dabs along its polyline (consecutive points joined by
    a thick line so a fast drag leaves no gap). r is width-based in both axes,
    so a spot rasterizes to a true pixel circle at any resolution.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if not spots or w <= 0 or h <= 0:
        return mask
    for spot in spots:
        pts = spot.get("pts") or []
        r_px = max(1, int(round(float(spot.get("r", 0.0)) * w)))
        prev = None
        for p in pts:
            try:
                x = int(round(float(p[0]) * w))
                y = int(round(float(p[1]) * h))
            except (TypeError, ValueError, IndexError):
                continue
            cv2.circle(mask, (x, y), r_px, 255, -1)
            if prev is not None:
                cv2.line(mask, prev, (x, y), 255, thickness=2 * r_px)
            prev = (x, y)
    return mask


def _crop_keep_mask(rect, angle, h: int, w: int):
    """uint8 {0,1} mask of the pixels a confirmed crop KEEPS — the crop
    rectangle (normalized x1, y1, x2, y2 over width/height), rotated by
    `angle` about its center (Qt clockwise convention; the raster twin of
    _extract_rotated_crop / apply_crop_to_image). None when degenerate."""
    x1, y1, x2, y2 = [float(v) for v in rect]
    bw, bh = (x2 - x1) * w, (y2 - y1) * h
    if bw <= 1 or bh <= 1:
        return None
    cx, cy = (x1 + x2) / 2.0 * w, (y1 + y2) / 2.0 * h
    a = np.deg2rad(float(angle or 0.0))
    ca, sa = float(np.cos(a)), float(np.sin(a))
    corners = np.array(
        [(cx + ca * ux - sa * uy, cy + sa * ux + ca * uy)
         for ux, uy in ((-bw / 2, -bh / 2), (bw / 2, -bh / 2),
                        (bw / 2, bh / 2), (-bw / 2, bh / 2))], np.int32)
    keep = np.zeros((h, w), np.uint8)
    cv2.fillPoly(keep, [corners], 1)
    return keep


# Clone-heal tuning. The fill copies the best-matching CLEAN patch from the
# spot's neighborhood (real texture and grain, 16-bit native) instead of
# diffusing an average inward — diffusion (cv2.inpaint) produces a smooth,
# grainless patch with radial fan-like streaks that stands out on film scans.
_HEAL_RING = 3            # min ring width of clean context (matching + tone)
_HEAL_GUARD = 2           # min gap (px) between the hole and its context ring
_HEAL_ANGLES = 16         # candidate source directions searched around a spot
                          # (the two NEAREST rings sample at double density)
# The automatic sampling rule (maintainer, verbatim — see _heal_patch): the
# sample area must TOUCH the patch area, and among the touching candidates
# the lowest combined Δhue+Δsat+ΔV (vs the patch's own background) plus the
# lowest texture wins. Total selection — no distance rings, no gates, no
# fallback ranking. The former ring-sweep constants died with the sweep.
_HEAL_MIN_RING_PX = 8     # need at least this much clean ring to heal
_HEAL_SEG_MIN = 32        # min heal-segment size (px) for long strokes
_HEAL_SEG_THICKNESS = 6.0  # segment size as a multiple of hole thickness
_HEAL_SEG_MAX = 96        # segment-size cap (px at plan scale): a big compact
                          # dab otherwise becomes ONE segment whose source
                          # window (bbox + thickness-scaled pad) rarely fits
                          # anywhere clean, dumping the whole blob into the
                          # flat, grainless diffusion fallback — a visible
                          # smooth disc on film grain. Tiles keep cloning
                          # real texture from beside the blob.
_DLIKE_SEP_MADS = 6.0     # defect-vs-surround separation (in ring-grain MADs)
                          # for the dlike_on plan verdict (metadata) and the
                          # source-interior tolerance in the SSD search
_HEAL_MIN_KEEP_FRAC = 0.2  # ring share the defect rejection must leave; below
                           # it the estimate is distrusted (see _heal_patch)
_HEAL_BLACK_FLOOR = 2000   # ~3% of white: below this a ring pixel reads as
                           # film holder / rebate, not scene (dust inverts
                           # bright; the unexposed border inverts to black)
DUST_FEATHER_DEFAULT = 0.25   # feather ramp width, fraction of each hole's
                              # own half-thickness (its local radius), so the
                              # fade scales with the brush size (user-adjustable
                              # per image via the dust panel)


def _box_sum(integ: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    """Sum of a uint8 map over [y0:y1, x0:x1] via its cv2.integral image."""
    return int(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])


def _feather_alpha(mask: np.ndarray, fmap: np.ndarray) -> np.ndarray:
    """Per-pixel blend alpha for the fill: rises from ~0 at each hole's
    boundary to 1 over `fmap` px inward (smoothstep ramp); exactly 0 outside
    the mask. `fmap` holds each hole's feather ramp width in px (float;
    <= 0.5 = hard fill). Depths are taken at PIXEL CENTERS (inside - 0.5) so
    the relative alpha profile is resolution-stable: a 1 px ramp at preview
    scale and its 3 px equivalent at export cover the same fraction of the
    hole with the same average alpha — an integer ramp with a boundary lift
    made thin-stroke rims visibly harder at preview than at export."""
    inside = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    f = np.maximum(fmap.astype(np.float32), 0.5)
    return _smoothstep((inside - 0.5) / f) * (mask > 0)


def _hsv1(rgb):
    """(h, s, v) of one RGB triple (float 0..65535) — all three in 0..1."""
    px = np.asarray(rgb, np.float32).reshape(1, 1, 3) / 65535.0
    hh, ss, vv = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0]
    return float(hh) / 360.0, float(ss), float(vv)


def _heal_patch(img16: np.ndarray, img16_c: np.ndarray, labels: np.ndarray,
                comp: int, bbox: tuple,
                half_th: float, mask_pad: np.ndarray, integ: np.ndarray,
                filled: np.ndarray, fmap: np.ndarray, feather: float,
                src_off=None, forced_flags=None,
                sample=None, manual=False, auto=False, ws_windowed=False,
                pref_off=None):
    """Fill one hole patch (a compact component, or one segment of a long
    stroke) by cloning the best-matching nearby clean patch.

    The hole's surrounding ring of known context is compared against candidate
    windows on rings of directions/distances around it; the source whose RING
    pixels best match (SSD) is cloned into the hole, with a smooth per-channel
    tone correction interpolated from the ring differences (a cheap membrane,
    so gradients continue through the patch while its grain is kept verbatim).
    `bbox` = (bx0, by0, bx1, by1) tight bounds of THIS patch's hole pixels;
    `half_th` = the component's half-thickness (max distance transform), its
    local scale. Writes the healed hole pixels into `filled` and returns
    `(offset, genuine, dlike_on)` — the chosen source-window offset
    `(dy, dx)` (relative to the match window, in this buffer's pixels) plus
    the patch's content-adaptive verdicts (rejection trusted; defect
    distinct — plan metadata), which the plan records so replays reproduce them —
    or None when no fully clean, in-bounds source window exists (caller
    falls back to diffusion inpaint for this patch).
    `src_off`: a pre-planned offset to REUSE (scale replay / a prior edit's
    plan) — always taken verbatim (clamped into bounds), never re-searched:
    once a patch's reference is set, nothing may move it. When dust has
    landed on the pinned source and `sample` is None, the patch returns a
    4th element "defer" instead of filling — the caller re-runs it after
    the Telea pass with `sample` = the healed buffer, so it clones healed
    content at the SAME location. `forced_flags`: the plan's
    (genuine, dlike_on) verdicts to reuse instead of re-deriving them from
    this buffer's pixels (grain statistics shift with resampling, and a
    flipped verdict is a visible preview/export difference). `img16_c`:
    img16 smoothed to the PLAN scale (identity on canonical buffers) — all
    PER-PIXEL threshold classifications (holder-black, defect-like) read it,
    so full-res grain crosses those thresholds no more often than the
    preview's area-averaged pixels did.

    Everything local is keyed off the thickness, never the bbox: a traced
    hair's bbox can span half the frame while the stroke is a few px wide.
    Three defenses keep the defect itself out of the tone anchor (a defect's
    soft edge leaks past the brushed mask; anchoring on it lifted whole
    strokes into bright ghosts):
      - the ring starts a thickness-scaled GUARD gap away from the hole;
      - ring pixels that are luma outliers (3 scaled-MAD) against the ring's
        OUTER half (farthest from the hole = least contaminated) are rejected
        from both matching and tone;
      - the membrane sigma is thickness-local, so any contamination that still
        survives can only tint its own neighborhood, not the whole stroke.
    """
    h, w = labels.shape[:2]
    bx0, by0, bx1, by1 = bbox

    guard = max(_HEAL_GUARD, int(round(0.5 * half_th)))
    ring_w = max(_HEAL_RING, guard + 3)
    pad = guard + ring_w
    wx0 = max(0, bx0 - pad)
    wy0 = max(0, by0 - pad)
    wx1 = min(w, bx1 + pad)
    wy1 = min(h, by1 + pad)

    # The hole is only THIS patch's pixels: component pixels inside the window
    # but outside the bbox belong to a neighboring segment's call.
    hole = labels[wy0:wy1, wx0:wx1] == comp
    hole[:by0 - wy0] = False
    hole[by1 - wy0:] = False
    hole[:, :bx0 - wx0] = False
    hole[:, bx1 - wx0:] = False
    away = cv2.distanceTransform(
        (~hole).astype(np.uint8), cv2.DIST_L2, 3)  # px from the hole
    ring = ((away > guard) & (away <= pad)
            & (mask_pad[wy0:wy1, wx0:wx1] == 0))

    dst = img16[wy0:wy1, wx0:wx1].astype(np.float32)
    dst_c = img16_c[wy0:wy1, wx0:wx1].astype(np.float32)
    # Film-holder / rebate pixels are not scene context: the unexposed
    # border inverts to near-black, and letting it into the ring drags the
    # membrane's tone correction toward black (a dark smudge on strokes
    # near the frame edge) and pollutes source matching. Drop near-black
    # pixels while they are the ring's minority; if they dominate, the
    # stroke really is in dark content and the ring stays.
    dark = dst_c.mean(axis=2) < _HEAL_BLACK_FLOOR
    if int((ring & dark).sum()) < 0.5 * int(ring.sum()):
        ring &= ~dark
    if int(ring.sum()) < _HEAL_MIN_RING_PX:
        return None  # boxed in by other spots — no context to match against

    # Robust rejection: the defect leaks past the mask into the ring — its
    # soft edge, or its whole continuation past the stroke's end, where it can
    # even be the ring's MAJORITY (so no ring-population statistic works). The
    # discriminative signal is the DEFECT'S COLOR, estimated from the hole's
    # own content: for a tight stroke the hole IS the defect; for a generous
    # brush the defect is the hole's outlier mode vs the ring (the background
    # majority inside the hole matches the ring, the defect does not). Ring
    # pixels colored like the defect are its leak: reject anything closer to
    # the defect color than HALF the clean cluster's distance (p95 of the
    # ring's defect-distances — valid for any contamination share below
    # ~95%). When the "defect" is indistinct from the surround (generous
    # brush over mostly-clean area) the distances collapse toward 0 and
    # nothing meaningful is rejected; legit bimodal structure (a sky/roof
    # edge) survives because both modes sit far from the defect color.
    ry, rx = np.nonzero(ring)
    ring_px = dst[ry, rx]
    hole_px = dst[hole]
    ring_med = np.median(ring_px, axis=0)
    dh = np.abs(hole_px - ring_med).mean(axis=1)
    defect = np.median(hole_px[dh >= np.percentile(dh, 75.0)], axis=0)
    d_def = np.abs(ring_px - defect).mean(axis=1)
    thr = 0.5 * float(np.percentile(d_def, 95.0))
    keep = d_def >= thr
    # The rejection is trusted only under the WHITE-DUST PRIOR, validated on
    # the SPLIT IT PRODUCED (no ring population is trustworthy a priori — a
    # hair's continuation can be the ring's majority, including its outer
    # band): film dust and the bright hairs this rejection was built for
    # read BRIGHTER than the context they leak over, so the REJECTED pixels
    # must be brighter than the KEPT ones by a noise margin. For a faint
    # speck or a clean-ish hole the "defect" estimate collapses onto the
    # BACKGROUND itself; the rejection then strips the background and
    # whatever foreign minority remains (a wall across an edge, a curb)
    # becomes the ENTIRE matching/tone anchor — the search then hunts for
    # wrong-toned patches far away while perfect same-tone context sits
    # right beside the spot. That inverted split fails this check (its
    # "rejected" side is the darker or equal one) and the full ring stays.
    rejected_brighter = False
    if bool(keep.any()) and not bool(keep.all()):
        kept_med = np.median(ring_px[keep], axis=0)
        kept_mad = float(np.median(
            np.abs(ring_px[keep] - kept_med).mean(axis=1))) + 1e-3
        rejected_brighter = (
            float(np.median(ring_px[~keep], axis=0).mean())
            > float(kept_med.mean()) + 2.0 * kept_mad)
    # Sanity cap on the rejection: discarding nearly the whole ring means
    # the "defect" was estimated as the ring's own MAJORITY — a generous
    # brush over clean content, where the top-quartile estimate collapses
    # onto the background. Whatever survives is then some small foreign
    # mode (a black film-holder border, a dark roofline) that becomes the
    # ENTIRE matching/tone anchor: clean sky strokes healed to solid black,
    # cloned from the border. A real defect's leak never crowds out this
    # much context (even a thick traced hair leaves ~a third of its ring
    # clean), so distrust the estimate and keep the full ring (`genuine`
    # also rides the plan as metadata).
    genuine = (rejected_brighter
               and int(keep.sum()) >= _HEAL_MIN_KEEP_FRAC * keep.size)
    if auto:
        # The strip is a MANUAL-stroke defense (a traced hair's leak). An
        # auto spot cannot leak — its hole is SPOT_SCALE× the speck's core,
        # the skirt sits inside it — so the strip's misfires (anchoring the
        # membrane tone on foreign ring structure across an edge) are its
        # only possible effect here.
        genuine = False
    if forced_flags is not None:
        genuine = bool(forced_flags[0])
    if genuine:
        if int(keep.sum()) < _HEAL_MIN_RING_PX:
            return None
        ring[:] = False
        ring[ry[keep], rx[keep]] = True
    dst_ring = dst[ring]
    ring_bg = np.median(dst_ring, axis=0)
    ring_mad = float(np.median(np.abs(dst_ring - ring_bg).mean(axis=1)))

    def _touches_patch(ty, tx):
        """Does the sample area (hole shifted by (ty, tx)) come within the
        rule's ±4 px of THE PATCH (labels == comp — the whole spot, not just
        this segment's tile)? Used by both the pinned-offset validation and
        the per-patch shared-offset (pref) path so plan time and replay
        agree: a shared offset that stops touching where the patch boundary
        curves away must re-pick at BOTH, or the plans drift."""
        gy0 = max(0, min(wy0, wy0 + ty) - 5)
        gx0 = max(0, min(wx0, wx0 + tx) - 5)
        gy1 = min(h, max(wy1, wy1 + ty) + 5)
        gx1 = min(w, max(wx1, wx1 + tx) + 5)
        comp_near = cv2.dilate(
            (labels[gy0:gy1, gx0:gx1] == comp).astype(np.uint8),
            np.ones((9, 9), np.uint8)).astype(bool)
        hy_, hx_ = np.nonzero(hole)
        py_ = hy_ + (wy0 + ty - gy0)
        px_ = hx_ + (wx0 + tx - gx0)
        okp = ((py_ >= 0) & (py_ < gy1 - gy0)
               & (px_ >= 0) & (px_ < gx1 - gx0))
        return bool(comp_near[py_[okp], px_[okp]].any())

    best_src, best_off, best_rv = None, None, None
    if src_off is not None:
        # Pinned source (scale replay, or a prior edit's plan): the offset
        # is taken VERBATIM — once a patch's reference is set, NOTHING may
        # move it. It is only clamped into bounds (scale rounding can push
        # the window a pixel past the frame). Same pixels-actually-read rule
        # as the search (so a plan picked beside a neighboring speck replays
        # identically at every scale): a clean hole projection samples raw
        # pixels, with the ring's clean subset driving the tone. Only when
        # dust has landed on the FILL pixels themselves (a new stroke
        # painted over the sampled patch) is the segment DEFERRED and re-run
        # after the Telea pass with `sample` = the healed buffer: it clones
        # the healed content at the very same location instead of
        # re-searching elsewhere.
        dy = min(max(int(src_off[0]), -wy0), h - wy1)
        dx = min(max(int(src_off[1]), -wx0), w - wx1)
        if not manual:
            # THE RULE RETROACTS ON AUTOMATIC PINS. Stored automatic offsets
            # that do not TOUCH their patch are pre-rule fossils: catalog
            # plans carry them, and sticky binding propagated them into
            # fresh, validly-keyed plans on every re-detect for as long as
            # the old search existed — pruning by spot identity can never
            # catch that. An automatic pin is honoured only if its sample
            # still touches THE PATCH — the whole spot (labels == comp), not
            # just this segment's tile: a long stroke shares ONE patch-
            # touching offset across its segments, and a tile-local test
            # wrongly rejected that shared offset on the stroke's far tiles.
            # ±4 px tolerance covers cross-scale raster rounding; a sample
            # overlapping dust routes through the existing clean/defer
            # machinery below. USER re-picks (manual=True) keep any
            # distance — that pick is the user's explicit choice.
            if not _touches_patch(dy, dx):
                src_off = None  # fossil pin — fall through to the rule
    if src_off is not None:
        best_off = (dy, dx)
        if _box_sum(integ, wy0 + dy, wx0 + dx, wy1 + dy, wx1 + dx) == 0:
            best_src = img16[wy0 + dy:wy1 + dy,
                             wx0 + dx:wx1 + dx].astype(np.float32)
        else:
            # Same cleanliness criterion as the fresh pick (RAW mask for the
            # sample area): a TOUCHING source hugs the patch border, and the
            # 1 px mask_pad dilation flipping the verdict at another scale
            # made replays reroute through the deferred pass — preview and
            # export diverged on the very segments the rule pins. Scale
            # rounding can still push a tangent offset ~1 px INTO the mask;
            # such an offset is rescued by the nearest clean offset within a
            # ≤4 px Chebyshev spiral (essentially the same patch) before the
            # deferred pass is considered.
            def _raw_clean(oy, ox):
                if (wy0 + oy < 0 or wx0 + ox < 0
                        or wy1 + oy > h or wx1 + ox > w):
                    return False
                win = labels[wy0 + oy:wy1 + oy, wx0 + ox:wx1 + ox]
                return not (win[hole] > 0).any()
            if not _raw_clean(dy, dx):
                for rr in range(1, 5):
                    hit = None
                    for ddy in range(-rr, rr + 1):
                        for ddx in range(-rr, rr + 1):
                            if max(abs(ddy), abs(ddx)) != rr:
                                continue
                            if _raw_clean(dy + ddy, dx + ddx):
                                hit = (dy + ddy, dx + ddx)
                                break
                        if hit:
                            break
                    if hit:
                        dy, dx = hit
                        best_off = (dy, dx)
                        break
            # Raw-mask ring subset, matching the fresh pick (scale-stable
            # tone anchor — see the fresh branch below).
            rv_pin = labels[wy0 + dy:wy1 + dy, wx0 + dx:wx1 + dx][ring] == 0
            if _raw_clean(dy, dx) and bool(rv_pin.any()):
                best_src = img16[wy0 + dy:wy1 + dy,
                                 wx0 + dx:wx1 + dx].astype(np.float32)
                if not rv_pin.all():
                    best_rv = rv_pin
            elif sample is not None:
                best_src = sample[wy0 + dy:wy1 + dy,
                                  wx0 + dx:wx1 + dx].astype(np.float32)
            else:
                g0, d0 = forced_flags if forced_flags is not None else (True,
                                                                        False)
                return best_off, g0, d0, "defer"
    if best_src is None:
        # ========== AUTOMATIC SAMPLING RULE (maintainer, verbatim) ==========
        # Applies to EVERY automatically picked initial sample — AI-detected
        # spots and painted strokes alike (only an explicit user re-pick is
        # exempt, and it arrives as a pinned src_off above):
        # 1. The sample area MUST TOUCH the patch area — the border of the
        #    sample area and the border of the patch share at least one
        #    pixel of contact (union contiguous). Not 500 px away, not
        #    50 px away: touching.
        # 2. Among the touching candidates pick the lowest combined
        #    delta-hue + delta-saturation + delta-V (vs the patch's own
        #    background) plus the lowest texture. A TOTAL selection: the
        #    best touching candidate always wins — no distance escalation,
        #    no "good enough" gate, no fallback ranking.
        # 3. (Again) the sample area and the patch area must touch.
        # Reference = the darker half of the HOLE's own pixels (film dust is
        # white, so the darker half is the background under/around the
        # defect). No ring statistic participates in the choice — a shadow
        # band or wall crossing the ring cannot redirect the anchor.
        # On a WINDOWED working-space buffer the hue/sat/value deltas are
        # computed on de-windowed display values (container codes compress
        # them 64×, hue degenerating first).
        if ws_windowed:
            dwk = 65535.0 / (WS_W - WS_B)

            def _dw(v):
                return np.clip((np.asarray(v, np.float32) - WS_B) * dwk,
                               0.0, 65535.0)
        else:
            dwk = 1.0

            def _dw(v):
                return v
        # ONE sample offset per PATCH: the rule speaks of "the patch area"
        # as a unit — a long stroke is healed in internal segments, and
        # letting each segment argmin independently put differently-phased
        # clones side by side (a visible seam inside one stroke, whose
        # raster position drifted between preview and export). The
        # component's first segment runs the argmin; its offset is reused
        # by the rest of the patch whenever it stays in-bounds and clean
        # (a curled stroke folding into the offset re-runs the argmin).
        if pref_off is not None:
            def _pref_ok(oy, ox):
                if (wy0 + oy < 0 or wx0 + ox < 0
                        or wy1 + oy > h or wx1 + ox > w):
                    return None
                lw = labels[wy0 + oy:wy1 + oy, wx0 + ox:wx1 + ox]
                if (lw[hole] > 0).any():
                    return None
                return lw[ring] == 0
            pdy, pdx = int(pref_off[0]), int(pref_off[1])
            rvp = _pref_ok(pdy, pdx) if _touches_patch(pdy, pdx) else None
            if rvp is None:
                # A raster-level graze (the offset has a small along-stroke
                # component) must NUDGE the shared offset, not abandon it —
                # an independent re-pick here puts a differently-phased
                # clone mid-stroke (a visible seam that also drifts between
                # preview and export). Same ≤4 px spiral as the replay
                # rescue: essentially the same patch.
                for rr in range(1, 5):
                    for ddy in range(-rr, rr + 1):
                        for ddx in range(-rr, rr + 1):
                            if max(abs(ddy), abs(ddx)) != rr:
                                continue
                            if not _touches_patch(pdy + ddy, pdx + ddx):
                                continue
                            rvp = _pref_ok(pdy + ddy, pdx + ddx)
                            if rvp is not None and bool(rvp.any()):
                                pdy, pdx = pdy + ddy, pdx + ddx
                                break
                            rvp = None
                        if rvp is not None:
                            break
                    if rvp is not None:
                        break
            if rvp is not None and bool(rvp.any()):
                best_src = img16[wy0 + pdy:wy1 + pdy,
                                 wx0 + pdx:wx1 + pdx].astype(np.float32)
                best_off = (pdy, pdx)
                best_rv = None if bool(rvp.all()) else rvp
    if best_src is None:
        hl = hole_px.mean(axis=1)
        ref = _dw(np.median(hole_px[hl <= np.median(hl)], axis=0))
        ref_h, ref_s, ref_v = _hsv1(ref)
        best_comb = None
        # THE FINITE TOUCHING SET: every translation that places a congruent
        # copy of the patch so its border shares contact with the patch
        # border — no angular sampling, no gaps. Translations that OVERLAP
        # the patch form the support of the shape's autocorrelation (the
        # Minkowski sum of the hole with its own reflection); the touching
        # set is that support's outer 8-connected boundary. Finite and
        # complete: every member touches by construction, and every possible
        # touching placement is a member. The sample-area dust check uses
        # the RAW mask (labels) — the 1 px mask_pad dilation would push the
        # sample off the border it must touch — and the membrane's ring
        # subset derives from the raw mask too (scale-stable, where
        # mask_pad's fixed dilation drifted preview vs export).
        hys, hxs = np.nonzero(hole)
        K = hole[int(hys.min()):int(hys.max()) + 1,
                 int(hxs.min()):int(hxs.max()) + 1].astype(np.float32)
        kh, kw = K.shape
        pyk, pxk = kh + 2, kw + 2
        canvas = np.zeros((kh + 2 * pyk, kw + 2 * pxk), np.float32)
        canvas[pyk:pyk + kh, pxk:pxk + kw] = K
        overlap = cv2.matchTemplate(canvas, K, cv2.TM_CCORR) > 0.5
        touching = (cv2.dilate(overlap.astype(np.uint8),
                               np.ones((3, 3), np.uint8)).astype(bool)
                    & ~overlap)
        t_ys, t_xs = np.nonzero(touching)
        for dy, dx in zip((t_ys - pyk).tolist(), (t_xs - pxk).tolist()):
            sy0, sx0, sy1, sx1 = wy0 + dy, wx0 + dx, wy1 + dy, wx1 + dx
            if sy0 < 0 or sx0 < 0 or sy1 > h or sx1 > w:
                continue
            # Pre-existing invariant: dust is never cloned — a touching
            # placement whose SAMPLE AREA overlaps any spot is not scene.
            lbl_win = labels[sy0:sy1, sx0:sx1]
            if (lbl_win[hole] > 0).any():
                continue
            rv2 = lbl_win[ring] == 0
            if not rv2.any():
                continue  # membrane needs ≥1 clean ring px (NaN guard)
            src = img16[sy0:sy1, sx0:sx1].astype(np.float32)
            area = src[hole]
            ah, as_, av = _hsv1(_dw(np.median(area, axis=0)))
            dh_ = abs(ah - ref_h)
            dh_ = min(dh_, 1.0 - dh_) * 2.0 * max(as_, ref_s)
            tex = float(area.mean(axis=1).std()) * dwk / 65535.0
            comb = dh_ + abs(as_ - ref_s) + abs(av - ref_v) + tex
            if best_comb is None or comb < best_comb:
                best_comb = comb
                best_src, best_off = src, (dy, dx)
                best_rv = None if bool(rv2.all()) else rv2
        if best_src is None:
            # No touching candidate EXISTS: every touching direction is out
            # of frame (a patch nearly as big as the buffer) or inside other
            # dust. There is no sample to select — dust pixels are never
            # cloned (pre-existing invariant) — so this degenerates to the
            # boundary-diffusion fill (Telea), which grows the fill from the
            # patch's own touching border.
            return None

    if best_src is None:
        return None

    if manual:
        # USER-PICKED source (the dragged reticle): clone-stamp semantics —
        # the patch is copied VERBATIM, color and texture alike, blended
        # only by the feather at the rim. The membrane below exists to make
        # AUTOMATIC picks seamless (texture from the source, low
        # frequencies from the destination); applied to an explicit re-pick
        # it fought the user — the old color stayed and the picked look
        # was toned away.
        heal = best_src
    else:
        # Smooth per-channel tone correction from the ring differences:
        # normalized Gaussian convolution interpolates (dst - src) on the
        # ring into the hole, so the clone keeps its grain but its low
        # frequencies land on the hole's boundary values (gradients continue
        # seamlessly through the patch). Sigma is thickness-local: a
        # bbox-sized sigma turned the correction into one constant for the
        # whole component, letting one bad segment tint it all. Where the
        # source ring is blocked by a neighboring spot (dust clusters), the
        # blocked positions contribute the CLEAN subset's median difference
        # instead of dropping out: dropping them left a spatially LOPSIDED
        # anchor, and on a gradient the one-sided difference interpolated
        # into a tone dome over the whole fill — the visible gray disc that
        # read as "it sampled from the wrong place" even though the clone
        # content was right.
        ring_m = ring
        dmap = np.zeros_like(dst)
        if best_rv is not None and not best_rv.all():
            diffs = dst[ring] - best_src[ring]
            med = np.median(diffs[best_rv], axis=0)
            diffs[~best_rv] = med
            dmap[ring] = diffs
        else:
            dmap[ring] = dst[ring] - best_src[ring]
        # MEDIAN-ANCHORED tone (maintainer decision): the normalized Gaussian
        # convolution below is a spatially-varying MEAN, and pattern
        # mismatches between the two rings (a bokeh highlight present in
        # only one of them) are heavy-tailed outliers that dragged the whole
        # fill's tone — the ~4%-dark gray discs. The differences are
        # winsorized to median ± 3 scaled-MADs per channel: the anchor's
        # center is the ring's MEDIAN difference, the bulk still varies
        # spatially (gradients continue through the fill), the tail cannot
        # tint. Median/MAD resample stably, so preview == export holds.
        diffs_all = dmap[ring]
        med_all = np.median(diffs_all, axis=0)
        mad_all = (np.median(np.abs(diffs_all - med_all), axis=0)
                   * 1.4826 + 1e-3)
        dmap[ring] = np.clip(diffs_all, med_all - 3.0 * mad_all,
                             med_all + 3.0 * mad_all)
        ind = ring_m.astype(np.float32)
        sigma = half_th + pad
        wsum = cv2.GaussianBlur(ind, (0, 0), sigma)
        dsum = cv2.GaussianBlur(dmap, (0, 0), sigma)
        mean_diff = dmap[ring_m].mean(axis=0)
        corr = np.where(wsum[..., None] > 1e-4,
                        dsum / np.maximum(wsum, 1e-6)[..., None], mean_diff)
        heal = best_src + corr
    filled[wy0:wy1, wx0:wx1][hole] = np.clip(
        np.rint(heal[hole]), 0, 65535).astype(np.uint16)

    # Feather: the ramp is the user-set FRACTION of this hole's depth (its
    # local radius), so the fade grows with the brush size instead of staying
    # a fixed few pixels; the depth cap keeps the core at full fill. The
    # feather governs EVERYTHING (maintainer rule: every pixel gets the
    # effect, opacity rolls off from the stroke center — no exceptions):
    # there is no defect force-fill floor, so a wide feather fades the heal
    # out even across the defect itself. The old dlike floor (alpha pinned
    # to 1 on defect-colored pixels) made the slider do nothing on real
    # dust — precisely where the user reaches for it.
    inside = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
    depth = max(1.0, float(inside.max()))
    fmap[wy0:wy1, wx0:wx1][hole] = \
        max(0.5, min(float(feather) * depth, depth))
    # dlike_on lives on as plan metadata only (records keep their shape and
    # the verdict stays resolution-stable via the replayed flags).
    sep = float(np.abs(defect - ring_bg).mean())
    dlike_on = bool(genuine and sep > _DLIKE_SEP_MADS * (ring_mad + 1e-3)
                    and float(defect.mean()) > float(ring_bg.mean()))
    if forced_flags is not None:
        dlike_on = bool(forced_flags[1])
    return best_off, genuine, dlike_on


# Canonical planning scale for the heal = the preview's long side. Buffers at
# or below it (preview, thumbnails) plan on themselves; larger buffers (hi-res
# zoom, export) REPLAY the plan computed at this scale so they reproduce the
# preview's healing structure. See apply_dust_removal.
_DUST_PLAN_LONG = 1080

# Centroid-match tolerance (normalized) when an EDIT re-heals with the
# previous plan as prior (sticky sources): unchanged segments sit at
# bit-identical centroids (the tile grid is absolute), so the bind is tight —
# a NEW stroke's segments must never inherit a neighbor's source without
# ever being scored. Cross-resolution replay keeps the looser default
# (raster rounding shifts centroids across scales).
_DUST_EDIT_TOL = 0.004


def _nearest_plan_record(plan, cyn, cxn, tol=0.06):
    """The plan record whose segment centroid is nearest to (cyn, cxn) in
    normalized coords, or None when nothing lies within `tol` (a component
    that only rasterizes at the finer scale — sub-pixel at plan resolution)."""
    best, best_d = None, tol
    for rec in plan:
        d = ((rec[0] - cyn) ** 2 + (rec[1] - cxn) ** 2) ** 0.5
        if d < best_d:
            best, best_d = rec, d
    return best


def apply_dust_removal(img16: np.ndarray, spots, inpaint_radius: int = 3,
                       feather: float = DUST_FEATHER_DEFAULT,
                       plan=None, collect_plan=None,
                       prior_plan=None, crop=None,
                       ws_windowed: bool = False) -> np.ndarray:
    """Heal the dust spots out of a 16-bit RGB image, non-destructively.

    `feather` is the edge fade width as a fraction of each hole's own
    half-thickness (the user's per-image Feather setting, 0..1): 0 =
    essentially hard edges, 1 = the cross-fade spans the hole's full depth.
    Keyed to the hole's local radius, the fade scales with the brush size —
    and with resolution, since the rasterized hole does. The blend is a pure
    opacity dome rolling off from the stroke's center, with NO exceptions
    (maintainer rule): a wide feather fades the heal out even across the
    defect itself — dial the feather down (0 = hard) for complete removal.

    Each mask component is filled by CLONING the best-matching clean patch
    from its neighborhood (see _heal_patch) — real texture and grain,
    16-bit native — instead of the old cv2.inpaint diffusion, whose averaged
    fill left a smooth round patch with fan-like streaks. Long strokes are
    healed in thickness-scaled SEGMENTS, each from its own local source strip
    (one whole-stroke window would demand a clean area the size of the
    stroke's bbox, which rarely exists — a curl would silently fall back to
    diffusion and ghost). Patches with no clean source window (image border,
    dense dust) fall back to cv2.inpaint (Telea, 8-bit). The fill is
    composited through a feathered alpha that ramps inward from each hole's
    boundary (thickness-relative, capped by the hole's defect-free rim); ONLY
    masked pixels change, the rest of the frame is bit-for-bit untouched.
    Returns a NEW uint16 array; img16 is never mutated. No-op (returns img16
    unchanged) when there are no spots or the rasterized mask is empty.

    Resolution consistency: the heal PLAN — stroke segmentation, each
    segment's chosen source window, and the diffusion-fallback decisions —
    is content-adaptive, so re-deriving it independently per resolution
    picked visibly different fills at the 1080px preview vs the full-res
    export ("the preview and the final don't look the same"). Buffers larger
    than the canonical scale therefore replay a plan computed at
    _DUST_PLAN_LONG (matching the preview) with all geometry scaled: same
    segments, same source patches, same fallbacks — the export heals with
    the same structure the user approved on screen.

    `collect_plan` (a list): on a canonical-scale buffer, receives that
    heal's plan records — CCRImage caches the PREVIEW render's plan this way.
    `plan`: replay the given plan instead of self-planning on a downscale,
    so exports/hi-res sample exactly the patches the on-screen preview did
    (self-planning re-decodes/resizes and near-tie source choices can flip
    on real content). Ignored on canonical-scale buffers.
    `prior_plan` (canonical-scale buffers only): the PREVIOUS edit's plan.
    Once a patch's source is set, nothing may move it — segments whose
    centroids are unchanged (tight _DUST_EDIT_TOL bind) reuse their prior
    source/verdicts VERBATIM instead of re-searching, so painting a new
    stroke cannot re-sample the strokes already on screen (a fresh search
    with the new mask in place flipped near-tie sources for any segment
    whose context ring the new stroke grazed). Even a stroke that lands ON
    a prior source does not move it: that segment defers to a second pass
    and clones the HEALED content at the same location (see _heal_patch).
    New segments match no prior record and search normally.
    `crop`: ((x1, y1, x2, y2) normalized, angle_deg) — the image's confirmed
    crop. Content OUTSIDE it (film holder, rebate, whatever the user cut
    away) is not scene: it is treated like dust for context purposes, so
    rings never anchor on it and fresh searches never sample it. Pinned
    sources keep their reference regardless (sticky) — one outside a newer
    crop just routes through the deferred pass onto the same pixels.
    """
    if not spots:
        return img16
    h, w = img16.shape[:2]
    long_side = max(h, w)
    if long_side <= _DUST_PLAN_LONG:
        return _heal_impl(img16, spots, inpaint_radius, feather,
                          collect=collect_plan, plan=prior_plan,
                          plan_tol=_DUST_EDIT_TOL, crop=crop,
                          ws_windowed=ws_windowed)
    scale = long_side / float(_DUST_PLAN_LONG)
    if plan is None:
        # No preview plan supplied — derive one from this buffer's own
        # downscale (plan-only: the healed small buffer would be discarded).
        t0 = time.time()
        pw = max(2, int(round(w / scale)))
        ph = max(2, int(round(h / scale)))
        small = cv2.resize(img16, (pw, ph), interpolation=cv2.INTER_AREA)
        plan = []
        _heal_impl(small, spots, inpaint_radius, feather, collect=plan,
                   plan_only=True, crop=crop, ws_windowed=ws_windowed)
        print(f"Dust plan (no cached preview plan): {time.time() - t0:.3f}s")
    return _heal_impl(img16, spots, inpaint_radius, feather,
                      plan=plan, scale_up=scale, crop=crop,
                      ws_windowed=ws_windowed)


def _heal_impl(img16: np.ndarray, spots, inpaint_radius: int,
               feather: float, plan=None, collect=None,
               scale_up: float = 1.0, plan_only: bool = False,
               plan_tol: float = 0.06, crop=None,
               ws_windowed: bool = False) -> np.ndarray:
    """apply_dust_removal's engine at ONE resolution. With `collect` (a list),
    appends a plan record per heal segment:
    (cy_norm, cx_norm, offset, genuine, dlike_on, manual) where offset is
    the chosen source displacement normalized over (h, w) — None for a
    diffusion fallback — genuine/dlike_on are the patch's content-adaptive
    verdicts (see _heal_patch), and manual marks a USER-PICKED source
    (cloned verbatim, no membrane re-toning; set by the overlay's
    source-ring drag and carried through every replay). With `plan` (+ `scale_up` = this buffer's
    size over the plan scale), segments reuse the planned
    offsets/fallbacks/verdicts instead of re-deriving them, and the
    pixel-based geometry (segment size, Telea radius) scales by `scale_up`
    so segmentation matches the plan's. `plan_tol` is the centroid-match
    tolerance: loose for cross-resolution replay (raster rounding), tight
    (_DUST_EDIT_TOL) when a prior edit's plan pins sources at the same scale.
    `plan_only` skips the Telea fill and the feathered composite (the caller
    only wants the collected plan, not the healed buffer) and returns img16."""
    if not spots:
        return img16
    t0 = time.time()
    h, w = img16.shape[:2]
    mask = rasterize_dust_mask(spots, h, w)
    if not mask.any():
        return img16

    # PER-SPOT PATCHES (maintainer rule: never auto-connect two strokes —
    # each spot samples separately). `labels` carries one id PER SPOT, with
    # drawing order owning shared pixels, so touching/overlapping strokes
    # keep their own segments, their own touching-sample argmin and their
    # own per-patch offset. Only the never-clone-dust checks (mask_pad /
    # integ / labels>0) and the feathered composite see the union. Each
    # spot rasterizes into its own bbox-local canvas with the exact integer
    # geometry of rasterize_dust_mask (rounded full-frame coords, integer
    # translation), so the stamped labels match the union mask bit for bit.
    labels = np.zeros((h, w), np.int32)
    boxes = {}
    for si, s in enumerate(spots, start=1):
        r_px = max(1, int(round(float(s.get("r", 0.0)) * w)))
        pxy = []
        for p in (s.get("pts") or []):
            try:
                pxy.append((int(round(float(p[0]) * w)),
                            int(round(float(p[1]) * h))))
            except (TypeError, ValueError, IndexError):
                continue
        if not pxy:
            continue
        bx0 = max(0, min(p[0] for p in pxy) - r_px - 1)
        bx1 = min(w, max(p[0] for p in pxy) + r_px + 2)
        by0 = max(0, min(p[1] for p in pxy) - r_px - 1)
        by1 = min(h, max(p[1] for p in pxy) + r_px + 2)
        if bx1 <= bx0 or by1 <= by0:
            continue
        local = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        prev = None
        for (x, y) in pxy:
            cv2.circle(local, (x - bx0, y - by0), r_px, 255, -1)
            if prev is not None:
                cv2.line(local, prev, (x - bx0, y - by0), 255,
                         thickness=2 * r_px)
            prev = (x - bx0, y - by0)
        if not local.any():
            continue
        labels[by0:by1, bx0:bx1][local > 0] = si
        lys, lxs = np.nonzero(local)
        boxes[si] = (bx0 + int(lxs.min()), by0 + int(lys.min()),
                     int(lxs.max()) - int(lxs.min()) + 1,
                     int(lys.max()) - int(lys.min()) + 1)
    n = len(spots) + 1
    # "Clean" excludes every spot plus 1 px (buries antialiased speck edges).
    mask_pad = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    if crop is not None and crop[0]:
        keep = _crop_keep_mask(crop[0], crop[1], h, w)
        if keep is not None and not keep.all():
            # Outside-crop content (holder/rebate/junk the user cut away) is
            # not scene: treat it like dust for context purposes — rings
            # never anchor on it and fresh searches never sample it. Pinned
            # sources keep their reference (sticky): one poking outside just
            # routes through the deferred pass onto the same pixels.
            mask_pad = np.maximum(
                mask_pad, np.where(keep > 0, 0, 255).astype(np.uint8))
    integ = cv2.integral((mask_pad > 0).astype(np.uint8))
    filled = img16.copy()
    fallback = np.zeros_like(mask)
    # Per-pixel feather ramp width (px): each patch writes feather × its own
    # hole depth (see _heal_patch), so the fade tracks the brush size and the
    # buffer's resolution alike. float32: fractional ramps keep the relative
    # alpha profile identical across scales (a rounded 1 px preview ramp vs
    # its 3 px export equivalent visibly diverged).
    fmap = np.full((h, w), 0.5, np.float32)
    # Pixel-based knobs scale with the buffer (relative to the plan scale) so
    # a stroke splits into the SAME segments here as it did in the plan, and
    # the diffusion fallback blurs over the same image fraction.
    seg_min = max(8, int(round(_HEAL_SEG_MIN * max(1.0, scale_up))))
    telea_r = max(1, int(round(inpaint_radius * max(1.0, scale_up))))
    # Plan-scale smoothing for per-pixel threshold classifications inside
    # _heal_patch: a box the size of the scale factor makes this buffer's
    # pixel statistics match the preview's area-averaged ones, so the same
    # pixels classify as holder-black / defect-like at every resolution.
    kb = int(round(max(1.0, scale_up)))
    kb += 1 - (kb % 2)  # nearest odd
    img16_c = img16 if kb <= 1 else cv2.blur(img16, (kb, kb))
    # Segments whose PINNED source is under newer dust: re-run after the
    # fills below so they clone the healed content at the same location.
    deferred = []
    # One sample offset per patch (component): the first segment's argmin
    # seeds the rest — see _heal_patch's pref_off.
    comp_off = {}
    t_setup = time.time() - t0
    t0 = time.time()
    n_segs = 0
    for i in range(1, n):  # 0 is background; one patch PER SPOT
        if i not in boxes:
            continue
        x0, y0, cw, ch = boxes[i]
        comp_win = labels[y0:y0 + ch, x0:x0 + cw] == i
        if not comp_win.any():
            continue  # fully overwritten by later strokes (shared pixels)
        comp_auto = spots[i - 1].get("kind") == "auto"
        # Half-thickness = the defect's local scale (1px zero border so a
        # component touching its bbox edge still measures correctly).
        hole0 = np.zeros((ch + 2, cw + 2), np.uint8)
        hole0[1:-1, 1:-1] = comp_win
        half_th = float(cv2.distanceTransform(hole0, cv2.DIST_L2, 3).max())
        seg_max = max(seg_min, int(round(_HEAL_SEG_MAX * max(1.0, scale_up))))
        seg = max(seg_min, min(int(round(_HEAL_SEG_THICKNESS * half_th)),
                               seg_max))
        # Local scale for the guard/ring/search geometry: bounded by the
        # tile, so a merged blob's growing thickness cannot resize EVERY
        # tile's window (a new dab used to re-sample segments far from the
        # edit). Inert for uncapped components (seg = 6·half_th there).
        loc_th = min(half_th, 0.5 * seg)
        # Components spanning multiple tiles anchor the grid to the IMAGE,
        # not the bbox: a dab extending the bbox origin used to shift every
        # tile of the blob, re-sampling segments the edit never touched.
        # Compact components keep the single un-split tile.
        if cw <= seg and ch <= seg:
            gy, gx = y0, x0
        else:
            gy, gx = (y0 // seg) * seg, (x0 // seg) * seg
        for ty in range(gy, y0 + ch, seg):
            for tx in range(gx, x0 + cw, seg):
                by, bx = max(ty, y0), max(tx, x0)
                sub = comp_win[by - y0:ty - y0 + seg, bx - x0:tx - x0 + seg]
                if not sub.any():
                    continue
                n_segs += 1
                ys, xs = np.nonzero(sub)
                bbox = (bx + int(xs.min()), by + int(ys.min()),
                        bx + int(xs.max()) + 1, by + int(ys.max()) + 1)
                cyn = (by + float(ys.mean())) / h
                cxn = (bx + float(xs.mean())) / w
                src_off, forced_fb, flags, manual = None, False, None, False
                if plan is not None:
                    rec = _nearest_plan_record(plan, cyn, cxn, tol=plan_tol)
                    if rec is not None:
                        if rec[2] is None:
                            forced_fb = True  # the plan chose diffusion here
                        else:
                            src_off = (int(round(rec[2][0] * h)),
                                       int(round(rec[2][1] * w)))
                            if len(rec) >= 5:
                                flags = (rec[3], rec[4])
                            if len(rec) >= 6:
                                manual = bool(rec[5])  # user-picked source
                res = None
                if not forced_fb:
                    res = _heal_patch(img16, img16_c, labels, i, bbox,
                                      loc_th, mask_pad, integ, filled, fmap,
                                      feather, src_off=src_off,
                                      forced_flags=flags, manual=manual,
                                      auto=comp_auto, ws_windowed=ws_windowed,
                                      pref_off=comp_off.get(i))
                if (res is not None and len(res) == 3 and src_off is None
                        and res[0] is not None):
                    comp_off.setdefault(i, res[0])
                if res is not None and len(res) == 4:
                    # Pinned source now under new dust: fill in a SECOND
                    # pass from the healed buffer (after Telea), so the
                    # reference location stays — no exceptions.
                    deferred.append((i, bbox, loc_th, res[0], flags, manual))
                off = None if res is None else res[0]
                if collect is not None:
                    collect.append(
                        (cyn, cxn, None, None, None, False) if res is None
                        else (cyn, cxn, (off[0] / h, off[1] / w),
                              res[1], res[2], manual))
                if off is None:
                    fallback[by:by + sub.shape[0],
                             bx:bx + sub.shape[1]][sub] = 255
                    # Same radius-relative ramp as the clone path, using the
                    # tile-bounded half-thickness as the local radius.
                    fmap[by:by + sub.shape[0], bx:bx + sub.shape[1]][sub] = \
                        max(0.5, min(float(feather) * loc_th, loc_th))

    t_segs = time.time() - t0
    mode = (" [plan-only]" if plan_only
            else (" [replay]" if plan is not None else ""))
    if plan_only:
        # The caller only wants the collected plan — the fill/composite work
        # below would be thrown away with the buffer.
        print(f"Dust heal {w}x{h}: setup {t_setup:.3f}s, "
              f"{n_segs} segments {t_segs:.3f}s{mode}")
        return img16

    t0 = time.time()
    if fallback.any():
        # Diffusion fallback (cv2.inpaint supports 8-bit only).
        bgr8 = cv2.cvtColor(cv2.convertScaleAbs(img16, alpha=255.0 / 65535.0),
                            cv2.COLOR_RGB2BGR)
        telea8 = cv2.inpaint(bgr8, fallback, telea_r, cv2.INPAINT_TELEA)
        telea16 = cv2.cvtColor(telea8, cv2.COLOR_BGR2RGB).astype(np.uint16) * 257
        fb = fallback > 0
        filled[fb] = telea16[fb]
    t_telea = time.time() - t0

    # Second pass for deferred segments: every hole now has SOME fill in
    # `filled` (clone or Telea), so a pinned source that sat under new dust
    # clones the healed content at its unchanged reference location.
    for i, bbox_d, loc_d, off_d, flags_d, man_d in deferred:
        _heal_patch(img16, img16_c, labels, i, bbox_d, loc_d, mask_pad,
                    integ, filled, fmap, feather,
                    src_off=off_d, forced_flags=flags_d, sample=filled,
                    manual=man_d)

    # Feathered composite: alpha rises 0 -> 1 from each hole's boundary inward
    # over its feather ramp (a fraction of the hole's own thickness, so it
    # scales with brush size and resolution alike), so the fill cross-fades
    # into the original instead of cutting hard at the mask edge. OUTSIDE the
    # mask alpha is exactly 0 —
    # those pixels are kept bit-for-bit. The float blend runs per COMPONENT
    # window (padded 2 px): one global mask bbox degenerates to nearly the
    # whole frame when spots are scattered, and the full-frame float blend
    # then dominates the heal (~0.9 s at 6000 px for 40 spots). Components
    # never touch (8-connectivity merges adjacent pixels), so each one's
    # alpha computed from its own labels is exactly the global answer.
    t0 = time.time()
    out = img16.copy()
    # The composite blends over the UNION's connected regions: sampling is
    # per spot, but two overlapping strokes still cross-fade as one filled
    # area — a per-spot alpha would ramp to zero along their shared border,
    # cutting a seam through the middle of the union's interior.
    nu, ulab, ustats, _ = cv2.connectedComponentsWithStats(mask,
                                                           connectivity=8)
    for i in range(1, nu):
        bx0 = int(ustats[i, cv2.CC_STAT_LEFT])
        by0 = int(ustats[i, cv2.CC_STAT_TOP])
        bw_ = int(ustats[i, cv2.CC_STAT_WIDTH])
        bh_ = int(ustats[i, cv2.CC_STAT_HEIGHT])
        ys = slice(max(0, by0 - 2), min(h, by0 + bh_ + 2))
        xs = slice(max(0, bx0 - 2), min(w, bx0 + bw_ + 2))
        comp = ulab[ys, xs] == i
        # The dome IS the whole story: no defect force-fill floor (see
        # _heal_patch's feather comment) — the slider's rolloff shows on
        # every heal, dust included.
        a = _feather_alpha(comp.astype(np.uint8) * 255, fmap[ys, xs])
        write = a > 0.0
        if not write.any():
            continue
        a3 = a[..., None]
        blend = (img16[ys, xs].astype(np.float32) * (1.0 - a3)
                 + filled[ys, xs].astype(np.float32) * a3)
        out[ys, xs][write] = np.clip(
            np.rint(blend[write]), 0, 65535).astype(np.uint16)
    print(f"Dust heal {w}x{h}: setup {t_setup:.3f}s, "
          f"{n_segs} segments {t_segs:.3f}s, telea {t_telea:.3f}s, "
          f"composite {time.time() - t0:.3f}s{mode}")
    return out


# --- Area editing (local masked adjustment layers) -------------------------
# Masks are rasterized from NORMALIZED geometry (fractions of width/height) so
# the same area definition reproduces identically at the 1080px preview, the
# hi-res zoom detail, and the full-res export — the same resolution-independent
# contract apply_crop_to_image and the reference-norm replay already honor.

def _smoothstep(a: np.ndarray) -> np.ndarray:
    """Cubic ease (3a^2 - 2a^3) clamped to [0,1]."""
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def build_circle_mask(h, w, geom, angle_deg=0.0, feather=0.25) -> np.ndarray:
    """Rotated, squishable ellipse -> float32 alpha[h,w] in [0,1].

    geom: {cx, cy, rx, ry} as fractions of width/height (rx of width, ry of
    height). angle_deg rotates the ellipse about its center (Qt clockwise
    convention). feather is the fraction of the radius over which alpha ramps
    from 1 (inside) to 0 (at the boundary). feather<=0 gives a hard edge.
    """
    cx = float(geom.get("cx", 0.5)) * w
    cy = float(geom.get("cy", 0.5)) * h
    rx = max(float(geom.get("rx", 0.3)) * w, 1e-3)
    ry = max(float(geom.get("ry", 0.3)) * h, 1e-3)
    t = np.deg2rad(float(angle_deg))
    cs, sn = np.cos(t), np.sin(t)
    # arange broadcasting (not mgrid) keeps the big intermediates to one (h,w)
    # float32 array — at full export resolution mgrid's two int64 grids would
    # waste hundreds of MB per area.
    ys = np.arange(h, dtype=np.float32)[:, None] - cy
    xs = np.arange(w, dtype=np.float32)[None, :] - cx
    xr = cs * xs + sn * ys          # rotate into the ellipse-local frame
    yr = -sn * xs + cs * ys
    d = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)   # 1.0 == ellipse boundary
    f = max(float(feather), 1e-4)
    return _smoothstep((1.0 - d) / f).astype(np.float32)


def build_gradient_mask(h, w, geom, feather=0.25) -> np.ndarray:
    """Linear gradient -> float32 alpha[h,w] in [0,1].

    geom: {x0,y0,x1,y1} normalized endpoints. alpha is 0 at p0 (no effect),
    1 at p1 (full effect), with a smooth ramp across the segment and constant
    full/zero plateaus beyond the endpoints. The two handles define the ramp
    extent, so feather is unused for gradients (accepted for API symmetry).
    """
    ax = float(geom.get("x0", 0.3)) * w
    ay = float(geom.get("y0", 0.5)) * h
    bx = float(geom.get("x1", 0.7)) * w
    by = float(geom.get("y1", 0.5)) * h
    vx, vy = bx - ax, by - ay
    L2 = max(vx * vx + vy * vy, 1e-6)
    ys = np.arange(h, dtype=np.float32)[:, None] - ay
    xs = np.arange(w, dtype=np.float32)[None, :] - ax
    t = (xs * vx + ys * vy) / L2
    return _smoothstep(t).astype(np.float32)


def build_area_mask(h, w, area) -> np.ndarray:
    """Dispatch on area['kind'] to the right mask builder."""
    geom = area.get("geometry") or {}
    feather = float(area.get("feather", 0.25))
    if area.get("kind") == "gradient":
        return build_gradient_mask(h, w, geom, feather)
    return build_circle_mask(h, w, geom, float(area.get("angle", 0.0)), feather)


def apply_area_layers(base_u16: np.ndarray, areas, layer_fn) -> np.ndarray:
    """Composite enabled area layers onto a globally-adjusted base.

    base_u16: the global adjustment result (uint16 RGB) at the CURRENT
        resolution — the implicit "whole image" layer.
    areas: list of area dicts (see spec/area-editing.md §4.1).
    layer_fn: callable(base_u16, settings) -> uint16 array, the full per-pixel
        adjustment pass for one area's own settings, computed against the SAME
        base (so each area's delta is independent).

    Additive blend: out = base + sum_i alpha_i * (layer_i - base), clipped.
    Order-independent; overlapping areas accumulate. Returns base_u16 unchanged
    when there are no enabled areas.
    """
    # Only areas that are enabled AND carry adjustments do anything — skip the
    # rest before allocating any full-resolution mask/delta. (A freshly created
    # area has empty settings until the user touches a slider.)
    active = [a for a in areas if a.get("enabled") and a.get("settings")]
    if not active:
        return base_u16
    h, w = base_u16.shape[:2]
    base = base_u16.astype(np.float32)
    acc = base.copy()
    for a in active:
        layer = np.asarray(layer_fn(base_u16, a["settings"]), dtype=np.float32)
        m = build_area_mask(h, w, a)[..., None]
        acc += m * (layer - base)
    return np.clip(acc, 0, 65535).astype(np.uint16)