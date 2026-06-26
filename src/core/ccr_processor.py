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

            // Per-channel levels controls (linear domain, post-conversion data).
            // Blackpoint/Gain mirror the regular Black/White Point sliders
            // (same /300 mapping) per channel; Shift is a uniform additive
            // offset; Input Gain is a pre-everything exposure multiplier.
            if (ch_input_gain != 0.0f || ch_master_shift != 0.0f || ch_master_gain != 0.0f ||
                ch_r_shift != 0.0f || ch_r_gain != 0.0f || ch_r_blackpoint != 0.0f ||
                ch_g_shift != 0.0f || ch_g_gain != 0.0f || ch_g_blackpoint != 0.0f ||
                ch_b_shift != 0.0f || ch_b_gain != 0.0f || ch_b_blackpoint != 0.0f) {

                float ig = pow(2.0f, ch_input_gain / 50.0f);
                float rs  = clamp(ch_master_shift + ch_r_shift, -100.0f, 100.0f) / 300.0f;
                float rg  = clamp(ch_master_gain + ch_r_gain, -100.0f, 100.0f) / 300.0f;
                float rbp = clamp(ch_r_blackpoint, -100.0f, 100.0f) / 300.0f;
                float gs  = clamp(ch_master_shift + ch_g_shift, -100.0f, 100.0f) / 300.0f;
                float gg  = clamp(ch_master_gain + ch_g_gain, -100.0f, 100.0f) / 300.0f;
                float gbp = clamp(ch_g_blackpoint, -100.0f, 100.0f) / 300.0f;
                float bs  = clamp(ch_master_shift + ch_b_shift, -100.0f, 100.0f) / 300.0f;
                float bg  = clamp(ch_master_gain + ch_b_gain, -100.0f, 100.0f) / 300.0f;
                float bbp = clamp(ch_b_blackpoint, -100.0f, 100.0f) / 300.0f;

                // Process each channel only when non-neutral, so the normalize
                // round-trip can't drift untouched channels by 1 LSB.
                if (ig != 1.0f || rs != 0.0f || rg != 0.0f || rbp != 0.0f) {
                    float xr = (r / 65535.0f) * ig + rs;
                    xr = (xr - rbp) / ((1.0f - rg) - rbp);
                    r = clamp(xr, 0.0f, 1.0f) * 65535.0f;
                }
                if (ig != 1.0f || gs != 0.0f || gg != 0.0f || gbp != 0.0f) {
                    float xg = (g / 65535.0f) * ig + gs;
                    xg = (xg - gbp) / ((1.0f - gg) - gbp);
                    g = clamp(xg, 0.0f, 1.0f) * 65535.0f;
                }
                if (ig != 1.0f || bs != 0.0f || bg != 0.0f || bbp != 0.0f) {
                    float xb = (b / 65535.0f) * ig + bs;
                    xb = (xb - bbp) / ((1.0f - bg) - bbp);
                    b = clamp(xb, 0.0f, 1.0f) * 65535.0f;
                }
            }

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


def ccr_normalize_with_reference(ccr_image,output_path=None,jpg_out=False,jpg_quality=95,max_long_side=None,output_colorspace="srgb") -> np.ndarray:
    """
    Normalize and align the image using the CCR algorithm, using a reference rectangle
    for percentile calculations instead of a crop factor.

    Args:
        ccr_image: CCRImage object

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
    if fine_angle != 0:
        center_ref = (img_ref.shape[1] // 2, img_ref.shape[0] // 2)
        w_ref, h_ref = img_ref.shape[1], img_ref.shape[0]
        rot_mat = cv2.getRotationMatrix2D(center_ref, -fine_angle, 1.0)
        img_ref = cv2.warpAffine(img_ref, rot_mat, (w_ref, h_ref), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        # Clean up rotation matrix as it's no longer needed
        del rot_mat
        # print(f"Rotated image by {angle} degrees")
        # if output_path is not None: # this is for output
        #     # when outputting rotate original image as well
        #     rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        #     img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    print(f"Image setup and rotation: {time.time() - step_start:.3f}s")

    # Reference frame
    step_start = time.time()
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

        final_mapped_rect = map_rect_to_original(
                img_ref.shape,
                rgb_brightness_normalized.shape,
                reference_rect
            )
        x1, y1, x2, y2 = final_mapped_rect
        del final_mapped_rect  # Clean up as it's no longer needed
        if getattr(ccr_image, 'crop_rect', None) is not None:
            # A crop invalidates the reference-rect scaling above — anchor
            # the watermark to the output's bottom-right corner instead
            # (same as the bwpoint pipeline).
            y2, x2 = rgb_brightness_normalized.shape[:2]
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


def _default_slope_invert(img_f: np.ndarray, black_point_bgr) -> np.ndarray:
    """Density-space inversion for the black-point-only mode, using the baked
    scalar DEFAULT_DENSITY_SLOPE. `img_f` is a float32 (H,W,3) BGR array.

    Per channel: out = clip(DEFAULT_DENSITY_SLOPE * max(log10(base/img), 0), 0, 1),
    then an optional display gamma, scaled to uint16. The per-channel base divide
    is the ONLY place colour balance enters, and log10(base/img) cancels any
    per-channel light-source scaling — so re-sampling the black point under a new
    light keeps colour consistent without a white point."""
    out = np.empty_like(img_f)
    for c in range(3):
        base = max(float(black_point_bgr[c]), 1.0)
        ch = np.maximum(img_f[..., c], _DENSITY_FLOOR)   # copy; avoids /0 & log(0)
        np.divide(base, ch, out=ch)                      # base / img
        np.log10(ch, out=ch)                             # optical density above base
        np.maximum(ch, 0.0, out=ch)                      # clamp (img brighter than base)
        out[..., c] = ch
    out *= np.float32(DEFAULT_DENSITY_SLOPE)             # single scalar = contrast
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


def compute_auto_exposure_gain(img_bgr: np.ndarray) -> float:
    """Auto-exposure for default-slope mode: return the EXPOSURE-base value (in
    exposure-slider units, where 2^(x/50) is the nominal stop multiplier) that
    would place the (film-holder-excluded) AUTO_EXPOSURE_PERCENTILE luminance at
    AUTO_EXPOSURE_TARGET of full scale.

    The value is applied through the TONE-AWARE exposure (not a uniform gain), so
    the boost mainly lifts midtones while highlights roll off near white instead
    of hard-clipping — i.e. this is the nominal target; the top compresses
    gracefully toward it. Pure-white pixels (luminance >= WHITE_EXCLUDE_FRACTION·
    65535) are the film holder / clear surround and are excluded so they don't
    peg the estimate. Returns 0.0 when there isn't enough non-white content."""
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


def ccr_normalize_with_bwpoint(ccr_image, black_point_bgr, white_point_bgr=None,
                               output_path=None, jpg_out=False,
                               jpg_quality=95, max_long_side=None,
                               output_colorspace="srgb"):
    """
    Film negative conversion using the same pipeline as ccr_normalize_with_reference
    but with explicit per-channel B/W points instead of auto-detected percentiles.

    black_point_bgr: (B,G,R) scan values of transparent/clear film area (HIGH values).
                     Transparent areas → output black (0) after inversion.
    white_point_bgr: (B,G,R) scan values of dense/exposed film area (LOW values).
                     Dense areas → output white (65535) after inversion.
                     If None, the default-slope mode is used: a density-space
                     inversion with the baked scalar DEFAULT_DENSITY_SLOPE is
                     applied to the black point alone (see _default_slope_invert).

    Pipeline: BWPN (user B/W points) → inversion → saturation boost → shadow correction
    ODAI is skipped because per-channel B/W point mapping already normalises channels.
    """
    total_start_time = time.time()

    # --- Load working image ---
    img = _load_export_source(ccr_image, output_path, max_long_side)
    if img is None:
        raise ValueError("CCRImage: could not load image data for B/W point conversion")

    # --- Fine rotation (same as main pipeline) ---
    fine_angle = ccr_image.fine_rotation_angle / 100.0
    h_flip = ccr_image.horizontal_mirrored
    v_flip = ccr_image.vertical_mirrored
    h, w = img.shape[:2]
    if fine_angle != 0:
        center_rot = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center_rot, -fine_angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        del rot_mat

    # --- BWPN: B/W point values are absolute anchors, constant across the whole roll ---
    #
    # With no_auto_bright=True in rawpy, film base and dense-area values are identical in every
    # frame. The sampled B/W points are applied directly as fixed per-channel anchors.
    img_f = img.astype(np.float32)
    if white_point_bgr is None:
        # --- Default-slope mode (black point only): density-space inversion ---
        rgb_inverted = _default_slope_invert(img_f, black_point_bgr)
        del img_f
        print(f"BWPN (default slope): {time.time() - total_start_time:.3f}s")
    else:
        norm = np.empty_like(img_f)
        for c in range(3):
            p_hi = max(float(black_point_bgr[c]), 1.0)   # transparent film (film base) → maps to black
            p_lo = max(float(white_point_bgr[c]), 1.0)   # dense film (exposed area) → maps to white

            denom = p_hi - p_lo
            if abs(denom) < 1.0:
                norm[..., c] = 0.0
                continue
            np.subtract(img_f[..., c], p_lo, out=norm[..., c])
            np.divide(norm[..., c], denom, out=norm[..., c])
            np.multiply(norm[..., c], 65535.0, out=norm[..., c])
            np.clip(norm[..., c], 0, 65535, out=norm[..., c])
        del img_f
        print(f"BWPN (user points): {time.time() - total_start_time:.3f}s")

        # --- Inversion (ODAI skipped: per-channel B/W points already equalise channels) ---
        rgb_inverted = (65535.0 - norm).clip(0, 65535).astype(np.uint16)
        del norm
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

    # Non-destructive base offsets applied through the adjustment pipeline (UI shows 0).
    # contrast_base set to 0 to disable the baked-in contrast S-curve (was 40).
    ccr_image.contrast_base = 0
    ccr_image.temperature_base = 0

    # Auto-exposure (default-slope mode only): compute once from the preview and
    # store as a non-destructive uniform-gain base; export/zoom reuse it. With a
    # white point the two-point map already sets the cut, so force it to 0.
    if output_path is None:
        ccr_image.exposure_base = (
            compute_auto_exposure_gain(rgb_result) if white_point_bgr is None else 0.0)

    # --- User adjustments (export only) ---
    if output_path is not None:
        rgb_result = ccr_image.apply_adjustments(rgb_result)

    # --- Export path: flips, rotation, watermark, file write ---
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
    orientation, optional watermark and output colour space — mirroring the
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

    if output_path is None:
        print(f"TOTAL ref-params normalization time: {time.time() - total_start_time:.3f}s")
        return rgb_result

    # --- Export path: adjustments, crop, flips, rotation, watermark, write ---
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


def apply_bwpoint_normalization(img: np.ndarray, black_point_bgr, white_point_bgr=None) -> np.ndarray:
    """B/W-point conversion at any resolution: absolute per-channel anchors +
    inversion + standard look — mirrors ccr_normalize_with_bwpoint's preview
    path (the anchors are global constants, so no rescaling is needed).

    When white_point_bgr is None, the default-slope mode is used: a density-space
    inversion with the baked scalar DEFAULT_DENSITY_SLOPE is applied to the black
    point alone (see _default_slope_invert). Otherwise the explicit two-point
    math is used verbatim (bit-identical to the prior behaviour)."""
    img_f = img.astype(np.float32)
    if white_point_bgr is None:
        # Default-slope mode (black point only): density-space inversion.
        return _default_slope_invert(img_f, black_point_bgr)
    norm = np.empty_like(img_f)
    for c in range(3):
        p_hi = max(float(black_point_bgr[c]), 1.0)
        p_lo = max(float(white_point_bgr[c]), 1.0)
        denom = p_hi - p_lo
        if abs(denom) < 1.0:
            norm[..., c] = 0.0
            continue
        np.subtract(img_f[..., c], p_lo, out=norm[..., c])
        np.divide(norm[..., c], denom, out=norm[..., c])
        np.multiply(norm[..., c], 65535.0, out=norm[..., c])
        np.clip(norm[..., c], 0, 65535, out=norm[..., c])
    inverted = (65535.0 - norm).clip(0, 65535).astype(np.uint16)
    # apply_postinvert_look DISABLED — return the neutral linear inversion so the
    # zoom/hi-res replay matches the look-less preview/export path.
    return inverted
    # return apply_postinvert_look(inverted)


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

    Inverts the same perceptual formulas adjust_image applies:
        temperature:  r *= (1 + s),  b *= (1 - s)
                      s = (slider/100) * 0.40 * tone_curve
        tint:         g *= (1 - t),  r *= (1 + 0.3t),  b *= (1 + 0.3t)
                      t = tanh(slider * 0.02) * 0.18 * tone_curve
                          * balance_factor * skin_tone_sensitivity
    Solving r(1+s) = b(1-s) gives s; then m(1+0.3t) = g(1-t) gives t, where
    m is the common R/B value after the temperature step.
    """
    eps = 1e-6
    r = max(float(r), eps)
    g = max(float(g), eps)
    b = max(float(b), eps)
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 65535.0

    # Same piecewise tone-aware strength curve as adjust_image (luminance is
    # taken from the un-adjusted pixel there too, so this matches apply time).
    if lum <= 0.3:
        tone = 0.8 + 0.2 * min(max(lum / 0.3, 0.0), 1.0)
    elif lum <= 0.6:
        tone = 1.0
    else:
        progress = (lum - 0.6) / 0.4
        sigmoid = 1.0 / (1.0 + np.exp(-8.0 * (progress - 0.5)))
        tone = 1.0 - 0.75 * sigmoid

    # Temperature: choose s so the R and B channels meet.
    s = (b - r) / (b + r)
    temp_slider = float(np.clip(s * 100.0 / (0.40 * tone), -100.0, 100.0))

    # Use the achieved (possibly clamped) scale for the tint step.
    s_eff = (temp_slider / 100.0) * 0.40 * tone
    m = (r * (1.0 + s_eff) + b * (1.0 - s_eff)) / 2.0

    # Tint: choose t so G meets the common R/B level m.
    t = (g - m) / (g + 0.3 * m)
    skin = 1.0 + 0.5 * np.exp(-12.0 * (lum - 0.35) ** 2)
    denom = 0.18 * tone * tint_balance_factor * skin
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
) -> np.ndarray:
    """
    Apply temperature, tint, exposure, brightness, blackpoint, whitepoint, highlights, shadows,
    contrast, and saturation adjustments to a 16-bit image.
    All input factors are in range [-100, 100], 0 = no change.
    band_settings optionally carries the per-color-band sliders
    (band_<color>_<param> keys), applied as the final step.
    Returns a 16-bit image.
    """
    img = img16.astype(np.float32)

    # Map input factors from [-100, 100] to useful ranges
    kelvin_scale = 0.003 * kelvin_shift      # -1.0 to +1.0
    tint_scale = 0.002 * tint_shift          # -1.0 to +1.0
    exposure_scale = exposure * 2.0 / 100.0  # -2.0 to +2.0 stops
    brightness_scale = brightness / 8.0
    blackpoint_scale = 1.0 - (blackpoint / 100.0)    # -1.0 to +1.0 (fraction of 65535)
    whitepoint_scale = 1.0 + (whitepoint / 100.0)  # 0.0 to 2.0 (scaling factor)
    contrast_scale = 1.0 + (contrast / 100.0)      # 0.0 to 2.0
    saturation_scale = 1.0 + (saturation / 100.0)  # 0.0 to 2.0

    # Temperature and Tint (Lightroom-like perceptual adjustments)
    if kelvin_shift != 0.0 or tint_shift != 0.0:
        # Calculate luminance for tone-aware masking
        img_norm = img / 65535.0
        luminance = np.dot(img_norm[..., :3], [0.299, 0.587, 0.114])
        
        # Create smooth asymmetric tone-aware strength curve (Lightroom-like)
        # Shadows get strong effect, midtones get maximum effect, highlights get minimal effect
        # Using piecewise smooth curves for natural transitions
        
        # Define strength levels for different tonal regions
        shadow_strength = 0.8      # 80% strength in shadows (0-30% luminance)
        midtone_strength = 1.0     # 100% strength in midtones (30-60% luminance)  
        highlight_strength = 0.25  # 25% strength in highlights (60-100% luminance)
        
        # Transition points
        shadow_to_mid = 0.3       # Shadows to midtones transition at 30% luminance
        mid_to_highlight = 0.6    # Midtones to highlights transition at 60% luminance
        
        # Create smooth asymmetric curve using sigmoid blending
        tone_curve = np.zeros_like(luminance)
        
        # Shadow region (0-30%): smooth transition from 80% to 100%
        shadow_mask = luminance <= shadow_to_mid
        shadow_progress = np.clip(luminance[shadow_mask] / shadow_to_mid, 0, 1)
        tone_curve[shadow_mask] = shadow_strength + (midtone_strength - shadow_strength) * shadow_progress
        
        # Midtone region (30-60%): stay at 100% strength
        midtone_mask = (luminance > shadow_to_mid) & (luminance <= mid_to_highlight)
        tone_curve[midtone_mask] = midtone_strength
        
        # Highlight region (60-100%): smooth sigmoid transition from 100% to 25%
        highlight_mask = luminance > mid_to_highlight
        highlight_progress = (luminance[highlight_mask] - mid_to_highlight) / (1.0 - mid_to_highlight)
        # Use sigmoid for smooth natural rolloff
        sigmoid_factor = 1.0 / (1.0 + np.exp(-8 * (highlight_progress - 0.5)))
        tone_curve[highlight_mask] = midtone_strength - (midtone_strength - highlight_strength) * sigmoid_factor
        
        # Expand tone_curve to match image dimensions for broadcasting
        tone_curve = np.expand_dims(tone_curve, axis=-1)
        
        # Temperature (R/B scaling with logarithmic perceptual response)
        if kelvin_shift != 0.0:
            # Map slider values [-100, 100] to Kelvin temperatures [2000K, 8000K]
            # Neutral point (slider 0) = 5000K
            neutral_kelvin = 5000.0
            if kelvin_shift > 0:
                # Positive shift: 0 to +100 maps to 5000K to 8000K
                current_kelvin = neutral_kelvin + (kelvin_shift / 100.0) * 3000.0
            else:
                # Negative shift: -100 to 0 maps to 2000K to 5000K
                current_kelvin = neutral_kelvin + (kelvin_shift / 100.0) * 3000.0
            
            # Calculate Kelvin delta from neutral
            kelvin_delta = current_kelvin - neutral_kelvin
            
            # Logarithmic scaling for Kelvin - stronger impact at low end
            # Simulate the fact that 3000K->4000K has more visual impact than 7000K->8000K
            kelvin_abs = abs(kelvin_delta)
            
            # Create logarithmic response curve based on actual Kelvin values
            # Linear scale: 1K delta ≈ 0.013% R/B shift; full 3000K = 40% shift
            perceptual_scale = (kelvin_delta / 3000.0) * 0.40

            # tone_curve already handles spatial (shadow/highlight) weighting
            effective_scale = perceptual_scale * tone_curve[..., 0]
            
            r_scale = 1.0 + effective_scale
            b_scale = 1.0 - effective_scale
            
            img[..., 0] *= r_scale  # R
            img[..., 2] *= b_scale  # B

        # Tint (G-M scaling with perceptual mapping and enhanced midtone sensitivity)
        if tint_shift != 0.0:
            # Perceptual mapping for tint - non-linear response based on existing color balance
            # Tint impact varies with the current white balance state
            
            # Use pre-calculated balance factor instead of calculating it here
            balance_factor = tint_balance_factor
            
            # Enhanced midtone and skin tone sensitivity for tint
            # Tint is most visible in skin tones and neutral areas
            skin_tone_sensitivity = 1.0 + 0.5 * np.exp(-12 * (luminance - 0.35)**2)  # Peak at 35% luminance
            skin_tone_sensitivity = np.expand_dims(skin_tone_sensitivity, axis=-1)
            
            # Create perceptual tint curve - stronger response in certain ranges
            tint_abs = abs(tint_shift)
            if tint_abs > 0:
                # Sigmoid-like curve for tint perception
                perceptual_tint = np.tanh(tint_abs * 0.02) * np.sign(tint_shift) * 0.18
            else:
                perceptual_tint = 0.0
            
            # Apply perceptual tint with tone awareness, balance factor, and skin tone sensitivity
            effective_tint = perceptual_tint * tone_curve[..., 0] * balance_factor * skin_tone_sensitivity[..., 0]
            
            # Tint primarily affects green, with complementary adjustments to R/B
            g_scale = 1.0 - effective_tint  # Green channel (inverse of tint shift)
            r_scale = 1.0 + (0.3 * effective_tint)  # Slight red compensation
            b_scale = 1.0 + (0.3 * effective_tint)  # Slight blue compensation
            
            img[..., 1] *= g_scale  # G
            img[..., 0] *= r_scale  # R  
            img[..., 2] *= b_scale  # B

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

    # Per-channel levels controls (linear domain, post-conversion data).
    # Runs only when at least one slider is non-zero; otherwise a no-op.
    #
    # Blackpoint and Gain are per-channel versions of the regular Black Point
    # and White Point sliders (same /300 mapping); Shift is a uniform additive
    # offset that translates the channel's histogram; Input Gain is an
    # exposure-style multiplier applied before everything else.
    #     out = ((in * input_gain + shift) - black_val) / (white_val - black_val)
    # where black_val anchors white and remaps the dark end (regular Black
    # Point behaviour) and white_val = 1 - gain anchors black and moves the
    # bright end (regular White Point behaviour).
    _ch_active = (ch_input_gain, ch_master_shift, ch_master_gain,
                  ch_r_shift, ch_r_gain, ch_r_blackpoint,
                  ch_g_shift, ch_g_gain, ch_g_blackpoint,
                  ch_b_shift, ch_b_gain, ch_b_blackpoint)
    if any(p != 0.0 for p in _ch_active):
        ig = 2.0 ** (ch_input_gain / 50.0)      # slider ±100 → ×0.25…×4 (±2 stops)
        # Master + channel combined, clamped to slider range, then /300 like
        # the regular Black/White Point sliders.
        shifts = (np.clip(ch_master_shift + ch_r_shift, -100, 100) / 300.0,
                  np.clip(ch_master_shift + ch_g_shift, -100, 100) / 300.0,
                  np.clip(ch_master_shift + ch_b_shift, -100, 100) / 300.0)
        gains  = (np.clip(ch_master_gain + ch_r_gain, -100, 100) / 300.0,
                  np.clip(ch_master_gain + ch_g_gain, -100, 100) / 300.0,
                  np.clip(ch_master_gain + ch_b_gain, -100, 100) / 300.0)
        blacks = (np.clip(ch_r_blackpoint, -100, 100) / 300.0,
                  np.clip(ch_g_blackpoint, -100, 100) / 300.0,
                  np.clip(ch_b_blackpoint, -100, 100) / 300.0)

        for c in range(3):
            # Skip neutral channels so the normalize round-trip can't introduce
            # ±1 LSB quantization drift on channels the user didn't touch.
            if ig == 1.0 and shifts[c] == 0.0 and gains[c] == 0.0 and blacks[c] == 0.0:
                continue
            ch = img[..., c] / 65535.0
            if ig != 1.0:
                ch = ch * ig
            if shifts[c] != 0.0:
                ch = ch + shifts[c]
            black_val = blacks[c]
            white_val = 1.0 - gains[c]
            if black_val != 0.0 or white_val != 1.0:
                ch = (ch - black_val) / (white_val - black_val)
            img[..., c] = np.clip(ch, 0.0, 1.0) * 65535.0

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
) -> np.ndarray:
    """
    GPU-accelerated (OpenCL) version of adjust_image.
    Uses cached OpenCL context and compiled program for better performance.
    """
    global _opencl_cache

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
    sample = np.arange(65536, dtype=np.float32)
    src_idx = np.arange(256, dtype=np.float32) * (65535.0 / 255.0)

    out = np.empty_like(img16)
    for c, key in enumerate(("r", "g", "b")):
        # Compose per-channel curve over the composite curve in 0..255 space.
        ch_lut = build_channel_lut(curves.get(key))
        composed = ch_lut[np.clip(np.rint(rgb_lut), 0, 255).astype(np.intp)]
        # Expand the 256-point composed curve to a full 16-bit LUT.
        lut16 = np.interp(sample, src_idx,
                          composed * (65535.0 / 255.0)).astype(np.uint16)
        out[..., c] = lut16[img16[..., c]]
    return out


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


def apply_dust_removal(img16: np.ndarray, spots, inpaint_radius: int = 3) -> np.ndarray:
    """Inpaint the dust spots out of a 16-bit RGB image, non-destructively.

    Only the masked pixels are replaced (the rest of the frame keeps full
    16-bit precision); the fill is computed by cv2.inpaint (Telea) in 8-bit and
    composited back through a feathered alpha so the patch blends seamlessly.
    Returns a NEW uint16 array; img16 is never mutated. No-op (returns img16
    unchanged) when there are no spots or the rasterized mask is empty.
    """
    if not spots:
        return img16
    h, w = img16.shape[:2]
    mask = rasterize_dust_mask(spots, h, w)
    if not mask.any():
        return img16

    # Inpaint in 8-bit BGR (what cv2.inpaint supports).
    bgr8 = cv2.cvtColor(cv2.convertScaleAbs(img16, alpha=255.0 / 65535.0),
                        cv2.COLOR_RGB2BGR)
    filled8 = cv2.inpaint(bgr8, mask, inpaint_radius, cv2.INPAINT_TELEA)
    filled16 = cv2.cvtColor(filled8, cv2.COLOR_BGR2RGB).astype(np.uint16) * 257

    # Feathered alpha: dilate a touch to bury the speck's antialiased edge,
    # then box-blur to a soft 0..1 ramp so the fill has no visible seam. Inside
    # the dilate+blur halo (a few px around the mask) pixels blend toward the
    # fill; OUTSIDE the halo alpha is exactly 0, so those pixels are kept
    # bit-for-bit. The halo is the intended seamless-blend region.
    k = np.ones((3, 3), np.uint8)
    a = cv2.dilate(mask, k, iterations=1)
    a = (cv2.blur(a, (5, 5)).astype(np.float32) / 255.0)[..., None]

    out = (img16.astype(np.float32) * (1.0 - a)
           + filled16.astype(np.float32) * a)
    return np.clip(out, 0, 65535).astype(np.uint16)


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