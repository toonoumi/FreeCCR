# Getting Started with FreeCCR

Welcome to **FreeCCR** - a professional desktop application designed specifically for photographers who work with negative film scans and need fast, accurate, physics-based image conversion and color cast correction.

## Quick Start Guide

### Step 1: Launch FreeCCR
FreeCCR launches without activation in this build.
The evaluation version allows unlimited use of all features, though exported images will include a watermark.

### Step 2: Load Your Images
- **File → Load Images from Folder** to batch load all images from a directory
- **File → Load Individual Images** to select specific files
- Supported formats: RAW files (CR3, CR2, NEF, ARW, DNG, etc.) and standard images (TIFF, JPEG, PNG)

### Step 3: Draw the Reference Frame
**Right-click and drag** on the image to draw a reference frame around the picture area plus some surrounding **film base** (the orange/brown border). A **right-click without dragging** clears the frame. The conversion samples this frame to neutralise the film's colour cast, so its placement matters.

### Step 4: Check the Reference Frame
A poorly placed frame can cause colour casts. Redraw it (right-click and drag) if it includes any of:
- Sprocket holes or portions of sprocket holes within the frame boundary
- Multiple images or unintended areas captured within the selection
- Film holder components included in the frame area
- Complete absence of film base (we strongly recommend including some film base area surrounding your image for optimal results)

### Step 5: Convert the Image
Click the **Convert** button in the top toolbar to run the negative conversion.

### Step 6: (Optional) Fine-tune with Color Correction
Use the adjustment sliders on the right panel:
- **Temperature**: Adjust color temperature (warm/cool)
- **Tint**: Correct green/magenta color cast
- **Exposure**: Brighten or darken the overall image
- **Brightness**: Adjust midtone brightness
- **White/Black Point**: Set highlight and shadow clipping points
- **Contrast**: Increase or decrease contrast
- **Saturation**: Adjust color intensity
You can use the "Sync to All" button to apply adjustments to all images, or use `Ctrl/Command+C` and `Ctrl/Command+V` to copy and paste settings to another image. Both ask which setting groups you mean — white balance, tone, saturation, crop, orientation (90° rotation and mirroring), curves and so on — so you can copy just the crop, or just the white balance, and leave everything else on the target image untouched.

### Step 7: Export Your Images
-  (Optional) Select Checkbox **Export Jpg** to export JPEG for web use
- **Export All** or **Export** for exporting all images or single image 
- Select destination folder and export

### Step 8: Edit Your Images with Your prefered Editing Software
The direct output of the images is scientific and color accurate, but you can always utilize the exported TIFF file for more comprehensive personal style processing. Editing the exported TIFF file using existing professional photo editing software is part of our designed workflow.

## System Requirements

- **Operating System**: macOS 15.0+ (Apple Silicon & Intel) or Windows 10/11
- **Memory**: 8GB RAM recommended (16GB for large batch processing)
- **Storage**: 500MB for application, additional space for processed images
- **Display**: 1920x1080 minimum resolution recommended

## Installation

### macOS
1. Download the `.dmg` file for your Mac architecture:
   - [Apple Silicon (M1/M2/M3/M4)](https://www.freeccr.com/download)
   - [Intel Macs](https://www.freeccr.com/download)
2. Open the `.dmg` file and drag FreeCCR to Applications
3. Launch the application from Applications folder

### Windows
1. Download the [Windows installer](https://www.freeccr.com/download)
2. Run the installer and follow the setup wizard
3. Launch FreeCCR from the Desktop shortcut

## Workflow Tips

### Batch Processing
1. **Load all images** from your scan folder
2. **Draw a reference frame** on each image (right-click and drag)
3. **Convert** each image and review the result
4. **Export batch** when satisfied with results

### Color Correction Best Practices
- Start with **Temperature and Tint** to establish proper white balance
- Use **Exposure** for overall brightness, **Brightness** for midtone adjustments
- Set **Black and White Points** to optimize contrast range
- Apply **Contrast and Saturation** as final creative adjustments
- Monitor the **histogram** to avoid clipping highlights or shadows

## Understanding the Technology

### Why CCR vs. Simple Inversion?
Traditional negative conversion simply inverts pixel values (255 - pixel_value), which doesn't account for:
- Film's non-linear response to light
- Color dye characteristics of different film stocks  
- Varying density ranges in different parts of the negative
- Proper color balance relationships

CCR's physics-based approach produces more natural, film-like results that better represent the original scene.

## Troubleshooting

### Common Issues

**Q: Images appear too dark or too bright after conversion**
A: Adjust the Reference Frame to include only the image area, + film base. Never include your film sprocket holes or scanning backlight. Use **Brightness** button to adjust the image tone to your liking.

**Q: Colors look unnatural or oversaturated**
A: If you use an RGB light source and adjust it to 5500K for scanning negatives, the output can sometimes be overly saturated. You can simply adjust the saturation slider to your liking, or when scanning, adjust the RGB light so that your film base appears gray.

**Q: How do I set the reference frame?**
A: Right-click and drag on the image to define the frame boundaries; right-click without dragging clears it. Include the picture plus some film base, and avoid sprocket holes and the scanning backlight.

**Q: Application runs slowly with large files**
A: CCR processes images at high quality. For faster preview, the app automatically resizes working images while maintaining full resolution for export.

### Performance Optimization
- **Close other memory-intensive applications** when processing large batches
- **Process in smaller batches** if working with very large files (>50MB RAW files)
- **Use SSD storage** for better file I/O performance

## Privacy & Data Security

FreeCCR prioritizes your privacy:
- **All image processing happens locally** on your computer
- **No images are uploaded** to external servers
- **Minimal data collection** limited to technical diagnostics
- **No tracking or analytics** of your photography workflow
- **Offline operation** for core features

## Support & Resources
- **Email Support**: Contact us using https://www.freeccr.com/contact
- **Feature Requests**: We welcome feedback for improving CCR

## What's Next?

Once you're comfortable with the basics:
- Experiment with different reference frame selections for creative effects
- Explore batch processing workflows for large scanning projects
- Consider upgrading to a full license for watermark-free exports

---

**Ready to get started?** Download FreeCCR today and transform your negative film scans into beautiful positive images with professional accuracy and control.