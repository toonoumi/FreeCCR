#!/bin/bash
# Build script for maximum macOS compatibility
set -e

SIGN_CERTIFICATE="Developer ID Application: Yindividual LLC (NW9XSB3U68)"

APP_NAME="FreeCCR"
BUNDLE_DIR="${APP_NAME}.app"
CONTENTS_DIR="${BUNDLE_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
PLIST_SRC="macos_build_scripts/Info.plist"
ENTITLEMENTS_SRC="macos_build_scripts/entitlements.plist"
ICON_SRC="src/icons/freeccr_logo.icns"
NUITKA_BIN="main.dist/main.bin"
DMG_NAME="${APP_NAME}.dmg"
DMG_TEMP_DIR="dmg_temp"
LICENSE_FILE="LICENSES/license-FreeCCR.txt"
DMG_SETTINGS="macos_build_scripts/dmg_settings.py"

echo "Building FreeCCR with maximum compatibility..."

# Check Python deployment target
PYTHON_DEPLOYMENT_TARGET=$(python -c "import sysconfig; print(sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET') or 'None')")
echo "Current Python deployment target: $PYTHON_DEPLOYMENT_TARGET"

if [[ "$PYTHON_DEPLOYMENT_TARGET" != "10.15" && "$PYTHON_DEPLOYMENT_TARGET" != "None" ]]; then
    echo "WARNING: Python was compiled with deployment target $PYTHON_DEPLOYMENT_TARGET"
    echo "This may prevent creating binaries compatible with macOS 10.15"
    echo "Consider using a Python version compiled with lower deployment target"
fi

# Set strict deployment target
export MACOSX_DEPLOYMENT_TARGET=10.15
export SYSTEM_VERSION_COMPAT=1

# Additional flags to ensure compatibility
export CC=clang
export CXX=clang++

echo "Environment variables set:"
echo "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
echo "CFLAGS=$CFLAGS"
echo "CXXFLAGS=$CXXFLAGS"
echo "LDFLAGS=$LDFLAGS"

# Install dependencies
echo "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade nuitka

# Write version
echo "Writing version..."
python write_version.py

# Get version for release directory
VERSION=$(python -c "import sys; sys.path.append('src'); from version import VERSION; print(VERSION)")
echo "Version: $VERSION"

# Build with Nuitka
echo "Building with Nuitka..."
# Detect number of CPU cores for parallel compilation
# NPROC=$(sysctl -n hw.ncpu)
# echo "Using $NPROC CPU cores for compilation..."

# Get PyOpenCL cl directory path
PYOPENCL_CL_DIR=$(python -c "import pyopencl, os; print(os.path.join(os.path.dirname(pyopencl.__file__), 'cl'))")
echo "PyOpenCL cl directory: $PYOPENCL_CL_DIR"

python -m nuitka \
    --clang \
    --standalone \
    --include-package=numpy \
    --include-package=pyopencl \
    --include-data-dir="$PYOPENCL_CL_DIR=pyopencl/cl" \
    --enable-plugin=pyside6 \
    --include-data-dir=src/icons=icons \
    --include-data-dir=LICENSES=LICENSES \
    --include-data-dir=src/models=models \
    --deployment \
    --nofollow-import-to=doctest \
    --nofollow-import-to=unittest \
    --nofollow-import-to=pytest \
    --nofollow-import-to=test \
    --nofollow-import-to=IPython \
    src/main.py

    # --macos-app-icon=src/icons/freeccr_logo.icns \
    # --macos-create-app-bundle \
    # --macos-app-name=FreeCCR \
    # --macos-sign-identity=auto \
    # --macos-sign-notarization \
    # --macos-app-version="$(git describe --tags)" \
    # --include-data-dir="$PYOPENCL_CL_DIR=pyopencl/cl" \

echo "Build completed!"

# Clean previous bundle and temp
rm -rf "$BUNDLE_DIR" "$DMG_TEMP_DIR" "$DMG_NAME"

# Create bundle structure
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

# Remove extended attributes from source files BEFORE copying
echo "Removing extended attributes from source files before copying..."
xattr -cr main.dist/ 2>/dev/null || true
xattr -cr "$PLIST_SRC" 2>/dev/null || true
xattr -cr "$ICON_SRC" 2>/dev/null || true

# Copy executable
cp "$NUITKA_BIN" "$MACOS_DIR/$APP_NAME"

# Copy Info.plist
cp "$PLIST_SRC" "$CONTENTS_DIR/Info.plist"

# Copy icon if it exists
if [ -f "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$RESOURCES_DIR/freeccr_logo.icns"
fi

# Copy all required .so, .dylib, and resource files from Nuitka build
cp -R main.dist/* "$MACOS_DIR/"
# Remove the copied binary duplicate
rm -f "$MACOS_DIR/main.bin"

#export CODESIGN_ALLOCATE="/Applications/Xcode.app/Contents/Developer/usr/bin/codesign_allocate"

# Comprehensive extended attributes removal from the .app bundle
echo "Comprehensive extended attributes removal from $BUNDLE_DIR..."
# Remove all extended attributes recursively
xattr -cr "$BUNDLE_DIR" 2>/dev/null || true

# Multiple passes to ensure complete removal
echo "Multiple passes of extended attribute removal..."
for i in {1..3}; do
    echo "Pass $i of extended attribute removal..."
    find "$BUNDLE_DIR" -exec xattr -c {} \; 2>/dev/null || true
    
    # Check for specific problematic attributes and remove them
    find "$BUNDLE_DIR" -exec xattr -d com.apple.FinderInfo {} \; 2>/dev/null || true
    find "$BUNDLE_DIR" -exec xattr -d com.apple.fileprovider.fpfs#P {} \; 2>/dev/null || true
    find "$BUNDLE_DIR" -exec xattr -d com.apple.metadata:_kMDItemUserTags {} \; 2>/dev/null || true
    find "$BUNDLE_DIR" -exec xattr -d com.apple.metadata:kMDItemWhereFroms {} \; 2>/dev/null || true
    find "$BUNDLE_DIR" -exec xattr -d com.apple.quarantine {} \; 2>/dev/null || true
    
    # Additional aggressive removal for persistent attributes
    xattr -d com.apple.FinderInfo "$BUNDLE_DIR" 2>/dev/null || true
    xattr -d com.apple.fileprovider.fpfs#P "$BUNDLE_DIR" 2>/dev/null || true
done

# # Final verification and forced removal if needed
# echo "Final verification of extended attributes..."
# REMAINING_ATTRS=$(xattr -r "$BUNDLE_DIR" 2>/dev/null || true)
# if [ -n "$REMAINING_ATTRS" ]; then
#     echo "WARNING: Some extended attributes still present:"
#     echo "$REMAINING_ATTRS"
#     echo "Attempting forced removal..."
    
#     # Try with sudo if needed
#     if command -v sudo >/dev/null 2>&1; then
#         sudo xattr -cr "$BUNDLE_DIR" 2>/dev/null || true
#         sudo find "$BUNDLE_DIR" -exec xattr -c {} \; 2>/dev/null || true
        
#         # Aggressive removal of specific persistent attributes with sudo
#         sudo xattr -d com.apple.FinderInfo "$BUNDLE_DIR" 2>/dev/null || true
#         sudo xattr -d com.apple.fileprovider.fpfs#P "$BUNDLE_DIR" 2>/dev/null || true
#     fi
    
#     # Nuclear option: recreate the bundle directory structure to avoid persistent attributes
#     if [ -n "$(xattr -r "$BUNDLE_DIR" 2>/dev/null || true)" ]; then
#         echo "Nuclear option: Recreating bundle structure to eliminate persistent attributes..."
#         TEMP_BUNDLE="${BUNDLE_DIR}_temp"
#         mkdir -p "$TEMP_BUNDLE/Contents/MacOS" "$TEMP_BUNDLE/Contents/Resources"
        
#         # Copy files without preserving extended attributes
#         cp "$CONTENTS_DIR/Info.plist" "$TEMP_BUNDLE/Contents/Info.plist"
#         cp -R "$MACOS_DIR"/* "$TEMP_BUNDLE/Contents/MacOS/"
#         cp -R "$RESOURCES_DIR"/* "$TEMP_BUNDLE/Contents/Resources/" 2>/dev/null || true
        
#         # Remove old bundle and replace with clean one
#         rm -rf "$BUNDLE_DIR"
#         mv "$TEMP_BUNDLE" "$BUNDLE_DIR"
        
#         echo "Bundle recreated without extended attributes"
#     fi
    
#     # Final check
#     FINAL_CHECK=$(xattr -r "$BUNDLE_DIR" 2>/dev/null || true)
#     if [ -n "$FINAL_CHECK" ]; then
#         echo "ERROR: Unable to remove all extended attributes:"
#         echo "$FINAL_CHECK"
#         echo "This may cause codesigning or notarization to fail."
#     else
#         echo "✅ All extended attributes successfully removed after forced cleanup"
#     fi
# else
#     echo "✅ All extended attributes successfully removed"
# fi

# Sign all binaries and libraries with hardened runtime
echo "Code signing all binaries with hardened runtime..."
find $BUNDLE_DIR -type f \( -name "*.so" -o -name "*.dylib" -o -perm +111 \) -exec codesign --force --verify --verbose --sign "$SIGN_CERTIFICATE" --options runtime --entitlements "$ENTITLEMENTS_SRC" {} \;

# Sign the main app bundle with entitlements
codesign --deep --force --verify --verbose --sign "$SIGN_CERTIFICATE" --options runtime --entitlements "$ENTITLEMENTS_SRC" $BUNDLE_DIR

# Verify code signature and hardened runtime
echo "Verifying code signature..."
codesign --verify --deep --strict $BUNDLE_DIR
echo "Checking hardened runtime..."
codesign --display --verbose $BUNDLE_DIR

echo "Bundle created at $BUNDLE_DIR"

# Prepare DMG temp folder with .app and license
mkdir -p "$DMG_TEMP_DIR"
cp -R "$BUNDLE_DIR" "$DMG_TEMP_DIR/"
cp "$LICENSE_FILE" "$DMG_TEMP_DIR/"  # <-- Copy license to DMG root

#ln -s /Applications "$DMG_TEMP_DIR/Applications"  # <-- Enable Applications symlink


# Create DMG using dmgbuild (requires dmgbuild to be installed)
if command -v dmgbuild >/dev/null 2>&1; then
    pushd "$DMG_TEMP_DIR"
    dmgbuild -s "../$DMG_SETTINGS" "$APP_NAME" "../$DMG_NAME"
    popd
    echo "DMG created at $DMG_NAME"
else
    echo "dmgbuild not found. Please install it with 'pip install dmgbuild' to generate a DMG."
fi

# now notarize the DMG
echo "Notarizing the DMG..."
# Verify hardened runtime is enabled on the main executable
echo "Checking hardened runtime on main executable..."
codesign --display --verbose=2 "$BUNDLE_DIR/Contents/MacOS/$APP_NAME" 2>&1 | grep -q "runtime" && echo "✅ Hardened runtime is enabled" || echo "❌ Hardened runtime is NOT enabled"



# Verify DMG is properly signed before notarization
echo "Verifying DMG signature before notarization..."
codesign --verify --deep --strict $DMG_NAME || echo "Warning: DMG signature verification failed"

# if first time notarizing, you need to store credentials, run below command once
# xcrun notarytool store-credentials "notaryccr" --apple-id "toono.umi@gmail.com" --team-id "NW9XSB3U68" --password "pgvb-utiz-mynm-qync"
echo "Submitting DMG for notarization... $(date +%Y-%m-%dT%H:%M:%S)"
xcrun notarytool submit $DMG_NAME --keychain-profile "notaryccr" --wait
if [ $? -eq 0 ]; then
    echo "Notarization successful! $(date +%Y-%m-%dT%H:%M:%S)"
    
    # Create release directory and move DMG
    RELEASE_DIR="./release/$VERSION"
    echo "Creating release directory: $RELEASE_DIR"
    mkdir -p "$RELEASE_DIR"
    
    echo "Moving DMG to release directory..."
    mv "$DMG_NAME" "$RELEASE_DIR/"
    echo "DMG moved to: $RELEASE_DIR/$DMG_NAME"
else
    echo "Notarization failed. Checking logs..."
    xcrun notarytool log $DMG_NAME --keychain-profile "notaryccr"
    exit 1
fi



# Clean up temp folder
rm -rf "$DMG_TEMP_DIR"

# Check the resulting binary's deployment target
# echo "Checking deployment target of the built binary..."
# if [ -f "FreeCCR.dist/FreeCCR.bin" ]; then
#     otool -l FreeCCR.dist/FreeCCR.bin | grep -A 3 LC_VERSION_MIN_MACOSX || echo "No deployment target found"
# elif [ -f "FreeCCR.app/Contents/MacOS/FreeCCR" ]; then
#     otool -l FreeCCR.app/Contents/MacOS/FreeCCR | grep -A 3 LC_VERSION_MIN_MACOSX || echo "No deployment target found"
# else
#     echo "Could not find built binary to check"
# fi
