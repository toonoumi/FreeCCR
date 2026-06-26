@echo off
python write_version.py

for /f "delims=" %%i in ('python -c "import pyopencl, os; print(os.path.join(os.path.dirname(pyopencl.__file__), 'cl'))"') do set PYOPENCL_CL_DIR=%%i
echo PyOpenCL cl directory: %PYOPENCL_CL_DIR%

REM onnxruntime ships native DLLs (capi/onnxruntime*.dll) + a .pyd extension
REM that Nuitka must bundle, or the AI dust-detection feature is unavailable in
REM the compiled build. --include-package pulls the modules + extension and
REM --include-package-data copies the native DLLs alongside it.
REM See spec/dust-removal.md.

REM Build with MSVC (Visual Studio), NOT mingw: MSVC compiles FreeCCR's huge
REM generated C files (ccr_processor, __helpers) that crashed gcc 14.2 / clang 19,
REM and matches onnxruntime's toolchain. --jobs=6 bounds parallel compilation
REM memory (24 cores would otherwise launch 24 cl.exe at once).
nuitka --msvc=latest --jobs=6 --standalone --assume-yes-for-downloads --include-package=numpy --include-package=utils --enable-plugin=pyside6 ^
--include-data-dir=src/icons=icons --include-data-dir=LICENSES=LICENSES --include-data-dir=src/models=models --windows-icon-from-ico=src/icons/freeccr_logo.ico ^
--windows-console-mode=attach --include-package=pyopencl --include-data-dir="%PYOPENCL_CL_DIR%=pyopencl/cl" ^
--include-package=onnxruntime --include-package-data=onnxruntime ^
--nofollow-import-to=doctest --nofollow-import-to=IPython ^
--nofollow-import-to=PIL.PdfParser --nofollow-import-to=PIL.PdfImagePlugin ^
--output-filename=freeccr.exe src/main.py

REM Nuitka bundles the MSVC runtime from the VC toolset redist, which is OLDER
REM (14.29/14.32) than onnxruntime.dll requires. Loaded against that stale
REM runtime, onnxruntime's .pyd fails its init ("DLL initialization routine
REM failed") and AI dust detection is disabled in the exe. Overwrite with the
REM system's current runtime so it initialises. See spec/dust-removal.md §8.
for %%d in (msvcp140.dll msvcp140_1.dll msvcp140_2.dll vcruntime140.dll vcruntime140_1.dll) do (
    if exist "%SystemRoot%\System32\%%d" copy /Y "%SystemRoot%\System32\%%d" "main.dist\%%d" >nul
)
echo Refreshed bundled MSVC runtime for onnxruntime.