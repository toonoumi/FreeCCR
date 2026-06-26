.PHONY: all install-deps install-nuitka build build-legacy build-compatible build-windows clean check-compatibility

# Python interpreter
PYTHON ?= python

# Build scripts directory
BUILD_SCRIPTS_DIR = macos_build_scripts

all: build

install-deps:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-nuitka:
	$(PYTHON) -m pip install --upgrade nuitka

# macOS builds (delegate to specialized scripts)
build: install-deps install-nuitka
	$(BUILD_SCRIPTS_DIR)/build_compatible.sh

build-compatible: install-deps install-nuitka
	$(BUILD_SCRIPTS_DIR)/build_compatible.sh

# Windows build (cross-platform)
build-windows: install-deps install-nuitka
	$(PYTHON) write_version.py
	MACOSX_DEPLOYMENT_TARGET=10.15 $(PYTHON) -m nuitka --mingw64 --clang --standalone --include-package=numpy --enable-plugin=pyside6 \
	--include-data-dir=src/icons=icons --windows-icon-from-ico=src/icons/freeccr_logo.ico \
	--include-data-dir=LICENSES=LICENSES --include-data-dir=src/models=models \
	--windows-console-mode=attach --output-filename=FreeCCR \
	src/main.py

# Utility targets
check-compatibility:
	./check_compatibility.sh

clean:
	rm -rf __pycache__ build dist *.dist-info *.spec *.build
	rm -rf main.build main.dist main.exe main.bin
	rm -rf FreeCCR.build FreeCCR.dist FreeCCR.exe FreeCCR.bin
	rm -rf *.app *.dmg