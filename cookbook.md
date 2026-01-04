# Development Cookbook: Chat With PDF - Local LLM Integration

**Goal:** Transform Chat With PDF into a fully local, cross-platform desktop application with automatic model management and optimal hardware utilization.

**Target:** Consumer-grade laptops (Windows/macOS) with 10GB+ RAM, distributed as standalone executables.

---

## Architecture Overview

### Stack Decision: Llama CPP + LlamaIndex RAG

**Backend:** `llama-cpp-python` with hardware detection
- Bundles into executable (~50-80MB)
- Metal support for Apple Silicon Macs
- CPU optimization for Windows
- Direct GGUF model loading

**Model Strategy:**
- **Chat LLM:** TBD after HuggingFace exploration (targeting 7B-8B models, ~6GB GGUF)
  - Candidates: Qwen2.5-7B, Qwen3-7B, Phi-3.5, Mistral-7B
  - Quantization: Q6_K or Q8_0 for quality
- **Embedding:** Nomic Embed Text v2 MOE Q8_0 (~1GB) - Already implemented
- **Download:** On first launch from HuggingFace with GUI progress bar
- **Cache:** `~/.chat-with-pdf/models/`

**Distribution:**
- **Bundle size:** ~200-300MB executable
- **Models:** Downloaded separately (not bundled)
- **Platforms:** macOS (Metal), Windows (CPU), Linux (CPU)

---

## Phase 1: Foundation (Backend Infrastructure)

### Step 1.1: Hardware Detection Module

**Create:** `src/backend/hardware.py`

**Purpose:** Detect system capabilities and determine optimal inference settings.

**Requirements:**
- Detect OS (macOS/Windows/Linux) using `platform.system()`
- Detect CPU architecture and capabilities
- Detect Apple Silicon (for Metal acceleration) by checking CPU brand string via `sysctl`
- Check available RAM using `psutil` library (add to requirements.txt)
- Determine optimal `n_gpu_layers` and `n_threads` based on detected hardware

**Implementation Strategy:**

Create a `HardwareDetector` class with static methods:
- `detect_platform()` - Returns 'macos', 'windows', or 'linux'
- `has_metal()` - Checks if running on Apple Silicon by examining CPU brand string for "Apple"
- `get_available_ram_gb()` - Returns available RAM in gigabytes
- `get_cpu_count()` - Returns logical CPU count (with fallback to 4)
- `get_optimal_settings()` - Returns dictionary with optimal configuration:
  1) CPU-only (OSX or Windows) - { 'n_gpu_layers' : 0, 'n_threads' : cpu_count }
  2) MacOS Apple Silicon - { 'n_gpu_layers' : -1, 'n_threads' : 1 }

**Testing Checklist:**
- [ ] Test on Mac Intel
- [ ] Test on Mac M1/M2/M3
- [ ] Test on Windows (various CPUs)
- [ ] Verify Metal detection accuracy
- [ ] Verify thread count calculation

---

### Step 1.2: Model Manager Module

**Create:** `src/backend/model_manager.py`

**Purpose:** Handle model downloading, caching, and validation.

**Model Management Plan:**
Total RAM → Reserve for LLM (50%) → Model Selection (all Q4 models)
8 GB      → 4 GB                   → Qwen3-Instruct-4B
16 GB     → 8 GB                   → Llama4-Scout-8B
32 GB     → 16 GB                  → Mistral-Small-3-24B
64 GB     → 32 GB                  → Qwen3-Instruct-32B

**User Messaging:**
At startup/first launch:

"ChatWithPDF works best when at least 50% of your RAM is free. Please close unnecessary applications before using the app for optimal performance."

Or in documentation:

"System Requirements: This app uses local AI models. For best performance, ensure you have at least 50% of your RAM available. Close browser tabs and other memory-intensive applications before starting."

**Optional Safety Check:**
You could still do a one-time check at startup:
If available RAM < (total RAM * 0.4):
    Show warning: "Low memory detected. Close some apps for better performance."
    [Continue Anyway] [Quit]

**Requirements:**
- Check if models exist locally
- Download models from HuggingFace with progress tracking
- Verify checksums/integrity
- Provide progress callbacks for GUI integration
- Download models from HuggingFace using `huggingface_hub` library
- Verify model integrity through file size validation
- Support resume of interrupted downloads (via `hf_hub_download` built-in feature)
- Check disk space before downloading using `shutil.disk_usage`

**Implementation Strategy:**

Create a `ModelManager` class that:
- Initializes with cache directory at `~/.chat-with-pdf/models/`
- Implements `check_models_exist()` to return status dict for each model type
- Implements `check_disk_space()` to verify sufficient free space (model size + 1GB buffer)
- Implements `download_model()` using `hf_hub_download` with resume support
- Implements `download_all_models()` to download missing models sequentially
- Implements `verify_model()` to check file size within 10% of expected value
- Implements `get_model_path()` to return absolute path to cached model
- Implements `clear_cache()` for troubleshooting (deletes all cached models)

Handle errors by raising `RuntimeError` with descriptive messages for insufficient disk space or download failures.

**Configuration File:** Create `src/backend/config.py`

Define `DEFAULT_MODELS` dictionary with structure:
- `chat` model: repo_id (TBD), filename (TBD.gguf), size_gb (6.0), context_window (8192)
- `embed` model: repo_id (nomic-ai/nomic-embed-text-v1.5-GGUF), filename, size_gb (1.0)

Define constants:
- `MODEL_CACHE_DIR = "~/.chat-with-pdf/models"`
- `APP_VERSION = "0.1.0"      # Initialize embedding model (existing custom class)

**Testing Checklist:**
- [ ] Test index creation with sample PDF
- [ ] Test streaming responses
- [ ] Verify Metal acceleration on Mac (check Activity Monitor GPU usage)
- [ ] Verify CPU usage on Windows
- [ ] Test with different PDF sizes (1 page vs 100 pages)
- [ ] Test multiple queries on same index

---

## Phase 2: User Experience (First Launch Flow)

### Step 2.1: First Launch Detection

**Modify:** `src/backend/service.py`

**Purpose:** Detect if models are available and trigger setup flow if needed.

### Step 2.2: GUI Progress Dialog

**Modify:** `main.py`

**Purpose:** Show user-friendly progress during model download.

**Add fir Required:**
1. Import `LlamaCPP` from `llama_index.llms.llama_cpp`
2. Import `HardwareDetector` from new hardware module
3. Update `__init__` to accept chat and embed model paths as parameters
4. Initialize `Settings.llm` with LlamaCPP instance instead of current implementation
5. Keep existing `LlamaCppEmbedding` for embeddings

**Implementation Strategy:**

Modify `PDFAgent` constructor to:
- Accept `chat_model_path` and `embed_model_path` as required parameters
- Call `HardwareDetector.get_optimal_settings()` to get hardware configuration
- Log initialization details (device type, threads, GPU layers)
- Initialize `Settings.embed_model` with existing `LlamaCppEmbedding` class
- Initialize `Settings.llm` with `LlamaCPP` wrapper configured with:
  - `temperature=0.1` for consistent outputs
  - `max_new_tokens=512` for response length
  - `context_window=8192` for document context
  - `model_kwargs` containing `n_gpu_layers` and `n_threads` from hardware detection

Keep existing methods:
- `create_index()` - Load PDF with `SimpleDirectoryReader`, create `VectorStoreIndex`
- `ask_agent()` - Query with streaming enabled, raise error if no index exists

Add logging throughout for debugging and monitoring. """Show main application UI"""
    # Your existing UI code...
    pass
```

**Testing Checklist:**
- [ ] Test dialog appearance and layout
- [ ] Test progress updates during download
- [ ] Test error display
- [ ] Test transition to main UI after success

---

### Step 2.3: Settings & Configuration

**Create:** Settings view for future model management

**Features to add later:**
- Model selection (switch between different models)
- Manual model update check
- Cache management (clear models)
- Hardware settings override
- Performance monitoring

**Implementation:** Defer to Phase 3

---

## Implementation Strategy:

Update `PDFService` class:
- Add `model_manager` attribute (instance of `ModelManager`)
- Add `agent` attribute (initially None)
- Add `_initialize_agent()` private method that:
  - Checks if all models exist using `model_manager.check_models_exist()`
  - If missing, logs and returns without initializing agent
  - If present, gets model paths and creates `PDFAgent` instance
  - Handles exceptions with logging

Add new public methods:
- `needs_setup()` - Returns True if agent is None (setup required)
- `run_setup(progress_callback)` - Downloads all models then initializes agent

Keep all existing PDF processing methods unchanged.
**Mac (Apple Silicon):**
- [ ] Fresh install (no models)
- [ ] Model download completes successfully
- [ ] Metal acceleration activates (check Activity Monitor)
- [ ] PDF indexing works
- [ ] Streaming responses work
- [ ] Relaunch uses cached models (fast startup)

**Mac (Intel):**
- [ ] CPU-only mode works
- [ ] Performance is acceptable

**Windows:**
- [ ] Fresh install
- [ ] Download works
- [ ] CPU-only inference
- [ ] All features functional

**Edge Cases:**
- [ ] Interrupt download mid-way (close app)
- [ ] Relaunch resumes download
- [ ] Try with very large PDF (100+ pages)
- [ ] Try with very small PDF (1 page)
- [ ] Multiple queries in sequence
- [ ] Delete models manually, app handles gracefully

**Performance Benchmarks:**
- [ ] Mac M1/M2: Target 20-40 tokens/sec
- [ ] Windows 8-core CPU: Target 5-10 tokens/sec
- [ ] Index creation: < 30 seconds for 50-page PDF
- [ ] First query: < 5 seconds to first token

---

## Phase 4: Packaging & Distribution

### Step 4.1: PyInstaller Configuration

**Goal:** Create standalone executables for each platform.

**Create:** `build_scripts/build.spec` (PyInstaller spec file)

```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
  Implementation Strategy:**

Update main application entry point:
- Instantiate `PDFService` on startup
- Check if setup is needed via `service.needs_setup()`
- If yes, show download dialog; if no, proceed to main UI

Create `show_model_download_dialog()` function:
- Create modal `AlertDialog` with:
  - Progress bar (Flet `ProgressBar` component)
  - Status text showing current operation
  - Detail text showing download progress (MB downloaded / total MB)
  - Hardware info from `HardwareDetector` (platform, Metal availability)
  - Total download size from `model_manager.get_total_download_size_gb()`
- Define `progress_callback` that updates UI elements as models download
- Define `run_setup()` to execute in background thread:
  - Check disk space first
  - Call `service.run_setup(progress_callback)`
  - Close dialog on success
  - Show error message on failure
- Launch setup in daemon thread using `threading.Thread`

Create `show_main_ui()` function:
- Transition from setup dialog to main application interface
- Initialize existing UI components
# Build with PyInstaller
pyinstaller build_scripts/build.spec

echo "Build complete: dist/ChatWithPDF.app"
```

**Create:** `build_scripts/build_windows.bat`
```batch
@echo off
echo Building Chat With PDF for Windows...

REM Clean build
rmdir /s /q build dist

REM Install dependencies
pip install llama-cpp-python --force-reinstall --no-cache-dir

REM Build with PyInstaller
pyinstaller build_scripts\build.spec

echo Build complete: dist\ChatWithPDF.exe
```

---

### Step 4.3: Installer Creation

**macOS: Create DMG**

Use `create-dmg` tool:
```bash
brew install create-dmg

create-dmg \
  --volname "Chat With PDF" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --app-drop-link 450 150 \
  "ChatWithPDF-v0.1.0-macOS.dmg" \
  "dist/ChatWithPDF.app"
```

**Windows: Create MSI Installer**

Use tools like:
- **Inno Setup** (recommended, free)
- **WiX Toolset** (more advanced)
- **NSIS** (lightweight)

Example with Inno Setup:
```ini
[Setup]
AppName=Chat With PDF
AppVersion=0.1.0
DefaultDirName={pf}\ChatWithPDF
OutputBaseFilename=ChatWithPDF-v0.1.0-Windows
Compression=lzma2
SolidCompression=yes

[Files]
Source: "dist\ChatWithPDF.exe"; DestDir: "{app}"

[Icons]
Name: "{group}\Chat With PDF"; Filename: "{app}\ChatWithPDF.exe"
Name: "{commondesktop}\Chat With PDF"; Filename: "{app}\ChatWithPDF.exe"
```

---

## Phase 5: CI/CD & Distribution

### Step 5.1: GitHub Actions Setup

**Create:** `.github/workflows/build.yml`

```yaml
name: Build and Release

on:
  push:
    tags:
      - 'v*'  # Trigger on version tags (v0.1.0, v1.0.0, etc.)

jobs:
  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
          pip install pyinstaller
      
      - name: Build app
        run: |
          pyinstaller build_scripts/build.spec
      
      - name: Create DMG
        run: |
          brew install create-dmg
          create-dmg \
            --volname "Chat With PDF" \
            --window-pos 200 120 \
            --window-size 600 400 \
            "ChatWithPDF-${{ github.ref_name }}-macOS.dmg" \
            "dist/ChatWithPDF.app"
      
      - name: Upload Release Asset
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ github.event.release.upload_url }}
          asset_path: ./ChatWithPDF-${{ github.ref_name }}-macOS.dmg
          asset_name: ChatWithPDF-${{ github.ref_name }}-macOS.dmg
          asset_content_type: application/octet-stream

  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install llama-cpp-python --force-reinstall
          pip install pyinstaller
      
      - name: Build app
        run: |
          pyinstaller build_scripts/build.spec
      
      - name: Create installer (Inno Setup)
        run: |
          choco install innosetup
          iscc build_scripts/installer.iss
      
      - name: Upload Release Asset
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ github.event.release.upload_url }}
          asset_path: ./Output/ChatWithPDF-${{ github.ref_name }}-Windows.exe
          asset_name: ChatWithPDF-${{ github.ref_name }}-Windows.exe
          asset_content_type: application/octet-stream

**Configuration Requirements:**

Set up PyInstaller spec with:
- Entry point: `main.py`
- Data files: Include entire `src/` directory
- Hidden imports (critical for bundling):
  - `llama_cpp`
  - `llama_index.llms.llama_cpp`
  - `llama_index.core`
  - `llama_index.embeddings.llama_cpp`
  - All `src.backend` modules
  - `huggingface_hub`
  - `psutil`
- Executable settings:
  - Name: `ChatWithPDF`
  - Console: False (no terminal window)
  - UPX compression: Enabled
- macOS: Create `.app` bundle with icon and bundle identifier

**Platform-Specific Notes:**

**macOS:**
- Must build on macOS machine
- Install `llama-cpp-python` with Metal support: `CMAKE_ARGS="-DLLAMA_METAL=on"`
- Run: `pyinstaller build_scripts/build.spec`
- Output: `.app` bundle in `dist/` folder

**Windows:**
- Must build on Windows machine
- `llama-cpp-python` auto-detects CPU features (AVX2/AVX512)
- Run: `pyinstaller build_scripts/build.spec`
- Output: `.exe` file in `dist/` folder

**Linux:**
- Build process similar to Windows
- Output: Binary executable in `dist/` folder
    )
    page.banner = banner
    page.banner.open = True
    page.update()
```

3. **Run Check on Startup (non-blocking):**
```python
def main(page: ft.Page):
    # ... setup code ...
    
    # Check for updates in background
    def check_updates():
        update_info = check_for_updates(APP_VERSION)
        if update_info["update_available"]:
            show_update_notification(page, update_info)
    
    threading.Thread(target=check_updates, daemon=True).start()
```

---

### Step 5.4: Release Process

**Steps for releasing a new version:**

1. **Update version:**
   - Update `APP_VERSION` in `src/backend/config.py`
   - Update `version` in `pyproject.toml`

2. **Create Git tag:**
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

3. **GitHub Actions builds automatically:**
   - Creates binaries for all platforms
   - Uploads to GitHub Releases

4. **Create release notes:**
   - Go to GitHub Releases
   - Edit the auto-created release
   - Add changelog and installation instructions

5. **Users get notified:**
   - App checks for updates on next launch
   - Shows update banner with download link

---

## Testing & Quality Assurance

### Pre-Release Checklist

Bash script that:
- Cleans previous build artifacts (`rm -rf build dist`)
- Reinstalls `llama-cpp-python` with Metal flags: `CMAKE_ARGS="-DLLAMA_METAL=on"`
- Runs PyInstaller with spec file
- Reports output location: `dist/ChatWithPDF.app`

**Create:** `build_scripts/build_windows.bat`

Batch script that:
- Cleans previous build artifacts
- Reinstalls `llama-cpp-python` (auto-detects CPU features)
- Runs PyInstaller with spec file
- Reports output location: `dist\ChatWithPDF.exe
---

## Future Enhancements (Nice-to-Haves)

1. **Multiple Model Support**
   - Let users choose between 3B/7B models
   - Quality vs speed tradeoff

2. **Model Updates**
   - Check HuggingFace for newer model versions
   - One-click model update

3. **Advanced Settings**
   - Adjust temperature, top_p, etc.
   - Change context window size
   - Performance tuning

4. **Chat History**
   - Persist conversations across sessions
   - Export chat history

5. **Multi-PDF Support**
   - Index multiple PDFs
   - Cross-document queries

6. **OCR Support**
   - Extract text from scanned PDFs
   - Image-based PDFs (install via Homebrew):
- Configure DMG with application name, window size/position, icon size
- Add symbolic link to Applications folder for easy installation
- Generate DMG file from `.app` bundle in `dist/`
- Output: `ChatWithPDF-v0.1.0-macOS.dmg`

**Windows: Create MSI Installer**

Recommended tools:
- **Inno Setup** (free, easiest)
- **WiX Toolset** (advanced, XML-based)
- **NSIS** (lightweight scripting)

Using Inno Setup:
- Create installer script (`.iss` file) with:
  - Application metadata (name, version)
  - Installation directory
  - Files to include (executable from `dist/`)
  - Desktop and Start Menu shortcuts
  - Compression settings
- Compile script to generate installer executable
- Output: `ChatWithPDF-v0.1.0-Windows.exe-

## Resources & References

### Documentation
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python Docs](https://llama-cpp-python.readthedocs.io/)
- [HuggingFace Hub Docs](https://huggingface.co/docs/huggingface_hub/)

### Model Sources
- [HuggingFace GGUF Models](https://huggingface.co/models?library=gguf)
- [TheBloke's GGUF Collection](https://huggingface.co/TheBloke)

### Build Tools
- [PyInstaller Manual](https://pyinstaller.org/en/stable/)
- [create-dmg](https://github.com/create-dmg/create-dmg)
- [Inno Setup](https://jrsoftware.org/isinfo.php)

### Code Signing
- [Apple Code Signing](https://developer.apple.com/support/code-signing/)
- [Windows Code Signing](https://docs.microsoft.com/en-us/windows/win32/seccrypto/signtool)

---

## Timeline Estimate
**Configuration Overview:**

Trigger: On Git tags matching pattern `v*` (e.g., v0.1.0, v1.0.0)

**Three parallel jobs:**

**Job 1: build-macos** (runs on macos-latest)
- Checkout repository
- Set up Python 3.11
- Install dependencies from requirements.txt
- Install llama-cpp-python with Metal support
- Build app with PyInstaller
- Create DMG with create-dmg tool
- Upload DMG to GitHub Release

**Job 2: build-windows** (runs on windows-latest)
- Checkout repository
- Set up Python 3.11
- Install dependencies from requirements.txt
- Install llama-cpp-python (CPU auto-detect)
- Build app with PyInstaller
- Install Inno Setup via Chocolatey
- Create installer with Inno Setup
- Upload installer to GitHub Release

**Job 3: build-linux** (runs on ubuntu-latest)
- Checkout repository
- Set up Python 3.11
- Install dependencies from requirements.txt
**1. Version Check on Startup:**
- Create function that queries GitHub API for latest release
- Compare current app version with latest using semantic versioning
- Return dictionary with update availability, version number, and download URL
- Handle network errors gracefully (timeout after 5 seconds, return no update available)

**2. Update Notification UI:**
- Create Flet Banner component with update icon
- Display message showing available version
- Add "Download" button that opens browser to release page
- Add "Later" button to dismiss
- Show banner at top of application window

**3. Background Check Integration:**
- Run update check in daemon thread on application startup
- Non-blocking - doesn't delay app initialization
- Only show notification if update is available
- User can continue using app while check happens