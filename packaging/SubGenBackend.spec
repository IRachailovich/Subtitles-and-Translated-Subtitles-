from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata


project_root = Path(SPECPATH).parent
datas = [
    (str(project_root / "web"), "web"),
    (str(project_root / "Amiri-Regular.ttf"), "."),
]
hiddenimports = []
binaries = []

for package in (
    "faster_whisper",
    "google.auth",
    "google_auth_httplib2",
    "google_auth_oauthlib",
    "googleapiclient",
    "torchcodec",
    "transformers",
    "whisperx",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hidden)

# Transformers resolves these lazily at runtime. Keep them explicit so a
# frozen build cannot omit the alignment classes while still building cleanly.
hiddenimports.extend(collect_submodules("transformers.models.wav2vec2"))
datas.extend(copy_metadata("torchcodec"))
hiddenimports.extend([
    "whisperx.alignment",
    "transformers.models.wav2vec2.modeling_wav2vec2",
    "transformers.models.wav2vec2.processing_wav2vec2",
])

hiddenimports.extend(collect_submodules("keyring.backends"))

a = Analysis(
    [str(project_root / "desktop_backend.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # scikit-learn treats pyarrow as optional, but if PyInstaller discovers the
    # host copy it bundles pyarrow DLLs that fail to load in the frozen app.
    # Leaving it absent exercises sklearn's supported ModuleNotFoundError path.
    excludes=["pyarrow"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SubGenBackend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SubGenBackend",
)
