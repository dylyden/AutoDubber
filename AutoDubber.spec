# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [
    ('index.html', '.'),
    ('app.js', '.'),
    ('style.css', '.'),
    ('favicon.ico', '.'),
    ('dubber.py', '.'),
    ('voice_converter.py', '.'),
    ('dialog_picker.py', '.'),
    ('run_voxcpm_studio.py', '.'),
    ('vertex_batch_processor.py', '.'),
    ('system_diagnostic.py', '.'),
]
binaries = []
hiddenimports = ['demucs', 'voice_converter', 'dialog_picker', 'vertex_batch_processor', 'librosa', 'scipy']

for pkg in ('demucs', 'faster_whisper', 'safehttpx', 'soundfile', 'edge_tts'):
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('voxcpm', filter_submodules=lambda name: 'training' not in name)
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

excludes = [
    'tensorflow', 'tensorboard', 'keras', 'playwright', 'selenium', 'frida', 'frida_tools',
    'unicorn', 'keystone', 'polars', '_polars_runtime_32', 'gradio', 'matplotlib',
    'ultralytics', 'mitmproxy', 'telethon', 'faker', 'cv2', 'pandas', 'pyarrow'
]

a = Analysis(
    ['server.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# Filter out static .lib files (saves >750 MB of C++ static link archives like dnnl.lib)
a.binaries = [b for b in a.binaries if not b[0].lower().endswith('.lib')]
a.datas = [d for d in a.datas if not d[0].lower().endswith('.lib')]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoDubber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['favicon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AutoDubber',
)
