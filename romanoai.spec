# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['romanoai.py'],
    pathex=[],
    binaries=[],
    datas=[('dist/datosModelos.json', 'dist'), ('dist/media', 'dist/media'), ('dist/dnn', 'dist/dnn'), ('dist/usuarios.db', 'dist')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='romanoai',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['dist\\media\\Img\\inicio.ico'],
)
