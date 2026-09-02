@echo off
echo Building MangaRelief Pro Executable...

pyinstaller --noconfirm --onedir --windowed --icon "icon.ico" ^
  --add-data "style.qss;." ^
  --add-data "icon.ico;." ^
  --add-data "assets;assets/" ^
  --hidden-import scipy ^
  --hidden-import sklearn ^
  --hidden-import trimesh ^
  --hidden-import lxml ^
  --hidden-import lxml.etree ^
  --hidden-import PIL ^
  --hidden-import pillow_heif ^
  --hidden-import fast_simplification ^
  --hidden-import shapely ^
  --hidden-import manifold3d ^
  --hidden-import mapbox_earcut ^
  "manga_to_3d.py"

echo Build complete! The executable is in the 'dist' folder.
pause
