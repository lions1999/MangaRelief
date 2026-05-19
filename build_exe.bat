@echo off
echo Building MangaRelief Pro Executable...

pyinstaller --noconfirm --onedir --windowed --icon "icon.ico" ^
  --add-data "style.qss;." ^
  --add-data "icon.ico;." ^
  --add-data "assets;assets/" ^
  --hidden-import scipy ^
  --hidden-import sklearn ^
  --hidden-import trimesh ^
  --hidden-import PIL ^
  "manga_to_3d.py"

echo Build complete! The executable is in the 'dist' folder.
pause
