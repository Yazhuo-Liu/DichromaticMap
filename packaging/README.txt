DichromaticMap portable viewer
=============================

Python, NumPy and the Qt viewer libraries are included. Extract the entire
archive before launching; keep the executable and its supporting files together.

Windows: double-click DichromaticMap.exe.
macOS: open DichromaticMap.app (choose the archive for Apple Silicon or Intel).
Linux: run ./DichromaticMap from a terminal or your file manager.

Windows and macOS packages are not signed with a publisher certificate; macOS
packages are not notarized. Follow your organization's application policy.
On macOS, if you trust the archive from this project's GitHub Releases, try
opening DichromaticMap.app once, then use System Settings > Privacy & Security
> Open Anyway and confirm Open. Do not override a warning that the app is
damaged or will harm your Mac. Apple's instructions:
https://support.apple.com/en-us/102445
If your policy blocks the application, use the Python installation or build
from source. These archives are not installers and do not register .dmap
file associations. Import sessions from the viewer's Import session control.

Command-line arguments are the same as for python -m dichromatic_map, for example:
  ./DichromaticMap --workers 2 --lattice BCC --axis 100 --save pattern.png
On macOS use ./DichromaticMap.app/Contents/MacOS/DichromaticMap.
Windows/macOS GUI builds do not print command-line output to a terminal.

Linux builds require desktop system libraries; on Ubuntu these include
libegl1, libopengl0, libxcb-cursor0 and the usual X11/xcb libraries.
Official builds use Ubuntu 22.04, Windows 2022 and macOS 15 runners.
Use the Python package for other operating systems or architectures.

See RELEASE_NOTES.md for changes and download details. DichromaticMap's license
is in LICENSE. See THIRD_PARTY_NOTICES.txt and licenses/ for interpreter,
bootloader and Qt license texts. Other bundled license texts are in package
metadata (*.dist-info), under _internal on Windows/Linux or inside the macOS app.
PySide6/Qt use their own license terms; they are not covered by the project's MIT
license. The bundled libraries remain separate from the executable.

Source and documentation: https://github.com/Yazhuo-Liu/DichromaticMap
Issues: https://github.com/Yazhuo-Liu/DichromaticMap/issues
