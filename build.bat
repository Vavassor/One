@echo off
cl /O2 /DNDEBUG main.cpp kernel32.lib user32.lib gdi32.lib opengl32.lib ole32.lib avrt.lib /link /out:One.exe
