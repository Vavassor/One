@echo off
cl /O2 main.cpp kernel32.lib user32.lib gdi32.lib opengl32.lib /link /out:One.exe
