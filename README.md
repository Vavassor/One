# ONE

This is a single source file game written in C++. It's still in development and currently is just a test program.

## Why only one file?

It's mostly an exercise in constraint, trying to see how much can be done without art assets like images, meshes, map layout files, etc. Also, since all of the code for the game has to be in the file, there can be no outside dependency on libraries or anything else. Barring, of course, the operating system libraries and C runtime.

Additional rules aside from the "single file" one are needed. Because, I could embed any file or pre-made data in a .cpp by encoding it as a table of integers or a string literal. So that kind of storage should be minimized. Also, downloading any data from the internet at runtime would allow the same sort of loophole, so that's also disallowed.

## Building

It can be compiled using GCC on Linux and Visual C++ on Windows. For convenience, scripts are included with compile commands already written so you can double-click one of those to compile; build.sh for Linux, build.bat for Windows. Though on Linux the .sh has to be given executable permission first!
