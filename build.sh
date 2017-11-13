#!/bin/sh
g++ -o One -std=c++0x -O3 -DNDEBUG main.cpp -lGL -lX11 -lpthread -lasound
