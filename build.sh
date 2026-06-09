#!/bin/bash
set -e

echo "Building ggml via CMake..."
cd ggml
mkdir -p build
cd build
cmake ..
make -j4
cd ../..

echo "Compiling esmf.c..."
gcc -O3 -march=native -c esmf.c -o esmf.o

echo "Building protein..."
RPATH="$(pwd)/ggml/build/src"
g++ -std=c++17 -O3 -march=native -o protein protein_ext.cpp esmf.o \
    -I ggml/include -L ggml/build/src -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,$RPATH -lm -lpthread

echo "Done! Run with: ./protein model.esmf <AA_sequence>"
