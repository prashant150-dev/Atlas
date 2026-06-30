#!/usr/bin/env bash
# Build the AVX2 ternary kernel into a shared library.
# Needs MinGW-w64 gcc on PATH (see README.md). Run from repo root or this dir.
set -e
cd "$(dirname "$0")"
gcc -O3 -mavx2 -mfma -funroll-loops -ffast-math -shared -o libternary.dll ternary_kernel.c
echo "built libternary.dll"
