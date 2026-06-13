#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install it first:"
    echo 'https://brew.sh'
    exit 1
fi

brew update

brew install \
    curl \
    cmake \
    ninja \
    git \
    git-lfs \
    pkg-config

git lfs install

echo "macOS dependencies installed"