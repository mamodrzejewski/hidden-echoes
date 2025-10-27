#!/bin/bash
set -e

EXTERNAL_DIR="external"
REPO_URL="https://github.com/Stability-AI/stable-audio-tools.git"
REPO_NAME="stable-audio-tools"
REPO_COMMIT="18b3dae6bf46e0fb71f09223ba5179432731b111"

mkdir -p "$EXTERNAL_DIR"

if [ ! -d "$EXTERNAL_DIR/$REPO_NAME" ]; then
    git clone "$REPO_URL" "$EXTERNAL_DIR/$REPO_NAME"
    cd "$EXTERNAL_DIR/$REPO_NAME"
    git checkout "$REPO_COMMIT"
    cd -
fi

if [ -f "$EXTERNAL_DIR/$REPO_NAME/requirements.txt" ]; then
    pip install -r "$EXTERNAL_DIR/$REPO_NAME/requirements.txt"
fi

pip install -e "$EXTERNAL_DIR/$REPO_NAME"

if [ -f "requirements.txt" ]; then
    pip install -r "requirements.txt"
fi

