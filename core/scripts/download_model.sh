#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="models/paraphrase-multilingual-MiniLM-L12-v2"
SENTENCE_TRANSFORMERS_BASE_URL="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main"
ONNX_BASE_URL="https://huggingface.co/onnx-models/paraphrase-multilingual-MiniLM-L12-v2-onnx/resolve/main"

MANIFEST_FILES=(
    "1_Pooling/config.json"
    "config.json"
    "config_sentence_transformers.json"
    "modules.json"
    "sentence_bert_config.json"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
)

WEIGHT_FILES=(
    "pytorch_model.bin"
)

ONNX_FILES=(
    "model.onnx"
)

mkdir -p "$MODEL_DIR"

download() {
    local base_url="$1"
    local file="$2"
    local mode="$3"
    local destination="$MODEL_DIR/$file"
    local tmp_file="$destination.download"

    if [[ "$mode" == "skip_existing" && -f "$destination" ]]; then
        echo "Already exists: $destination"
        return
    fi

    mkdir -p "$(dirname "$destination")"

    echo "Downloading $file..."
    curl -L \
        --fail \
        --retry 5 \
        --retry-delay 2 \
        "$base_url/$file" \
        -o "$tmp_file"
    mv "$tmp_file" "$destination"
}

for file in "${MANIFEST_FILES[@]}"; do
    download "$SENTENCE_TRANSFORMERS_BASE_URL" "$file" "overwrite"
done

for file in "${WEIGHT_FILES[@]}"; do
    download "$SENTENCE_TRANSFORMERS_BASE_URL" "$file" "skip_existing"
done

for file in "${ONNX_FILES[@]}"; do
    download "$ONNX_BASE_URL" "$file" "skip_existing"
done

echo "Model downloaded to: $MODEL_DIR"
echo "Available runtimes: SentenceTransformers, ONNX"
