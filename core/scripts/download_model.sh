#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="models/paraphrase-multilingual-MiniLM-L12-v2"
BASE_URL="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main"

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

# fp32 baseline, required. Lives in the repo's onnx/ subfolder; downloaded
# flat into MODEL_DIR so it stays at MODEL_DIR/model.onnx.
ONNX_FILES=(
    "model.onnx"
)

# INT8 builds, optional. One per CPU target — pick the right one in .env via
# MODEL_FILE. A variant missing upstream is skipped, not fatal.
# NOTE: bash arrays use NO commas between elements.
ONNX_INT8_FILES=(
    "model_qint8_arm64.onnx"        # Apple Silicon / ARM (local dev)
#    "model_qint8_avx512_vnni.onnx"  # Intel with VNNI (target prod)
#    "model_quint8_avx2.onnx"        # Intel, AVX2 only (older CPUs)
)

mkdir -p "$MODEL_DIR"

# download <remote_path> <dest_path_under_model_dir> <mode>
download() {
    local remote="$1"
    local dest="$2"
    local mode="$3"
    local destination="$MODEL_DIR/$dest"
    local tmp_file="$destination.download"

    if [[ "$mode" == "skip_existing" && -f "$destination" ]]; then
        echo "Already exists: $destination"
        return
    fi

    mkdir -p "$(dirname "$destination")"

    echo "Downloading $remote -> $dest ..."
    curl -L \
        --fail \
        --retry 5 \
        --retry-delay 2 \
        "$BASE_URL/$remote" \
        -o "$tmp_file"
    mv "$tmp_file" "$destination"
}

for file in "${MANIFEST_FILES[@]}"; do
    download "$file" "$file" "overwrite"
done

for file in "${WEIGHT_FILES[@]}"; do
    download "$file" "$file" "skip_existing"
done

#for file in "${ONNX_FILES[@]}"; do
#    download "onnx/$file" "$file" "skip_existing"
#done

for file in "${ONNX_INT8_FILES[@]}"; do
    download "onnx/$file" "$file" "skip_existing" \
        || echo "Skipping unavailable INT8 build: onnx/$file"
done

echo "Model downloaded to: $MODEL_DIR"
echo "Available runtimes: SentenceTransformers, ONNX (fp32 + INT8)"
echo "Select the ONNX file to benchmark via MODEL_FILE in .env"
