import gc
import importlib
import os
import time

import numpy as np
import pytest
from tokenizers import Tokenizer

from conftest import MODEL_PATH, missing_model_files


def _get_onnxruntime():
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime is not installed")

    return importlib.import_module("onnxruntime")


def _make_session():
    ort = _get_onnxruntime()
    session_options = ort.SessionOptions()

    intra_op_threads = int(os.getenv("ORT_INTRA_OP_THREADS", "0"))
    inter_op_threads = int(os.getenv("ORT_INTER_OP_THREADS", "0"))
    if intra_op_threads:
        session_options.intra_op_num_threads = intra_op_threads
    if inter_op_threads:
        session_options.inter_op_num_threads = inter_op_threads

    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = os.getenv("ORT_PROVIDERS", "CPUExecutionProvider").split(",")
    return ort.InferenceSession(
        f"{MODEL_PATH}/model.onnx",
        sess_options=session_options,
        providers=providers,
    )


def _make_tokenizer():
    return Tokenizer.from_file(f"{MODEL_PATH}/tokenizer.json")


def _tokenize(tokenizer, texts):
    encoded = tokenizer.encode_batch(texts, add_special_tokens=True)

    input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
    attention_mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
    token_type_ids = np.asarray([item.type_ids for item in encoded], dtype=np.int64)

    if token_type_ids.size == 0:
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def _print_stats(title, batch_size, measured_iterations, times_ms):
    mean_ms = times_ms.mean()
    throughput = batch_size / (mean_ms / 1000.0)

    print()
    print(title)
    print(f"batch_size: {batch_size}")
    print(f"total time: {times_ms.sum():.3f} ms")
    print(f"iterations: {measured_iterations}")
    print(f"mean latency: {mean_ms:.3f} ms")
    print(f"latency per text: {mean_ms / batch_size:.3f} ms")
    print(f"median latency: {np.median(times_ms):.3f} ms")
    print(f"p95 latency: {np.percentile(times_ms, 95):.3f} ms")
    print(f"throughput: {throughput:.1f} texts/sec")


@pytest.fixture(scope="session")
def onnxruntime_session():
    missing_files = missing_model_files()
    assert not missing_files, (
        "Local model is incomplete. Run core/scripts/download_model.sh. "
        f"Missing files: {missing_files}"
    )
    return _make_session()


@pytest.fixture(scope="session")
def hf_tokenizer():
    return _make_tokenizer()


@pytest.mark.performance
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("measured_iterations", [10])
def test_onnxruntime_batch_embedding(
    onnxruntime_session,
    hf_tokenizer,
    en_texts,
    batch_size,
    measured_iterations,
):
    batch = [en_texts[i % len(en_texts)] for i in range(batch_size)]
    warmup_iterations = 10

    for _ in range(warmup_iterations):
        inputs = _tokenize(hf_tokenizer, batch)
        onnxruntime_session.run(["sentence_embedding"], inputs)

    gc.disable()
    try:
        times_ns = []
        for _ in range(measured_iterations):
            start_ns = time.perf_counter_ns()
            inputs = _tokenize(hf_tokenizer, batch)
            embedding = onnxruntime_session.run(["sentence_embedding"], inputs)[0]
            end_ns = time.perf_counter_ns()

            assert embedding.shape[0] == batch_size
            assert embedding.shape[1] == 384
            assert np.isfinite(embedding).all()
            times_ns.append(end_ns - start_ns)
    finally:
        gc.enable()

    times_ms = np.asarray(times_ns, dtype=np.float64) / 1_000_000.0
    _print_stats(
        "onnxruntime batch embedding performance",
        batch_size,
        measured_iterations,
        times_ms,
    )


@pytest.mark.performance
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("measured_iterations", [10])
def test_onnxruntime_batch_inference_only(
    onnxruntime_session,
    hf_tokenizer,
    en_texts,
    batch_size,
    measured_iterations,
):
    batch = [en_texts[i % len(en_texts)] for i in range(batch_size)]
    inputs = _tokenize(hf_tokenizer, batch)
    warmup_iterations = 10

    for _ in range(warmup_iterations):
        onnxruntime_session.run(["sentence_embedding"], inputs)

    gc.disable()
    try:
        times_ns = []
        for _ in range(measured_iterations):
            start_ns = time.perf_counter_ns()
            embedding = onnxruntime_session.run(["sentence_embedding"], inputs)[0]
            end_ns = time.perf_counter_ns()

            assert embedding.shape[0] == batch_size
            assert embedding.shape[1] == 384
            assert np.isfinite(embedding).all()
            times_ns.append(end_ns - start_ns)
    finally:
        gc.enable()

    times_ms = np.asarray(times_ns, dtype=np.float64) / 1_000_000.0
    _print_stats(
        "onnxruntime batch inference-only performance",
        batch_size,
        measured_iterations,
        times_ms,
    )
