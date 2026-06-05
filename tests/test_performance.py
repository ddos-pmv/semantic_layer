import pytest
import gc
import time

import numpy as np


@pytest.mark.performance
@pytest.mark.parametrize("measured_iterations", [1, 10, 100, 1000])
def test_single_embedding(model, en_texts, measured_iterations):
    text = en_texts[0]
    warmup_iterations = 10

    for _ in range(warmup_iterations):
        model.encode(text)

    gc.disable()
    try:
        times_ns = []
        for _ in range(measured_iterations):
            start_ns = time.perf_counter_ns()
            embedding = model.encode(text)
            end_ns = time.perf_counter_ns()

            times_ns.append(end_ns - start_ns)
    finally:
        gc.enable()

    times_ms = np.array(times_ns, dtype=np.float64) / 1_000_000.0

    print("\nsingle embedding latency")
    print(f"warmup_iterations: {warmup_iterations}")
    print(f"measured_iterations: {measured_iterations}")
    print(f"mean:   {times_ms.mean():.3f} ms")
    print(f"median: {np.median(times_ms):.3f} ms")
    print(f"p90:    {np.percentile(times_ms, 90):.3f} ms")
    print(f"p95:    {np.percentile(times_ms, 95):.3f} ms")
    print(f"p99:    {np.percentile(times_ms, 99):.3f} ms")
    print(f"min:    {times_ms.min():.3f} ms")
    print(f"max:    {times_ms.max():.3f} ms")


@pytest.mark.performance
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("measured_iterations", [10])
def test_batch_embedding(model, en_texts,batch_size, measured_iterations):
    text = en_texts[0]
    texts = [en_texts[0]] * batch_size
    warmup_iterations = 10

    for _ in range(warmup_iterations):
        model.encode(text)

    gc.disable()
    try:
        times_ns = []
        for _ in range(measured_iterations):
            start_ns = time.perf_counter_ns()
            embedding = model.encode(texts, batch_size=batch_size)
            end_ns = time.perf_counter_ns()

            times_ns.append(end_ns - start_ns)
    finally:
        gc.enable()

    times_ms = np.array(times_ns, dtype=np.float64) / 1_000_000.0
    mean_ms = times_ms.mean()
    throughput = batch_size / (mean_ms / 1000.0)

    print()
    print("batch embedding performance")
    print(f"batch_size: {batch_size}")
    print(f"iterations: {measured_iterations}")
    print(f"mean latency: {mean_ms:.3f} ms")
    print(f"latency per text: {mean_ms/batch_size :.3f} ms")
    print(f"median latency: {np.median(times_ms):.3f} ms")
    print(f"p95 latency: {np.percentile(times_ms, 95):.3f} ms")
    print(f"throughput: {throughput:.1f} texts/sec")