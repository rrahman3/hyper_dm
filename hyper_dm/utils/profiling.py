"""Hardware & timing profiling utilities.

Provides :class:`Profiler` – a lightweight, always-on helper that captures:

* **Per-epoch wall-clock time** and cumulative training time
* **GPU memory** (allocated / reserved / peak) after every epoch
* **GPU utilisation %** (via ``pynvml`` when available, else ``nvidia-smi``)
* **Per-batch iteration time** statistics (mean / p95 / max)
* **Inference latency** per sample and throughput

All numbers are returned as plain dicts so they can be forwarded straight
to :class:`RunTracker.log_metrics`.
"""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import torch


# ─── GPU utilisation via pynvml (optional) ──────────────────────

def _gpu_utilisation_pynvml(device_idx: int = 0) -> dict[str, float]:
    """Query live GPU utilization & temperature via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return {
            "gpu/utilisation_pct": float(util.gpu),
            "gpu/mem_used_GB": mem_info.used / 1e9,
            "gpu/mem_total_GB": mem_info.total / 1e9,
            "gpu/temperature_C": float(temp),
        }
    except Exception:
        return {}


def _gpu_utilisation_smi() -> dict[str, float]:
    """Fallback: parse ``nvidia-smi`` output."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split(",")
        return {
            "gpu/utilisation_pct": float(out[0]),
            "gpu/mem_used_GB": float(out[1]) / 1024,
            "gpu/mem_total_GB": float(out[2]) / 1024,
            "gpu/temperature_C": float(out[3]),
        }
    except Exception:
        return {}


def gpu_snapshot(device_idx: int = 0) -> dict[str, float]:
    """Return a dict of current GPU metrics (works on CUDA only)."""
    metrics: dict[str, float] = {}

    if not torch.cuda.is_available():
        return metrics

    # PyTorch-level memory stats (always available)
    metrics["gpu/mem_allocated_GB"] = torch.cuda.memory_allocated(device_idx) / 1e9
    metrics["gpu/mem_reserved_GB"] = torch.cuda.memory_reserved(device_idx) / 1e9
    metrics["gpu/mem_peak_GB"] = torch.cuda.max_memory_allocated(device_idx) / 1e9

    # Live utilisation (best-effort)
    live = _gpu_utilisation_pynvml(device_idx)
    if not live:
        live = _gpu_utilisation_smi()
    metrics.update(live)

    return metrics


# ─── Timer helpers ──────────────────────────────────────────────

@contextmanager
def cuda_timer() -> Generator[dict[str, float], None, None]:
    """Context manager that times a CUDA block with proper synchronisation.

    Yields a dict that will contain ``"elapsed_s"`` after the block exits::

        with cuda_timer() as t:
            model(x)
        print(t["elapsed_s"])
    """
    result: dict[str, float] = {}
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        yield result
        end_event.record()
        torch.cuda.synchronize()
        result["elapsed_s"] = start_event.elapsed_time(end_event) / 1000.0
    else:
        t0 = time.perf_counter()
        yield result
        result["elapsed_s"] = time.perf_counter() - t0


# ─── Profiler ───────────────────────────────────────────────────

@dataclass
class Profiler:
    """Accumulates timing and GPU stats across an epoch / run.

    Typical usage in a training loop::

        prof = Profiler()
        for ep in range(epochs):
            prof.epoch_start()
            for batch in loader:
                prof.batch_start()
                ...
                prof.batch_end()
            metrics = prof.epoch_end()
            tracker.log_metrics(metrics, step=ep)

        summary = prof.run_summary()
        tracker.log_metrics(summary)
    """

    device_idx: int = 0

    # ── internal state
    _epoch_t0: float = field(default=0.0, init=False, repr=False)
    _batch_t0: float = field(default=0.0, init=False, repr=False)
    _batch_times: list[float] = field(default_factory=list, init=False, repr=False)
    _epoch_times: list[float] = field(default_factory=list, init=False, repr=False)
    _run_t0: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._run_t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_idx)

    # ── epoch-level ──────────────────────────────────────────────

    def epoch_start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.device_idx)
        self._epoch_t0 = time.perf_counter()
        self._batch_times = []

    def epoch_end(self) -> dict[str, float]:
        """Return a dict of all epoch-level metrics."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._epoch_t0
        self._epoch_times.append(elapsed)

        m: dict[str, float] = {
            "time/epoch_s": elapsed,
            "time/total_s": sum(self._epoch_times),
        }

        if self._batch_times:
            bt = sorted(self._batch_times)
            m["time/batch_mean_ms"] = sum(bt) / len(bt) * 1000
            m["time/batch_p95_ms"] = bt[int(len(bt) * 0.95)] * 1000
            m["time/batch_max_ms"] = bt[-1] * 1000
            m["throughput/batches_per_s"] = len(bt) / max(elapsed, 1e-9)

        m.update(gpu_snapshot(self.device_idx))
        return m

    # ── batch-level ──────────────────────────────────────────────

    def batch_start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_t0 = time.perf_counter()

    def batch_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_times.append(time.perf_counter() - self._batch_t0)

    # ── run-level ────────────────────────────────────────────────

    def run_summary(self) -> dict[str, float]:
        """Return total-run metrics (call once after all epochs)."""
        total = time.perf_counter() - self._run_t0
        m: dict[str, float] = {
            "time/total_run_s": total,
            "time/total_run_min": total / 60,
            "time/num_epochs": float(len(self._epoch_times)),
        }
        if self._epoch_times:
            m["time/epoch_mean_s"] = sum(self._epoch_times) / len(self._epoch_times)
        m.update(gpu_snapshot(self.device_idx))
        return m


# ─── Inference profiler ─────────────────────────────────────────

@dataclass
class InferenceProfiler:
    """Captures per-sample latency and throughput during inference.

    Usage::

        iprof = InferenceProfiler()
        for sample in dataset:
            iprof.sample_start()
            result = model(sample)
            iprof.sample_end()
        print(iprof.summary())
    """

    device_idx: int = 0

    _t0: float = field(default=0.0, init=False, repr=False)
    _run_t0: float = field(default=0.0, init=False, repr=False)
    _latencies: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._run_t0 = time.perf_counter()

    def sample_start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def sample_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._latencies.append(time.perf_counter() - self._t0)

    def summary(self) -> dict[str, float]:
        total = time.perf_counter() - self._run_t0
        m: dict[str, float] = {
            "infer/total_s": total,
            "infer/num_samples": float(len(self._latencies)),
        }
        if self._latencies:
            lats = sorted(self._latencies)
            m["infer/latency_mean_ms"] = sum(lats) / len(lats) * 1000
            m["infer/latency_median_ms"] = lats[len(lats) // 2] * 1000
            m["infer/latency_p95_ms"] = lats[int(len(lats) * 0.95)] * 1000
            m["infer/latency_max_ms"] = lats[-1] * 1000
            m["infer/throughput_samples_per_s"] = len(lats) / max(total, 1e-9)
        m.update(gpu_snapshot(self.device_idx))
        return m
