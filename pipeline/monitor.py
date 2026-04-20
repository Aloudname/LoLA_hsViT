from __future__ import annotations

# lightweight runtime monitor utilities.
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from typing import Iterator, Optional
from concurrent.futures import ProcessPoolExecutor

import psutil


def tprint(*args, **kwargs) -> None:
    """print logs with [hh:mm:ss] timestamp."""
    print(datetime.now().strftime("[%H:%M:%S]"), *args, **kwargs)


@contextmanager
def _managed_pool(max_workers: int, desc: str = "pool") -> Iterator[ProcessPoolExecutor]:
    """safe process pool context used by plotting/export tasks."""
    pool = ProcessPoolExecutor(max_workers=max_workers)
    try:
        yield pool
    except Exception:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


@dataclass
class MonitorSnapshot:
    timestamp: str
    cpu_percent: float
    mem_percent: float
    mem_used_gb: float
    mem_total_gb: float


class Monitor:
    """simple cpu/memory monitor for training loops."""

    def snapshot(self) -> MonitorSnapshot:
        mem = psutil.virtual_memory()
        return MonitorSnapshot(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cpu_percent=psutil.cpu_percent(interval=None),
            mem_percent=float(mem.percent),
            mem_used_gb=float(mem.used / (1024 ** 3)),
            mem_total_gb=float(mem.total / (1024 ** 3)),
        )

    def log(self, prefix: str = "monitor") -> None:
        s = self.snapshot()
        tprint(
            f"{prefix}: cpu={s.cpu_percent:.1f}% "
            f"mem={s.mem_percent:.1f}% ({s.mem_used_gb:.2f}/{s.mem_total_gb:.2f} gb)"
        )


def monitor(interval_seconds: float = 1.0, max_steps: Optional[int] = None) -> None:
    """run interactive monitor loop.

    input:
        interval_seconds(float): print interval.
        max_steps(optional int): stop after steps when provided.
    """
    import time

    m = Monitor()
    step = 0
    while True:
        m.log(prefix="resource")
        time.sleep(interval_seconds)
        step += 1
        if max_steps is not None and step >= max_steps:
            break
