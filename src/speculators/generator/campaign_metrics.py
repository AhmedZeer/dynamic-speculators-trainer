"""Lightweight host/IO and optional NVML GPU utilization sampling."""

from threading import Event, Lock, Thread

import psutil
import torch


class ProcessMonitor:
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.stop = Event()
        self.lock = Lock()
        self.peak_rss = 0
        self.gpu_samples = []
        self.nvml_available = device.type == "cuda"
        self.initial_io = self.process.io_counters()
        self.thread = Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            rss = self.process.memory_info().rss
            gpu = None
            if self.nvml_available:
                try:
                    gpu = torch.cuda.utilization(self.device)
                except (ModuleNotFoundError, RuntimeError, AttributeError):
                    self.nvml_available = False
            with self.lock:
                self.peak_rss = max(self.peak_rss, rss)
                if gpu is not None:
                    self.gpu_samples.append(gpu)
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def snapshot(self):
        io = self.process.io_counters()
        with self.lock:
            result = {
                "host_peak_rss_bytes": self.peak_rss,
                "disk_read_bytes": io.read_bytes - self.initial_io.read_bytes,
                "disk_write_bytes": io.write_bytes - self.initial_io.write_bytes,
            }
            if self.gpu_samples:
                result["gpu_utilization_percent"] = sum(self.gpu_samples) / len(
                    self.gpu_samples
                )
        return result

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)
