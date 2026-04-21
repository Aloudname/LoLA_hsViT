#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced system monitor with:
- Configurable refresh rate
- CLI flag to enable/disable logging (state_true style)
- Smooth CPU usage (EMA)
- Aggregated memory & multi-GPU VRAM
- Colored progress bars (green/yellow/red thresholds)
- Clean, structured terminal UI (auto clear)

Usage:
    python monitor.py --interval 1.0 --log
    python monitor.py --interval 0.5        # no log

Dependencies:
    pip install psutil pynvml
"""

import os, time, psutil, argparse, platform
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced system monitor with:
- Configurable refresh rate
- CLI flag to enable/disable logging (state_true style)
- Smooth CPU usage (EMA)
- Aggregated memory & multi-GPU VRAM
- Colored progress bars (green/yellow/red thresholds)
- Clean, structured terminal UI (auto clear)

Usage:
    python monitor.py --interval 1.0 --log
    python monitor.py --interval 0.5        # no log

Dependencies:
    pip install psutil pynvml
"""

import os, time, psutil, argparse, platform
from datetime import datetime

# Optional NVML (multi-GPU)
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


def tprint(*args, **kwargs) -> None:
    """print logs with [hh:mm:ss] timestamp."""
    print(datetime.now().strftime("[%H:%M:%S]"), *args, **kwargs)


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def colorize(pct: float, text: str) -> str:
    """Color by threshold: <75 green, 75-90 yellow, >90 red"""
    if pct < 75:
        return f"\033[92m{text}\033[0m"  # green
    elif pct < 90:
        return f"\033[93m{text}\033[0m"  # yellow
    else:
        return f"\033[91m{text}\033[0m"  # red


def progress_bar(pct: float, width: int = 40) -> str:
    filled = int(width * pct / 100)
    bar = '█' * filled + ' ' * (width - filled)
    return colorize(pct, f"[{bar}] {pct:6.2f}%")


class Monitor:
    """Exponential moving average for smoothing"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


def get_cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except:
        return platform.processor() or "Unknown CPU"


def get_gpu_names_and_count():
    if not NVML_AVAILABLE:
        return "No GPU", 0

    names = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        names.append(name.decode() if isinstance(name, bytes) else name)

    if not names:
        return "No GPU", 0

    if len(set(names)) == 1:
        return names[0], len(names)
    else:
        return " / ".join(names), len(names)


def get_cpu_info(ema: Monitor):
    per_core = psutil.cpu_percent(percpu=True)
    avg = sum(per_core) / len(per_core)
    smooth = ema.update(avg)
    return len(per_core), avg, smooth


def get_mem_info():
    vm = psutil.virtual_memory()
    return vm.total, vm.used, vm.percent


def get_gpu_info():
    gpus = []
    if not NVML_AVAILABLE:
        return gpus

    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        gpus.append({
            'total': mem.total,
            'used': mem.used,
            'percent': mem.used / mem.total * 100,
            'util': util.gpu,
            'temp': temp
        })
    return gpus


def render(mem, cpu, gpus, cpu_name, gpu_name, gpu_count):
    total_mem, used_mem, mem_pct = mem
    cores, cpu_avg, cpu_smooth = cpu

    # Aggregate GPU
    if gpus:
        total_vram = sum(g['total'] for g in gpus)
        used_vram = sum(g['used'] for g in gpus)
        vram_pct = used_vram / total_vram * 100
        avg_temp = sum(g['temp'] for g in gpus) / len(gpus)
        avg_util = sum(g['util'] for g in gpus) / len(gpus)
    else:
        total_vram = used_vram = vram_pct = avg_temp = avg_util = 0

    # Header
    print(f"\033[93m{'System Monitor  |  '}\033[0m", f"\033[93m{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")

    # Memory
    print(f"\n[Memory & CPU: \033[95m{cpu_name}\033[0m")
    print(progress_bar(mem_pct))
    print(f"Total: \033[97m{total_mem/1e9:.2f}\033[0m GB \t Used: \033[97m{used_mem/1e9:.2f}\033[0m GB")
    print(f"CPU Cores: \033[97m{cores}\033[0m \t\t Avg: \033[97m{cpu_smooth:.2f}\033[0m%")

    # GPU
    print(f"\n[GPU: \033[95m{gpu_name} x {str(gpu_count)}\033[0m")
    print(progress_bar(vram_pct))
    print(f"Total VRAM: \033[97m{total_vram/1e9:.2f}\033[0m GB \t Used: \033[97m{used_vram/1e9:.2f}\033[0m GB")
    print(f"GPU Temp: \033[97m{avg_temp:.1f}\033[0m°C \t Occupated: \033[97m{avg_util:.2f}\033[0m%")
    print("\n")


def monitor():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', '-i', type=float, default=0.2, help='refresh interval (seconds)')
    parser.add_argument('--log', '-l', action='store_true', help='enable logging')
    args = parser.parse_args()

    ema = Monitor(alpha=0.2)
    cpu_name = get_cpu_name()
    gpu_name, gpu_count = get_gpu_names_and_count()

    log_file = None
    if args.log:
        log_file = open('monitor.log', 'a')

    try:
        while True:
            mem = get_mem_info()
            cpu = get_cpu_info(ema)
            gpus = get_gpu_info()

            clear()
            render(mem, cpu, gpus, cpu_name, gpu_name, gpu_count)

            if log_file:
                log_file.write(f"{datetime.now()} | MEM {mem[2]:.2f}% | CPU {cpu[2]:.2f}%\n")
                log_file.flush()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if log_file:
            log_file.close()


if __name__ == '__main__':
    monitor()
