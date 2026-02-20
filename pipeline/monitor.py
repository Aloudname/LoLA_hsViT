#!/usr/bin/env python3
"""
Monitor GPU and memory.
"""

import os
import time
import psutil
import argparse
import threading
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    timestamp: str
    sys_used_gb: float
    sys_available_gb: float
    sys_percent: float
    sys_total_gb: float
    process_mb: float
    process_percent: float
    gpu_allocated_gb: Optional[float] = None
    gpu_reserved_gb: Optional[float] = None
    gpu_total_gb: Optional[float] = None
    gpu_percent: Optional[float] = None
    gpu_temp: Optional[float] = None
    gpu_util: Optional[float] = None
    peak_sys_gb: float = 0.0
    peak_gpu_gb: float = 0.0


class Monitor:
    """
    Monitor GPU and memory usage.
    """
    
    def __init__(self, 
                 log_file: str = "outputs/logs/monitor.log",
                 interval: float = 2.0,
                 gpu_ids: Optional[List[int]] = None,
                 show_gpu_process: bool = True,
                 threshold_warning: float = 80.0,
                 threshold_critical: float = 90.0,
                 enable_gpu: bool = True):
        
        self.log_file = log_file
        self.interval = interval
        self.monitoring = False
        self.memory_snapshots: List[MemorySnapshot] = []
        self.peak_sys_memory = 0.0
        self.peak_gpu_memory = 0.0
        self.enable_gpu = enable_gpu and self._check_gpu_available()
        
        self.gpu_ids = gpu_ids
        self.show_gpu_process = show_gpu_process
        self.threshold_warning = threshold_warning
        self.threshold_critical = threshold_critical
        self.gpu_handles = {}
        self.gpu_device_count = 0
        
        if self.enable_gpu:
            self._init_gpu_monitor()
    
    def _check_gpu_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return device_count > 0
            except:
                return False
    
    def _init_gpu_monitor(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_device_count = pynvml.nvmlDeviceGetCount()
            
            if self.gpu_ids is None:
                self.gpu_ids = list(range(self.gpu_device_count))
            else:
                self.gpu_ids = [i for i in self.gpu_ids if i < self.gpu_device_count]
            
            if not self.gpu_ids:
                print("No valid GPU, disable GPU monitoring.")
                self.enable_gpu = False
                return
            
            for gpu_id in self.gpu_ids:
                self.gpu_handles[gpu_id] = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            print(f" GPU monitor initialized successfully with {len(self.gpu_ids)} GPU monitored.")
            
        except Exception as e:
            print(f"Error during GPU monitor initialization: {e}")
            self.enable_gpu = False
    
    def get_system_memory(self) -> Dict[str, float]:
        mem = psutil.virtual_memory()
        return {
            'used': mem.used / (1024**3),
            'available': mem.available / (1024**3),
            'total': mem.total / (1024**3),
            'percent': mem.percent
        }
    
    def get_gpu_memory_pynvml(self) -> List[Dict]:
        if not self.enable_gpu:
            return []
        
        try:
            import pynvml
            gpu_info = []
            
            for gpu_id in self.gpu_ids:
                handle = self.gpu_handles[gpu_id]
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                processes = []
                if self.show_gpu_process:
                    try:
                        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        for proc in procs:
                            try:
                                cmdline = subprocess.check_output(
                                    ['ps', '-p', str(proc.pid), '-o', 'cmd='],
                                    universal_newlines=True
                                ).strip()
                                cmdline = cmdline[:50] + '...' if len(cmdline) > 50 else cmdline
                            except:
                                cmdline = 'Unknown'
                            
                            processes.append({
                                'pid': proc.pid,
                                'used_memory': proc.usedGpuMemory / (1024**3),
                                'cmd': cmdline
                            })
                    except:
                        pass
                
                used_gb = memory_info.used / (1024**3)
                total_gb = memory_info.total / (1024**3)
                percent = (used_gb / total_gb) * 100 if total_gb > 0 else 0
                
                gpu_info.append({
                    'gpu_id': gpu_id,
                    'used': used_gb,
                    'total': total_gb,
                    'percent': percent,
                    'util': utilization.gpu,
                    'mem_util': utilization.memory,
                    'temp': temperature,
                    'processes': processes
                })
            
            return gpu_info
            
        except Exception as e:
            print(f"Fail to get GPU info: {e}")
            return []
    
    def get_gpu_memory_torch(self) -> List[Dict]:
        try:
            import torch
            gpu_info = []
            
            for gpu_id in self.gpu_ids:
                if gpu_id < torch.cuda.device_count():
                    allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                    total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                    
                    gpu_info.append({
                        'gpu_id': gpu_id,
                        'allocated': allocated,
                        'reserved': reserved,
                        'total': total,
                        'percent': (allocated / total) * 100 if total > 0 else 0
                    })
            
            return gpu_info
            
        except:
            return []
    
    def get_gpu_memory(self) -> List[Dict]:
        try:
            return self.get_gpu_memory_pynvml()
        except:
            return self.get_gpu_memory_torch()
    
    def get_process_memory(self, pid: Optional[int] = None) -> Tuple[float, float]:
        if pid is None:
            pid = os.getpid()
        try:
            process = psutil.Process(pid)
            mem_info = process.memory_info()
            mem_percent = process.memory_percent()
            return mem_info.rss / (1024**3), mem_percent
        except:
            return 0.0, 0.0
    
    def take_snapshot(self) -> MemorySnapshot:
        sys_mem = self.get_system_memory()
        proc_mem_gb, proc_percent = self.get_process_memory()
        gpu_info = self.get_gpu_memory() if self.enable_gpu else []
        
        self.peak_sys_memory = max(self.peak_sys_memory, sys_mem['used'])
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sys_used_gb=sys_mem['used'],
            sys_available_gb=sys_mem['available'],
            sys_percent=sys_mem['percent'],
            sys_total_gb=sys_mem['total'],
            process_mb=proc_mem_gb * 1024,
            process_percent=proc_percent,
            peak_sys_gb=self.peak_sys_memory
        )
        
        if gpu_info:
            total_used = sum(g['used'] for g in gpu_info)
            total_total = sum(g['total'] for g in gpu_info)
            avg_percent = (total_used / total_total) * 100 if total_total > 0 else 0
            
            snapshot.gpu_allocated_gb = total_used
            snapshot.gpu_total_gb = total_total
            snapshot.gpu_percent = avg_percent
            
            self.peak_gpu_memory = max(self.peak_gpu_memory, total_used)
            snapshot.peak_gpu_gb = self.peak_gpu_memory
            
            if gpu_info[0].get('temp'):
                snapshot.gpu_temp = gpu_info[0]['temp']
            if gpu_info[0].get('util'):
                snapshot.gpu_util = gpu_info[0]['util']
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def get_color(self, percent: float) -> str:
        if percent >= self.threshold_critical:
            return '\033[91m'
        elif percent >= self.threshold_warning:
            return '\033[93m'
        else:
            return '\033[92m'
    
    def get_progress_bar(self, percent: float, width: int = 30) -> str:
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
        color = self.get_color(percent)
        return f"{color}{bar}\033[0m"
    
    def display_snapshot(self, snapshot: MemorySnapshot):
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # header
        print('\033[1;36m' + '=' * 20 + '\033[0m')
        print(f'\033[1;33m Resource monitoring - {snapshot.timestamp} | interval: {self.interval}s\033[0m')
        print('\033[1;36m' + '=' * 20 + '\033[0m')
        
        sys_bar = self.get_progress_bar(snapshot.sys_percent)
        sys_color = self.get_color(snapshot.sys_percent)
        print(f'\n\033[1;34mSystem memory (RAM):\033[0m')
        print(f'  [{sys_bar}] {sys_color}{snapshot.sys_percent:.1f}%\033[0m')
        print(f'  ├─ Used: {sys_color}{snapshot.sys_used_gb:.1f}GB\033[0m / {snapshot.sys_total_gb:.1f}GB')
        print(f'  ├─ Available: {snapshot.sys_available_gb:.1f}GB')
        print(f'  └─ Process memory: {snapshot.process_mb:.0f}MB ({snapshot.process_percent:.1f}%)')
        
        if snapshot.gpu_total_gb:
            gpu_bar = self.get_progress_bar(snapshot.gpu_percent)
            gpu_color = self.get_color(snapshot.gpu_percent)
            print(f'\n\033[1;35m GPU memory:\033[0m')
            print(f'  [{gpu_bar}] {gpu_color}{snapshot.gpu_percent:.1f}%\033[0m')
            print(f'  ├─ Allocated: {gpu_color}{snapshot.gpu_allocated_gb:.1f}GB\033[0m / {snapshot.gpu_total_gb:.1f}GB')
            
            if snapshot.gpu_temp is not None:
                temp_color = self.get_temp_color(snapshot.gpu_temp)
                print(f'  ├─ GPU temperature: {temp_color}{snapshot.gpu_temp:.0f}°C\033[0m')
            
            if snapshot.gpu_util is not None:
                print(f'  └─ Occupation rate: {snapshot.gpu_util:.0f}%')
        else:
            print(f'\n\033[1;35m GPU memory undefined\033[0m')
        
        print(f'\n\033[1;33m Peaks:\033[0m')
        peak_sys_color = self.get_color((snapshot.peak_sys_gb / snapshot.sys_total_gb) * 100)
        print(f'  ├─ System memory peak: {peak_sys_color}{snapshot.peak_sys_gb:.1f}GB\033[0m')
        
        if snapshot.peak_gpu_gb > 0:
            peak_gpu_color = self.get_color((snapshot.peak_gpu_gb / snapshot.gpu_total_gb) * 100)
            print(f'  └─ GPU memory peak: {peak_gpu_color}{snapshot.peak_gpu_gb:.1f}GB\033[0m')
        
        print(f'\n\033[1;30m Got {len(self.memory_snapshots)} samples | Press Ctrl+C to stop\033[0m')
    
    def get_temp_color(self, temp: float) -> str:
        if temp >= 85:
            return '\033[91m'  # red
        elif temp >= 75:
            return '\033[93m'  # yellow
        else:
            return '\033[92m'  # green
    
    def start_monitoring(self):
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                snapshot = self.take_snapshot()
                self.display_snapshot(snapshot)
                time.sleep(self.interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print(f"monitor starts with interval {self.interval}s.")
    
    def stop_monitoring(self):
        self.monitoring = False
        print("\nmonitoring stopped.")
        self.save_log()
    
    def save_log(self):
        """
        save as log.
        """
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write(f"\n")
            f.write(f"Start at: {self.memory_snapshots[0].timestamp if self.memory_snapshots else 'N/A'}\n")
            f.write(f"Terminate at: {self.memory_snapshots[-1].timestamp if self.memory_snapshots else 'N/A'}\n")
            f.write(f"Sample interval: {self.interval}s\n")
            f.write(f"Total samples: {len(self.memory_snapshots)}\n")
                
            # table head
            f.write("Timestamp          | RAM_Used(GB) | RAM_Avail(GB) | RAM_%  | Proc_MB | GPU_Used(GB) | GPU_%  | GPU_Temp\n")
            f.write("-" * 100 + "\n")
            
            for s in self.memory_snapshots:
                gpu_str = f"{s.gpu_allocated_gb:.1f}/{s.gpu_total_gb:.1f}" if s.gpu_allocated_gb else "N/A"
                gpu_percent = f"{s.gpu_percent:.1f}%" if s.gpu_percent else "N/A"
                gpu_temp = f"{s.gpu_temp:.0f}°C" if s.gpu_temp else "N/A"
                
                f.write(f"{s.timestamp} | {s.sys_used_gb:>10.1f} | {s.sys_available_gb:>11.1f} | "
                        f"{s.sys_percent:>5.1f} | {s.process_mb:>7.0f} | {gpu_str:>13} | "
                        f"{gpu_percent:>6} | {gpu_temp:>7}\n")
        
        print(f"Log saved at {self.log_file}")


def monitor():
    parser = argparse.ArgumentParser(description='GPU and memory monitor')
    parser.add_argument('--interval', '-i', type=float, default=2.0, help='monitor interval (seconds)')
    parser.add_argument('--log', '-l', type=str, default='outputs/logs/monitor.log', help='log file path')
    parser.add_argument('--gpus', type=str, default='all', help='gpu ids to monitor, separated by commas')
    parser.add_argument('--no-gpu', action='store_true', help='disable gpu monitoring')
    parser.add_argument('--no-process', action='store_true', help='disable gpu process monitoring')
    parser.add_argument('--warning', type=float, default=80.0, help='warning threshold (percentage)')
    parser.add_argument('--critical', type=float, default=90.0, help='critical threshold (percentage)')
    
    args = parser.parse_args()
    
    if args.gpus.lower() == 'all':
        gpu_ids = None
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    monitor = Monitor(
        log_file=args.log,
        interval=args.interval,
        gpu_ids=gpu_ids,
        show_gpu_process=not args.no_process,
        threshold_warning=args.warning,
        threshold_critical=args.critical,
        enable_gpu=not args.no_gpu
    )
    
    try:
        monitor.start_monitoring()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        
if __name__ == "__main__":
    monitor()