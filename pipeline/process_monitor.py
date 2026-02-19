import os

class ProcessMonitor:
    """
    Monitor processes,
    clean up excess ones.
    """
    
    def __init__(self, max_processes=10):
        self.max_processes = max_processes
        self.child_pids = []
        
    def check_process_count(self):
        current_count = len(self.child_pids)
        if current_count > self.max_processes:
            print(f"Warning: Too many subprocesses:\n Current {current_count} > max {self.max_processes}!")
            self.cleanup_excess_processes()
    
    def cleanup_excess_processes(self):
        excess = len(self.child_pids) - self.max_processes
        if excess > 0:
            for pid in self.child_pids[-excess:]:
                try:
                    os.kill(pid, 9)
                    print(f"Cleaned excesses process: {pid}")
                except:
                    pass
            self.child_pids = self.child_pids[:-excess]
    
    def register_child(self, pid):
        """register a subprocess."""
        self.child_pids.append(pid)
        self.check_process_count()
    
    def cleanup_all(self):
        """clean up a processes."""
        for pid in self.child_pids:
            try:
                os.kill(pid, 9)
            except:
                pass
        self.child_pids = []