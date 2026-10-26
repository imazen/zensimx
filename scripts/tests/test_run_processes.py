"""Live Linux controls for trainer identity and threaded process discovery."""
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rev4_featpot"))
from run_processes import bound_worker, process_snapshot, process_tree


class RunProcesses(unittest.TestCase):
    def test_pid_reuse_and_zombie_are_not_live_evidence(self):
        current = process_snapshot(os.getpid())
        self.assertIsNotNone(bound_worker(os.getpid(),current["start_ticks"]))
        self.assertIsNone(bound_worker(os.getpid(),current["start_ticks"]+1))
        child = subprocess.Popen([sys.executable,"-c","pass"])
        try:
            # waitid WNOWAIT observes exit without reaping, leaving a real zombie.
            os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOWAIT)
            zombie = process_snapshot(child.pid)
            self.assertEqual(zombie["state"],"Z")
            self.assertIsNone(bound_worker(child.pid,zombie["start_ticks"]))
        finally:
            child.wait()

    def test_child_spawned_by_worker_thread_is_visible(self):
        ready, release = queue.Queue(), threading.Event()
        def worker():
            child = subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])
            try:
                ready.put(child.pid)
                release.wait(10)
            finally:
                child.terminate()
                child.wait()
        thread = threading.Thread(target=worker)
        thread.start()
        try:
            pid = ready.get(timeout=5)
            self.assertIn(pid,{p["pid"] for p in process_tree(os.getpid())})
        finally:
            release.set()
            thread.join()


if __name__ == "__main__":
    unittest.main()
