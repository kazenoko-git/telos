"""Run the local Telos/LM Studio SSE benchmark with macOS memory guards."""

import argparse
import json
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

LMS = Path.home() / ".lmstudio/bin/lms"
SCRIPT = Path(__file__).with_name("benchmark_serve_vs_lmstudio.py")
MIN_MEMORY_FREE_PERCENT = 45
MIN_SWAP_FREE_MIB = 512
MAX_SWAP_GROWTH_MIB = 192
MAX_TELOS_RSS_MIB = 4096
MLX_ALLOCATOR_MIB = 3072
MLX_CACHE_MIB = 512
POLL_SECONDS = 0.25


def read_command(*args):
    return subprocess.check_output(args, text=True, timeout=2)


def memory_sample():
    pressure = read_command("memory_pressure", "-Q")
    swap = read_command("sysctl", "-n", "vm.swapusage")
    free_percent = re.search(r"System-wide memory free percentage: (\d+)%", pressure)
    swap_mib = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M\s+free = ([\d.]+)M", swap)
    if free_percent is None or swap_mib is None:
        raise RuntimeError("Could not read macOS memory and swap telemetry")
    total, used, free = map(float, swap_mib.groups())
    return int(free_percent[1]), used, free if total else None


class MemoryGuard:
    def __init__(self):
        self.free_percent, self.swap_start, self.swap_free = memory_sample()
        self.min_free = self.free_percent
        self.max_swap_used = self.swap_start
        self.max_telos_rss = 0.0
        self.telos = None

    def check(self):
        free, used, swap_free = memory_sample()
        self.min_free = min(self.min_free, free)
        self.max_swap_used = max(self.max_swap_used, used)
        if free < MIN_MEMORY_FREE_PERCENT:
            raise RuntimeError(f"Memory guard stopped at {free}% free")
        if used - self.swap_start >= MAX_SWAP_GROWTH_MIB:
            raise RuntimeError(f"Memory guard stopped after {used - self.swap_start:.1f} MiB swap growth")
        if swap_free is not None and swap_free < MIN_SWAP_FREE_MIB:
            raise RuntimeError(f"Memory guard stopped at {swap_free:.1f} MiB free swap")
        if self.telos is not None and self.telos.poll() is None:
            proc = subprocess.run(["ps", "-o", "rss=", "-p", str(self.telos.pid)],
                                  capture_output=True, text=True, timeout=2, check=False)
            if self.telos.poll() is None:
                proc.check_returncode()
                rss = int(proc.stdout.strip()) / 1024
                self.max_telos_rss = max(self.max_telos_rss, rss)
                if rss >= MAX_TELOS_RSS_MIB:
                    raise RuntimeError(f"Memory guard stopped at {rss:.1f} MiB Telos RSS")
        return free, used

    def run(self, command, *, timeout, stdout=None):
        self.check()
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT if stdout else None)
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                self.check()
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Timed out running {Path(command[0]).name}")
                time.sleep(POLL_SECONDS)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
            self.check()
        finally:
            stop_process(process)

    def summary(self):
        return {
            "minimum_free_percent": MIN_MEMORY_FREE_PERCENT,
            "minimum_free_swap_mib": MIN_SWAP_FREE_MIB,
            "maximum_swap_growth_mib": MAX_SWAP_GROWTH_MIB,
            "maximum_telos_rss_mib": MAX_TELOS_RSS_MIB,
            "telos_mlx_allocator_guideline_mib": MLX_ALLOCATOR_MIB,
            "telos_mlx_free_cache_limit_mib": MLX_CACHE_MIB,
            "start_free_percent": self.free_percent,
            "lowest_free_percent": self.min_free,
            "start_swap_used_mib": self.swap_start,
            "peak_swap_used_mib": self.max_swap_used,
            "peak_telos_rss_mib": self.max_telos_rss,
        }


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_for_telos(url, guard):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        guard.check()
        if guard.telos.poll() is not None:
            raise RuntimeError("Telos exited before its health check passed")
        try:
            with urlopen(url + "/health", timeout=0.5) as response:
                if json.load(response)["status"] == "ok":
                    return
        except (URLError, TimeoutError):
            time.sleep(POLL_SECONDS)
    raise RuntimeError("Telos health check timed out")


def require_idle(telos_port, lmstudio_port):
    if json.loads(read_command(LMS, "server", "status", "--json"))["running"]:
        raise RuntimeError("Close the existing LM Studio server before running this launcher")
    if json.loads(read_command(LMS, "ps", "--json")):
        raise RuntimeError("Unload the existing LM Studio models before running this launcher")
    for port in {telos_port, lmstudio_port}:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already in use")


def run_benchmark(args):
    require_idle(args.telos_port, args.lmstudio_port)
    guard = MemoryGuard()
    guard.check()
    print(f"Memory guard armed at {guard.free_percent}% free and {guard.swap_start:.1f} MiB used swap", flush=True)
    model = "bonsai-4b"
    telos_url = f"http://127.0.0.1:{args.telos_port}"
    lmstudio_url = f"http://127.0.0.1:{args.lmstudio_port}"
    bootstrap = (
        "import runpy; import mlx.core as mx; "
        f"mx.set_memory_limit({MLX_ALLOCATOR_MIB} * 2**20); "
        f"mx.set_cache_limit({MLX_CACHE_MIB} * 2**20); "
        "runpy.run_module('telos', run_name='__main__')"
    )
    with tempfile.TemporaryDirectory(prefix="telos-guarded-benchmark-") as folder:
        path = Path(folder)
        lms_started = False
        try:
            with (path / "telos.log").open("w+") as log:
                guard.telos = subprocess.Popen(
                    [sys.executable, "-c", bootstrap, "serve", "--device", "mlx", "--checkpoint",
                     args.checkpoint, "--port", str(args.telos_port), "--model-name", model,
                     "--context-length", "1024", "--prefix-cache-entries", "4"],
                    stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    wait_for_telos(telos_url, guard)
                except RuntimeError:
                    if guard.telos.poll() is not None:
                        log.seek(0)
                        print(log.read(), file=sys.stderr)
                    raise
                lms_started = True
                guard.run([LMS, "server", "start", "--port", str(args.lmstudio_port)], timeout=15)
                guard.run([LMS, "load", args.lmstudio_key, "--identifier", model,
                           "--context-length", "1024", "--parallel", "1",
                           "--no-speculative-draft-mtp", "--ttl", "180", "--yes"], timeout=30)
                result = path / "benchmark.json"
                guard.run([sys.executable, SCRIPT, "--telos-url", telos_url + "/v1", "--telos-model", model,
                           "--lmstudio-url", lmstudio_url + "/v1", "--lmstudio-model", model,
                           "--max-tokens", "192", "--trials", "4", "--output", result], timeout=100)
                report = json.loads(result.read_text())
        finally:
            stop_process(guard.telos)
            if lms_started:
                for command in ([LMS, "unload", model], [LMS, "server", "stop"]):
                    try:
                        result = subprocess.run(command, timeout=8, check=False)
                        if result.returncode != 0:
                            print(f"LM Studio cleanup exited {result.returncode}: {command[1:]}", file=sys.stderr)
                    except subprocess.TimeoutExpired:
                        print(f"LM Studio cleanup timed out: {command[1:]}", file=sys.stderr)
            print(json.dumps(guard.summary()), flush=True)
    report["memory_guard"] = guard.summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    def request_stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lmstudio-key", required=True)
    parser.add_argument("--telos-port", type=int, default=8001)
    parser.add_argument("--lmstudio-port", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.telos_port == args.lmstudio_port:
        parser.error("Telos and LM Studio need distinct ports")
    try:
        run_benchmark(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Guarded serve benchmark stopped: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Guarded serve benchmark interrupted\n")


if __name__ == "__main__":
    main()
