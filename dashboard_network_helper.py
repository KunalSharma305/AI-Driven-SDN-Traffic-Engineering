#!/usr/bin/python3
"""Dashboard-controlled persistent Mininet topology for AI-SDN evaluator demo.
Commands are JSON lines on stdin. This process must run as root (dashboard uses sudo).
"""
import json, os, sys, time, signal, subprocess
from pathlib import Path
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel

BASE = Path(os.environ.get("SDN_PROJECT_DIR", str(Path(__file__).resolve().parent))).expanduser().resolve()
# Ensure imports resolve from the actual project folder even when launched via sudo.
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from run_experiments import AITopo

LOG = BASE / "dashboard_network.log"
active_traffic = []
net = None
download_proc = None
http_server_proc = None
download_started_at = None
download_last_bytes = 0
download_last_sample_time = None
download_peak_mbps = 0.0
download_total_bytes = 0
DOWNLOAD_STATUS = BASE / "dashboard_download_status.json"

def log(msg):
    # stdout is redirected to dashboard_network.log by dashboard.
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)

def cleanup_traffic():
    global active_traffic
    for p in active_traffic:
        try:
            if p.poll() is None: p.terminate()
        except Exception: pass
    active_traffic = []

def write_download_status(status):
    try:
        tmp = DOWNLOAD_STATUS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status), encoding="utf-8")
        tmp.replace(DOWNLOAD_STATUS)
    except Exception as exc:
        log(f"WARN: could not write download status: {exc}")

def update_download_status(h1):
    global download_last_bytes, download_last_sample_time, download_peak_mbps, download_proc
    now = time.time()
    try:
        raw = h1.cmd("stat -c %s /tmp/sdn_demo_download.bin 2>/dev/null || echo 0").strip().splitlines()[-1]
        downloaded = max(0, int(raw))
    except Exception:
        downloaded = 0
    elapsed = max(0.0, now - (download_started_at or now))
    dt = max(0.001, now - (download_last_sample_time or now))
    speed = max(0.0, (downloaded - download_last_bytes) * 8 / dt / 1_000_000)
    download_peak_mbps = max(download_peak_mbps, speed)
    download_last_bytes, download_last_sample_time = downloaded, now
    running = download_proc is not None and download_proc.poll() is None
    return_code = None if running or download_proc is None else download_proc.poll()
    state = "downloading" if running else ("completed" if download_total_bytes and downloaded >= download_total_bytes else "stopped_or_failed")
    remaining = max(0, download_total_bytes - downloaded)
    eta = (remaining * 8 / (speed * 1_000_000)) if speed > 0 else None
    message = "Transfer complete" if state == "completed" else ("Download running" if running else f"Transfer stopped or failed (curl exit={return_code})")
    if state == "stopped_or_failed":
        try:
            err = h1.cmd("cat /tmp/sdn_demo_curl_error.log 2>/dev/null | tail -n 2").strip().replace("\n", " | ")
            if err:
                message += " · " + err[-240:]
        except Exception:
            pass
    status = {"state": state, "total_bytes": download_total_bytes, "downloaded_bytes": min(downloaded, download_total_bytes),
              "elapsed_seconds": round(elapsed, 2), "current_mbps": round(speed, 3), "peak_mbps": round(download_peak_mbps, 3),
              "eta_seconds": round(eta, 1) if eta is not None else None, "started_epoch": download_started_at,
              "return_code": return_code, "message": message}
    write_download_status(status)
    return status

def stop_download():
    global download_proc, http_server_proc
    for proc in (download_proc, http_server_proc):
        try:
            if proc is not None and proc.poll() is None: proc.terminate()
        except Exception: pass
    download_proc = None
    http_server_proc = None

def main():
    global net, active_traffic, download_proc, http_server_proc, download_started_at, download_last_bytes, download_last_sample_time, download_peak_mbps, download_total_bytes
    setLogLevel("warning")
    if os.geteuid() != 0:
        log("ERROR: Mininet helper must run as root")
        return 2
    if not (BASE / "run_experiments.py").is_file():
        log(f"ERROR: run_experiments.py missing from project directory: {BASE / 'run_experiments.py'}")
        return 2
    try:
        log(f"Project directory: {BASE}")
        log("Starting AITopo and connecting to Ryu at 127.0.0.1:6633...")
        topo = AITopo()
        net = Mininet(topo=topo, controller=None, switch=OVSKernelSwitch, link=TCLink,
                      autoSetMacs=True, autoStaticArp=True)
        net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6633)
        net.start()
        h1, h2 = net.get("h1"), net.get("h2")
        time.sleep(2)
        # Separate classic iperf servers: TCP port 5001, UDP port 5002.
        h2.popen("iperf -s -p 5001", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        h2.popen("iperf -s -u -p 5002", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log("READY: topology h1-s1-{s2,s3}-s4-h2 started; TCP/UDP iperf servers active.")
        log("Connectivity check: " + h1.cmd("ping -c 1 -W 1 10.0.0.2").strip().replace("\n", " | "))
        for raw in sys.stdin:
            try: command = json.loads(raw)
            except Exception:
                log("WARN: ignored malformed dashboard command")
                continue
            action = command.get("action")
            if action == "ping":
                count = max(1, min(100, int(command.get("count", 10))))
                log(f"PING started: {count} packets h1 -> h2")
                output = h1.cmd(f"ping -c {count} -W 1 10.0.0.2")
                log("PING result: " + output.strip().replace("\n", " | "))
            elif action == "iperf":
                protocol = "udp" if command.get("protocol") == "udp" else "tcp"
                rate = max(0.1, min(100.0, float(command.get("rate_mbps", 5))))
                duration = max(5, min(60, int(command.get("duration", 15))))
                port = 5002 if protocol == "udp" else 5001
                cmd = f"iperf -c 10.0.0.2 -p {port} -t {duration}"
                if protocol == "udp": cmd += f" -u -b {rate:g}M"
                log(f"TRAFFIC started: {protocol.upper()} iperf, requested={rate:g} Mbps, duration={duration}s")
                output_path = BASE / f"dashboard_iperf_{protocol}.log"
                out = open(output_path, "a", buffering=1)
                proc = h1.popen(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT)
                active_traffic.append(proc)
                log(f"iperf client PID={proc.pid}; output saved to {output_path.name}")
            elif action == "start_download":
                size_mb = max(10, min(250, int(command.get("size_mb", 100))))
                stop_download()
                # Use a real HTTP transfer between Mininet hosts; no topology or route logic changes.
                curl_ok = h1.cmd("command -v curl >/dev/null 2>&1; echo $?").strip().splitlines()[-1] == "0"
                if not curl_ok:
                    log("ERROR: curl is not installed in the Mininet host environment; download demo not started.")
                    write_download_status({"state":"error", "message":"curl is not available on h1", "total_bytes":0, "downloaded_bytes":0})
                    continue
                h2.cmd("mkdir -p /tmp/sdn_demo_http")
                made = h2.cmd(f"dd if=/dev/zero of=/tmp/sdn_demo_http/test_payload.bin bs=1M count={size_mb} status=none; stat -c %s /tmp/sdn_demo_http/test_payload.bin").strip().splitlines()[-1]
                download_total_bytes = int(made)
                # Start the server as a direct process (no shell wrapper), then verify HTTP is reachable before timing the transfer.
                http_server_proc = h2.popen(["python3", "-m", "http.server", "8000", "--bind", "10.0.0.2"],
                                            cwd="/tmp/sdn_demo_http", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.8)
                if http_server_proc.poll() is not None:
                    log("ERROR: HTTP server exited before download started.")
                    write_download_status({"state":"error", "message":"HTTP server failed to start on h2:8000", "total_bytes":download_total_bytes, "downloaded_bytes":0})
                    continue
                probe = h1.cmd("curl -sSI --connect-timeout 3 --max-time 5 http://10.0.0.2:8000/test_payload.bin 2>&1 | head -n 1").strip()
                if "200" not in probe:
                    log(f"ERROR: HTTP probe from h1 failed: {probe or 'no HTTP response'}")
                    write_download_status({"state":"error", "message":f"HTTP server unreachable from h1: {probe or 'no HTTP response'}", "total_bytes":download_total_bytes, "downloaded_bytes":0})
                    try: http_server_proc.terminate()
                    except Exception: pass
                    http_server_proc = None
                    continue
                download_started_at = time.time()
                download_last_bytes = 0
                download_last_sample_time = download_started_at
                download_peak_mbps = 0.0
                h1.cmd("rm -f /tmp/sdn_demo_download.bin /tmp/sdn_demo_curl_error.log")
                download_proc = h1.popen(["curl", "--fail", "--silent", "--show-error", "--connect-timeout", "5", "--max-time", "600", "--stderr", "/tmp/sdn_demo_curl_error.log", "http://10.0.0.2:8000/test_payload.bin", "-o", "/tmp/sdn_demo_download.bin"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                write_download_status({"state":"downloading", "message":"HTTP download started; waiting for first bytes", "total_bytes":download_total_bytes, "downloaded_bytes":0, "started_epoch":download_started_at, "current_mbps":0.0, "peak_mbps":0.0, "elapsed_seconds":0.0, "eta_seconds":None})
                log(f"DOWNLOAD started: h2 -> h1 HTTP transfer, file={size_mb} MiB, total_bytes={download_total_bytes}, curl PID={download_proc.pid}, HTTP probe={probe}")
            elif action == "download_status":
                update_download_status(h1)
            elif action == "stop_download":
                update_download_status(h1)
                stop_download()
                write_download_status({"state":"stopped", "message":"Download stopped by user", "total_bytes":download_total_bytes, "downloaded_bytes":download_last_bytes, "started_epoch":download_started_at, "current_mbps":0.0, "peak_mbps":download_peak_mbps, "elapsed_seconds":round(time.time()-(download_started_at or time.time()),2), "eta_seconds":None})
                log("DOWNLOAD stopped by dashboard request.")
            elif action == "stop_traffic":
                cleanup_traffic()
                h1.cmd("pkill -f 'iperf -c' >/dev/null 2>&1 || true")
                log("Traffic stop requested; active iperf clients terminated.")
            elif action == "shutdown":
                log("Dashboard requested Mininet shutdown.")
                break
            else:
                log(f"WARN: unknown action {action}")
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    finally:
        cleanup_traffic()
        stop_download()
        if net is not None:
            try: net.stop()
            except Exception as exc: log(f"WARN during net.stop: {exc}")
        log("Mininet helper stopped.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
