from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import sys
import threading
import queue
import time
import json
import socket
import shutil
import os

# AI-SDN Traffic Engineering Dashboard v6 — dashboard-controlled live demo
# Place this file in ~/sdn-project and run: streamlit run dashboard_v4.py

BASE_DIR = Path(__file__).resolve().parent
AI_FILE = BASE_DIR / "experiment_results_ai.csv"
BASELINE_FILE = BASE_DIR / "experiment_results_baseline.csv"
TRAFFIC_FILE = BASE_DIR / "traffic_logs.csv"
COMPARISON_FILE = BASE_DIR / "comparison_results" / "comparison_report.txt"
CONSOLE_LOG = BASE_DIR / "dashboard_console.log"
CONTROLLER_LOG = BASE_DIR / "controller_console.log"
RUNNER_FILE = BASE_DIR / "run_experiments.py"
VALIDATION_FILE = BASE_DIR / "independent_sdn_test_v4.py"
NETWORK_HELPER = BASE_DIR / "dashboard_network_helper.py"
NETWORK_LOG = BASE_DIR / "dashboard_network.log"
DOWNLOAD_STATUS_FILE = BASE_DIR / "dashboard_download_status.json"
CONTROLLER_FILE = BASE_DIR / "ai_traffic_engineering.py"

st.set_page_config(
    page_title="AI-SDN Control Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme / motion ----------
st.markdown("""
<style>
:root { --bg:#080d19; --panel:#101a2d; --panel2:#14223a; --line:#263752;
        --text:#edf4ff; --muted:#9eacc3; --cyan:#54e5d5; --blue:#6ba7ff;
        --green:#43d9a3; --red:#ff6878; }
.stApp { background: radial-gradient(ellipse at 80% 0%, #142b4a 0%, #0b1222 38%, #080d19 76%); color:var(--text); }
.block-container { padding-top: 2.1rem !important; padding-bottom: 3rem !important; max-width: 1500px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#111d33 0%,#0c1425 100%);
  border-right:1px solid #263752; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.7rem; }
h1,h2,h3 { letter-spacing:-.025em; color:#f4f8ff; }
h1 { font-size:2.25rem !important; }
p, label, [data-testid="stMetricLabel"] { color:#c0cde0; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#14223a,#0e1728);
 border:1px solid #2a3d5b; border-radius:16px; padding:17px 19px;
 transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
div[data-testid="stMetric"]:hover { transform:translateY(-3px); border-color:#54e5d5;
 box-shadow:0 10px 28px #0003; }
div[data-testid="stButton"] button { border-radius:11px; border:1px solid #2a3d5b;
 transition:all .18s ease; font-weight:600; }
div[data-testid="stButton"] button:hover { border-color:#54e5d5; color:#54e5d5;
 transform:translateY(-1px); }
div[data-testid="stDownloadButton"] button { border-radius:10px; }
[data-testid="stDataFrame"] { border:1px solid #293b56; border-radius:12px; overflow:hidden; }
.hero { background:linear-gradient(115deg,#14233c 0%,#183b5c 58%,#12414e 100%);
 border:1px solid #2b4b68; padding:30px 34px; border-radius:22px; margin:0 0 26px 0;
 box-shadow:0 16px 45px #0002; animation:fadeUp .45s ease both; }
.eyebrow { text-transform:uppercase; letter-spacing:.19em; color:#65f0df; font-size:.76rem; font-weight:800; }
.hero h1 { margin:.45rem 0 .5rem 0; }
.hero p { margin:0; font-size:1.02rem; color:#c8d7ec; }
.section-head { margin:22px 0 12px; font-size:1.25rem; font-weight:750; }
.panel { background:#101a2d; border:1px solid #263752; border-radius:16px; padding:18px 20px; }
.console-box { background:#070b12; border:1px solid #263752; border-radius:14px; padding:14px; color:#c8f7e8; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; overflow-wrap:anywhere; max-height:460px; overflow-y:auto; font-size:.82rem; line-height:1.55; }
.status-pill { display:inline-block; padding:6px 11px; border-radius:999px;
 background:#103a3b; color:#66f2dc; border:1px solid #286a68; font-size:.8rem; font-weight:700; }
.nav-caption { color:#7f91ad; font-size:.72rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; margin:1.2rem 0 .55rem; }
.brand { font-size:1.15rem; font-weight:850; color:#f5f8ff; }
.brand-sub { color:#9eacc3; margin-top:3px; font-size:.9rem; }
hr { border-color:#2a3b55 !important; }
@keyframes fadeUp { from {opacity:0; transform:translateY(8px)} to {opacity:1; transform:translateY(0)} }
@media (prefers-reduced-motion: reduce) { * {animation:none !important; transition:none !important;} }
</style>
""", unsafe_allow_html=True)


# ---------- Data helpers ----------
def read_csv_safe(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = make_unique_columns([str(c).strip() for c in df.columns])
        return df
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()

def make_unique_columns(columns):
    seen = {}
    out = []
    for col in columns:
        count = seen.get(col, 0)
        out.append(col if count == 0 else f"{col}_{count+1}")
        seen[col] = count + 1
    return out

def display_local_timestamps(values):
    """Normalize timestamps to IST without shifting naive timestamps twice.

    Naive timestamps in this project CSVs are already local wall-clock times (IST).
    Timestamps carrying an explicit timezone/offset are converted to Asia/Kolkata.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    def _to_ist(value):
        if pd.isna(value):
            return pd.NaT
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                # CSV timestamps without an offset are stored as local IST values.
                return ts
            return ts.tz_convert("Asia/Kolkata").tz_localize(None)
        except (TypeError, ValueError, OverflowError):
            return pd.NaT
    converted = series.map(_to_ist)
    return pd.to_datetime(converted, errors="coerce")

def format_table_timestamps(df):
    """Format already-localized timestamp columns consistently for display/export."""
    out = df.copy()
    for col in out.columns:
        if str(col).strip().lower() in {"timestamp", "time_stamp", "datetime", "date_time"}:
            parsed = pd.to_datetime(out[col], errors="coerce")
            # Keep invalid/unparseable source values visible instead of silently inventing a time.
            formatted = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
            out[col] = formatted.where(parsed.notna(), out[col].astype(str))
    return out

def num_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(dtype="float64")

def get_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df), index=df.index)

def normalize_experiment(df, mode):
    """Normalize known experiment fields without inventing measurements."""
    if df.empty:
        return df.copy()
    out = df.copy()
    aliases = {
        "requested_rate_mbps": ["requested_rate_mbps", "Requested rate (Mbps)", "requested_rate", "rate_mbps"],
        "measured_bandwidth_mbps": ["measured_bandwidth_mbps", "measured_bandwidth_mbps_2", "measured_bandwidth", "throughput_mbps"],
        "baseline_average_ping_ms": ["baseline_average_ping_ms", "baseline_avg_ping_ms"],
        "loaded_average_ping_ms": ["loaded_average_ping_ms", "loaded_average_ping_ms_2", "loaded_avg_ping_ms"],
        "baseline_packet_loss_percent": ["baseline_packet_loss_percent", "baseline_loss_percent"],
        "loaded_packet_loss_percent": ["loaded_packet_loss_percent", "loaded_packet_loss_percent_2", "loaded_loss_percent"],
        "post_average_ping_ms": ["post_average_ping_ms", "post_load_average_ping_ms"],
        "post_packet_loss_percent": ["post_packet_loss_percent", "post_load_packet_loss_percent"],
        "old_path": ["old_path"],
        "new_path": ["new_path"],
        "reroute_improvement": ["reroute_improvement"],
        "timestamp": ["timestamp"],
    }
    for target, options in aliases.items():
        if target not in out.columns:
            for c in options:
                if c in out.columns:
                    out[target] = out[c]
                    break
    if "requested_rate_mbps" not in out.columns:
        out["requested_rate_mbps"] = pd.NA
    if "timestamp" in out.columns:
        out["timestamp"] = display_local_timestamps(out["timestamp"])
    out["Mode"] = mode
    if mode == "Baseline":
        # Baseline mode is configured to hold PATH_1 fixed; do not display
        # stale/legacy AI reroute fields as if baseline actually rerouted.
        out["old_path"] = "PATH_1"
        out["new_path"] = "PATH_1"
        out["reroute_improvement"] = 0.0
    return out

ai_raw = read_csv_safe(AI_FILE)
baseline_raw = read_csv_safe(BASELINE_FILE)
traffic = read_csv_safe(TRAFFIC_FILE)
ai = normalize_experiment(ai_raw, "AI")
baseline = normalize_experiment(baseline_raw, "Baseline")
combined = pd.concat([ai, baseline], ignore_index=True, sort=False)
# Ensure Streamlit/Arrow never receives duplicate labels.
if not combined.empty:
    combined.columns = make_unique_columns([str(c) for c in combined.columns])

def fmt(value, suffix="", digits=2):
    try:
        value = float(value)
        return f"{value:,.{digits}f}{suffix}" if pd.notna(value) else "—"
    except (TypeError, ValueError):
        return "—"

def metric_mean(df, candidates):
    s = num_col(df, candidates).dropna()
    return s.mean() if not s.empty else None

def metric_median(df, candidates):
    s = num_col(df, candidates).dropna()
    return s.median() if not s.empty else None

def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def start_local_script(script_path, extra_args=None):
    """Start a known local Python script and capture its stdout/stderr for the dashboard."""
    if not script_path.exists():
        return False, f"Script not found: {script_path.name}"
    existing = st.session_state.get("runner_process")
    if existing is not None and existing.poll() is None:
        return False, "A dashboard-launched process is already running. Stop it before starting another."
    try:
        command = [sys.executable, "-u", str(script_path)] + list(extra_args or [])
        # Mininet requires root. Use non-interactive sudo so the dashboard never hangs
        # waiting for a password; if sudo is not pre-authorized, show the exact terminal command.
        if script_path == RUNNER_FILE:
            command = ["sudo", "-n", "/usr/bin/python3", "-u", str(script_path)] + list(extra_args or [])
        proc = subprocess.Popen(
            command, cwd=str(BASE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, bufsize=1, env={**__import__("os").environ, "PYTHONUNBUFFERED":"1"}
        )
        st.session_state.runner_process = proc
        st.session_state.runner_script = script_path.name
        st.session_state.runner_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.console_lines = []
        st.session_state.console_queue = queue.Queue()
        q = st.session_state.console_queue
        def reader():
            try:
                for line in proc.stdout:
                    q.put(line.rstrip("\n"))
            except Exception as exc:
                q.put(f"[dashboard reader] {exc}")
            finally:
                q.put("[dashboard] Process output stream closed.")
        threading.Thread(target=reader, daemon=True).start()
        return True, f"Started {script_path.name} (PID {proc.pid})"
    except Exception as exc:
        return False, f"Could not start script: {exc}"

def drain_runner_queue():
    q = st.session_state.get("console_queue")
    if q is not None:
        lines = st.session_state.setdefault("console_lines", [])
        while True:
            try:
                lines.append(q.get_nowait())
            except queue.Empty:
                break
        st.session_state.console_lines = lines[-1200:]

def read_console_file():
    if not CONSOLE_LOG.exists():
        return []
    try:
        return CONSOLE_LOG.read_text(errors="replace").splitlines()[-1200:]
    except Exception as exc:
        return [f"Unable to read console log: {exc}"]

@st.fragment(run_every="1s")
def render_console_output(source):
    """Refresh only the console panel, so streamed output appears without manual page refresh."""
    if source == "Dashboard-launched process":
        drain_runner_queue()
        lines = st.session_state.get("console_lines", [])
        if lines:
            st.markdown('<div class="console-box">' + __import__("html").escape("\n".join(lines[-500:])) + '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="console-box">Console ready. Start an experiment or validation script to stream output here.</div>', unsafe_allow_html=True)
        proc = st.session_state.get("runner_process")
        if proc is not None:
            status = "RUNNING" if proc.poll() is None else f"FINISHED · exit code {proc.returncode}"
            st.caption(f"{st.session_state.get('runner_script', 'Process')} · PID {proc.pid} · {status}")
    else:
        lines = read_console_file()
        if lines:
            st.markdown('<div class="console-box">' + __import__("html").escape("\n".join(lines[-500:])) + '</div>', unsafe_allow_html=True)
        else:
            st.warning("dashboard_console.log not found yet. Start the terminal command above first.")

# ---------- Dashboard-controlled controller / Mininet ----------
def port_is_open(host="127.0.0.1", port=6633, timeout=0.4):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def start_controller_from_dashboard():
    existing = st.session_state.get("controller_process")
    if existing is not None and existing.poll() is None:
        return False, f"Controller already launched by dashboard (PID {existing.pid})."
    if port_is_open():
        return False, "Port 6633 is already open. Ryu may already be running; avoid starting a duplicate controller."
    ryu = shutil.which("ryu-manager")
    if not ryu:
        return False, "ryu-manager not found in this Python environment. Activate ryu-env before launching Streamlit."
    try:
        log = open(CONTROLLER_LOG, "a", buffering=1)
        env = {**os.environ, "SDN_ROUTING_MODE": "ai", "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen([ryu, "--ofp-tcp-listen-port", "6633", str(CONTROLLER_FILE)],
                                cwd=str(BASE_DIR), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        st.session_state.controller_process = proc
        time.sleep(1.2)
        if proc.poll() is not None:
            return False, f"Ryu exited during startup (code {proc.returncode}). Check controller_console.log and model path."
        return True, f"Ryu AI controller started (PID {proc.pid}); log: {CONTROLLER_LOG.name}"
    except Exception as exc:
        return False, f"Could not start Ryu: {exc}"

def stop_controller_from_dashboard():
    proc = st.session_state.get("controller_process")
    if proc is None or proc.poll() is not None:
        return False, "No dashboard-launched Ryu process is running. If Ryu was started in another terminal, stop it there."
    try:
        proc.terminate()
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True, "Dashboard-launched Ryu controller stopped."
    except Exception as exc:
        return False, f"Could not stop Ryu: {exc}"

def authenticate_sudo(password):
    if not password:
        return False, "Enter your Ubuntu sudo password."
    try:
        result = subprocess.run(["sudo", "-S", "-p", "", "-v"], input=password + "\n",
                                text=True, capture_output=True, timeout=10, cwd=str(BASE_DIR))
        if result.returncode != 0:
            return False, "Sudo authentication failed. Check the Ubuntu password."
        return True, "sudo authenticated"
    except Exception as exc:
        return False, f"Sudo authentication error: {exc}"

def send_network_command(action, **kwargs):
    proc = st.session_state.get("network_process")
    if proc is None or proc.poll() is not None:
        return False, "Mininet control process is not running. Click Start Mininet first."
    try:
        proc.stdin.write(json.dumps({"action": action, **kwargs}) + "\n")
        proc.stdin.flush()
        return True, f"Sent {action} command to Mininet."
    except Exception as exc:
        return False, f"Could not send Mininet command: {exc}"

def start_mininet_from_dashboard(password):
    if not port_is_open():
        return False, "Ryu controller is not detected on port 6633. Click Start AI controller first."
    proc = st.session_state.get("network_process")
    if proc is not None and proc.poll() is None:
        return False, f"Mininet is already running (PID {proc.pid})."
    if not NETWORK_HELPER.exists():
        return False, f"Missing {NETWORK_HELPER.name}; keep it beside the dashboard file."
    ok, msg = authenticate_sudo(password)
    if not ok:
        return False, msg
    try:
        log = open(NETWORK_LOG, "a", buffering=1)
        # Authenticate the actual long-running sudo process via stdin. A separate
        # `sudo -v` call is not reliable on all WSL/sudo timestamp configurations.
        # sudo consumes the first line (password); the helper then receives JSON
        # control commands from the same pipe after the password line is consumed.
        proc = subprocess.Popen(["sudo", "-k", "-S", "-p", "", "/usr/bin/python3", "-u", str(NETWORK_HELPER)],
                                cwd=str(BASE_DIR), stdin=subprocess.PIPE, stdout=log,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env={**os.environ, "PYTHONUNBUFFERED": "1", "SDN_PROJECT_DIR": str(BASE_DIR)}, start_new_session=True)
        proc.stdin.write(password + "\n")
        proc.stdin.flush()
        st.session_state.network_process = proc
        st.session_state.network_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(2)
        if proc.poll() is not None:
            return False, f"Mininet helper exited (code {proc.returncode}). Check dashboard_network.log."
        return True, f"Mininet topology startup requested (PID {proc.pid})."
    except Exception as exc:
        return False, f"Could not start Mininet: {exc}"

def stop_network_from_dashboard():
    proc = st.session_state.get("network_process")
    if proc is None or proc.poll() is not None:
        return False, "No dashboard-launched Mininet process is running."
    try:
        proc.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
        proc.stdin.flush()
        return True, "Mininet shutdown requested."
    except Exception as exc:
        return False, f"Could not stop Mininet: {exc}"

# ---------- Sidebar navigation ----------
with st.sidebar:
    st.markdown('<div class="brand">🛰️ AI-SDN</div><div class="brand-sub">Traffic Engineering Control Center</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="nav-caption">Workspace</div>', unsafe_allow_html=True)
    pages = [
        ("Overview", "◈", "Overview"),
        ("AI vs Baseline", "⇄", "AI vs Baseline"),
        ("Live Traffic & ML", "⌁", "Live Traffic & ML"),
        ("Data & Reports", "⇩", "Data & Reports"),
    ]
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    for page, glyph, label in pages:
        is_active = st.session_state.page == page
        if st.button(f"{glyph}   {label}", key=f"nav_{page}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = page
            st.rerun()
    st.markdown("---")
    st.markdown('<div class="nav-caption">System status</div>', unsafe_allow_html=True)
    st.markdown('<span class="status-pill">● Dashboard ready</span>', unsafe_allow_html=True)
    st.caption(f"AI experiment rows: {len(ai_raw):,}")
    st.caption(f"Baseline rows: {len(baseline_raw):,}")
    st.caption(f"Traffic log rows: {len(traffic):,}")
    st.markdown("---")
    st.caption("Data is read from local CSV files in the project folder.")
    st.caption("Use Data & Reports to export the evidence shown here.")

if st.session_state.page == "Controller Console":
    st.session_state.page = "Overview"
page = st.session_state.page

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Software Defined Networking · ML-assisted routing</div>
  <h1>AI-Driven SDN Traffic Engineering</h1>
  <p>Experiment monitoring, congestion predictions, route decisions, and evidence for evaluation.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Overview ----------
if page == "Overview":
    st.markdown("## System overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI experiments", f"{len(ai_raw):,}")
    c2.metric("Baseline experiments", f"{len(baseline_raw):,}")
    rates = pd.to_numeric(get_col(combined, ["requested_rate_mbps"]), errors="coerce").dropna().nunique() if not combined.empty else 0
    c3.metric("Matched test rates", f"{min(len(ai), len(baseline)):,}")
    c4.metric("Traffic samples", f"{len(traffic):,}")

    st.markdown("### Actual Mininet topology")
    st.caption("Configured topology: h1–s1–s2–s4–h2 (PATH_1, 10 Mbps core links) and h1–s1–s3–s4–h2 (PATH_2, 20 Mbps core links). Shared host-edge links are configured at 20 Mbps.")
    topology = go.Figure()
    # Hand-positioned nodes reflect the actual AITopo graph and keep the two paths distinct.
    pos = {"h1": (0.0, 0.0), "s1": (1.0, 0.0), "s2": (2.0, 0.78),
           "s3": (2.0, -0.78), "s4": (3.0, 0.0), "h2": (4.0, 0.0)}
    edges = [
        ("h1", "s1", "Shared edge · 20 Mbps", "#94a3b8", 0.16),
        ("s1", "s2", "PATH_1 · 10 Mbps", "#39a8ff", 0.12),
        ("s2", "s4", "PATH_1 · 10 Mbps", "#39a8ff", 0.12),
        ("s1", "s3", "PATH_2 · 20 Mbps", "#ffad42", -0.12),
        ("s3", "s4", "PATH_2 · 20 Mbps", "#ffad42", -0.12),
        ("s4", "h2", "Shared edge · 20 Mbps", "#94a3b8", 0.16),
    ]
    for a, b, label, color, offset in edges:
        x1, y1 = pos[a]; x2, y2 = pos[b]
        topology.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines",
            line=dict(color=color, width=5 if "Shared" not in label else 3, dash="solid" if "Shared" not in label else "dot"),
            hovertemplate=f"{a} ↔ {b}<br>{label}<extra></extra>", showlegend=False))
        mx, my = (x1+x2)/2, (y1+y2)/2 + offset
        topology.add_annotation(x=mx, y=my, text=label, showarrow=False,
            font=dict(size=10, color=color), bgcolor="rgba(8,13,25,0.82)", borderpad=3)
    hosts = ["h1", "h2"]
    switches = ["s1", "s2", "s3", "s4"]
    topology.add_trace(go.Scatter(x=[pos[n][0] for n in hosts], y=[pos[n][1] for n in hosts],
        mode="markers+text", text=hosts, textposition="bottom center", name="Hosts",
        marker=dict(symbol="circle", size=34, color="#43d9a3", line=dict(color="#d5fff0", width=2)),
        textfont=dict(color="#eafff7", size=13), hovertemplate="Host %{text}<extra></extra>"))
    topology.add_trace(go.Scatter(x=[pos[n][0] for n in switches], y=[pos[n][1] for n in switches],
        mode="markers+text", text=switches, textposition="bottom center", name="OpenFlow switches",
        marker=dict(symbol="square", size=32, color="#477fff", line=dict(color="#dce8ff", width=2)),
        textfont=dict(color="#eaf1ff", size=13), hovertemplate="OpenFlow switch %{text}<extra></extra>"))
    topology.update_layout(template="plotly_dark", height=410, margin=dict(l=20, r=20, t=15, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,13,25,0.35)",
        xaxis=dict(visible=False, range=[-0.35, 4.35]), yaxis=dict(visible=False, range=[-1.35, 1.35], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest")
    st.plotly_chart(topology, use_container_width=True, config={"displayModeBar": False})
    legend_cols = st.columns(4)
    legend_cols[0].markdown("🟢 **Hosts** · h1, h2")
    legend_cols[1].markdown("🟦 **Switches** · s1–s4")
    legend_cols[2].markdown("🔵 **PATH_1** · s1–s2–s4")
    legend_cols[3].markdown("🟠 **PATH_2** · s1–s3–s4")

    @st.fragment(run_every="10s")
    def render_evaluation_snapshot():
        # Re-read result files on every fragment refresh so new completed tests update the means.
        ai_now = normalize_experiment(read_csv_safe(AI_FILE), "AI")
        base_now = normalize_experiment(read_csv_safe(BASELINE_FILE), "Baseline")
        st.markdown('<div class="section-head">Evaluation snapshot</div>', unsafe_allow_html=True)
        if not ai_now.empty and not base_now.empty:
            cols = st.columns(3)
            pairs = [
                ("Throughput", ["measured_bandwidth_mbps"], " Mbps", 3),
                ("Loaded latency", ["loaded_average_ping_ms"], " ms", 2),
                ("Loaded packet loss", ["loaded_packet_loss_percent"], "%", 2),
            ]
            for col, (label, keys, suffix, digits) in zip(cols, pairs):
                av = metric_mean(ai_now, keys)
                bv = metric_mean(base_now, keys)
                col.metric(f"{label} · AI mean", fmt(av, suffix, digits), delta=(fmt(av-bv, suffix, digits) + " vs baseline") if av is not None and bv is not None else None)
                col.caption(f"Baseline mean: {fmt(bv, suffix, digits)}")
            st.caption(f"Auto-updates every 10 seconds · AI tests: {len(ai_now)} · Baseline tests: {len(base_now)} · Last refreshed: {time.strftime('%H:%M:%S')}")
        else:
            st.info("Run at least one AI and one Baseline test to populate the comparison snapshot.")
    render_evaluation_snapshot()

    @st.fragment(run_every="5s")
    def render_recent_test_results():
        ai_live = normalize_experiment(read_csv_safe(AI_FILE), "AI")
        base_live = normalize_experiment(read_csv_safe(BASELINE_FILE), "Baseline")
        tests = pd.concat([ai_live, base_live], ignore_index=True, sort=False)
        st.markdown("### Latest experiment results · auto-refresh 5s")
        if tests.empty:
            st.info("No experiment results yet. Complete an AI or Baseline run to populate this table.")
            return
        if "timestamp" in tests.columns:
            # normalize_experiment already converts raw UTC timestamps to IST; do not convert twice.
            tests = tests.sort_values("timestamp", ascending=False, na_position="last")
        preferred = ["Mode", "timestamp", "requested_rate_mbps", "measured_bandwidth_mbps",
                     "baseline_average_ping_ms", "baseline_packet_loss_percent", "loaded_average_ping_ms",
                     "loaded_packet_loss_percent", "post_average_ping_ms", "post_packet_loss_percent",
                     "old_path", "new_path", "reroute_improvement"]
        cols = [c for c in preferred if c in tests.columns] + [c for c in tests.columns if c not in preferred]
        tests = tests[cols]
        tests_display = format_table_timestamps(tests.head(100))
        st.dataframe(tests_display, use_container_width=True, hide_index=True)
        st.download_button("Download all test logs + metrics (CSV)", format_table_timestamps(tests).to_csv(index=False).encode("utf-8"),
                           file_name="all_test_logs_metrics.csv", mime="text/csv", key="download_live_test_metrics")
        st.caption(f"Showing latest 100 of {len(tests)} tests · refreshed {time.strftime('%H:%M:%S')}")
    render_recent_test_results()

    st.markdown('<div class="section-head">Experiment files</div>', unsafe_allow_html=True)
    status_rows = [
        {"File": AI_FILE.name, "Status": "Available" if AI_FILE.exists() else "Missing", "Rows": len(ai_raw)},
        {"File": BASELINE_FILE.name, "Status": "Available" if BASELINE_FILE.exists() else "Missing", "Rows": len(baseline_raw)},
        {"File": TRAFFIC_FILE.name, "Status": "Available" if TRAFFIC_FILE.exists() else "Missing", "Rows": len(traffic)},
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

# ---------- Comparison ----------
elif page == "AI vs Baseline":
    st.markdown("## AI mode vs fixed-route baseline")
    st.caption("Baseline is displayed as fixed PATH_1. The baseline CSV's stale path fields are intentionally overridden for display.")
    if ai.empty or baseline.empty:
        st.warning("Both experiment CSV files are required for this view.")
    else:
        ai_rate = pd.to_numeric(ai["requested_rate_mbps"], errors="coerce")
        base_rate = pd.to_numeric(baseline["requested_rate_mbps"], errors="coerce")
        ai_rates = set(ai_rate.dropna().tolist())
        base_rates = set(base_rate.dropna().tolist())
        matched = sorted(ai_rates & base_rates)
        ai_cmp = ai[ai_rate.isin(matched)].copy()
        base_cmp = baseline[base_rate.isin(matched)].copy()
        st.markdown(f"**Matched requested traffic rates:** {', '.join(str(x) for x in matched) if matched else 'No matching rates found'}")
        cols = st.columns(3)
        metrics = [
            ("Throughput (Mbps)", "measured_bandwidth_mbps", True),
            ("Loaded latency (ms)", "loaded_average_ping_ms", False),
            ("Loaded packet loss (%)", "loaded_packet_loss_percent", False),
        ]
        for col, (label, key, higher_better) in zip(cols, metrics):
            a = metric_mean(ai_cmp, [key])
            b = metric_mean(base_cmp, [key])
            delta = None
            if a is not None and b is not None:
                delta = f"{((a-b)/abs(b)*100):+.2f}% vs baseline" if b else "baseline mean is zero"
            col.metric(f"AI mean · {label}", fmt(a, "", 3), delta=delta, delta_color="normal" if higher_better else "inverse")
            col.caption(f"Baseline mean: {fmt(b, '', 3)}")

        # Prefer repeated evaluation runs when present: the top-level live CSVs may only
        # contain the latest session and can omit loss events recorded by the evaluator.
        chart_ai, chart_base = ai_cmp.copy(), base_cmp.copy()
        eval_dir = BASE_DIR / "evaluation_results"
        if eval_dir.exists():
            ai_files = sorted(eval_dir.glob("ai_run*.csv"))
            base_files = sorted(eval_dir.glob("baseline_run*.csv"))
            def load_runs(paths, mode):
                frames = []
                for path in paths:
                    raw = read_csv_safe(path)
                    if not raw.empty:
                        frames.append(normalize_experiment(raw, mode))
                return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
            eval_ai = load_runs(ai_files, "AI")
            eval_base = load_runs(base_files, "Baseline")
            if not eval_ai.empty and not eval_base.empty:
                chart_ai, chart_base = eval_ai, eval_base
                st.caption("Charts use repeated evaluation runs from evaluation_results/ when available; detailed table below shows the current session CSVs.")

        chart_rows = []
        for mode, df in [("AI mode", chart_ai), ("Baseline mode", chart_base)]:
            for _, row in df.iterrows():
                chart_rows.append({
                    "Mode": mode,
                    "Timestamp": row.get("timestamp", pd.NaT),
                    "Requested rate (Mbps)": pd.to_numeric(pd.Series([row.get("requested_rate_mbps")]), errors="coerce").iloc[0],
                    "Throughput (Mbps)": pd.to_numeric(pd.Series([row.get("measured_bandwidth_mbps")]), errors="coerce").iloc[0],
                    "Loaded latency (ms)": pd.to_numeric(pd.Series([row.get("loaded_average_ping_ms")]), errors="coerce").iloc[0],
                    "Loaded packet loss (%)": pd.to_numeric(pd.Series([row.get("loaded_packet_loss_percent")]), errors="coerce").iloc[0],
                    "Baseline ping packet loss (%)": pd.to_numeric(pd.Series([row.get("baseline_packet_loss_percent")]), errors="coerce").iloc[0],
                })
        plot_df = pd.DataFrame(chart_rows)
        if "Timestamp" in plot_df:
            plot_df["Timestamp"] = pd.to_datetime(plot_df["Timestamp"], errors="coerce")
        tab1, tab2, tab3 = st.tabs(["Throughput", "Latency", "Packet loss"])
        bar_colors = {"AI mode": "#22C55E", "Baseline mode": "#F97316"}
        for tab, metric in [(tab1, "Throughput (Mbps)"), (tab2, "Loaded latency (ms)"), (tab3, "Loaded packet loss (%)")]:
            with tab:
                if not plot_df.empty and plot_df[metric].notna().any():
                    chart_data = plot_df.dropna(subset=[metric]).copy()
                    # Group repeated runs by requested rate so non-zero loss events remain visible.
                    summary = (chart_data.groupby(["Requested rate (Mbps)", "Mode"], dropna=False, as_index=False)
                               .agg(**{metric: (metric, "mean"), "Run count": (metric, "count")}))
                    fig = px.bar(summary, x="Requested rate (Mbps)", y=metric, color="Mode",
                                 barmode="group", color_discrete_map=bar_colors,
                                 hover_data={"Run count": True, metric: ":.3f"}, template="plotly_dark",
                                 category_orders={"Mode": ["AI mode", "Baseline mode"]})
                    if metric == "Loaded packet loss (%)":
                        # Pre-load ping loss is a separate metric; show it as a separate bar series.
                        base_ping = chart_data[chart_data["Mode"] == "Baseline mode"].copy()
                        if not base_ping.empty and base_ping["Baseline ping packet loss (%)"].notna().any():
                            ping_summary = (base_ping.groupby("Requested rate (Mbps)", as_index=False)
                                            .agg(**{"Packet loss (%)": ("Baseline ping packet loss (%)", "mean")}))
                            fig.add_trace(go.Bar(x=ping_summary["Requested rate (Mbps)"],
                                y=ping_summary["Packet loss (%)"], name="Baseline pre-load ping loss",
                                marker_color="#FBBF24", opacity=0.9,
                                hovertemplate="Rate: %{x} Mbps<br>Pre-load ping loss: %{y:.3f}%<extra></extra>"))
                        vals = pd.concat([pd.to_numeric(summary[metric], errors="coerce"),
                                          pd.to_numeric(chart_data["Baseline ping packet loss (%)"], errors="coerce")]).dropna()
                        ymax = max(1.0, float(vals.max()) * 1.2 if not vals.empty else 1.0)
                        fig.update_yaxes(range=[0, ymax], rangemode="tozero")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      margin=dict(l=10,r=10,t=20,b=10), legend_title_text="",
                                      xaxis_title="Requested traffic rate (Mbps)", bargap=0.22, bargroupgap=0.08)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Bars show mean across available runs for each requested rate. Hover for exact values and run counts.")
                else:
                    st.info(f"No valid {metric} values found.")

        st.markdown("### Detailed experiment table")
        detail = pd.concat([ai_cmp, base_cmp], ignore_index=True, sort=False)
        # Select stable, useful columns and rename for presentation.
        wanted = ["Mode", "timestamp", "requested_rate_mbps", "measured_bandwidth_mbps",
                  "baseline_average_ping_ms", "loaded_average_ping_ms", "loaded_packet_loss_percent",
                  "post_average_ping_ms", "post_packet_loss_percent", "old_path", "new_path", "reroute_improvement"]
        present = [c for c in wanted if c in detail.columns]
        detail = detail[present].copy()
        detail = detail.rename(columns={
            "timestamp":"Timestamp", "requested_rate_mbps":"Requested rate (Mbps)",
            "measured_bandwidth_mbps":"Measured throughput (Mbps)",
            "baseline_average_ping_ms":"Baseline ping (ms)", "loaded_average_ping_ms":"Loaded latency (ms)",
            "loaded_packet_loss_percent":"Loaded loss (%)", "post_average_ping_ms":"Post-load latency (ms)",
            "post_packet_loss_percent":"Post-load loss (%)", "old_path":"Old path", "new_path":"New path",
            "reroute_improvement":"Route score improvement",
        })
        detail.columns = make_unique_columns([str(c) for c in detail.columns])
        detail = format_table_timestamps(detail)
        st.dataframe(detail, use_container_width=True, hide_index=True)
        st.download_button("Download all experiment metrics (CSV)", detail.to_csv(index=False).encode("utf-8"),
                           file_name="all_experiment_metrics.csv", mime="text/csv")

# ---------- Live traffic ----------
elif page == "Live Traffic & ML":
    @st.fragment(run_every="5s")
    def render_live_traffic():
        st.markdown("## Live traffic & ML telemetry")
        traffic_now = read_csv_safe(TRAFFIC_FILE)
        st.caption("Auto-refreshes every 5 seconds. Charts show the most recent 15 minutes where available; large timestamp gaps are not interpreted as continuous traffic.")
        refresh_col1, refresh_col2 = st.columns([1, 5])
        with refresh_col1:
            if st.button("↻ Refresh telemetry", use_container_width=True):
                st.rerun()

        # Real controls: all displayed model predictions below are read from controller telemetry CSV.
        st.markdown("### Evaluator controls · real Mininet traffic + trained model")
        st.caption("No synthetic predictions are generated here. Prediction/confidence values come from the running Ryu controller and traffic_logs.csv.")
        pw_col, ctrl_col, net_col = st.columns([1.4, 1, 1])
        with pw_col:
            sudo_password = st.text_input("Ubuntu sudo password (used only for Mininet start)", type="password", key="mininet_sudo_password")
        with ctrl_col:
            st.write("**Ryu controller**")
            if st.button("▶ Start AI controller", use_container_width=True, key="start_ryu_live"):
                ok, msg = start_controller_from_dashboard()
                (st.success if ok else st.warning)(msg)
            if st.button("■ Stop dashboard Ryu", use_container_width=True, key="stop_ryu_live"):
                ok, msg = stop_controller_from_dashboard()
                (st.success if ok else st.warning)(msg)
            cproc = st.session_state.get("controller_process")
            if cproc is not None and cproc.poll() is None:
                st.caption(f"Dashboard-managed · PID {cproc.pid}")
            elif port_is_open():
                st.caption("Port 6633 open · external controller may be running")
            else:
                st.caption("Not detected")
        with net_col:
            st.write("**Mininet topology**")
            if st.button("▶ Start Mininet", type="primary", use_container_width=True, key="start_mininet_live"):
                ok, msg = start_mininet_from_dashboard(sudo_password)
                (st.success if ok else st.error)(msg)
            nproc = st.session_state.get("network_process")
            if nproc is not None and nproc.poll() is None:
                st.caption(f"Dashboard-managed · PID {nproc.pid}")
            else:
                st.caption("Not running from this dashboard")

        st.markdown("#### Generate traffic")
        tc1, tc2, tc3, tc4 = st.columns([1.1, 1.1, 1.1, 1.2])
        with tc1:
            traffic_kind = st.selectbox("Traffic type", ["Ping", "UDP iperf", "TCP iperf"], key="traffic_kind")
        with tc2:
            traffic_rate = st.number_input("iperf rate (Mbps)", min_value=0.1, max_value=100.0, value=5.0, step=0.5, key="traffic_rate")
        with tc3:
            traffic_duration = st.slider("Duration (seconds)", min_value=5, max_value=60, value=15, step=5, key="traffic_duration")
        with tc4:
            ping_count = st.selectbox("Ping packets", [5, 10, 20, 30], index=1, key="ping_count")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            if st.button("▶ Generate selected traffic", type="primary", use_container_width=True, key="generate_live_traffic"):
                if traffic_kind == "Ping":
                    ok, msg = send_network_command("ping", count=int(ping_count))
                elif traffic_kind == "UDP iperf":
                    ok, msg = send_network_command("iperf", protocol="udp", rate_mbps=float(traffic_rate), duration=int(traffic_duration))
                else:
                    ok, msg = send_network_command("iperf", protocol="tcp", rate_mbps=float(traffic_rate), duration=int(traffic_duration))
                (st.success if ok else st.error)(msg)
        with bc2:
            if st.button("■ Stop traffic", use_container_width=True, key="stop_live_traffic"):
                ok, msg = send_network_command("stop_traffic")
                (st.success if ok else st.warning)(msg)
        with bc3:
            if st.button("■ Stop Mininet", use_container_width=True, key="stop_mininet_live"):
                ok, msg = stop_network_from_dashboard()
                (st.success if ok else st.warning)(msg)
        st.markdown("### Smart Download + AI Routing Demo")
        st.caption("Real HTTP file transfer between Mininet hosts h2 → h1. Download progress/speed are measured from transferred bytes; model/path values come from controller telemetry. This adds a demo action only and does not change topology or routing policy.")
        dl_a, dl_b = st.columns([1, 1])
        with dl_a:
            demo_size_mb = st.selectbox("Test file size", [25, 50, 100, 200], index=2, format_func=lambda x: f"{x} MiB", key="demo_download_size_mb")
        with dl_b:
            if st.button("⬇ Start smart download", type="primary", use_container_width=True, key="start_smart_download"):
                ok, msg = send_network_command("start_download", size_mb=int(demo_size_mb))
                if ok:
                    st.session_state["smart_download_start_epoch"] = time.time()
                    st.success("Download command sent. Progress will update automatically.")
                else:
                    st.error(msg)
            if st.button("■ Stop smart download", use_container_width=True, key="stop_smart_download"):
                ok, msg = send_network_command("stop_download")
                (st.success if ok else st.warning)(msg)
        # Ask the helper to sample real bytes and speed, then render the last status file.
        nproc_demo = st.session_state.get("network_process")
        if nproc_demo is not None and nproc_demo.poll() is None:
            send_network_command("download_status")
        dl_status = {}
        try:
            if DOWNLOAD_STATUS_FILE.exists():
                dl_status = json.loads(DOWNLOAD_STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            dl_status = {}
        if dl_status:
            total_b = max(0, int(dl_status.get("total_bytes", 0) or 0))
            done_b = max(0, int(dl_status.get("downloaded_bytes", 0) or 0))
            pct = min(1.0, done_b / total_b) if total_b else 0.0
            status_label = str(dl_status.get("state", "idle")).replace("_", " ").title()
            st.markdown(f"**Transfer status:** {status_label} · {dl_status.get('message', '')}")
            st.progress(pct, text=f"Download progress · {pct*100:.1f}% ({done_b/1048576:.1f} / {total_b/1048576:.1f} MiB)" if total_b else "No transfer data yet")
            dm1, dm2, dm3, dm4 = st.columns(4)
            dm1.metric("Current download speed", f"{float(dl_status.get('current_mbps',0) or 0):.2f} Mbps")
            dm2.metric("Peak observed speed", f"{float(dl_status.get('peak_mbps',0) or 0):.2f} Mbps")
            dm3.metric("Elapsed", f"{float(dl_status.get('elapsed_seconds',0) or 0):.1f} s")
            eta = dl_status.get("eta_seconds")
            dm4.metric("Estimated remaining", f"{float(eta):.1f} s" if eta is not None and pct < 1 else ("Complete" if pct >= 1 else "—"))
        # Surface actual controller inference alongside the transfer; no fabricated GNN values.
        st.markdown("#### Live ML / route evidence during demo")
        latest_sample = traffic_now.copy()
        if not latest_sample.empty and "timestamp" in latest_sample.columns:
            latest_sample["timestamp"] = display_local_timestamps(latest_sample["timestamp"])
            latest_sample = latest_sample.sort_values("timestamp")
        if not latest_sample.empty:
            last = latest_sample.iloc[-1]
            rm1, rm2, rm3, rm4 = st.columns(4)
            rm1.metric("Latest congestion prediction", str(last.get("prediction", "—")))
            try: conf_txt = f"{float(last.get('confidence'))*100:.1f}%"
            except (TypeError, ValueError): conf_txt = "—"
            rm2.metric("Model confidence", conf_txt)
            rm3.metric("Preferred path", str(last.get("preferred_path", "—")))
            rm4.metric("Reported active path", str(last.get("active_path", "—")))
            if "active_path" in latest_sample.columns:
                observed_paths = [str(v) for v in latest_sample["active_path"].dropna().tail(50).unique()]
                st.caption("Recent active-path values in controller telemetry: " + (" → ".join(observed_paths) if observed_paths else "No path samples"))
            st.caption("These are the controller's latest logged values, not a guarantee that rerouting occurred during this particular download. GNN shadow predictions are not represented as routing decisions.")
        else:
            st.info("Start Ryu and Mininet, then generate traffic. The demo will display prediction/confidence/path values when traffic_logs.csv has controller samples.")

        # GNN_FORECAST_GRID_V1: display real per-switch forecasts only.
        st.markdown("#### GNN Forecaster — Next Utilization Prediction")
        gnn_csv = Path(__file__).resolve().parent / "gnn_live_predictions.csv"

        if not gnn_csv.exists():
            st.info("GNN forecast CSV not found yet. Generate traffic to produce forecasts.")
        else:
            try:
                gnn_df = pd.read_csv(gnn_csv)
                col_lookup = {str(c).strip().lower(): c for c in gnn_df.columns}

                switch_col = next(
                    (col_lookup[k] for k in ("switch", "switch_id", "node")
                     if k in col_lookup), None
                )
                forecast_col = next(
                    (col_lookup[k] for k in (
                        "predicted_next_utilization_percent",
                        "predicted_utilization_percent",
                        "predicted_utilization",
                        "predicted_next_utilization"
                    ) if k in col_lookup), None
                )
                timestamp_col = next(
                    (col_lookup[k] for k in ("timestamp", "time", "datetime")
                     if k in col_lookup), None
                )

                if gnn_df.empty or switch_col is None or forecast_col is None:
                    st.info(
                        "No readable per-switch GNN forecast is available yet. "
                        "The CSV must contain switch and predicted-utilization columns."
                    )
                else:
                    gnn_df[switch_col] = gnn_df[switch_col].astype(str).str.strip().str.lower()
                    gnn_df[forecast_col] = pd.to_numeric(
                        gnn_df[forecast_col], errors="coerce"
                    )
                    gnn_df = gnn_df.dropna(subset=[forecast_col])

                    if timestamp_col is not None:
                        gnn_df["_gnn_time"] = pd.to_datetime(
                            gnn_df[timestamp_col], errors="coerce", utc=True
                        )
                        if gnn_df["_gnn_time"].notna().any():
                            gnn_df = gnn_df.sort_values("_gnn_time")

                    latest_gnn = (
                        gnn_df.drop_duplicates(subset=[switch_col], keep="last")
                        .set_index(switch_col)
                    )
                    gnn_cards = st.columns(4)

                    for card, switch_name in zip(gnn_cards, ("s1", "s2", "s3", "s4")):
                        if switch_name in latest_gnn.index:
                            value = float(latest_gnn.loc[switch_name, forecast_col])
                            if not pd.notna(value):
                                card.metric(f"{switch_name} forecast", "—")
                            else:
                                card.metric(
                                    f"{switch_name} predicted utilization",
                                    f"{value:.2f}%"
                                )
                        else:
                            card.metric(f"{switch_name} forecast", "No data")

                    st.caption(
                        "Values are the latest per-switch forecasts recorded by the "
                        "GNN predictor; they are predictions, not measured utilization."
                    )
            except Exception as exc:
                st.warning(f"Could not read GNN forecast values: {exc}")

        if NETWORK_LOG.exists():
            with st.expander("Mininet control status (not raw model output)"):
                st.code("\n".join(NETWORK_LOG.read_text(errors="replace").splitlines()[-25:]), language="text")
        if CONTROLLER_LOG.exists():
            with st.expander("Ryu controller status"):
                st.code("\n".join(CONTROLLER_LOG.read_text(errors="replace").splitlines()[-20:]), language="text")
        st.markdown("---")

        if traffic_now.empty:
            st.warning("traffic_logs.csv was not found or contains no rows. Run the controller and generate traffic first.")
        else:
            t = traffic_now.copy()
            for col in ["rx_mbps", "tx_mbps", "utilization_percent", "confidence", "path1_score", "path2_score", "port"]:
                if col in t.columns:
                    t[col] = pd.to_numeric(t[col], errors="coerce")
            if "timestamp" in t.columns:
                t["timestamp"] = display_local_timestamps(t["timestamp"])
                t = t.dropna(subset=["timestamp"]).sort_values("timestamp")
                if not t.empty:
                    newest = t["timestamp"].max()
                    # Show a recent window; if old logs are all that exists, fall back to last 500 rows.
                    recent = t[t["timestamp"] >= newest - pd.Timedelta(minutes=15)].copy()
                    t = recent if len(recent) >= 2 else t.tail(500).copy()
            if t.empty:
                st.warning("No valid timestamped telemetry samples found in traffic_logs.csv.")
                st.stop()
            latest = t.iloc[-1]
            a,b,c,d = st.columns(4)
            a.metric("Latest prediction", str(latest.get("prediction", "—")))
            b.metric("ML confidence", fmt(latest.get("confidence"), "", 3))
            c.metric("Link utilization", fmt(latest.get("utilization_percent"), "%", 2))
            d.metric("Active path", str(latest.get("active_path", "—")))
            st.caption(f"Latest logged sample: {latest.get('timestamp', 'timestamp unavailable')}")

            switch_values = sorted(t["switch"].dropna().astype(str).unique().tolist()) if "switch" in t.columns else []
            filter_col, agg_col = st.columns([2, 1])
            with filter_col:
                selected_switches = st.multiselect("Switch filter", switch_values, default=switch_values, key="live_switch_filter") if switch_values else []
            with agg_col:
                aggregation = st.selectbox("Aggregation", ["Mean per switch/time", "Maximum per switch/time"], index=0)
            view = t[t["switch"].astype(str).isin(selected_switches)].copy() if switch_values and selected_switches else (t.copy() if not switch_values else t.iloc[0:0].copy())
            agg_fn = "mean" if aggregation.startswith("Mean") else "max"
            plot_tabs = st.tabs(["Utilization", "RX / TX rate", "Path scores", "Predictions"])
            with plot_tabs[0]:
                if "utilization_percent" in view and not view.empty:
                    group_keys = (["timestamp"] if "timestamp" in view else []) + (["switch"] if "switch" in view else [])
                    chart_data = view.groupby(group_keys, as_index=False, dropna=False)["utilization_percent"].agg(agg_fn) if group_keys else view[["utilization_percent"]].copy()
                    if "timestamp" in chart_data:
                        chart_data = chart_data.sort_values("timestamp")
                        # Break lines across large telemetry gaps instead of drawing misleading diagonals.
                        if "switch" in chart_data:
                            chart_data["_gap"] = chart_data.groupby("switch")["timestamp"].diff().dt.total_seconds().gt(120)
                            chart_data["_segment"] = chart_data.groupby("switch")["_gap"].cumsum().astype(str)
                            chart_data["series"] = chart_data["switch"].astype(str) + " · " + chart_data["_segment"]
                        else:
                            chart_data["_gap"] = chart_data["timestamp"].diff().dt.total_seconds().gt(120)
                            chart_data["series"] = chart_data["_gap"].cumsum().astype(str)
                        fig = px.line(chart_data, x="timestamp", y="utilization_percent",
                                      color="switch" if "switch" in chart_data else None,
                                      markers=True, template="plotly_dark")
                    else:
                        fig = px.line(chart_data, y="utilization_percent", markers=True, template="plotly_dark")
                    fig.add_hline(y=50, line_dash="dash", line_color="#f5bd4f", annotation_text="Medium ≥ 50%", annotation_position="top left")
                    fig.add_hline(y=80, line_dash="dash", line_color="#ff6878", annotation_text="Congested ≥ 80%", annotation_position="top left")
                    fig.update_yaxes(title="Utilization (%)", range=[0, 105])
                    fig.update_xaxes(title="Timestamp (IST)", tickformat="%d %b<br>%H:%M:%S", hoverformat="%Y-%m-%d %H:%M:%S")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend_title_text="Switch", margin=dict(l=10,r=10,t=25,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Thresholds are operational reference levels used by this project: 50% medium and 80% congested.")
                else:
                    st.info("No utilization samples match the selected switches.")
            with plot_tabs[1]:
                rate_cols = [c for c in ["rx_mbps", "tx_mbps"] if c in view.columns]
                if rate_cols and not view.empty:
                    group_keys = (["timestamp"] if "timestamp" in view else []) + (["switch"] if "switch" in view else [])
                    chart_data = view.groupby(group_keys, as_index=False, dropna=False)[rate_cols].agg(agg_fn) if group_keys else view[rate_cols].copy()
                    if "timestamp" in chart_data:
                        long = chart_data.melt([c for c in ["timestamp", "switch"] if c in chart_data], var_name="Direction", value_name="Mbps")
                        fig = px.line(long, x="timestamp", y="Mbps", color="Direction", line_dash="switch" if "switch" in long else None, markers=True, template="plotly_dark")
                    else:
                        fig = px.line(chart_data, y=rate_cols, template="plotly_dark")
                    fig.update_xaxes(title="Timestamp (IST)", tickformat="%d %b<br>%H:%M:%S", hoverformat="%Y-%m-%d %H:%M:%S")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", margin=dict(l=10,r=10,t=25,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No RX/TX samples match the selected switches.")
            with plot_tabs[2]:
                score_cols = [c for c in ["path1_score", "path2_score"] if c in view.columns]
                if score_cols and not view.empty:
                    group_keys = ["timestamp"] if "timestamp" in view else []
                    chart_data = view.groupby(group_keys, as_index=False, dropna=False)[score_cols].agg(agg_fn) if group_keys else view[score_cols].copy()
                    fig = px.line(chart_data, x="timestamp" if "timestamp" in chart_data else chart_data.index, y=score_cols, markers=True, template="plotly_dark")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", margin=dict(l=10,r=10,t=25,b=10), legend_title_text="Path score")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No path-score samples match the selected switches.")
            with plot_tabs[3]:
                if "prediction" in view.columns and not view.empty:
                    pred_counts = view["prediction"].fillna("Unknown").value_counts().rename_axis("Prediction").reset_index(name="Samples")
                    fig = px.pie(pred_counts, names="Prediction", values="Samples", hole=.58, template="plotly_dark")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=25,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(pred_counts, use_container_width=True, hide_index=True)
            st.markdown("### Latest controller samples")
            latest_cols = [c for c in ["timestamp", "switch", "port", "rx_mbps", "tx_mbps", "utilization_percent", "prediction", "confidence", "path1_score", "path2_score", "preferred_path", "active_path"] if c in view.columns]
            recent_table = view[latest_cols].tail(500).sort_values("timestamp", ascending=False) if "timestamp" in latest_cols else view[latest_cols].tail(500)
            st.dataframe(format_table_timestamps(recent_table), use_container_width=True, hide_index=True)
            export_cols = [c for c in ["timestamp", "switch", "port", "rx_mbps", "tx_mbps", "utilization_percent", "prediction", "confidence", "path1_score", "path2_score", "preferred_path", "active_path"] if c in view.columns]
            export_table = view[export_cols].copy() if export_cols else view.copy()
            export_table = format_table_timestamps(export_table)
            st.download_button("Download ALL prediction & telemetry rows (CSV)", export_table.to_csv(index=False).encode("utf-8"),
                               file_name="all_prediction_telemetry.csv", mime="text/csv")

    render_live_traffic()

# ---------- Data / reports ----------
elif page == "Data & Reports":
    st.markdown("## Data & reports")
    st.caption("Download the raw data and a normalized combined table for the evaluator.")
    d1, d2, d3 = st.columns(3)
    if AI_FILE.exists():
        d1.download_button("Download AI experiments", AI_FILE.read_bytes(), file_name=AI_FILE.name, mime="text/csv", use_container_width=True)
    else:
        d1.warning("AI CSV missing")
    if BASELINE_FILE.exists():
        d2.download_button("Download baseline experiments", BASELINE_FILE.read_bytes(), file_name=BASELINE_FILE.name, mime="text/csv", use_container_width=True)
    else:
        d2.warning("Baseline CSV missing")
    if TRAFFIC_FILE.exists():
        d3.download_button("Download traffic logs", TRAFFIC_FILE.read_bytes(), file_name=TRAFFIC_FILE.name, mime="text/csv", use_container_width=True)
    else:
        d3.warning("Traffic log missing")
    if not combined.empty:
        st.markdown("### Combined normalized experiments")
        view = combined.copy()
        if "Mode" in view:
            # Baseline path fields are fixed for display and evaluation.
            mask = view["Mode"].eq("Baseline")
            for c, value in [("old_path","PATH_1"),("new_path","PATH_1"),("reroute_improvement",0.0)]:
                if c in view.columns:
                    view.loc[mask, c] = value
        view.columns = make_unique_columns([str(c) for c in view.columns])
        view = format_table_timestamps(view)
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download combined normalized CSV", view.to_csv(index=False).encode("utf-8"),
                           file_name="ai_vs_baseline_normalized.csv", mime="text/csv")
    if COMPARISON_FILE.exists():
        with st.expander("Comparison report text"):
            st.code(COMPARISON_FILE.read_text(errors="replace"), language="text")

st.markdown("---")
st.caption("AI-SDN Control Center · Metrics are derived from local CSV evidence. Small sample sizes and measurement outliers should be disclosed during evaluation.")

# Refresh the live demo once per second while a real file transfer is active.
# This only reruns the Streamlit view; it does not restart Ryu, Mininet, or traffic.
if page == "Live Traffic & ML" and DOWNLOAD_STATUS_FILE.exists():
    try:
        _dl_refresh = json.loads(DOWNLOAD_STATUS_FILE.read_text(encoding="utf-8"))
        if _dl_refresh.get("state") == "downloading":
            time.sleep(1.0)
            st.rerun()
    except Exception:
        pass
