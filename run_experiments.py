#!/usr/bin/env python3

"""
Automated experiment runner for the AI-Driven SDN Traffic Engineering project.

Run this while the Ryu AI controller is already running on TCP port 6633.

It creates:
    h1 -- s1 -- s2 -- s4 -- h2
           \-- s3 --/

All links are 10 Mbps.

Experiments:
    1, 3, 5, 7, 9 Mbps UDP traffic

For each rate it records:
    - requested traffic rate
    - measured UDP bandwidth
    - packet loss
    - average ping delay
    - active path (read from controller CSV when available)

Results:
    ~/sdn-project/experiment_results.csv
"""

import csv
import os
import re
import time
import subprocess
import argparse

from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel, info


PROJECT_DIR = os.path.expanduser("~/sdn-project")
RESULT_FILE = os.path.join(PROJECT_DIR, "experiment_results.csv")

DEFAULT_RATES = [1, 3, 5, 7, 9]
TRAFFIC_SECONDS = 20
PING_COUNT = 20


class AITopo(Topo):
    def build(self):
        h1 = self.addHost("h1", ip="10.0.0.1/24")
        h2 = self.addHost("h2", ip="10.0.0.2/24")

        s1 = self.addSwitch("s1", protocols="OpenFlow13")
        s2 = self.addSwitch("s2", protocols="OpenFlow13")
        s3 = self.addSwitch("s3", protocols="OpenFlow13")
        s4 = self.addSwitch("s4", protocols="OpenFlow13")

        opts = dict(bw=10, delay="1ms", loss=0)

        self.addLink(h1, s1, **opts)
        self.addLink(s1, s2, **opts)
        self.addLink(s1, s3, **opts)
        self.addLink(s2, s4, **opts)
        self.addLink(s3, s4, **opts)
        self.addLink(s4, h2, **opts)


def run_cmd(cmd):
    return subprocess.run(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def parse_ping(output):
    loss = None
    avg = None

    m = re.search(r"(\d+(?:\.\d+)?)%\s*packet loss", output)
    if m:
        loss = float(m.group(1))

    m = re.search(
        r"(?:rtt|round-trip).*?=\s*"
        r"[\d.]+/([\d.]+)/[\d.]+/[\d.]+\s*ms",
        output,
    )
    if m:
        avg = float(m.group(1))

    return loss, avg


def parse_iperf(output):
    bandwidth = None

    # Matches both:
    # 8.43 Mbits/sec
    # 9.01 Mbits/sec
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s+Mbits/sec",
        output
    )

    if matches:
        bandwidth = float(matches[-1])

    return bandwidth


def read_latest_controller_event():
    """
    Reads the latest rerouting event if the controller has already
    written rerouting_events.csv.
    """
    path = os.path.join(PROJECT_DIR, "rerouting_events.csv")

    if not os.path.exists(path):
        return "", "", ""

    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            return "", "", ""

        row = rows[-1]
        return (
            row.get("old_path", ""),
            row.get("new_path", ""),
            row.get("improvement", ""),
        )
    except Exception:
        return "", "", ""


def parse_rates(value):
    """Parse comma-separated custom Mbps rates, e.g. 2,4,6,11."""
    try:
        rates = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Rates must be comma-separated numbers, e.g. 2,4,6,11") from exc
    if not rates or any(rate <= 0 or rate > 100 for rate in rates):
        raise argparse.ArgumentTypeError("Provide at least one rate between 0.1 and 100 Mbps")
    # Keep integer values looking like integers in logs/CSV where possible.
    return [int(rate) if rate.is_integer() else rate for rate in rates]


def main():
    parser = argparse.ArgumentParser(description="Run SDN traffic experiments at configurable Mbps rates.")
    parser.add_argument("--rates", type=parse_rates, default=DEFAULT_RATES,
                        help="Comma-separated rates in Mbps (default: 1,3,5,7,9), e.g. --rates 2,4,6,11")
    args = parser.parse_args()
    rates = args.rates
    os.makedirs(PROJECT_DIR, exist_ok=True)

    setLogLevel("info")

    topo = AITopo()

    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True,
    )

    controller = net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6633,
    )

    info("\n*** Starting network\n")
    net.start()

    h1 = net.get("h1")
    h2 = net.get("h2")

    # Give the Ryu controller time to connect and install the initial path.
    info("*** Waiting for controller...\n")
    time.sleep(5)

    # Start UDP iperf server once.
    info("*** Starting iperf UDP server on h2\n")
    h2.cmd("pkill -f 'iperf -s' || true")
    h2.cmd("iperf -s -u > /tmp/iperf_server.log 2>&1 &")
    time.sleep(2)

    fieldnames = [
        "timestamp",
        "requested_rate_mbps",
        "measured_bandwidth_mbps",
        "packet_loss_percent",
        "average_ping_ms",
        "old_path",
        "new_path",
        "reroute_improvement",
    ]

    with open(RESULT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rate in rates:
            info("\n")
            info("=" * 60 + "\n")
            info(f"*** EXPERIMENT: {rate} Mbps UDP\n")
            info("=" * 60 + "\n")

            # Clear previous iperf client/server state.
            h1.cmd("pkill -f 'iperf -c' || true")

            # Warm-up / settle.
            time.sleep(3)

            # Ping while the requested load is running.
            ping_proc = h1.popen(
                f"ping -c {PING_COUNT} 10.0.0.2",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            iperf_output = h1.cmd(
                f"iperf -c 10.0.0.2 -u -b {rate}M -t {TRAFFIC_SECONDS}"
            )

            ping_output = ping_proc.communicate(timeout=15)[0]

            measured = parse_iperf(iperf_output)
            loss, avg_ping = parse_ping(ping_output)

            old_path, new_path, improvement = read_latest_controller_event()

            writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "requested_rate_mbps": rate,
                "measured_bandwidth_mbps": measured,
                "packet_loss_percent": loss,
                "average_ping_ms": avg_ping,
                "old_path": old_path,
                "new_path": new_path,
                "reroute_improvement": improvement,
            })
            f.flush()

            info(f"Requested traffic : {rate} Mbps\n")
            info(f"Measured bandwidth: {measured}\n")
            info(f"Packet loss       : {loss}%\n")
            info(f"Average ping      : {avg_ping} ms\n")

            if new_path:
                info(f"Latest reroute    : {old_path} -> {new_path}\n")

            # Let controller statistics settle before next experiment.
            time.sleep(5)

    h1.cmd("pkill -f 'iperf -c' || true")
    h2.cmd("pkill -f 'iperf -s' || true")

    info("\n*** Stopping network\n")
    net.stop()

    info(f"\n*** Results saved to:\n{RESULT_FILE}\n")


if __name__ == "__main__":
    setLogLevel("info")
    main()
