from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    MAIN_DISPATCHER,
    DEAD_DISPATCHER,
    CONFIG_DISPATCHER,
    set_ev_cls
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib.packet import arp
from ryu.lib.packet import ipv4

import joblib
import os
import time
import csv
from datetime import datetime
import pandas as pd
from gnn_live_predictor import LiveGNNPredictor


class AITrafficEngineering(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(self, *args, **kwargs):

        super(AITrafficEngineering, self).__init__(
            *args,
            **kwargs
        )

        # ======================================================
        # DATAPATHS / SWITCHES
        # ======================================================

        self.routing_mode = os.getenv("SDN_ROUTING_MODE", "ai").strip().lower()
        if self.routing_mode not in ("ai", "baseline"):
            self.routing_mode = "ai"

        self.datapaths = {}

        # ======================================================
        # MAC LEARNING
        # ======================================================

        self.mac_to_port = {}

        # ======================================================
        # PREVIOUS PORT STATISTICS
        # ======================================================

        self.previous_stats = {}

        # ======================================================
        # CURRENT UTILIZATION
        # ======================================================

        self.port_utilization = {}

        # ======================================================
        # ML PREDICTION
        # ======================================================

        self.port_prediction = {}

        self.port_probability = {}

        # ======================================================
        # LOAD TRAINED ML MODEL
        # ======================================================

        model_path = os.path.expanduser(
            "~/sdn-project/congestion_model.pkl"
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                "ML model not found: "
                + model_path
            )

        loaded_model = joblib.load(
            model_path
        )

        # ------------------------------------------------------
        # Support dictionary-based saved model
        # ------------------------------------------------------

        if isinstance(
            loaded_model,
            dict
        ):

            self.model = loaded_model.get(
                "model"
            )

            self.features = loaded_model.get(
                "features"
            )

        else:

            self.model = loaded_model

            # Fallback feature order
            self.features = [
                "rx_packets",
                "tx_packets",
                "rx_bytes",
                "tx_bytes",
                "rx_mbps",
                "tx_mbps",
                "utilization_percent"
            ]

        if self.model is None:

            raise ValueError(
                "Could not load ML classifier"
            )

        if not self.features:

            self.features = [
                "rx_packets",
                "tx_packets",
                "rx_bytes",
                "tx_bytes",
                "rx_mbps",
                "tx_mbps",
                "utilization_percent"
            ]

        self.logger.info("[AI-TE] Routing mode: %s", self.routing_mode.upper())
        self.logger.info("[AI-TE] ML model loaded successfully")

        # GNN-assisted routing state. Forecasts are used only while fresh.
        # If GNN inference is unavailable/stale, path scoring falls back to ML.
        self.gnn_predicted_utilization = {}
        self.gnn_forecast_timestamp = 0.0
        self.gnn_forecast_max_age_seconds = 12.0
        self.gnn_predictor = None
        try:
            self.gnn_predictor = LiveGNNPredictor(
                model_path="~/sdn-project/gnn_traffic_forecaster.joblib",
                csv_path="~/sdn-project/gnn_live_predictions.csv",
                max_snapshot_skew_seconds=3.0,
            )
            self.logger.info(
                "[GNN] Forecaster loaded successfully (shadow mode)"
            )
        except Exception:
            self.logger.exception(
                "[GNN] Initialization failed; existing routing unchanged"
            )

        self.logger.info(
            "[AI-TE] Model features: %s",
            self.features
        )

        # ======================================================
        # NETWORK TOPOLOGY
        # ======================================================
        #
        #                    s2
        #                   /  \
        #                  /    \
        #                 /      \
        #                /        \
        #               s1        s4
        #                \        /
        #                 \      /
        #                  \    /
        #                   \  /
        #                    s3
        #
        # h1 -- s1
        # h2 -- s4
        #
        # PATH_1:
        # s1 -> s2 -> s4
        #
        # PATH_2:
        # s1 -> s3 -> s4
        #
        # ======================================================

        self.paths = {
            "PATH_1": [
                (1, 2),   # s1 -> s2
                (2, 2),   # s2 -> s4
                (4, 3)    # s4 -> h2
            ],
            "PATH_2": [
                (1, 3),   # s1 -> s3
                (3, 2),   # s3 -> s4
                (4, 3)    # s4 -> h2
            ]
        }

        # REVERSE PATHS
        self.reverse_paths = {
            "PATH_1": [
                (4, 1),   # s4 -> s2
                (2, 1),   # s2 -> s1
                (1, 1)    # s1 -> h1
            ],
            "PATH_2": [
                (4, 2),   # s4 -> s3
                (3, 1),   # s3 -> s1
                (1, 1)    # s1 -> h1
            ]
        }

        # ======================================================
        # CURRENT ACTIVE PATH
        # ======================================================

        self.active_path = "PATH_1"

        # ======================================================
        # ANTI-FLAPPING / STABLE ROUTING
        # ======================================================

        # Alternate path must improve the ML score
        # by at least this amount before rerouting.

        self.path_change_threshold = 0.75

        # Candidate path waiting for confirmation

        self.pending_path = None

        # Number of consecutive confirmations

        self.pending_count = 0

        # Required consecutive confirmations

        self.required_confirmations = 2

        # Cooldown to prevent route flapping
        self.reroute_cooldown_seconds = 30
        self.last_reroute_time = 0.0

        # ======================================================
        # CSV LOGGING
        # ======================================================

        self.log_directory = os.path.expanduser(
            "~/sdn-project"
        )

        self.traffic_log_file = os.path.join(
            self.log_directory,
            "traffic_logs.csv"
        )

        self.rerouting_log_file = os.path.join(
            self.log_directory,
            "rerouting_events.csv"
        )

        self._initialize_csv_logs()

        # ======================================================
        # START MONITORING THREAD
        # ======================================================

        self.monitor_thread = hub.spawn(
            self._monitor
        )

    # ==========================================================
    # CSV LOGGING HELPERS
    # ==========================================================

    def _initialize_csv_logs(self):

        os.makedirs(
            self.log_directory,
            exist_ok=True
        )

        traffic_headers = [
            "timestamp",
            "switch",
            "port",
            "rx_mbps",
            "tx_mbps",
            "utilization_percent",
            "prediction",
            "confidence",
            "path1_score",
            "path2_score",
            "preferred_path",
            "active_path"
        ]

        rerouting_headers = [
            "timestamp",
            "old_path",
            "new_path",
            "path1_score",
            "path2_score",
            "improvement",
            "confirmation_count"
        ]

        if not os.path.exists(self.traffic_log_file):
            with open(
                self.traffic_log_file,
                "w",
                newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerow(traffic_headers)

        if not os.path.exists(self.rerouting_log_file):
            with open(
                self.rerouting_log_file,
                "w",
                newline=""
            ) as file:
                writer = csv.writer(file)
                writer.writerow(rerouting_headers)

        self.logger.info(
            "[AI-TE] CSV logging enabled: %s",
            self.traffic_log_file
        )

    def _log_traffic_row(
        self,
        dpid,
        port,
        rx_mbps,
        tx_mbps,
        utilization,
        prediction,
        confidence
    ):

        path1_score = self.path_score(
            self.paths["PATH_1"]
        )

        path2_score = self.path_score(
            self.paths["PATH_2"]
        )

        if path1_score < path2_score:
            preferred_path = "PATH_1"
        elif path2_score < path1_score:
            preferred_path = "PATH_2"
        else:
            preferred_path = self.active_path

        with open(
            self.traffic_log_file,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                f"s{dpid}",
                port,
                round(rx_mbps, 3),
                round(tx_mbps, 3),
                round(utilization, 3),
                prediction,
                round(confidence, 4),
                round(path1_score, 3),
                round(path2_score, 3),
                preferred_path,
                self.active_path
            ])

    def _log_rerouting_event(
        self,
        old_path,
        new_path,
        path1_score,
        path2_score,
        improvement
    ):

        with open(
            self.rerouting_log_file,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                old_path,
                new_path,
                round(path1_score, 3),
                round(path2_score, 3),
                round(improvement, 3),
                self.required_confirmations
            ])

    # ==========================================================
    # SWITCH STATE CHANGE
    # ==========================================================

    @set_ev_cls(
        ofp_event.EventOFPStateChange,
        [MAIN_DISPATCHER, DEAD_DISPATCHER]
    )
    def state_change(
        self,
        ev
    ):

        datapath = ev.datapath

        if ev.state == MAIN_DISPATCHER:

            if datapath.id not in self.datapaths:

                self.datapaths[
                    datapath.id
                ] = datapath

                self.mac_to_port.setdefault(
                    datapath.id,
                    {}
                )

                self.logger.info(
                    "[AI-TE] Switch connected: s%d",
                    datapath.id
                )

                # --------------------------------------------------
                # Install initial path after all switches connect
                # --------------------------------------------------

                if len(self.datapaths) == 4:

                    hub.spawn(
                        self._initial_path_setup
                    )

        elif ev.state == DEAD_DISPATCHER:

            if datapath.id in self.datapaths:

                del self.datapaths[
                    datapath.id
                ]

                self.logger.info(
                    "[AI-TE] Switch disconnected: s%d",
                    datapath.id
                )

    # ==========================================================
    # INITIAL PATH SETUP
    # ==========================================================

    def _initial_path_setup(self):

        hub.sleep(2)

        if len(self.datapaths) < 4:

            return

        self.logger.info(
            "[AI-TE] Initial active path: %s",
            self.active_path
        )

        self.install_selected_path()

    # ==========================================================
    # SWITCH FEATURES
    # ==========================================================

    @set_ev_cls(
        ofp_event.EventOFPSwitchFeatures,
        CONFIG_DISPATCHER
    )
    def switch_features_handler(
        self,
        ev
    ):

        datapath = ev.msg.datapath

        parser = datapath.ofproto_parser

        ofproto = datapath.ofproto

        # ------------------------------------------------------
        # Table miss -> controller
        # ------------------------------------------------------

        match = parser.OFPMatch()

        actions = [

            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]

        self.add_flow(
            datapath,
            0,
            match,
            actions
        )

        self.logger.info(
            "[AI-TE] Table-miss installed on s%d",
            datapath.id
        )

    # ==========================================================
    # PACKET IN
    # ==========================================================

    @set_ev_cls(
        ofp_event.EventOFPPacketIn,
        MAIN_DISPATCHER
    )
    def packet_in_handler(
        self,
        ev
    ):

        msg = ev.msg

        datapath = msg.datapath

        dpid = datapath.id

        parser = datapath.ofproto_parser

        ofproto = datapath.ofproto

        in_port = msg.match[
            "in_port"
        ]

        pkt = packet.Packet(
            msg.data
        )

        eth = pkt.get_protocol(
            ethernet.ethernet
        )

        if eth is None:

            return

        # ------------------------------------------------------
        # Ignore LLDP
        # ------------------------------------------------------

        if eth.ethertype == (
            ether_types.ETH_TYPE_LLDP
        ):

            return

        src = eth.src

        dst = eth.dst

        # ------------------------------------------------------
        # MAC learning
        # ------------------------------------------------------

        self.mac_to_port.setdefault(
            dpid,
            {}
        )

        self.mac_to_port[
            dpid
        ][src] = in_port

        # ======================================================
        # ARP
        # ======================================================

        if eth.ethertype == (
            ether_types.ETH_TYPE_ARP
        ):

            self._handle_arp(
                datapath,
                msg,
                pkt,
                in_port
            )

            return

        # ======================================================
        # IPv4
        # ======================================================

        if eth.ethertype == (
            ether_types.ETH_TYPE_IP
        ):

            self._handle_ipv4(
                datapath,
                msg,
                pkt,
                in_port
            )

            return

        # ======================================================
        # OTHER ETHERNET TRAFFIC
        # ======================================================

        out_port = self.mac_to_port[
            dpid
        ].get(
            dst,
            ofproto.OFPP_FLOOD
        )

        actions = [

            parser.OFPActionOutput(
                out_port
            )
        ]

        self._send_packet_out(
            datapath,
            msg,
            in_port,
            actions
        )

    # ==========================================================
    # ARP HANDLER
    # ==========================================================

    def _handle_arp(
        self,
        datapath,
        msg,
        pkt,
        in_port
    ):

        dpid = datapath.id

        parser = datapath.ofproto_parser

        ofproto = datapath.ofproto

        arp_packet = pkt.get_protocol(
            arp.arp
        )

        if arp_packet is None:

            return

        src_ip = arp_packet.src_ip

        # ======================================================
        # H1 -> H2
        # ======================================================

        if src_ip == "10.0.0.1":

            if dpid == 1:

                if self.active_path == "PATH_1":

                    out_port = 2

                else:

                    out_port = 3

            elif dpid == 2:

                out_port = 2

            elif dpid == 3:

                out_port = 2

            elif dpid == 4:

                out_port = 1

            else:

                out_port = (
                    self.mac_to_port[
                        dpid
                    ].get(
                        arp_packet.dst_mac,
                        ofproto.OFPP_FLOOD
                    )
                )

        # ======================================================
        # H2 -> H1
        # ======================================================

        elif src_ip == "10.0.0.2":

            if dpid == 4:

                if self.active_path == "PATH_1":

                    out_port = 2

                else:

                    out_port = 3

            elif dpid == 2:

                out_port = 1

            elif dpid == 3:

                out_port = 1

            elif dpid == 1:

                out_port = 1

            else:

                out_port = (
                    self.mac_to_port[
                        dpid
                    ].get(
                        arp_packet.dst_mac,
                        ofproto.OFPP_FLOOD
                    )
                )

        else:

            out_port = (
                self.mac_to_port[
                    dpid
                ].get(
                    arp_packet.dst_mac,
                    ofproto.OFPP_FLOOD
                )
            )

        actions = [

            parser.OFPActionOutput(
                out_port
            )
        ]

        self._send_packet_out(
            datapath,
            msg,
            in_port,
            actions
        )

    # ==========================================================
    # IPv4 HANDLER
    # ==========================================================

    def _handle_ipv4(
        self,
        datapath,
        msg,
        pkt,
        in_port
    ):

        dpid = datapath.id

        parser = datapath.ofproto_parser

        ofproto = datapath.ofproto

        ip_packet = pkt.get_protocol(
            ipv4.ipv4
        )

        if ip_packet is None:

            return

        src_ip = ip_packet.src

        dst_ip = ip_packet.dst

        # ======================================================
        # H1 -> H2
        # ======================================================

        if (
            src_ip == "10.0.0.1"
            and
            dst_ip == "10.0.0.2"
        ):

            if dpid == 1:

                if self.active_path == "PATH_1":

                    out_port = 2

                else:

                    out_port = 3

            elif dpid == 2:

                out_port = 2

            elif dpid == 3:

                out_port = 2

            elif dpid == 4:

                out_port = 1

            else:

                out_port = ofproto.OFPP_FLOOD

        # ======================================================
        # H2 -> H1
        # ======================================================

        elif (
            src_ip == "10.0.0.2"
            and
            dst_ip == "10.0.0.1"
        ):

            if dpid == 4:

                if self.active_path == "PATH_1":

                    out_port = 2

                else:

                    out_port = 3

            elif dpid == 2:

                out_port = 1

            elif dpid == 3:

                out_port = 1

            elif dpid == 1:

                out_port = 1

            else:

                out_port = ofproto.OFPP_FLOOD

        # ======================================================
        # OTHER IP TRAFFIC
        # ======================================================

        else:

            out_port = (
                self.mac_to_port[
                    dpid
                ].get(
                    self._destination_mac(
                        dst_ip
                    ),
                    ofproto.OFPP_FLOOD
                )
            )

        actions = [

            parser.OFPActionOutput(
                out_port
            )
        ]

        self._send_packet_out(
            datapath,
            msg,
            in_port,
            actions
        )

    # ==========================================================
    # DESTINATION MAC
    # ==========================================================

    def _destination_mac(
        self,
        dst_ip
    ):

        if dst_ip == "10.0.0.1":

            return "00:00:00:00:00:01"

        if dst_ip == "10.0.0.2":

            return "00:00:00:00:00:02"

        return ""

    # ==========================================================
    # SEND PACKET OUT
    # ==========================================================

    def _send_packet_out(
        self,
        datapath,
        msg,
        in_port,
        actions
    ):

        parser = datapath.ofproto_parser

        ofproto = datapath.ofproto

        data = None

        if msg.buffer_id == (
            ofproto.OFP_NO_BUFFER
        ):

            data = msg.data

        out = parser.OFPPacketOut(

            datapath=datapath,

            buffer_id=msg.buffer_id,

            in_port=in_port,

            actions=actions,

            data=data
        )

        datapath.send_msg(
            out
        )

    # ==========================================================
    # MONITOR THREAD
    # ==========================================================

    def _monitor(self):

        while True:

            # --------------------------------------------------
            # Request statistics
            # --------------------------------------------------

            for datapath in list(
                self.datapaths.values()
            ):

                self.request_stats(
                    datapath
                )

            # --------------------------------------------------
            # Evaluate paths
            # --------------------------------------------------

            if len(self.datapaths) == 4:

                self.evaluate_paths()

            # --------------------------------------------------
            # Wait
            # --------------------------------------------------

            hub.sleep(5)

    # ==========================================================
    # REQUEST PORT STATISTICS
    # ==========================================================

    def request_stats(
        self,
        datapath
    ):

        parser = datapath.ofproto_parser

        ofproto = datapath.ofproto

        request = parser.OFPPortStatsRequest(

            datapath,

            0,

            ofproto.OFPP_ANY
        )

        datapath.send_msg(
            request
        )

    # ==========================================================
    # PORT STATISTICS REPLY
    # ==========================================================

    @set_ev_cls(
        ofp_event.EventOFPPortStatsReply,
        MAIN_DISPATCHER
    )
    def stats_reply(
        self,
        ev
    ):

        datapath = ev.msg.datapath

        dpid = datapath.id

        now = time.time()

        # Aggregate port telemetry to match GNN training data.
        switch_rx_mbps = []
        switch_tx_mbps = []
        switch_utilizations = []

        for stat in ev.msg.body:

            port = stat.port_no

            if port == (
                datapath.ofproto.OFPP_LOCAL
            ):

                continue

            # --------------------------------------------------
            # Raw statistics
            # --------------------------------------------------

            rx_bytes = stat.rx_bytes

            tx_bytes = stat.tx_bytes

            rx_packets = stat.rx_packets

            tx_packets = stat.tx_packets

            key = (
                dpid,
                port
            )

            # --------------------------------------------------
            # Current rates
            # --------------------------------------------------

            rx_mbps = 0.0

            tx_mbps = 0.0

            if key in self.previous_stats:

                old = self.previous_stats[
                    key
                ]

                elapsed = (
                    now
                    -
                    old["time"]
                )

                if elapsed > 0:

                    rx_diff = max(
                        0,
                        rx_bytes
                        -
                        old["rx_bytes"]
                    )

                    tx_diff = max(
                        0,
                        tx_bytes
                        -
                        old["tx_bytes"]
                    )

                    rx_mbps = (
                        rx_diff
                        *
                        8
                        /
                        elapsed
                        /
                        1_000_000
                    )

                    tx_mbps = (
                        tx_diff
                        *
                        8
                        /
                        elapsed
                        /
                        1_000_000
                    )

            # --------------------------------------------------
            # Link utilization
            #
            # Mininet virtual links are treated as 10 Mbps.
            # --------------------------------------------------

            utilization = (

                max(
                    rx_mbps,
                    tx_mbps
                )

                /

                10.0

            ) * 100.0

            utilization = min(
                max(
                    utilization,
                    0.0
                ),
                100.0
            )

            self.port_utilization[
                key
            ] = utilization

            switch_rx_mbps.append(rx_mbps)
            switch_tx_mbps.append(tx_mbps)
            switch_utilizations.append(utilization)

            # ==================================================
            # PREPARE ML FEATURES
            # ==================================================

            values = {

                "rx_packets":
                    rx_packets,

                "tx_packets":
                    tx_packets,

                "rx_bytes":
                    rx_bytes,

                "tx_bytes":
                    tx_bytes,

                "rx_mbps":
                    rx_mbps,

                "tx_mbps":
                    tx_mbps,

                "utilization_percent":
                    utilization
            }

            # --------------------------------------------------
            # Build DataFrame using EXACT training features
            # --------------------------------------------------

            row = {}

            for feature in self.features:

                row[
                    feature
                ] = values.get(
                    feature,
                    0
                )

            sample = pd.DataFrame(

                [row],

                columns=self.features
            )

            # ==================================================
            # ML PREDICTION
            # ==================================================

            try:

                prediction = self.model.predict(
                    sample
                )[0]

                probabilities = (
                    self.model.predict_proba(
                        sample
                    )[0]
                )

                confidence = float(
                    max(
                        probabilities
                    )
                )

            except Exception as e:

                self.logger.error(
                    "[AI-TE] ML prediction error "
                    "on s%d p%d: %s",
                    dpid,
                    port,
                    str(e)
                )

                continue

            # --------------------------------------------------
            # Save prediction
            # --------------------------------------------------

            self.port_prediction[
                key
            ] = prediction

            self.port_probability[
                key
            ] = confidence

            # ==================================================
            # LIVE AI OUTPUT
            # ==================================================

            self.logger.info(

                "[AI] s%d p%d | "
                "RX %.2f Mbps | "
                "TX %.2f Mbps | "
                "Util %.2f%% | "
                "Prediction=%s | "
                "Confidence=%.2f",

                dpid,

                port,

                rx_mbps,

                tx_mbps,

                utilization,

                prediction,

                confidence
            )

            # --------------------------------------------------
            # Save traffic sample to CSV
            # --------------------------------------------------

            self._log_traffic_row(
                dpid,
                port,
                rx_mbps,
                tx_mbps,
                utilization,
                prediction,
                confidence
            )

            # --------------------------------------------------
            # Save current statistics
            # --------------------------------------------------

            self.previous_stats[
                key
            ] = {

                "rx_bytes":
                    rx_bytes,

                "tx_bytes":
                    tx_bytes,

                "time":
                    now
            }

    # ==========================================================
        # GNN next-observation forecast (shadow mode only).
        if self.gnn_predictor is not None and switch_utilizations:
            try:
                result = self.gnn_predictor.update_switch(
                    switch=f"s{dpid}",
                    rx_mbps=float(
                        sum(switch_rx_mbps) / len(switch_rx_mbps)
                    ),
                    tx_mbps=float(
                        sum(switch_tx_mbps) / len(switch_tx_mbps)
                    ),
                    utilization_percent=float(
                        sum(switch_utilizations) /
                        len(switch_utilizations)
                    ),
                    timestamp=now,
                )
                if result is not None:
                    # Only complete, aligned snapshots are returned by the helper.
                    self.gnn_predicted_utilization = {
                        str(name): float(value)
                        for name, value in result.items()
                        if 0.0 <= float(value) <= 100.0
                    }
                    self.gnn_forecast_timestamp = time.time()
                    self.logger.info(
                        "[GNN] Fresh utilization forecast saved for routing: %s",
                        {
                            name: round(value, 2)
                            for name, value in self.gnn_predicted_utilization.items()
                        }
                    )
            except Exception:
                self.logger.exception(
                    "[GNN] Prediction failed; routing falls back to classifier scores"
                )

    # ML PATH SCORE
    # ==========================================================

    def path_score(
        self,
        path
    ):
        # GNN_ROUTING_INTEGRATION_V1
        """Hybrid path cost: classifier congestion + fresh GNN forecast.

        Classifier cost preserves existing severity/confidence behavior.
        GNN cost adds predicted utilization (0..100%) scaled to 0..2 per
        distinct switch. Stale/unavailable forecasts are ignored safely.
        """
        score = 0.0
        classifier_score = 0.0
        gnn_score = 0.0
        seen_switches = set()

        gnn_fresh = (
            self.gnn_forecast_timestamp > 0.0
            and time.time() - self.gnn_forecast_timestamp
            <= self.gnn_forecast_max_age_seconds
        )

        for switch_id, port in path:
            key = (switch_id, port)
            prediction = self.port_prediction.get(key, "NORMAL")
            confidence = float(self.port_probability.get(key, 0.0))

            if prediction == "NORMAL":
                severity = 0.0
            elif prediction == "MEDIUM":
                severity = 1.0
            elif prediction == "CONGESTED":
                severity = 2.0
            else:
                severity = 0.0

            # Existing classifier score remains active.
            classifier_component = severity * confidence
            classifier_score += classifier_component
            score += classifier_component

            # Add each switch's GNN forecast once per path (not once per port).
            if gnn_fresh and switch_id not in seen_switches:
                seen_switches.add(switch_id)
                predicted_util = self.gnn_predicted_utilization.get(f"s{switch_id}")
                if predicted_util is not None:
                    # 0..100% becomes a 0..2 cost, comparable to classifier severity.
                    gnn_component = max(0.0, min(100.0, float(predicted_util))) / 50.0
                    gnn_score += gnn_component
                    score += gnn_component

        self.logger.info(
            "[GNN-SCORE] path=%s classifier=%.3f gnn=%.3f total=%.3f fresh=%s",
            "PATH_1" if path == self.paths.get("PATH_1") else "PATH_2",
            classifier_score, gnn_score, score, gnn_fresh
        )
        return score

    # ==========================================================
    # EVALUATE BOTH PATHS
    # ==========================================================

    def evaluate_paths(
        self
    ):

        path1_score = self.path_score(
            self.paths["PATH_1"]
        )

        path2_score = self.path_score(
            self.paths["PATH_2"]
        )

        # ======================================================
        # SHOW ML PATH SCORES
        # ======================================================

        self.logger.info(

            "[AI-TE] PATH_1 hybrid score = %.3f | "
            "PATH_2 hybrid score = %.3f",

            path1_score,

            path2_score
        )

        # ======================================================
        # SELECT LOWER-CONGESTION PATH
        # ======================================================

        # Fixed-route baseline: always use PATH_1, without ML rerouting.
        if self.routing_mode == "baseline":
            self.pending_path = None
            self.pending_count = 0

            if self.active_path != "PATH_1":
                old_path = self.active_path
                self.active_path = "PATH_1"
                self.logger.warning(
                    "[AI-TE] BASELINE FIXED ROUTE: %s -> PATH_1",
                    old_path
                )
                self.install_selected_path()
            else:
                self.logger.info(
                    "[AI-TE] BASELINE FIXED ROUTE: PATH_1 remains active"
                )
            return

        if path1_score < path2_score:

            selected = "PATH_1"

        elif path2_score < path1_score:

            selected = "PATH_2"

        else:

            selected = self.active_path

        self.logger.info(

            "[AI-TE] Hybrid preferred path: %s",

            selected
        )

        # ======================================================
        # SAME PATH
        # ======================================================

        if selected == self.active_path:

            self.pending_path = None

            self.pending_count = 0

            self.logger.info(

                "[AI-TE] Current path remains: %s",

                self.active_path
            )

            return

        # ======================================================
        # CURRENT PATH SCORE
        # ======================================================

        if self.active_path == "PATH_1":

            current_score = path1_score

        else:

            current_score = path2_score

        # ======================================================
        # SELECTED PATH SCORE
        # ======================================================

        if selected == "PATH_1":

            selected_score = path1_score

        else:

            selected_score = path2_score

        # ======================================================
        # IMPROVEMENT
        # ======================================================

        improvement = (

            current_score
            -
            selected_score

        )

        self.logger.info(

            "[AI-TE] Potential hybrid-score improvement = %.3f",

            improvement
        )

        # ======================================================
        # SMALL IMPROVEMENT
        # ======================================================

        if (
            improvement
            <
            self.path_change_threshold
        ):

            self.pending_path = None

            self.pending_count = 0

            self.logger.info(

                "[AI-TE] No reroute: "
                "hybrid-score improvement below threshold"
            )

            return

        # ======================================================
        # REROUTE COOLDOWN / ANTI-FLAPPING
        # ======================================================

        now = time.time()
        elapsed = now - self.last_reroute_time

        if elapsed < self.reroute_cooldown_seconds:
            self.pending_path = None
            self.pending_count = 0
            self.logger.info(
                "[AI-TE] Reroute skipped: cooldown active "
                "(%.1fs remaining)",
                self.reroute_cooldown_seconds - elapsed
            )
            return

        # CONFIRM CANDIDATE PATH
        # ======================================================

        if self.pending_path == selected:

            self.pending_count += 1

        else:

            self.pending_path = selected

            self.pending_count = 1

        self.logger.info(

            "[AI-TE] Reroute candidate=%s | "
            "confirmation=%d/%d",

            selected,

            self.pending_count,

            self.required_confirmations
        )

        # ======================================================
        # CONFIRMED REROUTING
        # ======================================================

        if (
            self.pending_count
            >=
            self.required_confirmations
        ):

            old_path = self.active_path

            self.active_path = selected
            self.last_reroute_time = time.time()

            self.pending_path = None

            self.pending_count = 0

            self.logger.warning(

                "[AI-TE] HYBRID REROUTING: "
                "%s -> %s",

                old_path,

                selected
            )

            self.install_selected_path()

            self._log_rerouting_event(
                old_path,
                selected,
                path1_score,
                path2_score,
                improvement
            )

    # ==========================================================
    # INSTALL SELECTED PATH
    # ==========================================================

    def install_selected_path(
        self
    ):

        path = self.paths[
            self.active_path
        ]

        reverse_path = self.reverse_paths[
            self.active_path
        ]

        self.logger.info(

            "[AI-TE] Installing %s",

            self.active_path
        )

        # ======================================================
        # REMOVE OLD PRIORITY-100 PATH FLOWS
        # ======================================================

        self._remove_path_flows()

        # ======================================================
        # FORWARD PATH
        # ======================================================

        for switch_id, out_port in path:

            if switch_id not in self.datapaths:

                continue

            datapath = self.datapaths[
                switch_id
            ]

            parser = (
                datapath.ofproto_parser
            )

            # --------------------------------------------------
            # H1 -> H2
            # --------------------------------------------------

            match = parser.OFPMatch(

                eth_type=0x0800,

                ipv4_src="10.0.0.1",

                ipv4_dst="10.0.0.2"
            )

            actions = [

                parser.OFPActionOutput(
                    out_port
                )
            ]

            self.add_flow(

                datapath,

                100,

                match,

                actions
            )

        # ======================================================
        # REVERSE PATH
        # ======================================================

        for switch_id, out_port in reverse_path:

            if switch_id not in self.datapaths:

                continue

            datapath = self.datapaths[
                switch_id
            ]

            parser = (
                datapath.ofproto_parser
            )

            # --------------------------------------------------
            # H2 -> H1
            # --------------------------------------------------

            match = parser.OFPMatch(

                eth_type=0x0800,

                ipv4_src="10.0.0.2",

                ipv4_dst="10.0.0.1"
            )

            actions = [

                parser.OFPActionOutput(
                    out_port
                )
            ]

            self.add_flow(

                datapath,

                100,

                match,

                actions
            )

        self.logger.info(

            "[AI-TE] Flow programming requested for %s | forward=%s | reverse=%s",

            self.active_path, path, reverse_path
        )

        self.logger.info(

            "[AI-TE] Installed %s successfully (FlowMods sent; verify OVS flow counters)",

            self.active_path
        )

    def _remove_path_flows(
        self
    ):

        """Remove old priority-100 IPv4 path flows from all switches."""

        for datapath in list(
            self.datapaths.values()
        ):

            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto

            for src_ip, dst_ip in [
                ("10.0.0.1", "10.0.0.2"),
                ("10.0.0.2", "10.0.0.1")
            ]:

                match = parser.OFPMatch(
                    eth_type=0x0800,
                    ipv4_src=src_ip,
                    ipv4_dst=dst_ip
                )

                flow_mod = parser.OFPFlowMod(
                    datapath=datapath,
                    command=ofproto.OFPFC_DELETE,
                    out_port=ofproto.OFPP_ANY,
                    out_group=ofproto.OFPG_ANY,
                    priority=100,
                    match=match
                )

                datapath.send_msg(
                    flow_mod
                )

        self.logger.info(
            "[AI-TE] Old path flows cleared"
        )

    # ==========================================================
    # ADD FLOW
    # ==========================================================

    def add_flow(
        self,
        datapath,
        priority,
        match,
        actions
    ):

        parser = (
            datapath.ofproto_parser
        )

        ofproto = datapath.ofproto

        instructions = [

            parser.OFPInstructionActions(

                ofproto.OFPIT_APPLY_ACTIONS,

                actions
            )
        ]

        flow_mod = parser.OFPFlowMod(

            datapath=datapath,

            priority=priority,

            match=match,

            instructions=instructions
        )

        datapath.send_msg(
            flow_mod
        )
