import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yaml
import os
import time
import random
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #080c14;
        color: #c9d8f0;
    }

    .stApp { background-color: #080c14; }

    .metric-card {
        background: linear-gradient(135deg, #0d1726 0%, #111d30 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #5a7fa8;
        margin-top: 6px;
    }

    .risk-LOW    { color: #2ecc71; }
    .risk-MEDIUM { color: #f39c12; }
    .risk-HIGH   { color: #e74c3c; }
    .risk-CRITICAL { color: #ff0040; text-shadow: 0 0 20px #ff004088; }

    .alert-box {
        background: linear-gradient(135deg, #1a0a0a, #2a0f0f);
        border-left: 4px solid #e74c3c;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    .normal-box {
        background: linear-gradient(135deg, #0a1a0f, #0f2a15);
        border-left: 4px solid #2ecc71;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    .header-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #4fc3f7, #2196f3, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .header-sub {
        color: #5a7fa8;
        font-size: 0.85rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .score-bar-container {
        background: #0d1726;
        border-radius: 6px;
        height: 10px;
        margin: 8px 0;
        overflow: hidden;
    }

    div[data-testid="stSidebar"] {
        background-color: #0a1020;
        border-right: 1px solid #1e3a5f;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        letter-spacing: 0.05em;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1976d2, #1565c0);
        transform: translateY(-1px);
    }

    hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_artifacts():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_path = config["data"]["model_path"]
    encoder_path = config["data"]["encoder_path"]

    if not os.path.exists(model_path):
        return None, None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        artifacts = pickle.load(f)

    return model, artifacts, config


def score_flow(model, artifacts, config, flow_dict: dict) -> dict:
    df = pd.DataFrame([flow_dict])

    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]
    medians = artifacts["medians"]

    for col in ["proto", "service", "state"]:
        if col in df.columns:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    num_cols = [c for c in config["features"]["numeric_columns"] if c in df.columns]
    for col in num_cols:
        df[col] = df[col].fillna(medians.get(col, 0))
    df[num_cols] = scaler.transform(df[num_cols])

    total_bytes = df["sbytes"] + df["dbytes"]
    df["byte_ratio"] = (df["sbytes"] / total_bytes.replace(0, np.nan)).fillna(0.5)
    total_pkts = df["spkts"] + df["dpkts"]
    df["packet_ratio"] = (df["spkts"] / total_pkts.replace(0, np.nan)).fillna(0.5)
    df["src_bytes_per_sec"] = (df["sbytes"] / df["dur"].replace(0, np.nan)).fillna(0)
    df["dst_bytes_per_sec"] = (df["dbytes"] / df["dur"].replace(0, np.nan)).fillna(0)
    df["jitter_ratio"] = (df["sjit"] / (df["djit"] + 1e-9)).clip(upper=1000)
    df["ttl_diff"] = (df["sttl"] - df["dttl"]).abs()
    diff = (df["spkts"] - df["dpkts"]).abs()
    total = (df["spkts"] + df["dpkts"]).replace(0, np.nan)
    df["conn_asymmetry"] = (diff / total).fillna(0)

    raw_score = model.score_samples(df)[0]
    anomaly_score = float(-raw_score)
    is_anomaly = model.predict(df)[0] == -1

    if anomaly_score < 0.40:
        risk = "LOW"
    elif anomaly_score < 0.55:
        risk = "MEDIUM"
    elif anomaly_score < 0.70:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {"score": anomaly_score, "is_anomaly": is_anomaly, "risk": risk}


def generate_simulated_flow(attack: bool = False) -> dict:
    if not attack:
        return {
            "dur": round(random.uniform(0.01, 5.0), 4),
            "proto": random.choice(["tcp", "tcp", "tcp", "udp"]),
            "service": random.choice(["http", "http", "dns", "ftp", "-"]),
            "state": random.choice(["FIN", "FIN", "CON", "INT"]),
            "spkts": random.randint(1, 20),
            "dpkts": random.randint(1, 20),
            "sbytes": random.randint(100, 5000),
            "dbytes": random.randint(500, 50000),
            "rate": round(random.uniform(0, 100), 2),
            "sttl": random.choice([64, 128]),
            "dttl": random.choice([64, 128]),
            "sload": round(random.uniform(0, 1000), 2),
            "dload": round(random.uniform(0, 5000), 2),
            "sloss": 0, "dloss": 0,
            "sinpkt": round(random.uniform(0, 1), 4),
            "dinpkt": round(random.uniform(0, 1), 4),
            "sjit": round(random.uniform(0, 10), 4),
            "djit": round(random.uniform(0, 10), 4),
            "swin": 255, "stcpb": 0, "dtcpb": 0, "dwin": 255,
            "tcprtt": round(random.uniform(0, 0.1), 4),
            "synack": round(random.uniform(0, 0.05), 4),
            "ackdat": round(random.uniform(0, 0.05), 4),
            "smean": random.randint(50, 500),
            "dmean": random.randint(200, 5000),
            "trans_depth": random.randint(0, 3),
            "response_body_len": random.randint(0, 10000),
            "ct_srv_src": random.randint(1, 10),
            "ct_state_ttl": random.randint(0, 4),
            "ct_dst_ltm": random.randint(1, 10),
            "ct_src_dport_ltm": random.randint(1, 5),
            "ct_dst_sport_ltm": random.randint(1, 5),
            "ct_dst_src_ltm": random.randint(1, 10),
            "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": random.randint(0, 3),
            "ct_src_ltm": random.randint(1, 10), "ct_srv_dst": random.randint(1, 10),
            "is_sm_ips_ports": 0
        }
    else:
        attack_type = random.choice(["portscan", "exfil", "dos"])
        if attack_type == "portscan":
            return {**generate_simulated_flow(False),
                    "spkts": random.randint(200, 1000), "dpkts": 0,
                    "sbytes": random.randint(5000, 20000), "dbytes": 0,
                    "dur": round(random.uniform(0.001, 0.1), 6),
                    "ct_dst_sport_ltm": random.randint(50, 200)}
        elif attack_type == "exfil":
            return {**generate_simulated_flow(False),
                    "sbytes": random.randint(500000, 2000000),
                    "dbytes": random.randint(100, 500),
                    "dur": round(random.uniform(1, 10), 2)}
        else:
            return {**generate_simulated_flow(False),
                    "rate": round(random.uniform(10000, 100000), 2),
                    "spkts": random.randint(1000, 5000),
                    "sbytes": random.randint(1000000, 5000000),
                    "dur": round(random.uniform(0.001, 0.5), 6)}


model, artifacts, config = load_model_and_artifacts()

if "flow_history" not in st.session_state:
    st.session_state.flow_history = []
if "total_scored" not in st.session_state:
    st.session_state.total_scored = 0
if "total_anomalies" not in st.session_state:
    st.session_state.total_anomalies = 0

st.markdown('<div class="header-title">🛡️ Network Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Real-time threat monitoring · Isolation Forest · UNSW-NB15</div>', unsafe_allow_html=True)
st.markdown("---")

if model is None:
    st.error("⚠️ Model not found. Please run `python pipeline.py` then `python train_model.py` first.")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    mode = st.radio("Mode", ["Live Simulation", "Manual Input"], label_visibility="collapsed")
    st.markdown("---")

    if mode == "Live Simulation":
        attack_rate = st.slider("Attack injection rate", 0, 100, 20, format="%d%%")
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        if st.button("▶ Score New Flow"):
            is_attack = random.random() < (attack_rate / 100)
            flow = generate_simulated_flow(attack=is_attack)
            result = score_flow(model, artifacts, config, flow)
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "score": result["score"],
                "is_anomaly": result["is_anomaly"],
                "risk": result["risk"],
                "proto": flow["proto"],
                "service": flow["service"],
                "sbytes": flow["sbytes"],
                "dbytes": flow["dbytes"],
            }
            st.session_state.flow_history.insert(0, entry)
            st.session_state.flow_history = st.session_state.flow_history[:50]
            st.session_state.total_scored += 1
            if result["is_anomaly"]:
                st.session_state.total_anomalies += 1

        if st.button("⚡ Simulate 10 Flows"):
            for _ in range(10):
                is_attack = random.random() < (attack_rate / 100)
                flow = generate_simulated_flow(attack=is_attack)
                result = score_flow(model, artifacts, config, flow)
                entry = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "score": result["score"],
                    "is_anomaly": result["is_anomaly"],
                    "risk": result["risk"],
                    "proto": flow["proto"],
                    "service": flow["service"],
                    "sbytes": flow["sbytes"],
                    "dbytes": flow["dbytes"],
                }
                st.session_state.flow_history.insert(0, entry)
                st.session_state.total_scored += 1
                if result["is_anomaly"]:
                    st.session_state.total_anomalies += 1
            st.session_state.flow_history = st.session_state.flow_history[:50]

        if st.button("🔄 Reset"):
            st.session_state.flow_history = []
            st.session_state.total_scored = 0
            st.session_state.total_anomalies = 0

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown(f"**Algorithm:** Isolation Forest")
    st.markdown(f"**Dataset:** UNSW-NB15")
    st.markdown(f"**Test F1:** 0.7492")
    st.markdown(f"**ROC-AUC:** 0.7601")

col1, col2, col3, col4 = st.columns(4)

total = st.session_state.total_scored
anomalies = st.session_state.total_anomalies
normal = total - anomalies
rate = (anomalies / total * 100) if total > 0 else 0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#4fc3f7">{total}</div>
        <div class="metric-label">Flows Scored</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value risk-CRITICAL">{anomalies}</div>
        <div class="metric-label">Anomalies Detected</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value risk-LOW">{normal}</div>
        <div class="metric-label">Normal Traffic</div>
    </div>""", unsafe_allow_html=True)

with col4:
    rate_color = "risk-LOW" if rate < 20 else "risk-HIGH" if rate < 50 else "risk-CRITICAL"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {rate_color}">{rate:.1f}%</div>
        <div class="metric-label">Anomaly Rate</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

if mode == "Manual Input":
    st.markdown("### 🔍 Score a Network Flow")
    c1, c2, c3 = st.columns(3)
    with c1:
        proto = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
        service = st.selectbox("Service", ["-", "http", "ftp", "dns", "smtp"])
        state = st.selectbox("State", ["FIN", "CON", "INT", "RST"])
        dur = st.number_input("Duration (sec)", value=0.5, min_value=0.0)
    with c2:
        sbytes = st.number_input("Source Bytes", value=500, min_value=0)
        dbytes = st.number_input("Dest Bytes", value=5000, min_value=0)
        spkts = st.number_input("Source Packets", value=5, min_value=0)
        dpkts = st.number_input("Dest Packets", value=5, min_value=0)
    with c3:
        sttl = st.number_input("Source TTL", value=64, min_value=0, max_value=255)
        dttl = st.number_input("Dest TTL", value=64, min_value=0, max_value=255)
        rate = st.number_input("Rate", value=10.0)
        sjit = st.number_input("Source Jitter", value=0.5)

    if st.button("🔎 Analyze This Flow"):
        flow = {
            "dur": dur, "proto": proto, "service": service, "state": state,
            "spkts": int(spkts), "dpkts": int(dpkts),
            "sbytes": int(sbytes), "dbytes": int(dbytes),
            "rate": rate, "sttl": int(sttl), "dttl": int(dttl),
            "sload": 0.0, "dload": 0.0, "sloss": 0, "dloss": 0,
            "sinpkt": 0.0, "dinpkt": 0.0, "sjit": sjit, "djit": 0.0,
            "swin": 255, "stcpb": 0, "dtcpb": 0, "dwin": 255,
            "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0,
            "smean": int(sbytes / max(spkts, 1)), "dmean": int(dbytes / max(dpkts, 1)),
            "trans_depth": 0, "response_body_len": int(dbytes),
            "ct_srv_src": 1, "ct_state_ttl": 0, "ct_dst_ltm": 1,
            "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1,
            "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0,
            "ct_src_ltm": 1, "ct_srv_dst": 1, "is_sm_ips_ports": 0
        }
        result = score_flow(model, artifacts, config, flow)
        score = result["score"]
        risk = result["risk"]

        risk_colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c", "CRITICAL": "#ff0040"}
        color = risk_colors[risk]

        st.markdown(f"""
        <div class="metric-card" style="border-color:{color}; margin-top:20px">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <div>
                    <div class="metric-value" style="color:{color}">{score:.4f}</div>
                    <div class="metric-label">Anomaly Score</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:1.5rem; font-weight:800; color:{color}">{risk}</div>
                    <div class="metric-label">{"⚠️ ANOMALY" if result["is_anomaly"] else "✅ NORMAL"}</div>
                </div>
            </div>
            <div class="score-bar-container" style="margin-top:16px">
                <div style="background:{color}; height:100%; width:{min(score*100, 100):.0f}%; border-radius:6px; transition:width 0.5s"></div>
            </div>
        </div>""", unsafe_allow_html=True)

if st.session_state.flow_history:
    st.markdown("### 📡 Recent Flow Activity")

    scores = [f["score"] for f in st.session_state.flow_history]
    chart_df = pd.DataFrame({
        "Flow": list(range(len(scores), 0, -1)),
        "Anomaly Score": scores
    })
    st.line_chart(chart_df.set_index("Flow"), color="#4fc3f7", height=180)

    st.markdown("### 🚨 Flow Log")
    for entry in st.session_state.flow_history[:15]:
        risk_colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c", "CRITICAL": "#ff0040"}
        color = risk_colors[entry["risk"]]
        box_class = "alert-box" if entry["is_anomaly"] else "normal-box"
        icon = "⚠️" if entry["is_anomaly"] else "✓"
        st.markdown(f"""
        <div class="{box_class}">
            <span style="color:{color}; font-weight:600">{icon} [{entry['time']}]</span>
            &nbsp;&nbsp;
            <span style="color:#8ab4d4">{entry['proto'].upper()}/{entry['service']}</span>
            &nbsp;·&nbsp;
            Score: <span style="color:{color}; font-weight:600">{entry['score']:.4f}</span>
            &nbsp;·&nbsp;
            Risk: <span style="color:{color}; font-weight:600">{entry['risk']}</span>
            &nbsp;·&nbsp;
            ↑{entry['sbytes']:,}B ↓{entry['dbytes']:,}B
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("### 📡 Flow Activity")
    st.info("No flows scored yet. Use the sidebar controls to start scoring traffic.")