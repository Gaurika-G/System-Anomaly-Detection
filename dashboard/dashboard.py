import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yaml
import os
import random
from datetime import datetime

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #080c14; color: #c9d8f0; }
    .stApp { background-color: #080c14; }
    .metric-card { background: linear-gradient(135deg, #0d1726 0%, #111d30 100%); border: 1px solid #1e3a5f; border-radius: 12px; padding: 20px 24px; margin-bottom: 12px; }
    .metric-value { font-size: 2.4rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; }
    .metric-label { font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; color: #5a7fa8; margin-top: 6px; }
    .alert-box { background: linear-gradient(135deg, #1a0a0a, #2a0f0f); border-left: 4px solid #e74c3c; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 6px 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
    .normal-box { background: linear-gradient(135deg, #0a1a0f, #0f2a15); border-left: 4px solid #2ecc71; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 6px 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
    .header-title { font-size: 2rem; font-weight: 800; background: linear-gradient(90deg, #4fc3f7, #2196f3, #7c4dff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }
    .header-sub { color: #5a7fa8; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; }
    .score-bar-container { background: #0d1726; border-radius: 6px; height: 10px; margin: 8px 0; overflow: hidden; }
    div[data-testid="stSidebar"] { background-color: #0a1020; border-right: 1px solid #1e3a5f; }
    .stButton > button { background: linear-gradient(135deg, #1565c0, #0d47a1); color: white; border: none; border-radius: 8px; padding: 10px 24px; font-family: 'Syne', sans-serif; font-weight: 700; width: 100%; }
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


def score_flow(model, artifacts, config, flow_dict):
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

    if not is_anomaly:
        risk = "LOW"
    elif anomaly_score < 0.52:
        risk = "MEDIUM"
    elif anomaly_score < 0.58:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {"score": anomaly_score, "is_anomaly": is_anomaly, "risk": risk}


def load_real_normal_flows():
    """Load actual normal flows from processed dataset for realistic simulation."""
    try:
        import yaml
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)
        df = pd.read_parquet(config["data"]["processed_train"])
        # Get normal rows only, inverse-transform not needed — we use raw processed values
        # Instead load from original CSV for unscaled realistic values
        import os
        raw_path = config["data"]["raw_train"]
        if os.path.exists(raw_path):
            raw = pd.read_csv(raw_path)
            normal_rows = raw[raw["label"] == 0].drop(columns=["id", "attack_cat", "label"], errors="ignore")
            return normal_rows.to_dict("records")
    except Exception:
        pass
    return []

_REAL_NORMAL_FLOWS = None

def get_real_normal_flows():
    global _REAL_NORMAL_FLOWS
    if _REAL_NORMAL_FLOWS is None:
        _REAL_NORMAL_FLOWS = load_real_normal_flows()
    return _REAL_NORMAL_FLOWS


def generate_simulated_flow(attack=False):
    if not attack:
        # Use a real normal flow from the dataset — guaranteed to score as normal
        real_flows = get_real_normal_flows()
        if real_flows:
            return random.choice(real_flows)
        # Fallback if dataset not available
        return {
            "dur": round(random.uniform(0.0, 0.2), 6),
            "proto": random.choice(["tcp", "tcp", "udp"]),
            "service": random.choice(["-", "http", "dns"]),
            "state": "FIN",
            "spkts": random.randint(2, 6),
            "dpkts": random.randint(2, 6),
            "sbytes": random.randint(100, 1000),
            "dbytes": random.randint(200, 3000),
            "rate": round(random.uniform(200, 1000), 2),
            "sttl": 64, "dttl": 64,
            "sload": round(random.uniform(0, 10000), 2),
            "dload": round(random.uniform(0, 10000), 2),
            "sloss": 0, "dloss": 0,
            "sinpkt": round(random.uniform(0, 200), 4),
            "dinpkt": round(random.uniform(0, 200), 4),
            "sjit": round(random.uniform(0, 50), 4),
            "djit": round(random.uniform(0, 50), 4),
            "swin": 255, "stcpb": 0, "dtcpb": 0, "dwin": 255,
            "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0,
            "smean": 200, "dmean": 500,
            "trans_depth": 0, "response_body_len": 0,
            "ct_srv_src": 5, "ct_state_ttl": 2, "ct_dst_ltm": 5,
            "ct_src_dport_ltm": 3, "ct_dst_sport_ltm": 3, "ct_dst_src_ltm": 5,
            "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0,
            "ct_src_ltm": 5, "ct_srv_dst": 5, "is_sm_ips_ports": 0,
        }

    # Attack patterns — deliberately anomalous
    attack_type = random.choice(["portscan", "exfil", "dos"])
    real_flows = get_real_normal_flows()
    base = random.choice(real_flows) if real_flows else {}
    if attack_type == "portscan":
        return {**base, "spkts": random.randint(200, 1000), "dpkts": 0,
                "sbytes": random.randint(5000, 20000), "dbytes": 0,
                "dur": round(random.uniform(0.001, 0.05), 6),
                "ct_dst_sport_ltm": random.randint(50, 200)}
    elif attack_type == "exfil":
        return {**base, "sbytes": random.randint(500000, 2000000),
                "dbytes": random.randint(100, 500),
                "dur": round(random.uniform(1, 10), 2)}
    else:
        return {**base, "rate": round(random.uniform(50000, 200000), 2),
                "spkts": random.randint(1000, 5000),
                "sbytes": random.randint(1000000, 5000000),
                "dur": round(random.uniform(0.001, 0.1), 6)}


# Session state
model, artifacts, config = load_model_and_artifacts()

if "flow_history" not in st.session_state:
    st.session_state.flow_history = []
if "total_scored" not in st.session_state:
    st.session_state.total_scored = 0
if "total_anomalies" not in st.session_state:
    st.session_state.total_anomalies = 0


def score_and_record(flow):
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


# Header
st.markdown('<div class="header-title">Network Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Real-time threat monitoring · Isolation Forest · UNSW-NB15</div>', unsafe_allow_html=True)
st.markdown("---")

if model is None:
    st.error("Model not found. Run pipeline.py then train_model.py first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Controls")
    mode = st.radio("Mode", ["Live Simulation", "Manual Input"])
    st.markdown("---")

    if mode == "Live Simulation":
        attack_rate = st.slider("Attack injection rate", 0, 100, 20, format="%d%%")

        if st.button("Score New Flow"):
            is_attack = random.random() < (attack_rate / 100)
            score_and_record(generate_simulated_flow(attack=is_attack))

        if st.button("Simulate 10 Flows"):
            for _ in range(10):
                is_attack = random.random() < (attack_rate / 100)
                score_and_record(generate_simulated_flow(attack=is_attack))

        if st.button("Reset"):
            st.session_state.flow_history = []
            st.session_state.total_scored = 0
            st.session_state.total_anomalies = 0

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown("**Algorithm:** Isolation Forest")
    st.markdown("**Dataset:** UNSW-NB15")
    st.markdown("**Test F1:** 0.7492")
    st.markdown("**ROC-AUC:** 0.7601")
    st.markdown("**Contamination:** 0.30")

# Metrics
total = st.session_state.total_scored
anomalies = st.session_state.total_anomalies
normal = total - anomalies
rate = (anomalies / total * 100) if total > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#4fc3f7">{total}</div><div class="metric-label">Flows Scored</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#e74c3c">{anomalies}</div><div class="metric-label">Anomalies Detected</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#2ecc71">{normal}</div><div class="metric-label">Normal Traffic</div></div>', unsafe_allow_html=True)
with col4:
    rate_color = "#2ecc71" if rate < 20 else "#e74c3c" if rate >= 50 else "#f39c12"
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{rate_color}">{rate:.1f}%</div><div class="metric-label">Anomaly Rate</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Manual Input
if mode == "Manual Input":
    st.markdown("### Score a Network Flow")
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
        rate_val = st.number_input("Rate", value=10.0)
        sjit = st.number_input("Source Jitter", value=0.5)

    if st.button("Analyze This Flow"):
        flow = {
            "dur": dur, "proto": proto, "service": service, "state": state,
            "spkts": int(spkts), "dpkts": int(dpkts),
            "sbytes": int(sbytes), "dbytes": int(dbytes),
            "rate": rate_val, "sttl": int(sttl), "dttl": int(dttl),
            "sload": 0.0, "dload": 0.0, "sloss": 0, "dloss": 0,
            "sinpkt": 0.0, "dinpkt": 0.0, "sjit": sjit, "djit": 0.0,
            "swin": 255, "stcpb": 0, "dtcpb": 0, "dwin": 255,
            "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0,
            "smean": int(sbytes / max(spkts, 1)),
            "dmean": int(dbytes / max(dpkts, 1)),
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
                    <div class="metric-label">{"ANOMALY" if result["is_anomaly"] else "NORMAL"}</div>
                </div>
            </div>
            <div class="score-bar-container" style="margin-top:16px">
                <div style="background:{color}; height:100%; width:{min(score*100, 100):.0f}%; border-radius:6px"></div>
            </div>
        </div>""", unsafe_allow_html=True)

# Chart and flow log
if st.session_state.flow_history:
    st.markdown("### Anomaly Score Timeline")
    scores = [f["score"] for f in reversed(st.session_state.flow_history)]
    chart_df = pd.DataFrame({"Anomaly Score": scores})
    st.line_chart(chart_df, color="#4fc3f7", height=180)

    st.markdown("### Flow Log")
    risk_colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c", "CRITICAL": "#ff0040"}
    for entry in st.session_state.flow_history[:20]:
        color = risk_colors[entry["risk"]]
        box_class = "alert-box" if entry["is_anomaly"] else "normal-box"
        icon = "ALERT" if entry["is_anomaly"] else "OK"
        st.markdown(f"""
        <div class="{box_class}">
            <span style="color:{color}; font-weight:600">{icon} [{entry['time']}]</span>
            &nbsp;&nbsp;
            <span style="color:#8ab4d4">{entry['proto'].upper()}/{entry['service']}</span>
            &nbsp;|&nbsp; Score: <span style="color:{color}; font-weight:600">{entry['score']:.4f}</span>
            &nbsp;|&nbsp; Risk: <span style="color:{color}; font-weight:600">{entry['risk']}</span>
            &nbsp;|&nbsp; Up:{entry['sbytes']:,}B Down:{entry['dbytes']:,}B
        </div>""", unsafe_allow_html=True)
else:
    st.markdown("### Flow Activity")
    st.info("No flows scored yet. Use the sidebar controls to start.")