import pickle
import yaml
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import time

app = FastAPI(
    title="Network Traffic Anomaly Detection API",
    description="Real-time anomaly scoring for network flows using Isolation Forest",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
ARTIFACTS = None
CONFIG = None


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_artifacts():
    global MODEL, ARTIFACTS, CONFIG
    CONFIG = load_config()

    model_path = CONFIG["data"]["model_path"]
    encoder_path = CONFIG["data"]["encoder_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoders not found at {encoder_path}. Run pipeline.py first.")

    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)

    with open(encoder_path, "rb") as f:
        ARTIFACTS = pickle.load(f)

    print(f"[STARTUP] Model loaded from {model_path}")
    print(f"[STARTUP] Encoders loaded from {encoder_path}")


@app.on_event("startup")
def startup_event():
    load_artifacts()


class NetworkFlow(BaseModel):
    dur: float = Field(default=0.0, description="Duration of connection in seconds")
    proto: str = Field(default="tcp", description="Protocol: tcp, udp, etc.")
    service: str = Field(default="-", description="Service: http, ftp, dns, etc.")
    state: str = Field(default="FIN", description="Connection state")
    spkts: int = Field(default=1, description="Source to destination packet count")
    dpkts: int = Field(default=1, description="Destination to source packet count")
    sbytes: int = Field(default=100, description="Source to destination bytes")
    dbytes: int = Field(default=100, description="Destination to source bytes")
    rate: float = Field(default=0.0)
    sttl: int = Field(default=64, description="Source TTL value")
    dttl: int = Field(default=64, description="Destination TTL value")
    sload: float = Field(default=0.0)
    dload: float = Field(default=0.0)
    sloss: int = Field(default=0)
    dloss: int = Field(default=0)
    sinpkt: float = Field(default=0.0)
    dinpkt: float = Field(default=0.0)
    sjit: float = Field(default=0.0)
    djit: float = Field(default=0.0)
    swin: int = Field(default=0)
    stcpb: int = Field(default=0)
    dtcpb: int = Field(default=0)
    dwin: int = Field(default=0)
    tcprtt: float = Field(default=0.0)
    synack: float = Field(default=0.0)
    ackdat: float = Field(default=0.0)
    smean: int = Field(default=0)
    dmean: int = Field(default=0)
    trans_depth: int = Field(default=0)
    response_body_len: int = Field(default=0)
    ct_srv_src: int = Field(default=1)
    ct_state_ttl: int = Field(default=0)
    ct_dst_ltm: int = Field(default=1)
    ct_src_dport_ltm: int = Field(default=1)
    ct_dst_sport_ltm: int = Field(default=1)
    ct_dst_src_ltm: int = Field(default=1)
    is_ftp_login: int = Field(default=0)
    ct_ftp_cmd: int = Field(default=0)
    ct_flw_http_mthd: int = Field(default=0)
    ct_src_ltm: int = Field(default=1)
    ct_srv_dst: int = Field(default=1)
    is_sm_ips_ports: int = Field(default=0)


class PredictionResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    risk_level: str
    confidence: str
    inference_time_ms: float
    explanation: dict


class BatchRequest(BaseModel):
    flows: list[NetworkFlow]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_flows: int
    anomalies_detected: int
    anomaly_rate: float
    inference_time_ms: float


def preprocess_flow(flow: NetworkFlow) -> pd.DataFrame:
    data = flow.model_dump()
    df = pd.DataFrame([data])

    encoders = ARTIFACTS["encoders"]
    scaler = ARTIFACTS["scaler"]
    medians = ARTIFACTS["medians"]

    for col in ["proto", "service", "state"]:
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(
            lambda x: x if x in known else le.classes_[0]
        )
        df[col] = le.transform(df[col])

    num_cols = CONFIG["features"]["numeric_columns"]
    existing_num = [c for c in num_cols if c in df.columns]
    for col in existing_num:
        if df[col].isnull().any():
            df[col] = df[col].fillna(medians.get(col, 0))
    df[existing_num] = scaler.transform(df[existing_num])

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

    return df


def score_to_risk(score: float) -> tuple[str, str]:
    if score < 0.40:
        return "LOW", "Normal traffic pattern"
    elif score < 0.55:
        return "MEDIUM", "Slightly unusual — worth monitoring"
    elif score < 0.70:
        return "HIGH", "Strong anomaly — recommend investigation"
    else:
        return "CRITICAL", "Severe anomaly — likely attack"


def build_explanation(flow: NetworkFlow, score: float) -> dict:
    flags = []
    total_bytes = flow.sbytes + flow.dbytes
    if total_bytes > 0:
        ratio = flow.sbytes / total_bytes
        if ratio > 0.8:
            flags.append("High outbound byte ratio — possible data exfiltration")
        elif ratio < 0.05:
            flags.append("Very low outbound ratio — possible C2 beacon")

    if flow.spkts > 0 and flow.dpkts == 0:
        flags.append("Zero response packets — possible port scan")

    if flow.dur > 0 and (flow.sbytes / flow.dur) > 1_000_000:
        flags.append("Extremely high transfer rate — possible flood attack")

    if abs(flow.sttl - flow.dttl) > 100:
        flags.append("Large TTL difference — possible spoofed source address")

    if not flags:
        flags.append("Composite feature anomaly — no single dominant indicator")

    return {"anomaly_indicators": flags, "raw_score": round(score, 6), "threshold": 0.5}


@app.get("/")
def root():
    return {
        "service": "Network Traffic Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "artifacts_loaded": ARTIFACTS is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(flow: NetworkFlow):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    try:
        df = preprocess_flow(flow)
        raw_score = MODEL.score_samples(df)[0]
        anomaly_score = float(-raw_score)
        is_anomaly = MODEL.predict(df)[0] == -1
        elapsed = (time.time() - start) * 1000
        risk_level, confidence = score_to_risk(anomaly_score)
        explanation = build_explanation(flow, anomaly_score)
        return PredictionResponse(
            anomaly_score=round(anomaly_score, 4),
            is_anomaly=bool(is_anomaly),
            risk_level=risk_level,
            confidence=confidence,
            inference_time_ms=round(elapsed, 3),
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(request.flows) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000 flows")
    start = time.time()
    try:
        predictions = []
        for flow in request.flows:
            df = preprocess_flow(flow)
            raw_score = MODEL.score_samples(df)[0]
            anomaly_score = float(-raw_score)
            is_anomaly = MODEL.predict(df)[0] == -1
            risk_level, confidence = score_to_risk(anomaly_score)
            explanation = build_explanation(flow, anomaly_score)
            predictions.append(PredictionResponse(
                anomaly_score=round(anomaly_score, 4),
                is_anomaly=bool(is_anomaly),
                risk_level=risk_level,
                confidence=confidence,
                inference_time_ms=0,
                explanation=explanation
            ))
        elapsed = (time.time() - start) * 1000
        anomalies = sum(1 for p in predictions if p.is_anomaly)
        return BatchResponse(
            predictions=predictions,
            total_flows=len(predictions),
            anomalies_detected=anomalies,
            anomaly_rate=round(anomalies / len(predictions), 4),
            inference_time_ms=round(elapsed, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))