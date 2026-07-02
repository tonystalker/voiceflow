"""
dashboard/streamlit_app.py

VoiceFlow AI — Call Analytics Dashboard
Auto-refreshes every 5 seconds.
Shows: call list, transcript viewer, intent breakdown, latency distribution.

Run with:
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceFlow AI — Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark premium look) ────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #16213e 100%);
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .transcript-box {
    background: #0f0f1a;
    border-left: 3px solid #7c3aed;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
  }
  .intent-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

LOG_PATH = Path(__file__).parents[1] / "logs" / "calls.jsonl"


# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_logs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    records = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("# 🎙️ VoiceFlow AI — Call Dashboard")
st.markdown("*Real-time analytics for your AI banking voice agent*")
st.divider()

df = load_logs()

if df.empty:
    st.info("No calls logged yet. Make a call or run the local pipeline test to see data here.", icon="📵")
    st.stop()

# ── KPI Row ────────────────────────────────────────────────────────────────
total_calls = df["call_sid"].nunique()
total_turns = len(df)
escalation_rate = df["escalated"].mean() * 100 if "escalated" in df.columns else 0
avg_latency = df["latency.total_ms"].mean() if "latency.total_ms" in df.columns else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📞 Total Calls", total_calls)
with col2:
    st.metric("🔄 Total Turns", total_turns)
with col3:
    st.metric("🚨 Escalation Rate", f"{escalation_rate:.1f}%")
with col4:
    st.metric("⚡ Avg Round-Trip", f"{avg_latency:.0f} ms" if avg_latency else "—")

st.divider()

# ── Sidebar — Call selector ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Call Sessions")
    call_sids = df["call_sid"].unique().tolist()
    selected_sid = st.selectbox("Select call", options=call_sids, format_func=lambda x: x[-12:])
    st.markdown("---")
    st.markdown("*Dashboard refreshes every 5 seconds*")
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# ── Main area: two columns ────────────────────────────────────────────────
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown(f"### 💬 Transcript — `{selected_sid[-12:]}`")
    call_df = df[df["call_sid"] == selected_sid].sort_values("turn_count")

    INTENT_COLORS = {
        "faq": "#10b981",
        "account_query": "#3b82f6",
        "dispute_query": "#f59e0b",
        "escalate": "#ef4444",
        "out_of_scope": "#6b7280",
    }

    for _, row in call_df.iterrows():
        intent = row.get("intent", "unknown") or "unknown"
        color = INTENT_COLORS.get(intent, "#6b7280")
        ts = row["timestamp"].strftime("%H:%M:%S") if pd.notna(row["timestamp"]) else ""
        st.markdown(
            f"""
            <div class="transcript-box">
              <div style="color:#888;font-size:0.75rem;margin-bottom:4px">
                Turn {int(row['turn_count'])} · {ts} · 
                <span style="background:{color};color:#fff;padding:2px 8px;border-radius:20px;font-size:0.72rem">{intent}</span>
              </div>
              <div style="color:#cdd6f4"><b>👤 Caller:</b> {row.get('transcript','')}</div>
              <div style="color:#a6e3a1;margin-top:6px"><b>🤖 Aria:</b> {row.get('response','')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown("### 📊 Intent Breakdown")
    if "intent" in df.columns:
        intent_counts = df["intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        fig_intent = px.bar(
            intent_counts,
            x="Intent",
            y="Count",
            color="Intent",
            color_discrete_map=INTENT_COLORS,
            template="plotly_dark",
        )
        fig_intent.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_intent, use_container_width=True)

    st.markdown("### ⚡ Latency Distribution (ms)")
    latency_cols = {
        "STT": "latency.stt_ms",
        "LLM": "latency.llm_ms",
        "Total": "latency.total_ms",
    }
    lat_data = []
    for label, col in latency_cols.items():
        if col in df.columns:
            for v in df[col].dropna():
                lat_data.append({"Stage": label, "Latency (ms)": v})

    if lat_data:
        lat_df = pd.DataFrame(lat_data)
        fig_lat = px.box(
            lat_df,
            x="Stage",
            y="Latency (ms)",
            color="Stage",
            template="plotly_dark",
        )
        fig_lat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

# ── Auto-refresh ─────────────────────────────────────────────────────────
st.markdown(
    "<script>setTimeout(function(){window.location.reload()}, 5000);</script>",
    unsafe_allow_html=True,
)
