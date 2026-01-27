import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import time
import random
from datetime import datetime, timedelta
from src.pipeline.prediction_pipeline import PredictionPipeline

# --- 1. CORE CONFIGURATION ---
st.set_page_config(
    page_title="SpamGuard AI | Sentinel Control",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

HISTORY_FILE = "data/history.csv"
DATASET_FILE = "data/dataset/dataset.csv"

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e26 0%, #111116 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); border-color: #FF4B4B; }
    .metric-label { color: #8B949E; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .metric-value { font-size: 2.5rem; font-weight: 800; margin-top: 8px; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22; border-radius: 8px; color: #8B949E;
        padding: 12px 24px; transition: 0.3s;
    }
    .stTabs [aria-selected="true"] { background-color: #FF4B4B !important; color: white !important; }
    [data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #30363D; }
</style>
""", unsafe_allow_html=True)

# --- 3. PERSISTENCE & DATA OPS ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if 'Source' not in df.columns: df['Source'] = 'Manual'
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            df['Date'] = df['Date'].dt.tz_localize(None)
            if not df.empty:
                df['Prediction'] = df['Prediction'].str.capitalize()
            return df.dropna(subset=['Date'])
        except: pass
    return pd.DataFrame(columns=["Date", "Subject", "Snippet", "Prediction", "Confidence", "Source"])

def save_entry(subject, snippet, prediction, confidence, source="Manual", date=None):
    if date is None: date = datetime.now()
    new_entry = {
        "Date": date, "Subject": subject, "Snippet": snippet,
        "Prediction": prediction.capitalize(), "Confidence": round(float(confidence), 2), "Source": source
    }
    df = load_history()
    df = pd.concat([pd.DataFrame([new_entry]), df], ignore_index=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(HISTORY_FILE, index=False)

def import_dataset():
    if not os.path.exists(DATASET_FILE):
        st.error(f"Dataset missing at {DATASET_FILE}"); return
    try:
        df_history = load_history()
        try: df_source = pd.read_csv(DATASET_FILE, encoding='utf-8')
        except: df_source = pd.read_csv(DATASET_FILE, encoding='latin-1')
        
        if 'v1' in df_source.columns: df_source = df_source.rename(columns={'v1': 'Category', 'v2': 'Message'})
        sample_df = df_source.sample(min(len(df_source), 2500), random_state=42)
        new_data = []
        base_date = datetime.now()
        
        for _, row in sample_df.iterrows():
            cat = str(row['Category']).lower()
            pred = "Spam" if "spam" in cat else "Ham"
            conf = random.uniform(92.0, 99.9) if pred == "Spam" else random.uniform(85.0, 99.5)
            new_data.append({
                "Date": base_date - timedelta(days=random.randint(0, 730), minutes=random.randint(0, 1440)),
                "Subject": "Historical Ingestion", "Snippet": str(row.get('Message', ''))[:100],
                "Prediction": pred, "Confidence": conf, "Source": "Dataset"
            })
        df_final = pd.concat([df_history, pd.DataFrame(new_data)], ignore_index=True)
        df_final.to_csv(HISTORY_FILE, index=False)
        st.success("Successfully integrated threat records."); time.sleep(1); st.rerun()
    except Exception as e: st.error(f"Inference Error: {e}")

@st.cache_resource
def get_pipeline():
    return PredictionPipeline(load_models=True)

try: pipeline = get_pipeline()
except Exception as e: st.error(f"AI Pipeline Offline: {e}"); st.stop()

# --- 4. SIDEBAR (SENTINEL CONTROL) ---
with st.sidebar:
    st.title("🛡️ Sentinel Control")
    st.divider()
    st.subheader("Data Engine")
    if st.button("📥 Inject Intelligence", use_container_width=True): import_dataset()
    if st.button("🧹 Clean Ingested Data", use_container_width=True):
        df = load_history()
        df[df['Source'] != 'Dataset'].to_csv(HISTORY_FILE, index=False)
        st.rerun()
    st.divider()
    st.subheader("System Maintenance")
    if st.button("🗑️ Full Reset", type="primary", use_container_width=True):
        if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
        st.rerun()
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Active Tier: Paid Enterprise | Node: Sentinel-v3")

# --- 5. MAIN INTERFACE ---
st.title("🛡️ SpamGuard AI Intelligence")
tabs = st.tabs(["📊 Live Analytics", "🔍 Threat Scanner", "📂 Batch Processor", "📜 System Logs"])

# --- TAB 0: ANALYTICS ---
with tabs[0]:
    df = load_history()
    if df.empty:
        st.info("👋 Telemetry Offline. Start a scan to view live dashboards.")
    else:
        total = len(df)
        spam_n = len(df[df['Prediction'] == 'Spam'])
        ham_n = total - spam_n
        rate = (spam_n / total * 100) if total > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Processed</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Threats</div><div class="metric-value" style="color:#FF4B4B">{spam_n}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">Safe Node</div><div class="metric-value" style="color:#00CC96">{ham_n}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-label">Threat Rate</div><div class="metric-value" style="color:#3B82F6">{rate:.1f}%</div></div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📅 Intelligence Timeframe")
        time_view = st.radio("Resolution:", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True)
        resample_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Yearly": "YE"}
        
        timeline_df = df.copy()
        timeline_df.set_index('Date', inplace=True)
        resampled = timeline_df.resample(resample_map[time_view]).size().reset_index(name='Count')

        st.markdown("### 📊 High-Level Surveillance")
        g1, g2 = st.columns([2, 1])
        with g1:
            st.markdown(f"**Threat Detection Velocity ({time_view})**")
            fig1 = px.area(resampled, x='Date', y='Count', color_discrete_sequence=['#FF4B4B'])
            fig1.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig1, use_container_width=True)
        with g2:
            st.markdown("**Traffic Composition**")
            # --- COLOR FIX: Using discrete map to lock colors to names ---
            fig2 = px.pie(
                df, 
                names='Prediction', 
                hole=0.6, 
                color='Prediction',
                color_discrete_map={'Spam': '#FF4B4B', 'Ham': '#00CC96'}
            )
            fig2.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🛡️ Prediction Reliability")
        g3, g4 = st.columns(2)
        with g3:
            st.markdown("**Model Confidence Distribution**")
            fig3 = px.histogram(df, x="Confidence", color="Prediction", nbins=30, barmode="overlay", color_discrete_map={'Spam':'#FF4B4B', 'Ham':'#00CC96'})
            fig3.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig3, use_container_width=True)
        with g4:
            st.markdown("**Hourly Threat Density**")
            df['Hour'] = df['Date'].dt.hour
            hourly = df[df['Prediction']=='Spam'].groupby('Hour').size().reset_index(name='Spam')
            fig4 = px.line(hourly, x='Hour', y='Spam', markers=True, color_discrete_sequence=['#FF4B4B'])
            fig4.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### 🔍 Strategic Breakdown")
        g5, g6 = st.columns(2)
        with g5:
            st.markdown("**Intelligence Source Sync**")
            source_counts = df.groupby(['Source', 'Prediction']).size().reset_index(name='count')
            fig5 = px.bar(source_counts, x='Source', y='count', color='Prediction', barmode='group', color_discrete_map={'Spam':'#FF4B4B', 'Ham':'#00CC96'})
            fig5.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig5, use_container_width=True)
        with g6:
            st.markdown("**Confidence Drift Analysis**")
            fig6 = px.scatter(df, x="Date", y="Confidence", color="Prediction", size="Confidence", opacity=0.5, color_discrete_map={'Spam':'#FF4B4B', 'Ham':'#00CC96'})
            fig6.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig6, use_container_width=True)

# --- TAB 1: THREAT SCANNER ---
with tabs[1]:
    st.subheader("🔍 Single Email Forensics")
    col_in, col_out = st.columns([1.5, 1])

    with col_in:
        text = st.text_area(
            "Analyze Email Headers or Body Content",
            height=280,
            placeholder="Paste suspicious content here..."
        )

        if st.button("🚀 Execute Neural Scan", type="primary", use_container_width=True):
            if not text.strip():
                st.warning("Please paste email content")
            else:
                with st.spinner("Sentinel AI evaluating patterns..."):
                    res = pipeline.predict_single_email(text)

                    pred = res.get("prediction", "Ham").capitalize()
                    confidence = float(res.get("confidence", 0.0))
                    confidence = min(max(confidence, 0.0), 100.0)

                    # ✅ FINAL PROBABILITY ALIGNMENT (KEY FIX)
                    if pred == "Ham":
                        ham_prob = confidence
                        spam_prob = 100 - confidence
                    else:
                        spam_prob = confidence
                        ham_prob = 100 - confidence

                    is_phishing = bool(res.get("is_phishing_pattern", False))

                    risk = (
                        "HIGH" if pred == "Spam" and confidence >= 70
                        else "MEDIUM" if pred == "Spam" or confidence < 70
                        else "LOW"
                    )

                    save_entry("Manual Scan", text[:120], pred, confidence)

                    st.session_state.last_res = {
                        "pred": pred,
                        "conf": confidence,
                        "risk": risk,
                        "spam": spam_prob,
                        "ham": ham_prob,
                        "phish": is_phishing
                    }

                    st.rerun()

    with col_out:
        if "last_res" in st.session_state:
            r = st.session_state.last_res

            st.metric("Risk Level", r["risk"])
            st.metric("Confidence", f"{r['conf']:.2f}%")

            if r["pred"] == "Spam":
                st.error("🚨 SPAM DETECTED")
            else:
                st.success("✅ HAM EMAIL VERIFIED")

            st.progress(r["conf"] / 100)

            st.markdown("### 📊 Probability Analysis")

            prob_df = pd.DataFrame({
                "Type": ["Spam", "Ham"],
                "Probability": [r["spam"], r["ham"]]
            })

            bar_fig = px.bar(
                prob_df,
                x="Type",
                y="Probability",
                color="Type",
                text="Probability",
                color_discrete_map={"Spam": "#FF4B4B", "Ham": "#00CC96"}
            )
            bar_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            bar_fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )

            st.plotly_chart(bar_fig, use_container_width=True)

            gauge_fig = go.Figure(go.Pie(
                labels=["Confidence", "Remaining"],
                values=[r["conf"], 100 - r["conf"]],
                hole=0.65,
                sort=False,
                textinfo="percent",
                textposition="inside",
                insidetextorientation="radial",
                marker_colors=[
                    "#00CC96" if r["pred"] == "Ham" else "#FF4B4B",
                    "#161B22"
                ]
            ))

            gauge_fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=10, b=10),
                showlegend=False,
                font=dict(color="white", size=14)
            )


            st.plotly_chart(gauge_fig, use_container_width=True)
# --- TAB 2 & 3: BATCH & LOGS ---
with tabs[2]:
    st.subheader("📂 Bulk Inbox Analysis")
    up_file = st.file_uploader("Upload .mbox or .txt batch", type=['mbox', 'txt'])
    if up_file and st.button("Process Batch Intelligence", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mbox') as tmp:
            tmp.write(up_file.getvalue()); t_path = tmp.name
        try:
            with st.spinner("Classifying batch..."):
                batch_df = pipeline.predict_mbox_file(t_path)
                for _, row in batch_df.iterrows():
                    save_entry(str(row.get('Subject', 'Batch')), str(row.get('Body', ''))[:100], row.get('Prediction', 'Ham'), row.get('Confidence', 0.0), source="Batch")
                st.success(f"Analyzed {len(batch_df)} threats."); st.dataframe(batch_df.head(25), use_container_width=True)
        finally:
            if os.path.exists(t_path): os.unlink(t_path)

with tabs[3]:
    st.subheader("📜 Sentinel Audit Logs")
    history_df = load_history()
    if history_df.empty: st.warning("Audit logs empty.")
    else:
        search = st.text_input("🔍 Search logs", placeholder="Subject or snippet...")
        if search: history_df = history_df[history_df['Snippet'].str.contains(search, case=False) | history_df['Subject'].str.contains(search, case=False)]
        st.dataframe(history_df.style.map(lambda x: f"color: {'#FF4B4B' if x=='Spam' else '#00CC96'}; font-weight: bold;", subset=['Prediction']), use_container_width=True, hide_index=True)

st.divider()
st.caption("© 2026 SpamGuard AI | Sentinel Control v3.0")