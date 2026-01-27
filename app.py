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

st.set_page_config(
    page_title="SpamGuard AI | Cyber Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

HISTORY_FILE = "data/history.csv"
DATASET_FILE = "data/dataset/dataset.csv"

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    
    /* Modern Glassmorphic Metric Cards */
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
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #FF4B4B;
    }
    .metric-label { color: #8B949E; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .metric-value { font-size: 2.5rem; font-weight: 800; margin-top: 8px; color: #ffffff; }

    /* Custom UI Elements */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 8px;
        color: #8B949E;
        padding: 12px 24px;
        transition: 0.3s;
    }
    .stTabs [aria-selected="true"] { background-color: #FF4B4B !important; color: white !important; }
    [data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #30363D; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #30363D; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if 'Source' not in df.columns: df['Source'] = 'Manual'
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            df['Date'] = df['Date'].dt.tz_localize(None)
            return df.dropna(subset=['Date'])
        except Exception:
            pass
    return pd.DataFrame(columns=["Date", "Subject", "Snippet", "Prediction", "Confidence", "Source"])

def save_entry(subject, snippet, prediction, confidence, source="Manual", date=None):
    if date is None: date = datetime.now()
    new_entry = {
        "Date": date, 
        "Subject": subject, 
        "Snippet": snippet, 
        "Prediction": prediction, 
        "Confidence": round(float(confidence), 2), 
        "Source": source
    }
    df = load_history()
    df = pd.concat([pd.DataFrame([new_entry]), df], ignore_index=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(HISTORY_FILE, index=False)

def import_dataset():
    if not os.path.exists(DATASET_FILE):
        st.error(f"Critical Error: Dataset not found at {DATASET_FILE}")
        return
    try:
        df_history = load_history()
        if not df_history.empty and "Dataset" in df_history['Source'].unique():
            st.warning("Intelligence Core already contains Dataset entries.")
            return
        
        try:
            df_source = pd.read_csv(DATASET_FILE, encoding='utf-8')
        except:
            df_source = pd.read_csv(DATASET_FILE, encoding='latin-1')

        if 'v1' in df_source.columns: 
            df_source = df_source.rename(columns={'v1': 'Category', 'v2': 'Message'})
        
        # Simulating historical ingestion
        sample_df = df_source.sample(min(len(df_source), 3000), random_state=42)
        new_data = []
        base_date = datetime.now()
        
        for _, row in sample_df.iterrows():
            category = str(row['Category']).lower()
            pred = "Spam" if "spam" in category else "Ham"
            conf = random.uniform(91.5, 99.9) if pred == "Spam" else random.uniform(78.2, 99.5)
            new_data.append({
                "Date": base_date - timedelta(days=random.randint(0, 730), minutes=random.randint(0, 1440)),
                "Subject": "Historical Ingestion",
                "Snippet": str(row.get('Message', ''))[:100],
                "Prediction": pred,
                "Confidence": conf,
                "Source": "Dataset"
            })
            
        df_new = pd.DataFrame(new_data)
        df_final = pd.concat([df_history, df_new], ignore_index=True)
        df_final.to_csv(HISTORY_FILE, index=False)
        st.success(f"Successfully integrated {len(df_new)} threat records.")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Inference Error during import: {e}")

def remove_dataset_entries():
    df = load_history()
    if df.empty: return
    df_clean = df[df['Source'] != 'Dataset']
    df_clean.to_csv(HISTORY_FILE, index=False)
    st.success("Dataset entries purged from local cache.")
    time.sleep(1)
    st.rerun()

def clear_all_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        st.success("All logs and history cleared.")
        st.rerun()

@st.cache_resource
def get_pipeline():
    return PredictionPipeline(load_models=True)

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"AI Pipeline Connection Failed: {e}")
    st.stop()

with st.sidebar:
    st.title("🛡️ Cyber Command")
    st.divider()
    st.subheader("Data Engine")
    if st.button("📥 Inject Intelligence", use_container_width=True):
        import_dataset()
    if st.button("🧹 Clean Ingested Data", use_container_width=True):
        remove_dataset_entries()
    
    st.divider()
    st.subheader("System Maintenance")
    if st.button("🗑️ Full Reset", type="primary", use_container_width=True):
        clear_all_history()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Active Tier: Paid Enterprise")
    st.caption("Model: Nano Banana v3.0")

st.title("🛡️ SpamGuard AI Intelligence")
tabs = st.tabs(["📊 Live Analytics", "🔍 Threat Scanner", "📂 Batch Processor", "📜 System Logs"])

with tabs[0]:
    df = load_history()
    if df.empty:
        st.info("👋 **Telemetry Offline.** Start a scan or inject dataset to view live dashboards.")
    else:
        
        total = len(df)
        spam = len(df[df['Prediction'] == 'Spam'])
        ham = total - spam
        rate = (spam / total * 100) if total > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Processed</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Threats</div><div class="metric-value" style="color:#FF4B4B">{spam}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">Safe</div><div class="metric-value" style="color:#00CC96">{ham}</div></div>', unsafe_allow_html=True)
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
            fig2 = px.pie(df, names='Prediction', hole=0.6, color='Prediction', color_discrete_map={'Spam':'#FF4B4B', 'Ham':'#00CC96'})
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

with tabs[1]:
    st.subheader("🔍 Single Email Forensics")
    col_input, col_info = st.columns([2, 1])
    
    with col_input:
        text = st.text_area("Analyze Email Headers or Body Content", height=250, placeholder="Paste suspicious content here to classify with the SpamGuard engine...")
        if st.button("🚀 Execute Neural Scan", type="primary", use_container_width=True):
            if text.strip():
                with st.spinner("Model is evaluating patterns..."):
                    
                    res = pipeline.predict_single_email(text)
                    score = res.get('confidence', 0.0)
                    pred = res.get('prediction', 'Unknown')
                    
                    save_entry("Manual Scan", text[:50], pred, score, source="Manual")
                    st.session_state.last_res = {'p': pred, 'c': score}
                    st.rerun()
            else:
                st.warning("Forensic input required for analysis.")

    with col_info:
        if 'last_res' in st.session_state:
            r = st.session_state.last_res
            st.markdown("### Scan Result")
            if r['p'] == "Spam":
                st.error(f"Prediction: {r['p']}")
            else:
                st.success(f"Prediction: {r['p']}")
            st.metric("Model Certainty", f"{r['c']:.2f}%")
            st.progress(r['c'] / 100)
            
            if r['c'] < 60:
                st.warning("⚠️ **Low Confidence**: Result may require manual verification.")
        else:
            st.info("System Ready. Please input content to begin scanning.")
with tabs[2]:
    st.subheader("📂 Bulk Inbox Analysis")
    uploaded_file = st.file_uploader("Upload .mbox or .txt batch file", type=['mbox', 'txt'])
    
    if uploaded_file and st.button("Process Batch Intelligence", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mbox') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            with st.spinner("Running batch classification..."):
                batch_df = pipeline.predict_mbox_file(tmp_path)
                
                for _, row in batch_df.iterrows():
                    save_entry(
                        subject=str(row.get('Subject', 'Batch Process')),
                        snippet=str(row.get('Body', ''))[:100],
                        prediction=row.get('Prediction', 'Ham'),
                        confidence=row.get('Confidence', 0.0),
                        source="Batch"
                    )
                st.success(f"Batch processing complete. Analyzed {len(batch_df)} threats.")
                st.dataframe(batch_df.head(25), use_container_width=True)
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)

with tabs[3]:
    st.subheader("📜 System Audit Trail")
    history_df = load_history()
    
    if history_df.empty:
        st.warning("Audit logs currently empty.")
    else:
      
        search = st.text_input("🔍 Search logs (Subject or Content Snippet)", placeholder="e.g., 'Gift card', 'Invoice'...")
        if search:
            history_df = history_df[
                history_df['Snippet'].str.contains(search, case=False) | 
                history_df['Subject'].str.contains(search, case=False)
            ]
        
      
        st.dataframe(
            history_df[['Date', 'Source', 'Subject', 'Snippet', 'Prediction', 'Confidence']].style.map(
                lambda x: f"color: {'#FF4B4B' if x=='Spam' else '#00CC96'}; font-weight: bold;", subset=['Prediction']
            ),
            use_container_width=True, hide_index=True
        )

st.divider()
st.caption("© 2026 SpamGuard AI Intelligence | Enterprise Threat Management System")