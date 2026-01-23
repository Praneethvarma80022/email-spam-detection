import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import os
import time
import random
from datetime import datetime, timedelta
from src.pipeline.prediction_pipeline import PredictionPipeline

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

HISTORY_FILE = "data/history.csv"
DATASET_FILE = "data/dataset/dataset.csv"

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3, h4, h5, h6, p, li, .stMarkdown, .stText, label, .stRadio label { color: #FFFFFF !important; }
    .metric-card {
        background-color: #262730;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value { color: #FFFFFF !important; font-size: 2rem; font-weight: 700; }
    .metric-label { color: #A0A0A0 !important; font-size: 0.9rem; text-transform: uppercase; }
    div.stButton > button { background-color: #262730; color: #FFFFFF; border: 1px solid #4F4F4F; }
    div.stButton > button:hover { border-color: #FF4B4B; color: #FF4B4B !important; }
    div.stButton > button[kind="primary"] { background-color: #FF4B4B; border: none; color: white !important; }
    [data-testid="stSidebar"] { background-color: #161920; border-right: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if 'Source' not in df.columns:
                df['Source'] = 'Manual'
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            df['Date'] = df['Date'].dt.tz_localize(None)
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Date", "Subject", "Snippet", "Prediction", "Confidence", "Source"])

def save_entry(subject, snippet, prediction, confidence, source="Manual", date=None):
    if date is None: date = datetime.now()
    new_entry = {"Date": date, "Subject": subject, "Snippet": snippet, "Prediction": prediction, "Confidence": confidence, "Source": source}
    df = load_history()
    df = pd.concat([pd.DataFrame([new_entry]), df], ignore_index=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(HISTORY_FILE, index=False)

def import_dataset():
    if not os.path.exists(DATASET_FILE):
        st.error(f"Dataset file not found at {DATASET_FILE}")
        return
    try:
        df_history = load_history()
        if not df_history.empty and "Dataset" in df_history['Source'].unique():
            st.warning("Dataset is already added.")
            return
        try:
            df_source = pd.read_csv(DATASET_FILE, encoding='utf-8')
        except:
            df_source = pd.read_csv(DATASET_FILE, encoding='latin-1')

        if 'v1' in df_source.columns: 
            df_source = df_source.rename(columns={'v1': 'Category', 'v2': 'Message'})
        
        sample_df = df_source.sample(min(len(df_source), 3000), random_state=42)
        new_data = []
        base_date = datetime.now()
        
        for _, row in sample_df.iterrows():
            category = str(row['Category']).lower()
            pred = "Spam" if "spam" in category else "Ham"
            conf = random.uniform(85, 99) if pred == "Spam" else random.uniform(70, 99)
            
            # Generate dates spanning last 2 years
            days_back = random.randint(0, 730)
            
            new_data.append({
                "Date": base_date - timedelta(days=days_back, minutes=random.randint(0, 1440)),
                "Subject": "Training Data",
                "Snippet": str(row.get('Message', ''))[:100],
                "Prediction": pred,
                "Confidence": conf,
                "Source": "Dataset"
            })
            
        df_new = pd.DataFrame(new_data)
        if not df_history.empty:
            df_final = pd.concat([df_history, df_new], ignore_index=True)
        else:
            df_final = df_new
        
        df_final['Date'] = pd.to_datetime(df_final['Date'], utc=True).dt.tz_localize(None)
        df_final = df_final.sort_values(by="Date", ascending=False)
        os.makedirs("data", exist_ok=True)
        df_final.to_csv(HISTORY_FILE, index=False)
        st.success(f"Added {len(df_new)} records!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Error importing dataset: {e}")

def remove_dataset_entries():
    df = load_history()
    if df.empty: return
    df_clean = df[df['Source'] != 'Dataset']
    if len(df_clean) == len(df):
        st.warning("No dataset entries found.")
    else:
        df_clean.to_csv(HISTORY_FILE, index=False)
        st.success("Removed dataset entries.")
        time.sleep(1)
        st.rerun()

def clear_all_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        st.rerun()

@st.cache_resource
def get_pipeline():
    return PredictionPipeline(load_models=True)

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

with st.sidebar:
    st.title("🛡️ SpamGuard AI")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ Add Dataset"):
            with st.spinner("Importing..."): import_dataset()
    with c2:
        if st.button("➖ Remove Data"): remove_dataset_entries()
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear All History", type="primary"): clear_all_history()

st.title("Email Threat Intelligence")
tabs = st.tabs(["📊 Dashboard", "🔍 Analyze Email", "📂 Batch Processing"])

with tabs[0]:
    df = load_history()
    
    if df.empty:
        st.info("👋 **History is Empty.** Use the sidebar to Add Dataset or start analyzing.")
    else:

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
        df['Date'] = df['Date'].dt.tz_localize(None)
        df = df.dropna(subset=['Date']) 

        total = len(df)
        spam_df = df[df['Prediction'] == 'Spam']
        spam = len(spam_df)
        safe = len(df[df['Prediction'] == 'Ham'])
        spam_rate = (spam / total * 100) if total > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        def card(label, value, color):
            return f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{color}!important">{value}</div></div>"""
        
        with c1: st.markdown(card("Total Emails", f"{total:,}", "#FFF"), unsafe_allow_html=True)
        with c2: st.markdown(card("Spam Detected", f"{spam:,}", "#FF4B4B"), unsafe_allow_html=True)
        with c3: st.markdown(card("Safe Emails", f"{safe:,}", "#00CC96"), unsafe_allow_html=True)
        with c4: st.markdown(card("Spam Rate", f"{spam_rate:.1f}%", "#3B82F6"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📅 Activity Timeline")
      
        time_view = st.radio("Select View:", ["Daily", "Weekly", "Monthly", "Yearly"], horizontal=True)

        if time_view == "Daily":
            cutoff = df['Date'].max() - timedelta(days=90)
            view_df = df[df['Date'] > cutoff].copy()
            view_spam = view_df[view_df['Prediction'] == 'Spam']
            grouped_spam = view_spam.groupby(view_spam['Date'].dt.date).size().reset_index(name='Count')
            grouped_all = view_df.groupby(view_df['Date'].dt.date).size().reset_index(name='Count')
            x_title = "Date (Last 90 Days)"
        
        elif time_view == "Weekly":
        
            cutoff = df['Date'].max() - timedelta(days=180)
            view_df = df[df['Date'] > cutoff].copy()
            view_spam = view_df[view_df['Prediction'] == 'Spam']
            
    
            grouped_spam = view_spam.groupby(pd.Grouper(key='Date', freq='W-MON')).size().reset_index(name='Count')
            grouped_all = view_df.groupby(pd.Grouper(key='Date', freq='W-MON')).size().reset_index(name='Count')
            x_title = "Week (Last 6 Months)"

        elif time_view == "Monthly":
            cutoff = df['Date'].max() - timedelta(days=365*3)
            view_df = df[df['Date'] > cutoff].copy()
            view_spam = view_df[view_df['Prediction'] == 'Spam']
            grouped_spam = view_spam.groupby(view_spam['Date'].dt.to_period('M').dt.to_timestamp()).size().reset_index(name='Count')
            grouped_all = view_df.groupby(view_df['Date'].dt.to_period('M').dt.to_timestamp()).size().reset_index(name='Count')
            x_title = "Month"
            
        else:
            view_df = df.copy()
            view_spam = view_df[view_df['Prediction'] == 'Spam']
            grouped_spam = view_spam.groupby(view_spam['Date'].dt.to_period('Y').dt.to_timestamp()).size().reset_index(name='Count')
            grouped_all = view_df.groupby(view_df['Date'].dt.to_period('Y').dt.to_timestamp()).size().reset_index(name='Count')
            x_title = "Year"

        col_charts_1, col_charts_2 = st.columns(2)

        with col_charts_1:
            st.markdown(f"**{time_view} Spam Count** (Bar)")
            if not grouped_spam.empty:
                fig_bar = px.bar(grouped_spam, x='Date', y='Count', color_discrete_sequence=['#FF4B4B'])
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color='white'), xaxis_title=x_title, yaxis_title="Spam Emails"
                )
                fig_bar.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No spam data for this period.")

        with col_charts_2:
            st.markdown(f"**{time_view} Total Activity** (Trend)")
            if not grouped_all.empty:
                fig_area = px.area(grouped_all, x='Date', y='Count', markers=True)
                fig_area.update_traces(line_color='#3B82F6', fillcolor="rgba(59, 130, 246, 0.3)")
                fig_area.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color='white'), xaxis_title=x_title, yaxis_title="Total Processed"
                )
                fig_area.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(fig_area, use_container_width=True)
            else:
                st.info("No activity data.")

        st.markdown("---")
        
        st.subheader("Distribution")
        c_p1, c_p2, c_p3 = st.columns([1, 2, 1])
        with c_p2:
            fig_pie = px.pie(df, names='Prediction', hole=0.6, color='Prediction', 
                            color_discrete_map={'Spam':'#FF4B4B', 'Ham':'#00CC96'})
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.markdown("---")
        st.subheader("📝 Recent Logs")
        st.dataframe(
            df[['Date', 'Source', 'Snippet', 'Prediction', 'Confidence']].head(20).style.map(
                lambda x: f"color: {'#FF4B4B' if x=='Spam' else '#00CC96'}; font-weight: bold;", subset=['Prediction']
            ),
            use_container_width=True, hide_index=True
        )

with tabs[1]:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    if 'last_scan' not in st.session_state: st.session_state.last_scan = None
        
    with c1:
        txt = st.text_area("Analyze Email", height=200, placeholder="Paste email content here...")
        if st.button("Scan Email", type="primary"):
            if txt.strip():
                res = pipeline.predict_single_email(txt)
                conf = res.get('confidence', 0.0) or 0.0
                save_entry("Manual Scan", txt[:50], res['prediction'], conf, source="Manual")
                st.session_state.last_scan = {'p': res['prediction'], 'c': conf, 't': txt}
                st.rerun()
            else: st.warning("Please enter text.")

        if st.session_state.last_scan:
            res = st.session_state.last_scan
            st.markdown("---")
            if res['p'] == "Spam": st.error(f"🚨 SPAM DETECTED ({res['c']:.1f}%)")
            else: st.success(f"✅ SAFE EMAIL ({res['c']:.1f}%)")

    with c2: st.info("The AI checks for phishing patterns and suspicious keywords.")

with tabs[2]:
    st.markdown("<br>", unsafe_allow_html=True)
    upl = st.file_uploader("Upload .mbox file", type=['mbox', 'txt'])
    if 'last_batch' not in st.session_state: st.session_state.last_batch = None
    
    if upl and st.button("Process Batch"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mbox') as tmp:
            tmp.write(upl.getvalue())
            tmp_path = tmp.name
        try:
            with st.spinner("Processing..."):
                res_df = pipeline.predict_mbox_file(tmp_path)
                new_rows = []
                for _, r in res_df.iterrows():
                    new_rows.append({
                        "Date": pd.to_datetime(r.get('Time')) if r.get('Time') else datetime.now(),
                        "Subject": str(r.get('Subject', 'Batch Entry')),
                        "Snippet": str(r.get('Body', ''))[:100],
                        "Prediction": r['Prediction'],
                        "Confidence": 0.0,
                        "Source": "Batch"
                    })
                if new_rows:
                    df_curr = load_history()
                    new_df = pd.DataFrame(new_rows)
                    new_df['Date'] = pd.to_datetime(new_df['Date'], utc=True).dt.tz_localize(None)
                    df_curr = pd.concat([new_df, df_curr], ignore_index=True)
                    df_curr.to_csv(HISTORY_FILE, index=False)
                st.session_state.last_batch = res_df
                st.rerun()
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)

    if st.session_state.last_batch is not None:
        st.success(f"Batch Complete! Saved {len(st.session_state.last_batch)} emails.")
        st.dataframe(st.session_state.last_batch.head(), use_container_width=True)
        if st.button("Clear Result"):
            st.session_state.last_batch = None
            st.rerun()