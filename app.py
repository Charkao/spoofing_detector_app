import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(page_title="AI Spoofing Detector", layout="wide")

st.title("🚨 AI-Powered Spoofing Detection Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Feature Dataset (CSV)", type=["csv"])
contamination = st.sidebar.slider("Anomaly Contamination (%)", 0.01, 0.15, 0.05)

# --- Feature List ---
feature_columns = [
    "cancel_ratio", "avg_time_to_cancel", "cancel_burst_count", "order_reversal_rate",
    "inter_order_time_std", "post_order_price_change", "volume_cancellation_ratio", "trader_entropy"
]

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["window_start"])
    st.success("✅ Dataset Loaded Successfully")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Explorer", "🌲 Anomaly Detection", "📈 Trader Timeline", "🤖 GPT Assistant"])

    # --- Feature Scaling ---
    df_clean = df.dropna(subset=feature_columns).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_columns])

    # --- Tab 1: Visual Explorer ---
    with tab1:
        st.subheader("🔍 Explore Feature Distributions")
        selected_feature = st.selectbox("Select a Feature", feature_columns)

        fig1 = px.histogram(df_clean, x=selected_feature, nbins=50, title=f"Histogram of {selected_feature}", marginal="box")
        st.plotly_chart(fig1, use_container_width=True)

        fig2, ax = plt.subplots()
        sns.boxplot(x=df_clean[selected_feature], ax=ax)
        ax.set_title(f"Boxplot of {selected_feature}")
        st.pyplot(fig2)

    # --- Tab 2: Anomaly Detection ---
    with tab2:
        st.subheader("🌲 Run Isolation Forest")
        if st.button("Detect Anomalies"):
            model = IsolationForest(contamination=contamination, random_state=42)
            model.fit(X_scaled)
            df_clean['anomaly_score'] = model.decision_function(X_scaled)
            df_clean['is_anomaly'] = model.predict(X_scaled).astype(int)
            df_clean['is_anomaly'] = df_clean['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)

            st.success(f"✅ Anomalies Detected: {df_clean['is_anomaly'].sum()} / {len(df_clean)}")

            fig3 = px.scatter(
                df_clean,
                x="cancel_ratio",
                y="volume_cancellation_ratio",
                color="is_anomaly",
                title="Cancel Ratio vs Volume Cancellation Ratio",
                labels={"is_anomaly": "Anomaly"},
                color_discrete_map={0: "blue", 1: "red"}
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.dataframe(df_clean[df_clean['is_anomaly'] == 1].sort_values("anomaly_score").head(20))

    # --- Tab 3: Trader Timeline ---
    with tab3:
        st.subheader("📈 Trader Time Series Viewer")
        trader_id = st.selectbox("Select Trader", sorted(df_clean['trader_id'].unique()))
        trader_data = df_clean[df_clean['trader_id'] == trader_id].sort_values("window_start")

        fig4 = px.line(
            trader_data,
            x="window_start",
            y="cancel_ratio",
            markers=True,
            title=f"Cancel Ratio Over Time: Trader {trader_id}"
        )
        anomaly_points = trader_data[trader_data['is_anomaly'] == 1]
        fig4.add_scatter(
            x=anomaly_points['window_start'],
            y=anomaly_points['cancel_ratio'],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Anomaly'
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- Tab 4: GPT Assistant ---
    with tab4:
        st.subheader("🤖 Explain Your Data with LLM")
        st.markdown("This tab requires GPT API integration. You can ask things like:")
        st.markdown("- Why is Trader TRD7 flagged as an anomaly?\n- What does high entropy mean in this context?\n- Compare TRD2 and TRD5\n")

        st.warning("🔑 GPT integration not active in this demo. To enable, connect OpenAI API.")

else:
    st.info("👈 Upload a feature dataset to get started.")
