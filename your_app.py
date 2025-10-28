# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt


# --------- CONFIG ----------
PREDICTOR_PATH = "delay_predictor.pkl"
ENCODERS_PATH = "encoders.pkl"
MERGED_CSV = "data/merged.csv"

# Prediction thresholds
RISK_THRESH_HIGH = 0.7
RISK_THRESH_MED = 0.5

# --------- UTILITIES ----------
@st.cache_data
def load_data(path=MERGED_CSV):
    df = pd.read_csv(path)
    return df

@st.cache_resource
def load_model(path=PREDICTOR_PATH):
    return joblib.load(path)

@st.cache_resource
def load_encoders(path=ENCODERS_PATH):
    return joblib.load(path)

def encode_row(row, encoders):
    # Copy to avoid changing original
    r = row.copy()
    for col, enc in encoders.items():
        if col in r.index:
            # if unseen category, map to -1 (XGBoost can handle ints; better would be target encoding)
            try:
                r[col] = enc.transform([r[col]])[0]
            except Exception:
                r[col] = -1
    return r

def prepare_features(df, encoders, drop_cols=None):
    """
    Prepare features for model (encode categorical features using encoders dict),
    and return X (DataFrame) matching training features.
    """
    if drop_cols is None:
        drop_cols = []
    X = df.copy()
    # Drop obviously leak columns if present
    leak_cols = ['Is_Delayed','Delivery_Status','Delay_Days','Actual_Delivery_Days','Rating_Category','Customer_Rating',
                 'Order_Id','Order_Date','Origin','Destination','Feedback_Text','Feedback_Date','Issue_Category','Special_Handling']
    for c in leak_cols + drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    # Encode using encoders (if present)
    for col, enc in encoders.items():
        if col in X.columns:
            # unseen values get -1
            X[col] = X[col].apply(lambda val: enc.transform([val])[0] if val in enc.classes_ else -1)
    # fillna numeric
    X = X.fillna(X.mean(numeric_only=True))
    return X

def get_alternative_carrier(current_carrier, carrier_stats, top_n=2):
    """
    Choose alternative carriers with lower delay rate than current.
    carrier_stats: pandas Series indexed by carrier with delay rate.
    """
    if current_carrier not in carrier_stats.index:
        # pick best carriers
        best = carrier_stats.sort_values().index.tolist()
        return best[:top_n]
    low = carrier_stats[carrier_stats < carrier_stats[current_carrier]].sort_values()
    if len(low) == 0:
        # no better carrier found; return top carriers
        return carrier_stats.sort_values().index.tolist()[:top_n]
    return low.index.tolist()[:top_n]

def suggest_actions(row, prob, carrier_stats, sustainability_mode=False):
    """
    Simple prescriptive rules:
    - If prob high and better carrier exists => suggest switching
    - If prob high & priority Express => add 1 day buffer
    - If sustainability_mode => suggest fuel-efficient carriers (by fuel consumption)
    """
    actions = []
    if prob >= RISK_THRESH_HIGH:
        actions.append("High risk: consider reassignment or deeper review.")
    elif prob >= RISK_THRESH_MED:
        actions.append("Medium risk: monitor and prioritize resources.")

    # Suggest carrier switch if beneficial
    cur = row.get('Carrier', None)
    if cur is not None:
        alt = get_alternative_carrier(cur, carrier_stats, top_n=2)
        if alt and alt[0] != cur:
            actions.append(f"Consider switching to {alt[0]} (lower delay rate).")

    # Priority suggestion
    pr = row.get('Priority', None)
    if pr is not None and pr.lower().startswith('express') and prob >= RISK_THRESH_MED:
        actions.append("Express order: add +1 day buffer or upgrade resources for on-time delivery.")

    # Sustainability suggestion
    if sustainability_mode:
        actions.append("Sustainability mode: prefer carriers/vehicles with lower fuel consumption / CO₂.")

    if not actions:
        actions.append("No action needed - low risk.")
    return actions

# --------- STREAMLIT UI ----------
st.set_page_config(page_title="NexGen Smart Delivery AI", layout="wide")
st.title("🚚 NexGen Smart Delivery AI — Predictive + Prescriptive Dashboard")

# Load baseline data
df = load_data()
model = load_model()
encoders = load_encoders()

# Prepare carrier stats (from historical data)
carrier_delay_rate = df.groupby('Carrier')['Is_Delayed'].mean().sort_values()

# Sidebar controls
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Mode", ["Overview Dashboard", "Predict & Recommend", "Alerts & Export", "Explain Model Predictions"])
sustainability_mode = st.sidebar.checkbox("Enable Sustainability Mode (prefer green carriers)", value=False)

# ---------- Overview Dashboard ----------
if mode == "Overview Dashboard":
    st.header("Operational Overview")
    col1, col2 = st.columns(2)
    with col1:
        # KPI cards
        total_orders = len(df)
        delayed_pct = df['Is_Delayed'].mean()
        avg_delay = df['Delay_Days'].mean() if 'Delay_Days' in df.columns else np.nan
        st.metric("Total Orders (in dataset)", total_orders)
        st.metric("Overall Delay Rate", f"{delayed_pct:.2%}")
        st.metric("Average Delay Days", f"{avg_delay:.2f}")

        # Carrier leaderboard
        st.subheader("Carrier Delay Rates")
        fig = px.bar(carrier_delay_rate.reset_index(), x='Carrier', y='Is_Delayed',
                     labels={'Is_Delayed':'Delay Rate'}, title="Delay Rate by Carrier")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Delay by Priority")
        priority = df.groupby('Priority')['Is_Delayed'].mean().sort_values(ascending=False)
        fig2 = px.bar(priority.reset_index(), x='Priority', y='Is_Delayed', labels={'Is_Delayed':'Delay Rate'},
                      title="Delay Rate by Priority")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Cost vs Delay")
        if 'Total_Cost' in df.columns:
            fig3 = px.box(df, x='Is_Delayed', y='Total_Cost', labels={'Is_Delayed':'Delayed (1=Yes)'}, title="Cost distribution by delay")
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(df.head(200))

# ---------- Predict & Recommend ----------
elif mode == "Predict & Recommend":
    st.header("Predict Delay Probability & Get Recommendations")

    predict_mode = st.radio("Predict for", ["Batch (all orders in merged.csv)", "Single Manual Order"])
    if predict_mode == "Batch (all orders in merged.csv)":
        st.write("Predicting on the merged dataset...")
        X = prepare_features(df, encoders)
        if 'Is_Delayed' in X.columns:
            X = X.drop(columns=['Is_Delayed'])
        probs = model.predict_proba(X)[:,1]

        df_preds = df.copy()
        df_preds['Delay_Prob'] = probs
        # categorize risk
        df_preds['Risk_Level'] = pd.cut(df_preds['Delay_Prob'], bins=[-0.01, RISK_THRESH_MED, RISK_THRESH_HIGH, 1.01],
                                        labels=['Low','Medium','High'])
        st.success("Predictions complete.")
        st.dataframe(df_preds[['Order_Id','Carrier','Priority','Delay_Prob','Risk_Level']].sort_values('Delay_Prob', ascending=False).head(50))

        # Prescriptive actions column
        st.subheader("Top high-risk orders and recommendations")
        carrier_stats = carrier_delay_rate
        high_risk = df_preds[df_preds['Delay_Prob'] >= RISK_THRESH_HIGH].copy()
        if high_risk.empty:
            st.info("No high-risk orders detected.")
        else:
            # Build actionable column
            high_risk['Recommendations'] = high_risk.apply(lambda r: suggest_actions(r, r['Delay_Prob'], carrier_stats, sustainability_mode), axis=1)
            st.dataframe(high_risk[['Order_Id','Carrier','Priority','Delay_Prob','Recommendations']].head(50))

        # Download option
        st.download_button("Download predictions (CSV)", data=df_preds.to_csv(index=False), file_name="predictions_with_risk.csv")

    else:
        st.write("Enter order features for manual single-order prediction.")
        # Build input form based on model features (inferred)
        # Get input columns
        X_sample = prepare_features(df.head(1), encoders)
        input_cols = X_sample.columns.tolist()

        with st.form("predict_form"):
            inputs = {}
            for c in input_cols:
                # lightweight widgets based on dtype
                if df[c].dtype == 'int64' or df[c].dtype == 'float64':
                    inputs[c] = st.number_input(label=c, value=float(df[c].median()), key=c)
                else:
                    # offer selectbox using unique values from dataset
                    opts = df[c].unique().tolist()
                    inputs[c] = st.selectbox(label=c, options=opts, index=0, key=c)

            submitted = st.form_submit_button("Predict & Recommend")
            if submitted:
                single = pd.DataFrame([inputs])
                # encode
                single_enc = prepare_features(single, encoders)
                if 'Is_Delayed' in single_enc.columns:
                    single_enc = single_enc.drop(columns=['Is_Delayed'])
                prob = model.predict_proba(single_enc)[:,1][0]

                risk = "High" if prob >= RISK_THRESH_HIGH else "Medium" if prob >= RISK_THRESH_MED else "Low"
                st.metric("Predicted Delay Probability", f"{prob:.2f}")
                st.metric("Risk Level", risk)
                # suggestions
                suggestions = suggest_actions(single.iloc[0], prob, carrier_delay_rate, sustainability_mode)
                st.write("Recommendations:")
                for s in suggestions:
                    st.write("- " + s)

# ---------- Alerts & Export ----------
elif mode == "Alerts & Export":
    st.header("Alerts & Actionable Summary")
    st.write("Run batch predictions first in 'Predict & Recommend' to populate results.")
    if st.button("Run batch predictions now"):
        X = prepare_features(df, encoders)
        probs = model.predict_proba(X)[:,1]
        df_preds = df.copy()
        df_preds['Delay_Prob'] = probs
        df_preds['Risk_Level'] = pd.cut(df_preds['Delay_Prob'], bins=[-0.01, RISK_THRESH_MED, RISK_THRESH_HIGH, 1.01],
                                        labels=['Low','Medium','High'])
        # Alert summary
        total = len(df_preds)
        high = (df_preds['Risk_Level']=='High').sum()
        med = (df_preds['Risk_Level']=='Medium').sum()
        est_cost_impact = df_preds.loc[df_preds['Risk_Level'].isin(['Medium','High']), 'Total_Cost'].sum() * 0.1
        st.metric("Total Orders", total)
        st.metric("High Risk Orders", high)
        st.metric("Medium Risk Orders", med)
        st.metric("Estimated Potential Cost Impact (approx)", f"₹{est_cost_impact:,.0f}")
        st.write("Top recommended quick actions:")
        st.write("- Reassign high-risk orders to carriers with lower delay rates.")
        st.write("- Prioritize resources / buffer time for Express shipments.")
        st.write("- Consider sustainability-mode reassignments if appropriate.")
        st.download_button("Download high-risk orders", data=df_preds[df_preds['Risk_Level']=='High'].to_csv(index=False), file_name="high_risk_orders.csv")

# ==============================
# 📊 EXPLAINABLE AI MODE SECTION
# ==============================
elif mode == "Explain Model Predictions":
    st.header("🧠 Explainable AI — Why Did the Model Predict This?")

    # --- Load the data and model ---
    df = pd.read_csv("data/merged.csv")
    model = joblib.load("delay_predictor.pkl")
    encoder = joblib.load("encoders.pkl")  # if you saved label encoders

    # --- Define features (same as used during training) ---
    feature_cols = [
    'Customer_Segment', 'Priority', 'Product_Category',
    'Order_Value_Inr', 'Carrier', 'Promised_Delivery_Days',
    'Quality_Issue', 'Delivery_Cost_Inr', 'Rating', 'Would_Recommend',
    'Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance', 'Insurance',
    'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead', 'Total_Cost'
    ]

    X = df[feature_cols]

    # --- Encode categorical columns ---
    cat_cols = ['Customer_Segment', 'Priority', 'Product_Category', 'Carrier', 'Quality_Issue', 'Would_Recommend']
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes

    st.write("Generating SHAP explanations (this may take a few seconds)...")

    try:
        explainer = shap.TreeExplainer(model.get_booster())
    except Exception:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # --- Select order for explanation ---
    order_ids = df['Order_Id'].tolist()
    selected = st.selectbox("Select Order ID", order_ids)
    idx = df.index[df['Order_Id'] == selected][0]
    sample = X.iloc[[idx]]

    # --- Model prediction ---
    prob = model.predict_proba(sample)[0, 1]
    RISK_THRESH_HIGH = 0.7
    RISK_THRESH_MED = 0.4
    risk = (
        "High" if prob >= RISK_THRESH_HIGH
        else "Medium" if prob >= RISK_THRESH_MED
        else "Low"
    )

    # --- Display prediction info ---
    st.metric("Predicted Delay Probability", f"{prob:.2f}")
    st.metric("Risk Level", risk)


    shap_values_single = explainer.shap_values(sample)



    # --- Optional Waterfall Plot (more visual) ---
    st.subheader("Waterfall View for SHAP Explanation")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_single[0],
            base_values=explainer.expected_value,
            data=sample.iloc[0],
            feature_names=sample.columns
        ),
        max_display=10,
        show=False
    )
    st.pyplot(fig2)

    # --- Top 10 Influencing Features (Plotly Bar) ---
    st.subheader("Top Influencing Features")
    shap_df = pd.DataFrame({
        'Feature': sample.columns,
        'Impact': np.abs(shap_values_single[0])
    }).sort_values('Impact', ascending=False).head(10)

    fig3 = px.bar(
        shap_df,
        x='Impact',
        y='Feature',
        orientation='h',
        title="Top 10 Features Driving This Prediction",
        color='Impact',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # ============================================
    # 🌍 GLOBAL MODEL EXPLAINABILITY SECTION
    # ============================================

    st.markdown("---")
    st.header("🌍 Global Model Explainability — What Factors Drive Delays Overall?")

    with st.spinner("Computing global SHAP summary across all deliveries..."):
        # Reuse previously prepared X (encoded features)
        shap_values_all = explainer.shap_values(X)

        # --- Summary Beeswarm Plot ---
        st.subheader("📊 SHAP Summary Plot (All Deliveries)")
        fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values_all, X, show=False, plot_type="dot")
        st.pyplot(fig_summary)

        # --- Global Feature Importance ---
        st.subheader("🏆 Top Features Driving Delays Across All Deliveries")
        mean_abs_shap = np.abs(shap_values_all).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Mean |SHAP| Impact': mean_abs_shap
        }).sort_values(by='Mean |SHAP| Impact', ascending=False)

        fig_importance = px.bar(
            feature_importance.head(10),
            x='Mean |SHAP| Impact',
            y='Feature',
            orientation='h',
            color='Mean |SHAP| Impact',
            color_continuous_scale='blues',
            title='Top 10 Features Influencing Delivery Delays Globally'
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    st.success("✅ Global explainability generated successfully!")

st.markdown("---")
st.caption("Built for OFI NexGen Logistics - Predictive & Prescriptive demo")
