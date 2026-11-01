import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import joblib
import os
import subprocess
import sys

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# GLOBAL CSS STYLING
# -------------------------------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f8fafc, #eff6ff); font-family: 'Segoe UI', sans-serif; }
.main-title { text-align: center; font-size: 58px !important; font-weight: 900 !important; color: #1e3a8a !important; }
.stNumberInput input { border-radius: 12px !important; height: 50px !important; }
.stButton>button { background: #1e3a8a !important; color: white !important; font-weight: 700 !important; border-radius: 12px !important; height: 55px !important; }
/* Pure white background for containers with borders */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: white !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODEL & DATA
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "best_model.joblib"
    if not os.path.exists(model_path):
        if os.path.exists("train.py"):
            with st.status("🚀 Training initial AI model (first-time setup)..."):
                try:
                    subprocess.run([sys.executable, "train.py"], check=True)
                    st.success("Model trained successfully!")
                except Exception as e:
                    st.error(f"Failed to train model: {e}")
                    return None
        else:
            st.error("Model file not found and 'train.py' is missing!")
            return None
    return joblib.load(model_path)

@st.cache_data
def load_data():
    file_path = "data/diabetes.csv"
    if not os.path.exists(file_path):
        st.error(f"Critical Error: {file_path} not found! Please ensure 'data/diabetes.csv' exists.")
        # Return empty df with expected columns to prevent page crash
        return pd.DataFrame(columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
        ])
    
    df = pd.read_csv(file_path, header=None)
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df.columns = columns
    # Handle zeros: Glucose, BP, BMI can't be 0
    clean_cols = ["Glucose", "BloodPressure", "BMI"]
    for col in clean_cols:
        df[col] = pd.to_numeric(df[col].replace(0, pd.NA), errors='coerce')
    return df

model = load_model()
df = load_data()

if model is None or df.empty:
    st.error("Please ensure the project is correctly configured (run `python train.py` first if not using auto-retrain).")
    st.stop()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.markdown("<h1 class='main-title' style='font-size: 26px !important; text-align: left;'>Navigation</h1>", unsafe_allow_html=True)
    page = st.radio("Go to:", ["System Description", "Risk Predictor", "Data Analytics", "Clinical Insights", "About Application"], key="nav_page")
    st.markdown("---")

# -------------------------------------------------
# SYSTEM DESCRIPTION PAGE
# -------------------------------------------------
if page == "System Description":
    st.components.v1.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>", height=0)
    st.markdown("<div id='top'></div>", unsafe_allow_html=True)
    # --- Hero Title ---
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <h1 style='font-size:2.4rem; color:#1e3a8a; font-weight:900; margin-bottom:15px;'>
            Diabetes Risk Prediction Model using Random Forest
        </h1>
    </div>
    <hr style='border: 2px solid #1e3a8a; margin-bottom: 30px;'>
    """, unsafe_allow_html=True)

    # --- Introduction ---
    st.markdown("""
    <div style='background:linear-gradient(135deg,#eff6ff,#dbeafe); border-left:5px solid #1e3a8a;
                border-radius:10px; padding:25px; margin-bottom:20px;'>
        <h3 style='color:#1e3a8a; margin-top:0;'>📖 1. Introduction</h3>
        <p style='color:#1e293b; font-size:1rem; line-height:1.7; text-align:justify;'>
            Diabetes mellitus is a widespread chronic metabolic condition characterized by elevated blood glucose levels. 
            If left unmanaged, it leads to devastating long-term health complications, including cardiovascular disease, 
            chronic kidney failure, and vision loss. With the rapid evolution of digital health, <b>Artificial Intelligence (AI)</b> 
            is now a critical tool for medical predictive analytics. Our system leverages ensemble Machine Learning to analyze 
            clinical markers—such as <b>Glucose, Insulin, and BMI</b>—providing a modern, data-driven approach to early risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Problem Statement ---
    st.markdown("""
    <div style='background:linear-gradient(135deg,#fff7ed,#fed7aa); border-left:5px solid #ea580c;
                border-radius:10px; padding:25px; margin-bottom:20px;'>
        <h3 style='color:#c2410c; margin-top:0;'>🚨 2. Problem Statement</h3>
        <p style='color:#1e293b; font-size:1rem; line-height:1.7; text-align:justify;'>
            The global prevalence of diabetes is rising at an alarming rate, yet millions remain unaware of their 
            elevated risk until serious health complications occur. <b>The core problem is the hidden nature of 
            early-stage diabetes;</b> its symptoms are often so subtle that people skip diagnostic consultations 
            entirely. This project bridges that critical gap by providing an accessible, AI-powered <b>preliminary 
            screening tool</b>. We use predictive modeling to identify early metabolic warning signs, empowering 
            individuals to understand their risk and seek professional medical diagnosis before it's too late.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Challenges & Scope side by side ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:20px; height:100%;'>
            <h3 style='color:#1e3a8a; margin-top:0;'>⚡ 3. Existing Challenges</h3>
            <ul style='color:#1e293b; font-size:0.95rem; line-height:2;'>
                <li><b>Data Complexity</b>: Clinical indicators are non-linearly related.</li>
                <li><b>Accessibility Barriers</b>: Facilities concentrated in urban areas.</li>
                <li><b>Delayed Intervention</b>: Late-stage consultations reduce effectiveness.</li>
                <li><b>Data Interpretation</b>: Raw results difficult for non-technical users.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:20px; height:100%;'>
            <h3 style='color:#1e3a8a; margin-top:0;'>🎯 4. Scope of Project</h3>
            <ul style='color:#1e293b; font-size:0.95rem; line-height:2;'>
                <li><b>Model Training</b>: High-performance Random Forest Classifier.</li>
                <li><b>Screening Interface</b>: Streamlit web app for rapid risk feedback.</li>
                <li><b>Real-Time Processing</b>: Automated preprocessing pipeline.</li>
                <li><b>Validation Metrics</b>: Accuracy, Precision, Recall, F1-Score.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Methodology ---
    st.markdown("""
    <div style='background:#f0fdf4; border-left:5px solid #16a34a; border-radius:10px; padding:20px 25px; margin-bottom:20px;'>
        <h3 style='color:#15803d; margin-top:0;'>⚙️ 5. System Methodology</h3>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("Data Collection", "PIMA Indians Diabetes Dataset — 768 records, 8 features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DPF, Age."),
        ("Data Cleaning", "Zero values in critical columns (Glucose, Blood Pressure, BMI) identified and handled to eliminate invalid entries."),
        ("Feature Engineering", "All 8 features selected and validated for their contribution to diabetes risk prediction."),
        ("Model Training", "Random Forest Classifier (500 trees, max_depth=6, class_weight=balanced) trained on 80% of the dataset."),
        ("Model Evaluation", "Evaluated using Accuracy (82%), Precision (80%), Recall (78%), and F1-Score (79%)."),
        ("Model Serialization", "Trained model pipeline saved using Joblib for efficient real-time loading."),
        ("Application Interface", "Interactive web interface built with Streamlit for instant risk prediction."),
    ]
    for i, (title, desc) in enumerate(steps):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        st.markdown(f"""
        <div style='background:{bg}; border:1px solid #e2e8f0; border-radius:8px; padding:12px 18px; margin-bottom:8px;'>
            <b style='color:#1e3a8a; font-size:1rem;'>Step {i+1}: {title}</b>
            <p style='color:#334155; margin:4px 0 0 0; font-size:0.92rem;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Model Performance ---
    st.markdown("""
    <div style='background:#faf5ff; border-left:5px solid #7c3aed; border-radius:10px; padding:20px 25px; margin-bottom:20px;'>
        <h3 style='color:#6d28d9; margin-top:0;'>📈 6. Model Performance</h3>
    </div>
    """, unsafe_allow_html=True)

    perf_col, _ = st.columns([1, 1])
    with perf_col:
        st.markdown("""
<table style='width:100%; border-collapse:collapse;'>
  <thead>
    <tr style='background-color:#1e3a8a; color:white;'>
      <th style='padding:12px 16px; text-align:left; border:1px solid #c7d2fe;'>Metric</th>
      <th style='padding:12px 16px; text-align:left; border:1px solid #c7d2fe;'>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr style='background-color:#f0f4ff;'>
      <td style='padding:11px 16px; border:1px solid #ddd; color:#000000;'>Accuracy</td>
      <td style='padding:11px 16px; border:1px solid #ddd; color:#000000;'><b>82%</b></td>
    </tr>
    <tr>
      <td style='padding:11px 16px; border:1px solid #ddd; color:#000000;'>Precision</td>
      <td style='padding:11px 16px; border:1px solid #ddd; color:#000000;'><b>80%</b></td>
    </tr>
    <tr style='background-color:#f0f4ff;'>
      <td style='padding:11px 16px; border:1px solid #ddd; color:#000000;'>Recall</td>
      <td style='padding:11px 16px; border:1px solid #ddd; color:#000000;'><b>78%</b></td>
    </tr>
    <tr>
      <td style='padding:11px 16px; border:1px solid #ddd; color:#000000;'>F1 Score</td>
      <td style='padding:11px 16px; border:1px solid #ddd; color:#000000;'><b>79%</b></td>
    </tr>
  </tbody>
</table>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Key Insights ---
    st.markdown("""
    <div style='background:linear-gradient(135deg,#fef9c3,#fef08a); border-left:5px solid #ca8a04;
                border-radius:10px; padding:20px 25px; margin-bottom:20px;'>
        <h3 style='color:#92400e; margin-top:0;'>🔍 7. Key Insights</h3>
        <ul style='color:#1e293b; font-size:1rem; line-height:2;'>
            <li>🔬 <b>Glucose Level</b> has the strongest correlation with diabetes.</li>
            <li>⚖️ <b>Higher BMI</b> increases diabetes risk significantly.</li>
            <li>👴 <b>Age and Pregnancies</b> also contribute to prediction accuracy.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- Disclaimer ---
    st.markdown("""
    <div style='background:#fef2f2; border:1px solid #fca5a5; border-radius:10px; padding:15px 20px;'>
        <p style='color:#991b1b; margin:0; font-size:0.95rem;'>
            ⚠️ <b>Important Disclaimer:</b> This system is designed as a support tool for preliminary risk screening 
            and is <b>not intended to replace professional medical diagnosis</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# RISK PREDICTOR PAGE
# -------------------------------------------------
elif page == "Risk Predictor":
    st.components.v1.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>", height=0)
    st.markdown("<div id='top'></div>", unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: #475569;'>Enter clinical metrics to assess the probability of metabolic dysfunction.</p>", unsafe_allow_html=True)

    if 'bmi_val' not in st.session_state:
        st.session_state['bmi_val'] = 25.0
    if 'bmi_msg' not in st.session_state:
        st.session_state['bmi_msg'] = None

    left, right = st.columns([1.2, 1])

    with left:
        with st.container(border=True):
            st.subheader("📋 Patient Clinical Data")
            with st.expander("📏 Don't know your BMI? (Calculator Tool)"):
                cw1, cw2 = st.columns(2)
                with cw1: cw_w = st.number_input("Weight (kg)", 20.0, 300.0, 70.0, key="calc_w")
                with cw2: cw_h = st.number_input("Height (cm)", 50.0, 250.0, 170.0, key="calc_h")
                if st.button("✅ Calculate & Apply BMI"):
                    bmi_calc = round(cw_w / ((cw_h/100)**2), 1)
                    st.session_state['bmi_val'] = bmi_calc
                    st.session_state['bmi_msg'] = f"Calculated BMI: **{bmi_calc}** and applied to form!"
                    st.rerun()
                
                if st.session_state['bmi_msg']:
                    st.success(st.session_state['bmi_msg'])
                    st.session_state['bmi_msg'] = None
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Gender", ["Female"])
                preg = st.number_input("Pregnancies", 0, 20, 1)
                gluc = st.number_input("Glucose (mg/dL)", 0, 300, 120)
                skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
                bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, st.session_state['bmi_val'])
            with col2:
                age = st.number_input("Age (Years)", 0, 120, 30)
                bp = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
                ins = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
                dpf = st.number_input("DPF (Genetics)", 0.0, 3.0, 0.5)
            predict_btn = st.button("🚀 Run Analysis")

    with right:
        st.markdown("<h3 style='margin-bottom: 0px;'>🧠 Intelligence Output</h3>", unsafe_allow_html=True)
        
        # Default state or results
        if predict_btn:
            input_df = pd.DataFrame([{
                "Pregnancies": preg, "Glucose": gluc, "BloodPressure": bp,
                "SkinThickness": skin, "Insulin": ins, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age
            }])
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
            prob = model.predict_proba(input_df)[0][1] * 100
            
            if prob < 35: lab, color, bg = "LOW RISK", "#10b981", "#dcfce7"
            elif prob < 65: lab, color, bg = "MODERATE RISK", "#f59e0b", "#fef3c7"
            else: lab, color, bg = "HIGH RISK", "#ef4444", "#fee2e2"

            # Result Risk Label exactly above the gauge
            st.markdown(f"<h3 style='text-align:center; color:{color}; margin-bottom:-65px; font-weight:bold;'>{lab}</h3>", unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob,
                number={'suffix': "%", 'font': {'color': color, 'size': 60}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#ddd",
                    'steps': [
                        {'range': [0, 35], 'color': '#dcfce7'},
                        {'range': [35, 65], 'color': '#fef3c7'},
                        {'range': [65, 100], 'color': '#fee2e2'}
                    ],
                }
            ))
            fig.update_layout(height=350, margin=dict(l=30, r=30, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
            
            # Status Box with centered text
            st.markdown(f"""
                <div style='background:{bg}; color:{color}; padding:10px 15px; border-radius:10px; 
                border:1px solid {color}; text-align:center; font-size:1rem; margin-bottom:20px;'>
                    <span style='font-weight:bold;'>Status:</span> {lab}
                </div>
            """, unsafe_allow_html=True)

            # Recommendations
            st.markdown(f"### 🥗 Recommendations for {lab}")
            
            if lab == "LOW RISK":
                st.markdown("""
                - **Physical Activity**: Maintain at least 150 minutes of moderate activity per week.
                - **Diet**: Continue a balanced diet rich in leafy greens and lean proteins.
                - **Checkups**: Schedule annual metabolic screenings to monitor trends.
                """)
            elif lab == "MODERATE RISK":
                st.markdown("""
                - **Monitor Levels**: Re-test blood glucose levels in 3-6 months.
                - **Dietary Adjustments**: Reduce processed sugar intake and increase fiber.
                - **Daily Activity**: Aim for 30 minutes of brisk walking or swimming daily.
                """)
            else: # HIGH RISK
                st.markdown("""
                - **Immediate Action**: Consult an **Endocrinologist or GP** immediately.
                - **Diagnostics**: Request a full metabolic panel and fasting insulin test.
                - **Lifestyle**: Begin a medically supervised low-carb nutritional plan.
                """)
        else:
            # Placeholder content when no prediction is made
            st.info("👈 Enter clinical data and click 'Run Analysis' to see results here.")

    st.caption("⚠ Educational use only. Consult a professional.")

# -------------------------------------------------
# DATA ANALYTICS PAGE
# -------------------------------------------------
elif page == "Data Analytics":
    st.components.v1.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>", height=0)
    st.markdown("<div id='top'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='display: flex; align-items: center; margin-bottom: 20px;'>
            <span style='font-size: 3rem; margin-right: 15px;'>📊</span>
            <div>
                <h1 style='margin: 0; color: #1e3a8a;'>Professional Clinical Intelligence</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Top Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            st.metric("Patient Records", len(df))
    with m2:
        with st.container(border=True):
            st.metric("Mean Glucose", f"{df['Glucose'].mean():.1f}")
    with m3:
        with st.container(border=True):
            st.metric("Critical Prevalence", f"{(df['Outcome'].mean()*100):.1f}%")
    with m4:
        with st.container(border=True):
            st.metric("Sample BMI Avg", f"{df['BMI'].mean():.1f}")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Benchmarking Section
    st.markdown("<h3>🏢 Patient Benchmarks (Average Values)</h3>", unsafe_allow_html=True)
    col_bench1, col_bench2 = st.columns([1.5, 1])
    
    with col_bench1:
        with st.container(border=True):
            st.markdown("**Healthy vs Diabetic Comparison**")
            avg_comparison = df.groupby('Outcome')[['Glucose', 'BMI', 'Age', 'BloodPressure']].mean().reset_index()
            avg_comparison['Outcome'] = avg_comparison['Outcome'].map({0: 'Healthy', 1: 'Diabetic'})
            
            # Melt to create the grouped bar chart
            melted_avg = avg_comparison.melt(id_vars='Outcome', var_name='variable', value_name='value')
            
            fig_bench = px.bar(
                melted_avg, 
                x='Outcome', y='value', color='variable',
                barmode='group',
                color_discrete_map={'Glucose': '#3b82f6', 'BMI': '#ef4444', 'Age': '#f59e0b', 'BloodPressure': '#475569'},
                height=400
            )
            fig_bench.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bench, width="stretch")

    with col_bench2:
        with st.container(border=True):
            st.markdown("**Benchmarking Indicators (Averages)**")
            full_avg = df.groupby('Outcome').mean().T
            full_avg.columns = ['Healthy', 'Diabetic']
            st.table(full_avg.style.format("{:.4f}"))

    st.markdown("---")
    
    # Interactive Tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Feature Analysis", "Population Profile", "🔥 Correlation Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Glucose & BMI Intelligence")
            df_plot = df.copy()
            df_plot['color'] = df_plot['Outcome'].astype(str)
            fig_scat = px.scatter(
                df_plot, x="Glucose", y="BMI", 
                color="color",
                color_discrete_map={"0": "#3b82f6", "1": "#ef4444"}, 
                opacity=0.6,
                labels={"color": "color"}
            )
            fig_scat.add_vline(x=140, line_dash="dash", line_color="red", annotation_text="Pre-Diabetes (140+)")
            fig_scat.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Obesity Range (30+)")
            fig_scat.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scat, width="stretch")
            
        with col2:
            st.markdown("### 📦 Feature Distribution")
            metric_option = st.selectbox("Compare metric:", options=["Glucose", "BMI", "Age", "BloodPressure", "Insulin"])
            
            fig_box = px.box(
                df_plot, x="color", y=metric_option, 
                color="color", 
                points="all",
                color_discrete_map={"0": "#3b82f6", "1": "#ef4444"},
                labels={"color": "color", "x": ""}
            )
            if metric_option == "Glucose":
                fig_box.add_hline(y=140, line_dash="dash", line_color="red", annotation_text="Threshold")
            else:
                fig_box.add_hline(y=df[metric_option].mean(), line_dash="dash", line_color="grey", annotation_text="Threshold")
                
            fig_box.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box, width="stretch")

    with tab2:
        col_age, col_target = st.columns(2)
        with col_age:
            st.markdown("### Age Profile")
            fig_age = px.histogram(
                df_plot, x="Age", color="color", 
                barmode="overlay",
                color_discrete_map={"0": "#3b82f6", "1": "#ef4444"},
                labels={"color": "color"}
            )
            fig_age.add_vline(x=45, line_dash="dash", line_color="black", annotation_text="Rising Risk (45+)")
            fig_age.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_age, width="stretch")
            
        with col_target:
            st.markdown("### ⚖ Target Balance")
            # Donut chart with percentages (Image 3)
            fig_pie = px.pie(
                df_plot, names="color", hole=0.6,
                color="color",
                color_discrete_map={"0": "#3b82f6", "1": "#ef4444"}
            )
            fig_pie.update_traces(textinfo='percent', textposition='inside')
            fig_pie.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, width="stretch")

    with tab3:
        st.markdown("### Health Correlation Matrix")
        corr = df.apply(pd.to_numeric, errors='coerce').corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", 
            color_continuous_scale='RdBu_r'
        )
        fig_corr.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, width="stretch")

    st.markdown("---")
    
    # Raw Data Section (Image 1)
    st.markdown("### 📋 Raw Clinical Dataset (Sample View)")
    # Resetting index to start from 1 as shown in image
    df_sample = df.head(15).copy()
    df_sample.index = df_sample.index + 1
    st.dataframe(
        df_sample.style.background_gradient(cmap='Blues'),
        use_container_width=True
    )

# -------------------------------------------------
# CLINICAL INSIGHTS PAGE
# -------------------------------------------------
elif page == "Clinical Insights":
    st.components.v1.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>", height=0)
    st.markdown("<div id='top'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align:center; padding: 0 0 20px 0;'>
            <h2 style='color:#000000; font-weight:800; margin-bottom:5px; margin-top:0;'>🩺 Clinical Insights Library</h2>
            <p style='color:#64748b; font-size:1rem; margin:0;'>Essential clinical facts and common myths about diabetes.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # --- Blue Info Cards Grid ---
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:#dbeafe; border:1px solid #bfdbfe; border-radius:12px; padding:20px; height:130px; margin-bottom:15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>
            <p style='color:#1e40af; font-size:0.95rem; margin:0; line-height:1.6;'>
                💡 <b style='color:#1e3a8a;'>Glucose Fact:</b> Normal fasting blood sugar is below 99 mg/dL. Pre-diabetes is between 100-125 mg/dL.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#dbeafe; border:1px solid #bfdbfe; border-radius:12px; padding:20px; height:130px; margin-bottom:15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>
            <p style='color:#1e40af; font-size:0.95rem; margin:0; line-height:1.6;'>
                💡 <b style='color:#1e3a8a;'>Insulin Fact:</b> High insulin (Hyperinsulinemia) often develops years before blood sugar starts to rise.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:#dbeafe; border:1px solid #bfdbfe; border-radius:12px; padding:20px; height:130px; margin-bottom:15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>
            <p style='color:#1e40af; font-size:0.95rem; margin:0; line-height:1.6;'>
                💡 <b style='color:#1e3a8a;'>BMI Fact:</b> A BMI over 30 is clinically classified as Obesity and is the #1 modifiable risk factor for type 2 diabetes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#dbeafe; border:1px solid #bfdbfe; border-radius:12px; padding:20px; height:130px; margin-bottom:15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>
            <p style='color:#1e40af; font-size:0.95rem; margin:0; line-height:1.6;'>
                💡 <b style='color:#1e3a8a;'>Genetics Fact:</b> DPF (Diabetes Pedigree Function) measures your genetic risk. Even with a high score, lifestyle changes can significantly lower your actual risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:25px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#000000; border-left:5px solid #000000; padding-left:15px;'>🔍 Common Myths</h3>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
    
    with st.expander("🛑 Myth: Eating too much sugar causes diabetes."):
        st.success("✅ **Fact:** Type 1 diabetes is autoimmune. Type 2 is complex and heavily involves insulin resistance, weight, and family history, not strictly sugar intake alone.")
        
    with st.expander("🛑 Myth: You will always display clear symptoms if you have diabetes."):
        st.success("✅ **Fact:** Type 2 diabetes often has mild or no symptoms in its early stages. Many live with it for years without knowing, which is why routine screening is crucial.")
        
    with st.expander("🛑 Myth: People with diabetes are not allowed to eat carbohydrates."):
        st.success("✅ **Fact:** Carbohydrates and sweets can be eaten safely. The key is strict portion control, monitoring the Glycemic Index, and pairing carb-heavy foods with fiber and protein.")
        
    with st.expander("🛑 Myth: Only overweight or obese people can develop type 2 diabetes."):
        st.success("✅ **Fact:** While clinical obesity is a major risk factor, up to 15% of people with type 2 diabetes are at a healthy weight. Age, genetics, and ethnicity play massive roles.")

    st.markdown("---")
    
    st.markdown("<h3 style='text-align:center;'>📊 The 'Lean Diabetes' Statistical Phenomenon</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#475569; font-size:1rem; margin-bottom:20px;'>A major clinical misconception is that only clinically overweight individuals develop Type 2 diabetes. However, up to 15% of diagnosed patients maintain a completely normal-range BMI.</p>", unsafe_allow_html=True)
    
    labels = ['Overweight / Obese (BMI > 25)', 'Lean / Normal (BMI < 25)']
    values = [85, 15]
    colors = ['#f43f5e', '#10b981']
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.65, 
        marker=dict(colors=colors, line=dict(color='#ffffff', width=3)),
        textinfo='label+percent',
        textposition='outside',
        showlegend=False,
        pull=[0, 0.05]
    )])
    
    fig_donut.update_layout(
        height=380,
        margin=dict(t=10, b=10, l=0, r=0),
        annotations=[dict(text='<b style="color:#1e3a8a;">Type 2<br>Diabetics</b>', x=0.5, y=0.5, font_size=17.5, showarrow=False)]
    )
    
    st.plotly_chart(fig_donut, width="stretch")
    st.caption("Clinical population data: Proportions are based on global Type 2 diabetes demographics.")

# -------------------------------------------------
# ABOUT APPLICATION PAGE
# -------------------------------------------------
elif page == "About Application":
    st.components.v1.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>", height=0)
    st.markdown("<div id='top'></div>", unsafe_allow_html=True)
    # --- Hero Title ---
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <h1 style='font-size:2.4rem; color:#1e3a8a; font-weight:900; margin-bottom:15px;'>
            Diabetes Risk Prediction Model using Random Forest
        </h1>
    </div>
    <hr style='border: 2px solid #1e3a8a; margin-bottom: 30px;'>
    """, unsafe_allow_html=True)

    # --- Overview Section ---
    st.markdown("""
    <div style='background:linear-gradient(135deg,#eff6ff,#dbeafe); border-left:5px solid #1e3a8a;
                border-radius:10px; padding:25px; margin-bottom:25px; text-align:center;'>
        <p style='color:#1e293b; font-size:1.1rem; font-weight:600; margin-bottom:15px;'>
            This project is a Machine Learning-powered Diabetes Risk Prediction System designed to identify 
            potential metabolic risks in patients and provide actionable health insights.
        </p>
        <p style='color:#334155; font-size:1rem; line-height:1.7; text-align:justify;'>
            This project takes in basic patient health data (like age, blood sugar, BMI, and genetic history) 
            and uses a trained Machine Learning algorithm to instantly calculate their risk of developing diabetes. 
            If the AI detects a moderate or high risk, the app alerts the user and provides personalized health 
            recommendations. Ultimately, this system helps catch early warning signs so people can take 
            preventative action before any serious symptoms appear.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Key Features & Components side by side ---
    f_col, c_col = st.columns(2)
    with f_col:
        st.markdown("""
        <div style='background:#ffffff; border:1px solid #dbeafe; border-radius:12px; padding:25px; height:480px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
            <h3 style='color:#1e3a8a; margin-top:0; border-bottom:2px solid #eff6ff; padding-bottom:12px; font-weight:800;'>🎯 Key Features</h3>
            <ul style='color:#334155; font-size:1rem; line-height:2; padding-left:20px;'>
                <li style='margin-bottom:15px;'><b style='color:#1e3a8a;'>Risk Predictor:</b> A real-time diagnostic tool where users enter clinical data (Glucose, BMI, Age, etc.) to get a risk score (Low, Moderate, or High) along with personalized medical recommendations.</li>
                <li style='margin-bottom:15px;'><b style='color:#1e3a8a;'>Data Analytics Dashboard:</b> A professional intelligence suite that visualizes patient trends, correlation matrices, and medical "danger zones" (e.g., obesity thresholds and pre-diabetes levels).</li>
                <li style='margin-bottom:15px;'><b style='color:#1e3a8a;'>Integrated Tools:</b> Includes a built-in BMI calculator and a clinical data entry form optimized for medical workflows.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c_col:
        st.markdown("""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:20px; height:420px;'>
            <h3 style='color:#1e3a8a; margin-top:0; border-bottom:2px solid #1e3a8a; padding-bottom:8px;'>🧪 Key Components</h3>
            <ul style='color:#1e293b; font-size:0.95rem; line-height:1.8; padding-left:20px;'>
                <li><b>Algorithm</b>: Random Forest Classifier (Supervised Learning).</li>
                <li><b>Data Source</b>: PIMA Indians Diabetes Dataset.</li>
                <li><b>Pipeline</b>: Automated StandardScaler & Joblib serialization.</li>
                <li><b>UI Interface</b>: Interactive Plotly gauges and Streamlit containers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Frontend & Backend side by side ---
    fe_col, be_col = st.columns(2)
    with fe_col:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#f0f9ff,#e0f2fe); border:1px solid #bae6fd; border-radius:10px; padding:20px; height:300px;'>
            <h3 style='color:#0369a1; margin-top:0;'>🖥️ Frontend</h3>
            <ul style='color:#0c4a6e; font-size:0.92rem; line-height:1.8; padding-left:20px;'>
                <li><b>Framework</b>: Streamlit (Python-powered UI).</li>
                <li><b>Styling</b>: Custom CSS & HTML (for clinical dashboard layout).</li>
                <li><b>Visualization</b>: Plotly (Interactive Gauge charts and scatter plots).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with be_col:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#f7fee7,#ecfccb); border:1px solid #d9f99d; border-radius:10px; padding:20px; height:300px;'>
            <h3 style='color:#3f6212; margin-top:0;'>⚙️ Backend</h3>
            <ul style='color:#1a2e05; font-size:0.92rem; line-height:1.8; padding-left:20px;'>
                <li><b>Language</b>: Python 3.10.</li>
                <li><b>ML Library</b>: Scikit-Learn (Random Forest Classifier).</li>
                <li><b>Data Processing</b>: Pandas & NumPy (for clinical datasets).</li>
                <li><b>Model Management</b>: Joblib (for loading pre-trained weights).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ML Algorithms Used ---
    st.markdown("""
    <div style='background:#faf5ff; border-left:5px solid #7c3aed; border-radius:10px; padding:20px 25px; margin-bottom:20px;'>
        <h2 style='color:#6d28d9; margin-top:0; font-size:1.6rem;'>🤖 ML Algorithms Used</h2>
        <p style='color:#5b21b6; font-size:1rem; line-height:1.7; margin-bottom:0;'>
            Our predictive engine uses <b>Random Forest Classification</b> to determine metabolic risk. 
            By averaging outputs from multiple decision trees, the system ensures maximum diagnostic 
            accuracy and stability across clinical medical datasets.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""
        <div style='background:white; border:1px solid #ddd; border-top:4px solid #10b981; border-radius:8px; padding:15px; height:220px;'>
            <b style='color:#047857;'>🌲 Random Forest Classifier</b><br>
            <p style='font-size:0.88rem; color:#374151; margin-top:8px; line-height:1.5;'>
                The core engine for prediction. It builds multiple <b>decision trees</b> and averages their outputs 
                to detect complex risk patterns, ensuring high diagnostic accuracy in medical settings.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div style='background:white; border:1px solid #ddd; border-top:4px solid #3b82f6; border-radius:8px; padding:15px; height:220px;'>
            <b style='color:#1d4ed8;'>📏 Standard Scaler</b><br>
            <p style='font-size:0.88rem; color:#374151; margin-top:8px; line-height:1.5;'>
               Normalizes metrics like <b>Insulin and Genetics</b> to a standard range. This ensures every 
               clinical marker—regardless of its numeric size—has a <b>balanced impact</b> on the final result.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown("""
        <div style='background:white; border:1px solid #ddd; border-top:4px solid #f59e0b; border-radius:8px; padding:15px; height:220px;'>
            <b style='color:#b45309;'>🔀 Train-Test Split</b><br>
            <p style='font-size:0.88rem; color:#374151; margin-top:8px; line-height:1.5;'>
                Partitions the <b>clinical dataset</b> into testing and training sets. This verifies the model 
                is truly <b>generalizing</b> and can accurately assess risk for completely new patients.
            </p>
        </div>
        """, unsafe_allow_html=True)