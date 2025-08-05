import os
import time

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import statsmodels.api as sm
from streamlit_folium import st_folium
from openai import OpenAI

# -----------------------------
# OPENAI API Setup
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Synthetic Sensor Data Generator
# -----------------------------
def generate_synthetic_data(temp=None, hum=None, stress=None, pest=None, nutrient=None):
    return {
        "Temperature (°C)": float(temp) if temp is not None else np.random.uniform(18, 30),
        "Humidity (%)":    float(hum)   if hum   is not None else np.random.uniform(50, 90),
        "Germination Rate (%)": np.random.uniform(70, 100),
        "Stress Index (%)":      float(stress) if stress is not None else np.random.uniform(0, 50),
        "Pest Risk (%)":         float(pest)   if pest   is not None else np.random.uniform(0, 30),
        "Nutrient Level (%)":    float(nutrient) if nutrient is not None else np.random.uniform(50, 100)
    }

# -----------------------------
# Gauge/Tachometer Helper
# -----------------------------
def create_gauge(title, val, lo, hi, unit, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': f" {unit}", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [lo, hi]},
            'bar': {'color': color},
            'steps': [
                {'range': [lo, (hi-lo)/2+lo], 'color': "#E5F5E0"},
                {'range': [(hi-lo)/2+lo, hi], 'color': "#FEE0D2"},
            ]
        }
    ))
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        # ensure all text has room
    )
    return fig

# -----------------------------
# OLS Monte Carlo
# -----------------------------
def run_econometric_model(data, sims=50):
    rows = [
        generate_synthetic_data(
            temp = data["Temperature (°C)"]   + np.random.normal(0,0.5),
            hum  = data["Humidity (%)"]        + np.random.normal(0,1.0),
            stress=data["Stress Index (%)"]    + np.random.normal(0,1.0),
            pest = data["Pest Risk (%)"]       + np.random.normal(0,1.0),
            nutrient=data["Nutrient Level (%)"]+ np.random.normal(0,1.0)
        )
        for _ in range(sims)
    ]
    df = pd.DataFrame(rows)
    X = sm.add_constant(df[[
        "Temperature (°C)",
        "Humidity (%)",
        "Stress Index (%)",
        "Pest Risk (%)",
        "Nutrient Level (%)"
    ]])
    y = df["Germination Rate (%)"]
    return sm.OLS(y, X).fit()

# -----------------------------
# Streamlit App Setup
# -----------------------------
st.set_page_config(page_title="SmartAgri AI Platform", layout="wide")
st.title("PhenoSense - Smart Greenhouse Dashboard")

tabs = st.tabs([
    "Dashboard", 
    "AI Agent", 
    "Econometric ROI", 
    "Geospatial Map"
])

# ───────────────────────────────────────────────────────────
# TAB 1: Dashboard + Toggles + Playback + Auto‐Refresh
# ───────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Real-time Dashboard")

    # Species / Zone / Refresh
    species = st.selectbox("Crop Species", ["Lettuce", "Strawberry", "Tomato", "Barley"])
    zone    = st.selectbox("Greenhouse Zone", ["Zone A", "Zone B", "Zone C"])
    auto_refresh = st.checkbox("Auto-Refresh (2s)", value=True)

    # Scenario sliders
    cols = st.columns(5)
    temp_control    = cols[0].slider("Temperature (°C)",  10, 40,  25)
    hum_control     = cols[1].slider("Humidity (%)",      30,100,  70)
    stress_control  = cols[2].slider("Stress Index (%)",   0,100,  20)
    pest_control    = cols[3].slider("Pest Risk (%)",      0,100,  10)
    nutrient_control= cols[4].slider("Nutrient Level (%)", 0,100,  80)

    # Time-series playback
    playback = st.checkbox("Enable Playback Slider")
    if playback:
        length = st.slider("Playback Length (steps)", 2, 20, 5)
        step   = st.slider("Playback Step", 1, length, 1)
        # Generate series on‐the‐fly (demo)
        series = [
            generate_synthetic_data(
                temp=temp_control,
                hum=hum_control,
                stress=stress_control,
                pest=pest_control,
                nutrient=nutrient_control
            )
            for _ in range(length)
        ]
        data = series[step-1]
    else:
        data = generate_synthetic_data(
            temp=temp_control,
            hum=hum_control,
            stress=stress_control,
            pest=pest_control,
            nutrient=nutrient_control
        )

    # Display gauges
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(create_gauge("Temperature",    data["Temperature (°C)"],    0,40,"°C","red"), use_container_width=True,
    config={"responsive": True})
        st.plotly_chart(create_gauge("Humidity",       data["Humidity (%)"],         0,100,"%","blue"), use_container_width=True,
    config={"responsive": True})
    with c2:
        st.plotly_chart(create_gauge("Germination",    data["Germination Rate (%)"], 0,100,"%","green"), use_container_width=True,
    config={"responsive": True})
        st.plotly_chart(create_gauge("Stress Index",   data["Stress Index (%)"],     0,100,"%","orange"), use_container_width=True,
    config={"responsive": True})
    with c3:
        st.plotly_chart(create_gauge("Pest Risk",      data["Pest Risk (%)"],        0,100,"%","purple"), use_container_width=True,
    config={"responsive": True})
        st.plotly_chart(create_gauge("Nutrient Level", data["Nutrient Level (%)"],   0,100,"%","teal"), use_container_width=True,
    config={"responsive": True})

    st.subheader("Sensor Data")
    st.dataframe(pd.DataFrame([data]).round(2))

    if auto_refresh and not playback:
     time.sleep(2)
     st.rerun()


# ───────────────────────────────────────────────────────────
# TAB 2: Agentic AI Insights
# ───────────────────────────────────────────────────────────
with tabs[1]:
    st.header("AI Phenotyping Agent")
    ai_prompt = (
        f"Crop: {species} | Zone: {zone}\n"
        f"Temp: {temp_control} °C | Humidity: {hum_control}%\n"
        f"Stress: {stress_control}% | Pest: {pest_control}% | Nutrient: {nutrient_control}%\n"
        "Provide actionable agronomic advice."
    )
    if st.button("Run AI Agent"):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are an expert AI agronomist."},
                    {"role":"user","content":ai_prompt}
                ]
            )
            st.success(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"API call failed: {e}")

# ───────────────────────────────────────────────────────────
# TAB 3: Econometric ROI Prediction
# ───────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Econometric ROI (OLS)")
    model = run_econometric_model(data, sims=50)
    pred  = model.predict()[0]
    st.metric("Predicted Germination Rate", f"{pred:.2f}%")
    st.subheader("Model Summary")
    st.text(model.summary())

# ───────────────────────────────────────────────────────────
# TAB 4: Geospatial Map of Ireland
# ───────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Ireland Greenhouse Network")
    m = folium.Map(location=[53.4, -8.2], zoom_start=6)
    points = {
        "Dublin":  [53.3331, -6.2489],
        "Cork":    [51.8985, -8.4756],
        "Galway":  [53.2707, -9.0568],
        "Limerick":[52.6680, -8.6305]
    }
    for city, coord in points.items():
        folium.Marker(
            coord,
            popup=f"<b>{city}</b><br>Stress: {stress_control}%"
        ).add_to(m)
    st_folium(m, width=700, height=500)

# ──────────────────────────────────────────────
# FOOTER ANNOTATIONS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .footer {
        position: Draggable;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(0, 0, 0, 0.85);
        text-align: center;
        padding: 10px 0;
    }
    </style>
    <div class='footer'>
        <h4 style='color: #2ecc71; margin: 0;'>Developed and Designed by <b>Jit</b></h4>
        <p style='font-size: 14px; color: #bdc3c7; margin: 3px 0;'>
            <i>PhenoSense © 2025 – AI & Econometric Intelligence for Next-Gen Agriculture</i>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

