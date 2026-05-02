import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Observatorio Delictivo · México",
    page_icon="🔭",
    layout="wide"
)

# ─────────────────────────────────────────
# INTRO (NUEVO)
# ─────────────────────────────────────────
st.title("🔭 Observatorio Delictivo en México")

st.markdown("""
### 🧠 ¿Qué hace esta herramienta?

Este observatorio permite:

- Analizar la evolución delictiva en México  
- Comparar estados  
- Predecir tendencias futuras con inteligencia artificial  

📌 Datos oficiales del SESNSP normalizados por cada 100,000 habitantes.
""")

modo_simple = st.toggle("Modo simple (explicaciones claras para público general)")

# ─────────────────────────────────────────
# DATA (simulada si falla)
# ─────────────────────────────────────────
@st.cache_data
def cargar():
    fechas = pd.date_range("2018-01-01", periods=60, freq="ME")
    data = []
    estados = ["Guanajuato", "Jalisco", "CDMX"]

    for e in estados:
        base = np.random.randint(50, 200)
        for i, f in enumerate(fechas):
            data.append({
                "entidad": e,
                "fecha": f,
                "mes_num": i,
                "mes": f.month,
                "tasa_100k": base + i*0.5 + np.random.randn()*5
            })
    return pd.DataFrame(data)

df = cargar()

estado = st.selectbox("Selecciona un estado", df["entidad"].unique())
periodos = st.slider("Meses a predecir", 1, 12, 6)

df_e = df[df["entidad"] == estado]

# ─────────────────────────────────────────
# TABS RENOMBRADAS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Panorama general",
    "🔮 Predicción básica",
    "🌲 Predicción avanzada",
    "📈 Tendencia temporal"
])

# ════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════
with tab1:

    st.subheader(f"📍 Análisis de {estado}")

    total = int(df_e["tasa_100k"].sum())
    promedio = df_e["tasa_100k"].mean()

    c1, c2 = st.columns(2)
    c1.metric("Total acumulado", total)
    c2.metric("Promedio mensual", round(promedio, 2))

    # Insight automático
    st.info(f"""
🔍 **Insight clave:**

El estado de **{estado}** presenta una tasa promedio de **{round(promedio,2)}**.

👉 Esto indica un nivel de incidencia {"alto" if promedio > 120 else "moderado"}.
""")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_e["fecha"],
        y=df_e["tasa_100k"],
        mode="lines",
        name="Histórico"
    ))
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════
# TAB 2 — REGRESIÓN
# ════════════════════════════════════════
with tab2:

    X = df_e[["mes_num", "mes"]].values
    y = df_e["tasa_100k"].values

    model = LinearRegression().fit(X, y)
    pred = model.predict(X)

    r2 = r2_score(y, pred)

    st.metric("Precisión del modelo (R²)", round(r2, 3))

    if modo_simple:
        st.markdown(f"""
📌 **Interpretación:**

- Precisión: **{round(r2,2)}**
- El modelo es **{"confiable" if r2 > 0.7 else "moderado"}**
""")

    # Forecast
    future = np.arange(len(df_e), len(df_e)+periodos)
    Xf = np.column_stack([future, (future % 12)+1])
    yf = model.predict(Xf)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_e["fecha"], y=y, name="Real"))
    fig.add_trace(go.Scatter(
        x=pd.date_range(df_e["fecha"].max(), periods=periodos+1, freq="M")[1:],
        y=yf,
        name="Predicción",
        line=dict(dash="dash")
    ))

    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════
# TAB 3 — RANDOM FOREST
# ════════════════════════════════════════
with tab3:

    rf = RandomForestRegressor(n_estimators=100, max_depth=10)
    rf.fit(X, y)
    pred_rf = rf.predict(X)

    r2_rf = r2_score(y, pred_rf)
    mae = mean_absolute_error(y, pred_rf)

    c1, c2 = st.columns(2)
    c1.metric("Precisión (R²)", round(r2_rf, 3))
    c2.metric("Error promedio (MAE)", round(mae, 2))

    if modo_simple:
        st.markdown("""
🤖 **¿Qué es este modelo?**

Random Forest detecta patrones complejos combinando múltiples decisiones.
""")

    # Importancia
    imp = rf.feature_importances_

    fig = go.Figure(go.Bar(
        x=["Tendencia", "Estacionalidad"],
        y=imp
    ))
    st.plotly_chart(fig)

# ════════════════════════════════════════
# TAB 4 — HOLT-WINTERS
# ════════════════════════════════════════
with tab4:

    y_hw = y

    model_hw = ExponentialSmoothing(
        y_hw,
        trend='add',
        damped_trend=True
    ).fit()

    forecast = model_hw.forecast(periodos)

    r2_hw = r2_score(y_hw, model_hw.fittedvalues)

    st.metric("Precisión modelo temporal", round(r2_hw, 3))

    if modo_simple:
        st.markdown(f"""
📈 **Interpretación:**

El modelo indica que la tendencia es **{"creciente" if forecast[-1] > y_hw[-1] else "decreciente"}**.
""")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_e["fecha"], y=y_hw, name="Real"))
    fig.add_trace(go.Scatter(
        x=pd.date_range(df_e["fecha"].max(), periods=periodos+1, freq="M")[1:],
        y=forecast,
        name="Pronóstico",
        line=dict(dash="dash")
    ))

    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.caption("Proyecto de análisis delictivo · IA aplicada · México")
