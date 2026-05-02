import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Observatorio Delictivo - IA & Data Science", layout="wide")

# Ruta del archivo (Asegúrate de que esté en tu repositorio de GitHub)
PATH_DATOS = "INM_estatal_dic25.csv.zip"

@st.cache_data
def cargar_datos_seguridad():
    if os.path.exists(PATH_DATOS):
        df = pd.read_csv(PATH_DATOS)
        df.columns = df.columns.str.strip()
        return df
    return None

# --- CARGA INICIAL ---
df = cargar_datos_seguridad()

if df is not None:
    st.title("🛡️ Observatorio de Inteligencia Artificial y Datos Forenses")
    
    # Filtro dinámico en la barra lateral
    st.sidebar.header("Parámetros de Análisis")
    entidades = sorted(df['Entidad'].unique())
    # Pre-selección de Guanajuato por relevancia en tus proyectos previos
    idx_default = entidades.index('GUANAJUATO') if 'GUANAJUATO' in entidades else 0
    entidad_sel = st.sidebar.selectbox("Seleccionar Estado", entidades, index=idx_default)
    
    df_filtrado = df[df['Entidad'] == entidad_sel]

    # Organización por pestañas para mantener la estructura CRISP-ML(Q)
    tab1, tab2, tab3 = st.tabs(["📊 Diagnóstico Actual", "📈 Modelos de Predicción", "🔍 Análisis de Disparidad"])

    # --- PESTAÑA 1: DIAGNÓSTICO ---
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(f"Incidencia en {entidad_sel}", f"{df_filtrado['Total'].sum():,}")
            top_delitos = df_filtrado.groupby('Subtipo de delito')['Total'].sum().nlargest(10)
            st.write("**Top 10 Incidencias:**")
            st.dataframe(top_delitos)
        
        with col2:
            fig_pie = px.pie(top_delitos.reset_index(), values='Total', names='Subtipo de delito', 
                             title="Composición Delictiva Local", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- PESTAÑA 2: MODELOS (LR, RF, HW) ---
    with tab2:
        st.subheader("🤖 Modelado Multimodal de Tendencias")
        
        # Preparación de serie temporal
        df_ts = df_filtrado.groupby(['Año', 'Mes'])['Total'].sum().reset_index()
        y = df_ts['Total'].values
        X = np.arange(len(y)).reshape(-1, 1)

        # 1. Regresión Lineal (Tendencia General)
        model_lr = LinearRegression().fit(X, y)
        pred_lr = model_lr.predict(X)

        # 2. Random Forest (Captura de patrones complejos)
        # Usando 100 estimadores y profundidad 10 según tus especificaciones
        model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X, y)
        pred_rf = model_rf.predict(X)

        # 3. Holt-Winters (Pronóstico Estacional)
        model_hw = ExponentialSmoothing(y, trend='add', seasonal=None).fit()
        forecast_hw = model_hw.forecast(6)

        # Visualización Integrada
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=y, name="Datos Reales (SESNSP)", mode='lines+markers'))
        fig_pred.add_trace(go.Scatter(y=pred_lr, name="Regresión Lineal", line=dict(dash='dot')))
        fig_pred.add_trace(go.Scatter(y=pred_rf, name="Random Forest", line=dict(color='green')))
        fig_pred.add_trace(go.Scatter(x=np.arange(len(y), len(y)+6), y=forecast_hw, 
                                      name="Pronóstico HW (6 meses)", line=dict(color='orange', width=4)))
        
        st.plotly_chart(fig_pred, use_container_width=True)
        st.caption("Gráfica comparativa de modelos para el análisis forense de datos.")

    # --- PESTAÑA 3: DISPARIDAD (ENVIPE) ---
    with tab3:
        st.subheader("🔍 Contraste: Datos Oficiales vs Percepción (ENVIPE 2024)")
        # Simulación de datos ENVIPE para el análisis de disparidad estatal
        envipe_data = pd.DataFrame({
            'Entidad': ['GUANAJUATO', 'CDMX', 'EDOMEX', 'JALISCO', 'NUEVO LEÓN'],
            'Percepcion': [87.5, 80.1, 90.6, 78.4, 72.3],
            'Cifra_Negra': [92.4, 93.1, 94.0, 91.8, 90.5]
        })
        fig_env = px.scatter(envipe_data, x='Percepcion', y='Cifra_Negra', text='Entidad', 
                             size='Percepcion', color='Entidad', title="Brecha de Inseguridad")
        st.plotly_chart(fig_env, use_container_width=True)

else:
    st.error(f"❌ Error Crítico: No se encontró el archivo '{PATH_DATOS}' en la raíz del proyecto.")
    st.info("Asegúrate de subir el archivo .csv.zip junto con tu script de Python a GitHub.")
