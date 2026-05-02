import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- CONFIGURACIÓN E IDENTIDAD ---
st.set_page_config(page_title="Observatorio Delictivo - IA", layout="wide")

# Ruta automática para despliegue
PATH_DATOS = "INM_estatal_dic25.csv.zip"

@st.cache_data
def cargar_datos():
    if os.path.exists(PATH_DATOS):
        df = pd.read_csv(PATH_DATOS)
        df.columns = df.columns.str.strip()
        return df
    return None

# --- LÓGICA DE NEGOCIO ---
df = cargar_datos()

if df is not None:
    st.title("🛡️ Observatorio de Inteligencia Artificial")
    
    # Sidebar: Filtros que antes ocupaban mucho espacio
    entidad = st.sidebar.selectbox("Estado:", sorted(df['Entidad'].unique()), index=10)
    df_filtrado = df[df['Entidad'] == entidad]

    tab1, tab2, tab3 = st.tabs(["📊 Diagnóstico", "📈 Modelos Predictivos", "🔍 Comparativa"])

    with tab1:
        # Aquí condensamos lo que antes eran múltiples gráficas manuales
        col1, col2 = st.columns(2)
        top_delitos = df_filtrado.groupby('Subtipo de delito')['Total'].sum().nlargest(10)
        col1.metric("Total Incidentes", f"{df_filtrado['Total'].sum():,}")
        col1.write("**Top Delitos:**")
        col1.dataframe(top_delitos)
        col2.plotly_chart(px.bar(top_delitos, orientation='h', title="Incidencia por Subtipo"), use_container_width=True)

    with tab2:
        st.subheader("🤖 Análisis Multimodelo (Regresión, RF y HW)")
        
        # Preparación de serie de tiempo rápida
        df_ts = df_filtrado.groupby(['Año', 'Mes'])['Total'].sum().reset_index()
        y = df_ts['Total'].values
        X = np.arange(len(y)).reshape(-1, 1)

        # 1. Regresión Lineal
        lr = LinearRegression().fit(X, y)
        # 2. Random Forest (Configuración según tus specs: 100 est, 10 depth)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10).fit(X, y.ravel())
        # 3. Holt-Winters
        hw = ExponentialSmoothing(y, trend='add').fit()

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y, name="Real", mode='lines+markers'))
        fig.add_trace(go.Scatter(y=lr.predict(X), name="Tendencia Lineal", line=dict(dash='dot')))
        fig.add_trace(go.Scatter(y=rf.predict(X), name="Random Forest", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=np.arange(len(y), len(y)+6), y=hw.forecast(6), name="Proyección HW (6m)", line=dict(color='orange')))
        
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.info("Espacio para integrar microdatos de ENVIPE 2024 y análisis de disparidad.")

else:
    st.error(f"Archivo {PATH_DATOS} no detectado. Verifica que esté en el repositorio de GitHub.")
