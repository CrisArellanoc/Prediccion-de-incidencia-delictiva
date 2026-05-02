import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Observatorio Delictivo - Inteligencia de Datos", layout="wide")

# --- RUTA DEL ARCHIVO ---
# Se utiliza la ruta del archivo que has manejado en tu tercer semestre
PATH_DATOS = "INM_estatal_dic25.csv.zip"

# --- FUNCIONES DE CARGA AUTOMÁTICA ---
@st.cache_data
def cargar_datos_locales(ruta):
    """Carga automática de datos del SESNSP (2015-2025)"""
    if os.path.exists(ruta):
        df = pd.read_csv(ruta)
        df.columns = df.columns.str.strip()
        return df
    return None

@st.cache_data
def cargar_datos_envipe():
    """Simulación de datos ENVIPE 2024 para contraste de disparidad"""
    data = {
        'Entidad': ['GUANAJUATO', 'CIUDAD DE MÉXICO', 'ESTADO DE MÉXICO', 'JALISCO', 'NUEVO LEÓN'],
        'Percepcion_Inseguridad': [87.5, 80.1, 90.6, 78.4, 72.3],
        'Tasa_Cifra_Negra': [92.4, 93.1, 94.0, 91.8, 90.5]
    }
    return pd.DataFrame(data)

# --- PROCESAMIENTO INICIAL ---
df_sesnsp = cargar_datos_locales(PATH_DATOS)

# --- INTERFAZ DE USUARIO ---
if df_sesnsp is not None:
    st.title("🛡️ Observatorio de Inteligencia Artificial y Datos Forenses")
    st.info(f"Base de datos cargada automáticamente: {PATH_DATOS}")

    # Sidebar para navegación
    st.sidebar.header("Configuración de Análisis")
    entidades_disponibles = sorted(df_sesnsp['Entidad'].unique())
    # Se busca pre-seleccionar Guanajuato por relevancia en tus proyectos
    indice_default = entidades_disponibles.index('GUANAJUATO') if 'GUANAJUATO' in entidades_disponibles else 0
    entidad_sel = st.sidebar.selectbox("Seleccionar Estado", entidades_disponibles, index=indice_default)

    # Filtrado de datos por entidad
    df_ent = df_sesnsp[df_sesnsp['Entidad'] == entidad_sel]

    # Tabs de análisis basados en tus modelos implementados
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Diagnóstico Actual", 
        "📈 Tendencia (Regresión)", 
        "🤖 Patrones (Random Forest)", 
        "🔍 Pronóstico y ENVIPE"
    ])

    # --- TAB 1: DIAGNÓSTICO ---
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            total = df_ent['Total'].sum()
            st.metric(f"Incidencia Total en {entidad_sel}", f"{total:,}")
            top_delitos = df_ent.groupby('Subtipo de delito')['Total'].sum().sort_values(ascending=False).head(10)
            st.write("**Top 10 Delitos:**")
            st.dataframe(top_delitos)
        
        with col2:
            fig_pie = px.pie(top_delitos.reset_index(), values='Total', names='Subtipo de delito', 
                             title=f"Distribución Delictiva en {entidad_sel}", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- TAB 2: REGRESIÓN LINEAL ---
    with tab2:
        st.subheader("Análisis de Tendencia con Regresión Lineal")
        df_anual = df_ent.groupby('Año')['Total'].sum().reset_index()
        
        X = df_anual['Año'].values.reshape(-1, 1)
        y = df_anual['Total'].values
        
        model_lr = LinearRegression().fit(X, y)
        pred_lr = model_lr.predict(X)
        r2 = r2_score(y, pred_lr)
        
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(x=df_anual['Año'], y=y, name="Datos Reales", mode='lines+markers'))
        fig_lr.add_trace(go.Scatter(x=df_anual['Año'], y=pred_lr, name="Tendencia", line=dict(dash='dash', color='red')))
        st.plotly_chart(fig_lr, use_container_width=True)
        st.success(f"Coeficiente de determinación $R^2$: {r2:.4f}")

    # --- TAB 3: RANDOM FOREST ---
    with tab3:
        st.subheader("Análisis de Importancia con Random Forest")
        # Preparación de datos (Año y Mes como variables predictoras)
        df_rf = df_ent.groupby(['Año', 'Mes'])['Total'].sum().reset_index()
        df_rf['Mes_ID'] = range(len(df_rf))
        
        X_rf = df_rf[['Año', 'Mes_ID']]
        y_rf = df_rf['Total']
        
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_rf, y_rf)
        
        importancias = pd.DataFrame({
            'Variable': ['Evolución Anual', 'Ciclo Mensual'],
            'Importancia': rf.feature_importances_
        })
        
        st.write("Este modelo identifica qué factor influye más en el volumen delictivo:")
        st.bar_chart(importancias.set_index('Variable'))

    # --- TAB 4: HOLT-WINTERS & DISPARIDAD ---
    with tab4:
        col_hw, col_env = st.columns(2)
        
        with col_hw:
            st.subheader("Pronóstico Holt-Winters")
            serie = df_ent.groupby(['Año', 'Mes'])['Total'].sum().values
            if len(serie) > 12:
                # Aplicando suavizado exponencial Holt-Winters
                hw = ExponentialSmoothing(serie, trend='add', seasonal=None).fit()
                proyeccion = hw.forecast(6)
                
                fig_hw = go.Figure()
                fig_hw.add_trace(go.Scatter(y=serie, name="Histórico"))
                fig_hw.add_trace(go.Scatter(x=np.arange(len(serie), len(serie)+6), y=proyeccion, name="Proyección (6m)"))
                st.plotly_chart(fig_hw, use_container_width=True)
            else:
                st.warning("Serie de tiempo insuficiente para Holt-Winters.")

        with col_env:
            st.subheader("Disparidad ENVIPE 2024")
            df_env = cargar_datos_envipe()
            fig_env = px.scatter(df_env, x='Percepcion_Inseguridad', y='Tasa_Cifra_Negra', 
                                 text='Entidad', size='Percepcion_Inseguridad', color='Entidad',
                                 labels={'Percepcion_Inseguridad': '% Percepción', 'Tasa_Cifra_Negra': '% Cifra Negra'})
            st.plotly_chart(fig_env, use_container_width=True)

else:
    st.error(f"No se encontró el archivo '{PATH_DATOS}'.")
    st.info("Asegúrate de que el archivo ZIP esté en la misma carpeta que este script.")
