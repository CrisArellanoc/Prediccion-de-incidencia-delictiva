import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import requests
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
st.set_page_config(page_title="Observatorio Real · SESNSP 2025", page_icon="🇲🇽", layout="wide")

# Estilo visual
st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# CARGA DE DATOS DESDE TU DRIVE
# ─────────────────────────────────────────
@st.cache_data(ttl=3600) # Se actualiza cada hora si cambias el archivo
def cargar_datos_reales():
    # ID extraído de tu enlace
    FILE_ID = "1vxqqM0-L1A5yIMs0vJbdIr1_dddEwMnn"
    url = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
    
    try:
        df = pd.read_csv(url)
        
        # --- LIMPIEZA AUTOMÁTICA ---
        # Convertir nombres de columnas a minúsculas y quitar espacios para evitar errores
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Renombrar columnas comunes si vienen distinto (ej: 'estado' -> 'entidad')
        rename_dict = {'estado': 'entidad', 'tasa': 'tasa_100k', 'valor': 'tasa_100k', 'date': 'fecha'}
        df = df.rename(columns=rename_dict)
        
        # Asegurar formato fecha
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.sort_values("fecha")
        
        # Crear mes_num para los modelos de predicción
        df["mes_num"] = np.arange(len(df)) 
        
        return df
    except Exception as e:
        st.error(f"Error al leer tu CSV de Drive: {e}")
        return None

df = cargar_datos_reales()

# ─────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────
if df is not None:
    st.title("🔭 Observatorio Delictivo México (Datos SESNSP)")
    st.caption(f"Visualizando datos reales desde el 2018 hasta **{df['fecha'].max().strftime('%B %Y')}**")

    # SIDEBAR
    with st.sidebar:
        st.image("https://www.gob.mx/cms/uploads/action_program/main_image/26135/post_sesnsp.png", use_container_width=True)
        st.divider()
        estado_sel = st.selectbox("📍 Selecciona Entidad Federativa", sorted(df["entidad"].unique()))
        meses_proy = st.slider("🔮 Meses a futuro (IA)", 1, 12, 6)
        st.info("Esta herramienta usa el modelo de Triple Suavizado Exponencial (Holt-Winters) para detectar estacionalidad.")

    # FILTRADO
    df_e = df[df["entidad"] == estado_sel]

    # TABS
    tab1, tab2 = st.tabs(["🗺️ Mapa de Calor Nacional", "📈 Análisis y Predicción por Estado"])

    # --- TAB 1: MAPA ---
    with tab1:
        st.subheader(f"Incidencia Delictiva Nacional (Corte: {df['fecha'].max().date()})")
        
        # Tomamos el último mes disponible en tu base de datos
        df_reciente = df[df["fecha"] == df["fecha"].max()]
        
        # GeoJSON de los estados de México
        geojson_url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
        
        fig_mapa = px.choropleth(
            df_reciente,
            geojson=geojson_url,
            locations="entidad",
            featureidkey="properties.name", # Esto mapea 'Guanajuato' con el mapa
            color="tasa_100k",
            color_continuous_scale="YlOrRd",
            labels={'tasa_100k': 'Tasa'},
            scope="mexico",
            template="plotly_white"
        )
        fig_mapa.update_geos(fitbounds="locations", visible=False)
        fig_mapa.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        
        st.plotly_chart(fig_mapa, use_container_width=True)

    # --- TAB 2: ANÁLISIS ---
    with tab2:
        col1, col2, col3 = st.columns(3)
        ultima_tasa = df_e["tasa_100k"].iloc[-1]
        promedio = df_e["tasa_100k"].mean()
        
        col1.metric("Última Tasa (2025)", f"{ultima_tasa:.2f}")
        col2.metric("Promedio Histórico", f"{promedio:.2f}")
        col3.metric("Estado", estado_sel)

        # Gráfico Histórico + Predicción
        st.subheader("Evolución y Proyección")
        
        try:
            # Entrenamiento del modelo con tus datos reales
            y = df_e["tasa_100k"].values
            # Holt-Winters: Maneja tendencia y estacionalidad de 12 meses
            modelo = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=12).fit()
            prediccion = modelo.forecast(meses_proy)
            
            # Generar fechas futuras
            ult_fecha = df_e["fecha"].max()
            fechas_futuras = pd.date_range(ult_fecha, periods=meses_proy + 1, freq="ME")[1:]
            
            # Visualización
            fig_final = go.Figure()
            # Datos Reales
            fig_final.add_trace(go.Scatter(x=df_e["fecha"], y=y, name="Datos Reales (Drive)", line=dict(color="#1f77b4", width=3)))
            # Predicción
            fig_final.add_trace(go.Scatter(x=fechas_futuras, y=prediccion, name="Predicción IA", line=dict(color="red", dash="dash")))
            
            fig_final.update_layout(hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig_final, use_container_width=True)
            
        except Exception as e:
            st.error(f"No se pudo generar la predicción: {e}")
            st.line_chart(df_e.set_index("fecha")["tasa_100k"])

else:
    st.error("No se pudo establecer conexión con el archivo en Google Drive.")
    st.info("Revisa que el archivo tenga activada la opción: 'Cualquier persona con el enlace puede leer'.")
