import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
st.set_page_config(page_title="Observatorio Real · SESNSP", page_icon="🇲🇽", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# CARGA DE DATOS DESDE GOOGLE DRIVE
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def cargar_datos_reales():
    FILE_ID = "1vxqqM0-L1A5yIMs0vJbdIr1_dddEwMnn"
    url = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
    
    try:
        # Descarga directa del CSV
        df = pd.read_csv(url)
        
        # --- LIMPIEZA EXTREMA DE COLUMNAS ---
        # 1. Convertir a minúsculas, quitar acentos y espacios vacíos a los lados
        df.columns = df.columns.str.lower().str.strip().str.replace('á', 'a').str.replace('é', 'e')
        
        # 2. Mapeo dinámico (busca palabras clave dentro de los nombres de tus columnas)
        col_mapping = {}
        for col in df.columns:
            if any(kw in col for kw in ['entidad', 'estado']):
                col_mapping[col] = 'entidad'
            elif any(kw in col for kw in ['fecha', 'date', 'mes']):
                col_mapping[col] = 'fecha'
            elif any(kw in col for kw in ['tasa', 'valor', 'total', 'incidencia']):
                col_mapping[col] = 'tasa_100k'
                
        # 3. Aplicar el renombramiento
        df = df.rename(columns=col_mapping)
        
        # Validación: Si no encontró las 3 columnas básicas, lanzamos un error claro
        columnas_requeridas = ['entidad', 'fecha', 'tasa_100k']
        faltantes = [c for c in columnas_requeridas if c not in df.columns]
        if faltantes:
            raise ValueError(f"No se pudieron detectar las columnas: {faltantes}. Columnas detectadas: {list(df.columns)}")

        # --- LIMPIEZA DE DATOS ---
        df["fecha"] = pd.to_datetime(df["fecha"])
        # Limpiar los nombres de los estados (quitar espacios extra que arruinan el mapa)
        df["entidad"] = df["entidad"].str.strip().str.title()
        
        # Ordenar por fecha
        df = df.sort_values("fecha").reset_index(drop=True)
        return df
    
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

df = cargar_datos_reales()

# ─────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ─────────────────────────────────────────
if df is not None:
    st.title("🔭 Observatorio Delictivo de México")
    st.caption(f"Datos oficiales del SESNSP hasta **{df['fecha'].max().strftime('%B %Y')}**")

    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://www.gob.mx/cms/uploads/action_program/main_image/26135/post_sesnsp.png", use_container_width=True)
        st.divider()
        estado_sel = st.selectbox("📍 Entidad Federativa", sorted(df["entidad"].unique()))
        meses_proy = st.slider("🔮 Meses a predecir", 1, 24, 6)

    # --- TABS ---
    tab1, tab2 = st.tabs(["🗺️ Mapa Nacional", "📈 Análisis Histórico y Predicción"])

    # ════════════════════════════════════════
    # TAB 1: MAPA DE CALOR
    # ════════════════════════════════════════
    with tab1:
        ultima_fecha = df["fecha"].max()
        df_reciente = df[df["fecha"] == ultima_fecha].copy()
        
        st.subheader(f"Incidencia Delictiva (Corte: {ultima_fecha.date()})")
        
        # GeoJSON oficial de México
        geojson_url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
        
        try:
            fig_mapa = px.choropleth(
                df_reciente,
                geojson=geojson_url,
                locations="entidad",
                featureidkey="properties.name",
                color="tasa_100k",
                color_continuous_scale="Reds",
                labels={'tasa_100k': 'Tasa por 100k hab.'},
                scope="mexico",
                template="plotly_white"
            )
            fig_mapa.update_geos(fitbounds="locations", visible=False)
            fig_mapa.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_mapa, use_container_width=True)
        except Exception as e:
            st.warning("No se pudo cargar el mapa. Verifica que los nombres de los estados coincidan con los oficiales.")
            st.exception(e)

    # ════════════════════════════════════════
    # TAB 2: PREDICCIÓN (HOLT-WINTERS)
    # ════════════════════════════════════════
    with tab2:
        df_e = df[df["entidad"] == estado_sel].copy()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Último Registro", f"{df_e['tasa_100k'].iloc[-1]:.2f}")
        c2.metric("Promedio Histórico", f"{df_e['tasa_100k'].mean():.2f}")
        c3.metric("Máximo Histórico", f"{df_e['tasa_100k'].max():.2f}")

        st.subheader(f"Proyección de Tendencias para {estado_sel}")
        
        try:
            y = df_e["tasa_100k"].values
            
            # Entrenar modelo (Trend y Estacionalidad)
            modelo = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=12).fit()
            prediccion = modelo.forecast(meses_proy)
            
            # Crear fechas futuras usando freq="ME" (Corregido)
            fechas_futuras = pd.date_range(df_e["fecha"].max(), periods=meses_proy + 1, freq="ME")[1:]
            
            # Graficar
            fig_linea = go.Figure()
            # Histórico
            fig_linea.add_trace(go.Scatter(
                x=df_e["fecha"], y=y, 
                name="Datos Reales", 
                mode="lines", 
                line=dict(color="#1f77b4", width=3)
            ))
            # Predicción
            fig_linea.add_trace(go.Scatter(
                x=fechas_futuras, y=prediccion, 
                name="Proyección Holt-Winters", 
                mode="lines", 
                line=dict(color="red", dash="dash", width=2)
            ))
            
            fig_linea.update_layout(hovermode="x unified", template="plotly_white", yaxis_title="Tasa Delictiva")
            st.plotly_chart(fig_linea, use_container_width=True)
            
        except ValueError as ve:
            st.error(f"Fallo en la proyección matemática: {ve}")
            st.info("Nota: El modelo necesita al menos 24 meses continuos de datos para detectar estacionalidad (ciclos anuales).")
            # Graficar solo el histórico si falla la predicción
            fig_linea = px.line(df_e, x="fecha", y="tasa_100k", title="Histórico de Datos")
            st.plotly_chart(fig_linea, use_container_width=True)

else:
    st.stop()
