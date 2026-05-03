import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Observatorio Delictivo SESNSP",
    page_icon="🇲🇽",
    layout="wide"
)

# Estilo para métricas
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #d32f2f; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# CARGA Y TRADUCCIÓN DE DATOS
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def cargar_datos_reales():
    FILE_ID = "1vxqqM0-L1A5yIMs0vJbdIr1_dddEwMnn"
    url = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
    
    try:
        df = pd.read_csv(url)
        df = df.loc[:, ~df.columns.duplicated()]
        df.columns = df.columns.str.lower().str.strip()
        
        meses_esp = {
            'enero': 'January', 'febrero': 'February', 'marzo': 'March', 
            'abril': 'April', 'mayo': 'May', 'junio': 'June',
            'julio': 'July', 'agosto': 'August', 'septiembre': 'September', 
            'octubre': 'October', 'noviembre': 'November', 'diciembre': 'December'
        }

        col_entidad = next((c for c in df.columns if any(k in c for k in ['entidad', 'estado', 'nom_ent'])), None)
        col_fecha = next((c for c in df.columns if any(k in c for k in ['fecha', 'date', 'mes', 'año'])), None)
        col_tasa = next((c for c in df.columns if any(k in c for k in ['tasa', 'valor', 'total', 'incidencia'])), None)
        
        if not all([col_entidad, col_fecha, col_tasa]):
            return None

        fecha_procesada = df[col_fecha].astype(str).str.lower()
        for esp, ing in meses_esp.items():
            fecha_procesada = fecha_procesada.str.replace(esp, ing)

        df_clean = pd.DataFrame({
            'entidad': df[col_entidad].astype(str).str.strip().str.title(),
            'fecha': pd.to_datetime(fecha_procesada, errors='coerce'),
            'tasa_100k': pd.to_numeric(df[col_tasa], errors='coerce')
        })
        
        df_clean = df_clean.dropna(subset=['fecha', 'tasa_100k'])
        return df_clean.sort_values("fecha").reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Error en carga: {e}")
        return None

df = cargar_datos_reales()

# ─────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────
if df is not None:
    st.title("🇲🇽 Observatorio de Incidencia Delictiva (SESNSP)")
    
    with st.sidebar:
        st.image("https://www.gob.mx/cms/uploads/action_program/main_image/26135/post_sesnsp.png", use_container_width=True)
        st.divider()
        estado_sel = st.selectbox("📍 Entidad Federativa", sorted(df["entidad"].unique()))
        periodos_pred = st.slider("🔮 Meses a proyectar", 1, 24, 12)

    tab1, tab2 = st.tabs(["🗺️ Mapa Coroplético", "📈 Análisis y Proyección"])

    # --- TAB 1: MAPA (CORREGIDO) ---
    with tab1:
        st.subheader("Distribución de Incidencia")
        ultima_fecha = df["fecha"].max()
        df_mapa = df[df["fecha"] == ultima_fecha].copy()
        
        geojson_url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
        
        # Eliminamos 'scope' y usamos un enfoque más manual para evitar el ValueError de validación
        fig_mapa = px.choropleth(
            df_mapa,
            geojson=geojson_url,
            locations="entidad",
            featureidkey="properties.name",
            color="tasa_100k",
            color_continuous_scale="Reds",
            labels={'tasa_100k': 'Tasa'}
        )
        
        # Ajuste manual de la vista para evitar conflictos de validación de Plotly
        fig_mapa.update_geos(
            visible=False, 
            resolution=50,
            fitbounds="locations"
        )
        
        fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=550)
        st.plotly_chart(fig_mapa, use_container_width=True)

    # --- TAB 2: ANÁLISIS ---
    with tab2:
        df_e = df[df["entidad"] == estado_sel]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Última Tasa", f"{df_e['tasa_100k'].iloc[-1]:.2f}")
        m2.metric("Promedio Histórico", f"{df_e['tasa_100k'].mean():.2f}")
        m3.metric("Máximo Histórico", f"{df_e['tasa_100k'].max():.2f}")

        try:
            serie = df_e["tasa_100k"].values
            modelo = ExponentialSmoothing(serie, trend='add', seasonal='add', seasonal_periods=12).fit()
            pronostico = modelo.forecast(periodos_pred)
            fechas_pred = pd.date_range(df_e["fecha"].max(), periods=periodos_pred + 1, freq="ME")[1:]
            
            fig_evol = go.Figure()
            fig_evol.add_trace(go.Scatter(x=df_e["fecha"], y=serie, name="Histórico", line=dict(color='#333333')))
            fig_evol.add_trace(go.Scatter(x=fechas_pred, y=pronostico, name="Proyección", line=dict(color='red', dash='dash')))
            fig_evol.update_layout(template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig_evol, use_container_width=True)
        except:
            st.line_chart(df_e.set_index("fecha")["tasa_100k"])

    st.caption(f"Última fecha detectada en el archivo: {df['fecha'].max().strftime('%d/%m/%Y')}")
else:
    st.error("No se pudo cargar la base de datos.")
