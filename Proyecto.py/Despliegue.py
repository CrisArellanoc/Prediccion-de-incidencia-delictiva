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

# Estilo para mejorar la lectura de métricas
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #d32f2f; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# CARGA Y TRADUCCIÓN DE DATOS (GOOGLE DRIVE)
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def cargar_datos_reales():
    FILE_ID = "1vxqqM0-L1A5yIMs0vJbdIr1_dddEwMnn"
    url = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
    
    try:
        df = pd.read_csv(url)
        
        # 1. Limpieza de columnas duplicadas y nombres
        df = df.loc[:, ~df.columns.duplicated()]
        df.columns = df.columns.str.lower().str.strip()
        
        # 2. Diccionario de traducción para fechas en español
        meses_esp = {
            'enero': 'January', 'febrero': 'February', 'marzo': 'March', 
            'abril': 'April', 'mayo': 'May', 'junio': 'June',
            'julio': 'July', 'agosto': 'August', 'septiembre': 'September', 
            'octubre': 'October', 'noviembre': 'November', 'diciembre': 'December'
        }

        # 3. Detección estricta de columnas
        col_entidad = next((c for c in df.columns if any(k in c for k in ['entidad', 'estado', 'nom_ent'])), None)
        col_fecha = next((c for c in df.columns if any(k in c for k in ['fecha', 'date', 'mes', 'año'])), None)
        col_tasa = next((c for c in df.columns if any(k in c for k in ['tasa', 'valor', 'total', 'incidencia'])), None)
        
        if not all([col_entidad, col_fecha, col_tasa]):
            st.error(f"⚠️ Estructura no reconocida. Columnas: {list(df.columns)}")
            return None

        # 4. Traducción de la columna de fecha (Manejo de "Abril", "Enero", etc.)
        fecha_procesada = df[col_fecha].astype(str).str.lower()
        for esp, ing in meses_esp.items():
            fecha_procesada = fecha_procesada.str.replace(esp, ing)

        # 5. Construcción del DataFrame limpio
        df_clean = pd.DataFrame({
            'entidad': df[col_entidad].astype(str).str.strip().str.title(),
            'fecha': pd.to_datetime(fecha_procesada, errors='coerce'),
            'tasa_100k': pd.to_numeric(df[col_tasa], errors='coerce')
        })
        
        # Eliminar filas con errores de conversión (NaT o NaN)
        df_clean = df_clean.dropna(subset=['fecha', 'tasa_100k'])
        
        return df_clean.sort_values("fecha").reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Fallo en la conexión o procesamiento: {e}")
        return None

# Ejecutar carga de datos
df = cargar_datos_reales()

# ─────────────────────────────────────────
# INTERFAZ Y VISUALIZACIÓN
# ─────────────────────────────────────────
if df is not None:
    st.title("🇲🇽 Observatorio de Incidencia Delictiva (Datos Reales)")
    
    with st.sidebar:
        st.image("https://www.gob.mx/cms/uploads/action_program/main_image/26135/post_sesnsp.png", use_container_width=True)
        st.divider()
        estado_sel = st.selectbox("📍 Entidad Federativa", sorted(df["entidad"].unique()))
        periodos_pred = st.slider("🔮 Meses a proyectar", 1, 24, 12)
        st.caption(f"Última actualización de datos: {df['fecha'].max().strftime('%B %Y')}")

    tab1, tab2 = st.tabs(["🗺️ Mapa Coroplético", "📈 Análisis Estacional y Proyección"])

    # --- TAB 1: MAPA ---
    with tab1:
        st.subheader("Distribución de Incidencia por Estado")
        
        ultima_fecha = df["fecha"].max()
        df_mapa = df[df["fecha"] == ultima_fecha]
        
        geojson_url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
        
        fig_mapa = px.choropleth(
            df_mapa,
            geojson=geojson_url,
            locations="entidad",
            featureidkey="properties.name",
            color="tasa_100k",
            color_continuous_scale="Reds",
            scope="mexico",
            labels={'tasa_100k': 'Tasa'}
        )
        fig_mapa.update_geos(fitbounds="locations", visible=False)
        fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=550)
        st.plotly_chart(fig_mapa, use_container_width=True)

    # --- TAB 2: ANÁLISIS ---
    with tab2:
        df_e = df[df["entidad"] == estado_sel]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Última Tasa", f"{df_e['tasa_100k'].iloc[-1]:.2f}")
        m2.metric("Promedio Histórico", f"{df_e['tasa_100k'].mean():.2f}")
        m3.metric("Punto Máximo", f"{df_e['tasa_100k'].max():.2f}")

        st.subheader(f"Evolución en {estado_sel}")
        
        try:
            # Modelo Holt-Winters para datos estacionales
            serie = df_e["tasa_100k"].values
            # Requerimos al menos 24 puntos para una estacionalidad de 12 meses
            modelo = ExponentialSmoothing(
                serie, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=12
            ).fit()
            
            pronostico = modelo.forecast(periodos_pred)
            fechas_pred = pd.date_range(df_e["fecha"].max(), periods=periodos_pred + 1, freq="ME")[1:]
            
            fig_evol = go.Figure()
            # Histórico
            fig_evol.add_trace(go.Scatter(
                x=df_e["fecha"], y=serie, 
                name="Datos Históricos", 
                line=dict(color='#333333', width=2)
            ))
            # Proyección
            fig_evol.add_trace(go.Scatter(
                x=fechas_pred, y=pronostico, 
                name="Proyección (IA)", 
                line=dict(color='red', dash='dash', width=3)
            ))
            
            fig_evol.update_layout(
                template="plotly_white",
                hovermode="x unified",
                yaxis_title="Tasa por 100k hab."
            )
            st.plotly_chart(fig_evol, use_container_width=True)
            
        except Exception as e:
            st.warning("No hay suficientes datos históricos para realizar una proyección estacional precisa.")
            st.line_chart(df_e.set_index("fecha")["tasa_100k"])

    st.divider()
    st.caption("Fuente: Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP). Procesamiento automático de fechas en español activado.")
else:
    st.warning("⚠️ No se pudieron cargar los datos. Verifica que el archivo en Google Drive tenga permisos de 'Cualquier persona con el enlace'.")
