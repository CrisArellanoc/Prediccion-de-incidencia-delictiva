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

# Estilo personalizado para las métricas
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #1f77b4; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# CARGA DE DATOS (CONEXIÓN GOOGLE DRIVE)
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def cargar_datos_reales():
    # ID de tu archivo basado en el enlace proporcionado
    FILE_ID = "1vxqqM0-L1A5yIMs0vJbdIr1_dddEwMnn"
    url = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
    
    try:
        # 1. Leer CSV original
        df = pd.read_csv(url)
        
        # 2. Eliminar columnas duplicadas físicas que puedan venir en el archivo
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 3. Normalizar nombres de columnas (minúsculas y sin espacios)
        df.columns = df.columns.str.lower().str.strip()
        
        # 4. Asignación ESTRICTA de columnas (evita duplicidad de llaves)
        # Buscamos la mejor coincidencia para cada categoría
        col_entidad = next((c for c in df.columns if any(k in c for k in ['entidad', 'estado', 'nom_ent'])), None)
        col_fecha = next((c for c in df.columns if any(k in c for k in ['fecha', 'date', 'mes', 'año'])), None)
        col_tasa = next((c for c in df.columns if any(k in c for k in ['tasa', 'valor', 'total', 'incidencia'])), None)
        
        if not all([col_entidad, col_fecha, col_tasa]):
            st.error(f"⚠️ El archivo no tiene el formato esperado. Columnas encontradas: {list(df.columns)}")
            return None

        # 5. Crear un nuevo DataFrame solo con lo necesario
        df_clean = pd.DataFrame({
            'entidad': df[col_entidad].astype(str).str.strip().str.title(),
            'fecha': pd.to_datetime(df[col_fecha]),
            'tasa_100k': pd.to_numeric(df[col_tasa], errors='coerce')
        })
        
        # Limpiar posibles nulos
        df_clean = df_clean.dropna(subset=['tasa_100k', 'fecha'])
        
        return df_clean.sort_values("fecha").reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Fallo en la conexión o procesamiento: {e}")
        return None

# Ejecutar carga
df = cargar_datos_reales()

# ─────────────────────────────────────────
# LÓGICA DE LA INTERFAZ
# ─────────────────────────────────────────
if df is not None:
    st.title("🔭 Observatorio Delictivo en México (SESNSP)")
    
    # Sidebar con controles
    with st.sidebar:
        st.image("https://www.gob.mx/cms/uploads/action_program/main_image/26135/post_sesnsp.png", use_container_width=True)
        st.divider()
        estado_sel = st.selectbox("📍 Selecciona Estado", sorted(df["entidad"].unique()))
        periodos_pred = st.slider("🔮 Meses a proyectar", 1, 24, 12)
        st.caption("Nota: La IA utiliza el método Holt-Winters para capturar patrones anuales.")

    tab1, tab2 = st.tabs(["🗺️ Mapa de Calor", "📈 Análisis Estacional"])

    # --- TAB 1: MAPA ---
    with tab1:
        st.subheader(f"Incidencia por Entidad (Corte: {df['fecha'].max().strftime('%m/%Y')})")
        
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
        
        # Métricas rápidas
        m1, m2, m3 = st.columns(3)
        m1.metric("Última Tasa", f"{df_e['tasa_100k'].iloc[-1]:.2f}")
        m2.metric("Promedio Estatal", f"{df_e['tasa_100k'].mean():.2f}")
        m3.metric("Punto más alto", f"{df_e['tasa_100k'].max():.2f}")

        # Predicción Matemática
        try:
            # Los datos de SESNSP suelen ser mensuales, usamos estacionalidad de 12
            serie = df_e["tasa_100k"].values
            modelo = ExponentialSmoothing(
                serie, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=12
            ).fit()
            
            pronostico = modelo.forecast(periodos_pred)
            fechas_pred = pd.date_range(df_e["fecha"].max(), periods=periodos_pred + 1, freq="ME")[1:]
            
            # Gráfica combinada
            fig_evol = go.Figure()
            # Histórico
            fig_evol.add_trace(go.Scatter(
                x=df_e["fecha"], y=serie, 
                name="Datos Reales (Drive)", 
                line=dict(color='#1f77b4', width=3)
            ))
            # Proyección
            fig_evol.add_trace(go.Scatter(
                x=fechas_pred, y=pronostico, 
                name="Proyección IA", 
                line=dict(color='red', dash='dash')
            ))
            
            fig_evol.update_layout(
                title=f"Evolución y Predicción para {estado_sel}",
                xaxis_title="Año",
                yaxis_title="Tasa por 100k hab.",
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_evol, use_container_width=True)
            
        except Exception as e:
            st.warning("No hay suficientes datos para generar una predicción estacional sólida.")
            st.line_chart(df_e.set_index("fecha")["tasa_100k"])

    # Footer
    st.divider()
    st.caption("Fuente: Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública (SESNSP). IA aplicada mediante Triple Suavizado Exponencial.")
else:
    st.info("☁️ Conectando con Google Drive... Asegúrate de que el enlace del archivo sea público.")
