import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import requests
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis de incidencia delictiva",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CSS — PALANTIR AESTHETIC
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    box-sizing: border-box;
}

/* ── App shell ── */
.stApp {
    background: #0c0d0f;
    color: #c8cdd6;
}

/* ── Scanline overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255,255,255,0.012) 2px,
        rgba(255,255,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0e0f12 !important;
    border-right: 1px solid #1e2330 !important;
    width: 280px !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #5a6478 !important;
    font-size: 0.72rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label {
    color: #3d4558 !important;
    font-size: 0.65rem !important;
}

/* Sidebar inputs */
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: #111318 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 2px !important;
    color: #8a95a8 !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #0e0f12;
    border: 1px solid #1a1e28;
    border-top: 1px solid #242938;
    border-radius: 2px;
    padding: 16px 20px 14px;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 2px; height: 100%;
    background: #1f6feb;
}
[data-testid="metric-container"] label {
    color: #3d4558 !important;
    font-size: 0.62rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8ecf4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #3d6b9a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
}

/* ── Header ── */
.orca-header {
    background: #0e0f12;
    border: 1px solid #1a1e28;
    border-top: 2px solid #1f6feb;
    border-radius: 2px;
    padding: 24px 32px 20px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.orca-header::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 280px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(31,111,235,0.03));
    pointer-events: none;
}
.orca-wordmark {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    color: #1f6feb;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.orca-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e8ecf4;
    letter-spacing: -0.01em;
    margin: 0 0 4px;
}
.orca-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3d4558;
    letter-spacing: 0.06em;
    margin: 0 0 14px;
}
.orca-tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.orca-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    color: #3d6b9a;
    border: 1px solid #1a2a3d;
    padding: 3px 10px;
    border-radius: 1px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: rgba(31,111,235,0.04);
}
.orca-tag.alert {
    color: #c0392b;
    border-color: #3d1a1a;
    background: rgba(192,57,43,0.04);
}
.orca-tag.ok {
    color: #27ae60;
    border-color: #1a3d2a;
    background: rgba(39,174,96,0.04);
}

/* ── Section headers ── */
.orca-section {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 12px;
}
.orca-section-line {
    flex: 1;
    height: 1px;
    background: #1a1e28;
}
.orca-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    color: #2d3448;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    white-space: nowrap;
}
.orca-section-dot {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #1f6feb;
    flex-shrink: 0;
}

/* ── Cards ── */
.orca-card {
    background: #0e0f12;
    border: 1px solid #1a1e28;
    border-radius: 2px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.78rem;
    color: #5a6478;
    line-height: 1.7;
}
.orca-card strong, .orca-card b {
    color: #8a95a8;
    font-weight: 500;
}
.orca-card .hl {
    color: #4a9eff;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Alert ── */
.orca-alert {
    background: #0f0a0a;
    border: 1px solid #2d1a1a;
    border-left: 2px solid #c0392b;
    border-radius: 2px;
    padding: 16px 20px;
    font-size: 0.78rem;
    color: #7a4040;
    line-height: 1.7;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a1e28;
    border-radius: 0;
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #3d4558 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent;
    border-radius: 0 !important;
    background: transparent !important;
    margin-right: 0;
}
.stTabs [aria-selected="true"] {
    color: #4a9eff !important;
    border-bottom: 2px solid #1f6feb !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px;
}

/* ── DataFrames ── */
.stDataFrame {
    border: 1px solid #1a1e28 !important;
    border-radius: 2px !important;
}
.stDataFrame th {
    background: #111318 !important;
    color: #3d4558 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.62rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1a1e28 !important;
}
.stDataFrame td {
    background: #0e0f12 !important;
    color: #6a7585 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    border-bottom: 1px solid #12151c !important;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #1a1e28;
    margin: 20px 0;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: #1f6feb transparent transparent transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0c0d0f; }
::-webkit-scrollbar-thumb { background: #1e2330; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #2a3040; }

/* ── Sidebar header override ── */
.sidebar-brand {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #1f6feb;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.sidebar-ver {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    color: #2a3040;
    letter-spacing: 0.12em;
    margin-bottom: 16px;
}

/* ── Status indicator ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    color: #2a3040;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 8px;
}
.status-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #27ae60;
    box-shadow: 0 0 6px #27ae6080;
    flex-shrink: 0;
}
.status-dot.warn { background: #f39c12; box-shadow: 0 0 6px #f39c1280; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────
POBLACION_ESTADOS = {
    'Aguascalientes': 1513000, 'Baja California': 3835000, 'Baja California Sur': 812000,
    'Campeche': 943000, 'Coahuila de Zaragoza': 3242000, 'Colima': 748000,
    'Chiapas': 5650000, 'Chihuahua': 3832000, 'Ciudad de México': 9209000,
    'Durango': 1858000, 'Guanajuato': 6258000, 'Guerrero': 3569000,
    'Hidalgo': 3156000, 'Jalisco': 8540000, 'México': 17200000,
    'Michoacán de Ocampo': 4825000, 'Morelos': 1993000, 'Nayarit': 1255000,
    'Nuevo León': 5956000, 'Oaxaca': 4165000, 'Puebla': 6735000,
    'Querétaro': 2490000, 'Quintana Roo': 1919000, 'San Luis Potosí': 2845000,
    'Sinaloa': 3060000, 'Sonora': 3004000, 'Tabasco': 2419000,
    'Tamaulipas': 3556000, 'Tlaxcala': 1369000, 'Veracruz de Ignacio de la Llave': 8125000,
    'Yucatán': 2355000, 'Zacatecas': 1645000
}

DELITOS_ALTO_IMPACTO = [
    'Homicidio', 'Secuestro', 'Extorsión', 'Feminicidio',
    'Robo de vehículo automotor', 'Robo a casa habitación',
    'Robo a transeúnte en vía pública'
]

# ── Palantir-style Plotly theme ──
COLORS = {
    'accent':  '#1f6feb',
    'accent2': '#388bfd',
    'green':   '#2ea043',
    'red':     '#c0392b',
    'amber':   '#d97706',
    'purple':  '#8b5cf6',
    'bg':      '#0e0f12',
    'bg2':     '#111318',
    'grid':    '#1a1e28',
    'text':    '#3d4558',
    'text2':   '#5a6478',
    'label':   '#8a95a8',
}

PALETTE = [
    '#1f6feb', '#388bfd', '#2ea043', '#c0392b',
    '#d97706', '#8b5cf6', '#0ea5e9', '#f59e0b',
]

def plotly_layout(fig, title=""):
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(family="IBM Plex Mono, monospace", color=COLORS['text2'], size=10),
        title=dict(
            text=f"<span style='font-size:10px;letter-spacing:0.1em;text-transform:uppercase'>{title}</span>",
            font=dict(color=COLORS['label'], size=10, family="IBM Plex Mono, monospace"),
            x=0.0, xanchor='left',
        ),
        margin=dict(l=16, r=16, t=44, b=16),
        legend=dict(
            bgcolor="#111318",
            bordercolor=COLORS['grid'],
            borderwidth=1,
            font=dict(size=9, family="IBM Plex Mono, monospace", color=COLORS['text2']),
        ),
        xaxis=dict(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            linecolor=COLORS['grid'],
            tickfont=dict(size=9, family="IBM Plex Mono, monospace", color=COLORS['text']),
            tickcolor=COLORS['grid'],
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            zerolinecolor=COLORS['grid'],
            linecolor=COLORS['grid'],
            tickfont=dict(size=9, family="IBM Plex Mono, monospace", color=COLORS['text']),
            tickcolor=COLORS['grid'],
        ),
    )
    return fig

# ─────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1vxqqM0-L1A5yIMs0vJbdIr1_dddEwMnn"
GDRIVE_URL     = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

@st.cache_data(ttl=3600)
def cargar_sesnsp():
    try:
        import requests, io, re

        session  = requests.Session()
        response = session.get(GDRIVE_URL, stream=True, timeout=60)

        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            if token is None:
                match = re.search(r'confirm=([0-9A-Za-z_\-]+)', response.text)
                token = match.group(1) if match else 't'
            response = session.get(f"{GDRIVE_URL}&confirm={token}", stream=True, timeout=120)

        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content), low_memory=False)
        df.columns = df.columns.str.strip()

        rename_map = {}
        cols_lower = {c.lower(): c for c in df.columns}

        for candidate in ['entidad', 'estado', 'entidad federativa', 'nom_ent']:
            if candidate in cols_lower:
                rename_map[cols_lower[candidate]] = 'entidad'
                break
        for candidate in ['tipo de delito', 'tipo_delito', 'delito']:
            if candidate in cols_lower:
                rename_map[cols_lower[candidate]] = 'tipo_delito'
                break
        for candidate in ['subtipo de delito', 'subtipo_delito', 'subtipo']:
            if candidate in cols_lower:
                rename_map[cols_lower[candidate]] = 'subtipo_delito'
                break
        for candidate in ['incidencia_delictiva', 'incidencia', 'total', 'valor']:
            if candidate in cols_lower:
                rename_map[cols_lower[candidate]] = 'incidencia_delictiva'
                break
        for candidate in ['fecha', 'date', 'periodo']:
            if candidate in cols_lower:
                rename_map[cols_lower[candidate]] = 'fecha'
                break

        df = df.rename(columns=rename_map)

        meses_es = ['enero','febrero','marzo','abril','mayo','junio',
                    'julio','agosto','septiembre','octubre','noviembre','diciembre']

        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            df = df.dropna(subset=['fecha'])
            df['anio']    = df['fecha'].dt.year
            df['mes']     = df['fecha'].dt.month
            df['mes_num'] = (df['anio'] - df['anio'].min()) * 12 + df['mes']
            df['incidencia_delictiva'] = pd.to_numeric(
                df['incidencia_delictiva'], errors='coerce').fillna(0)
            df_long = df.copy()
        else:
            cols_meses = [c for c in df.columns if c.lower() in meses_es]
            if not cols_meses:
                return None, "No se reconoció el formato del CSV."

            id_vars = [c for c in df.columns if c not in cols_meses]
            df_long = df.melt(id_vars=id_vars, value_vars=cols_meses,
                              var_name='mes_nombre', value_name='incidencia_delictiva')
            df_long['incidencia_delictiva'] = pd.to_numeric(
                df_long['incidencia_delictiva'], errors='coerce').fillna(0)

            mes_num_map = {m: i+1 for i, m in enumerate(meses_es)}
            df_long['mes'] = df_long['mes_nombre'].str.lower().map(mes_num_map)

            anio_col = next((c for c in df_long.columns if 'año' in c.lower() or 'anio' in c.lower()), None)
            df_long['anio'] = pd.to_numeric(df_long[anio_col], errors='coerce') if anio_col else 2024
            df_long = df_long.dropna(subset=['anio', 'mes'])
            df_long['anio']    = df_long['anio'].astype(int)
            df_long['mes']     = df_long['mes'].astype(int)
            df_long['fecha']   = pd.to_datetime(dict(year=df_long['anio'], month=df_long['mes'], day=1))
            df_long['mes_num'] = (df_long['anio'] - df_long['anio'].min()) * 12 + df_long['mes']

        def tasa(row):
            pob = POBLACION_ESTADOS.get(row.get('entidad', ''), 1_000_000)
            return (row['incidencia_delictiva'] / pob) * 100_000 if pob > 0 else 0

        df_long['tasa_100k'] = df_long.apply(tasa, axis=1)
        return df_long, None

    except Exception as e:
        return None, f"Error al procesar el archivo: {e}"


@st.cache_data(ttl=86400)
def cargar_geojson():
    """
    Descarga el GeoJSON de estados de México desde GitHub (naturalearthdata vía public repo).
    Se cachea 24h — pesa ~500KB, negligible una vez cacheado.
    """
    url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        geojson = r.json()
        # Normalizar claves de propiedades para hacer match con df_raw['entidad']
        NOMBRE_MAP = {
            'Aguascalientes': 'Aguascalientes',
            'Baja California': 'Baja California',
            'Baja California Sur': 'Baja California Sur',
            'Campeche': 'Campeche',
            'Chiapas': 'Chiapas',
            'Chihuahua': 'Chihuahua',
            'Coahuila': 'Coahuila de Zaragoza',
            'Colima': 'Colima',
            'Distrito Federal': 'Ciudad de México',
            'Ciudad de México': 'Ciudad de México',
            'Durango': 'Durango',
            'Guanajuato': 'Guanajuato',
            'Guerrero': 'Guerrero',
            'Hidalgo': 'Hidalgo',
            'Jalisco': 'Jalisco',
            'México': 'México',
            'Mexico': 'México',
            'Michoacán': 'Michoacán de Ocampo',
            'Morelos': 'Morelos',
            'Nayarit': 'Nayarit',
            'Nuevo León': 'Nuevo León',
            'Oaxaca': 'Oaxaca',
            'Puebla': 'Puebla',
            'Querétaro': 'Querétaro',
            'Quintana Roo': 'Quintana Roo',
            'San Luis Potosí': 'San Luis Potosí',
            'Sinaloa': 'Sinaloa',
            'Sonora': 'Sonora',
            'Tabasco': 'Tabasco',
            'Tamaulipas': 'Tamaulipas',
            'Tlaxcala': 'Tlaxcala',
            'Veracruz': 'Veracruz de Ignacio de la Llave',
            'Yucatán': 'Yucatán',
            'Zacatecas': 'Zacatecas',
        }
        for feature in geojson.get('features', []):
            props = feature.get('properties', {})
            nombre_raw = props.get('name', props.get('NAME', props.get('NOMBRE', '')))
            props['entidad_norm'] = NOMBRE_MAP.get(nombre_raw, nombre_raw)
        return geojson, None
    except Exception as e:
        return None, str(e)


def horizonte_futuro(df, periodos):
    ultima = df['fecha'].max()
    max_mn = df['mes_num'].max()
    return pd.DataFrame([{
        'fecha':   ultima + pd.DateOffset(months=i),
        'mes_num': max_mn + i,
        'mes':    (ultima + pd.DateOffset(months=i)).month,
        'anio':   (ultima + pd.DateOffset(months=i)).year,
    } for i in range(1, periodos + 1)])


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">◈ By Cristóbal Arellano Carranza</div>
    <div class="sidebar-ver">ANÁLISIS DE INCIDENCIA DELICTIVA · v2.1</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    df_raw, error_carga = cargar_sesnsp()

    if df_raw is not None:
        df_hi = df_raw[df_raw['tipo_delito'].isin(DELITOS_ALTO_IMPACTO)].copy() \
            if 'tipo_delito' in df_raw.columns else df_raw.copy()

        entidades = sorted(df_raw['entidad'].dropna().unique()) if 'entidad' in df_raw.columns else []
        idx_gto   = entidades.index('Guanajuato') if 'Guanajuato' in entidades else 0

        entidad_sel = st.selectbox("Estado de análisis", entidades, index=idx_gto)
        periodos_fc = st.slider("Períodos a pronosticar (HW)", 1, 12, 6)

        estados_contraste = st.multiselect(
            "Estados para contraste",
            entidades,
            default=[entidades[i] for i in [0, idx_gto, -1] if i < len(entidades)][:3]
        )

        st.markdown("---")
        # Filtro extra para el mapa
        delitos_disponibles = ['Todos los delitos de alto impacto'] + DELITOS_ALTO_IMPACTO
        delito_mapa = st.selectbox("Delito para mapa de calor", delitos_disponibles, index=0)

        st.markdown("---")
        st.markdown(f"""
        <div class="status-row">
            <div class="status-dot"></div>
            <span>SESNSP conectado</span>
        </div>
        <div class="status-row" style="margin-top:6px">
            <div class="status-dot"></div>
            <span>{len(df_raw):,} registros cargados</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        entidad_sel       = "Guanajuato"
        periodos_fc       = 6
        estados_contraste = []
        delito_mapa       = 'Todos los delitos de alto impacto'
        df_hi             = None
        st.markdown("""
        <div class="status-row">
            <div class="status-dot warn"></div>
            <span>Sin datos</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#2a3040;letter-spacing:0.08em;line-height:1.8">
    FUENTE // SESNSP · INEGI<br>
    METODOLOGÍA // CRISP-ML(Q)<br>
    MODELOS // LR · RF · HW<br>
    <br>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="orca-header">
    <div class="orca-wordmark">◈ ORCA · Observatorio de Riesgo Criminal Analítico</div>
    <div class="orca-title">Sistema Nacional de Inteligencia Delictiva</div>
    <div class="orca-sub">SESNSP · Incidencia estatal 2015–2025 · Análisis CRISP-ML(Q) · Clasificado: uso institucional</div>
    <div class="orca-tags">
        <span class="orca-tag">SESNSP</span>
        <span class="orca-tag">REG. LINEAL</span>
        <span class="orca-tag">RANDOM FOREST</span>
        <span class="orca-tag">HOLT-WINTERS</span>
        <span class="orca-tag ok">● SISTEMA ACTIVO</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# ERROR DE CARGA
# ─────────────────────────────────────────────────────────
if df_raw is None:
    st.markdown(f"""
    <div class="orca-alert">
    ⚠ ERROR DE INGESTA — FUENTE NO DISPONIBLE<br><br>
    {error_carga}<br><br>
    Verificar: INM_estatal_dic25.csv.zip debe estar en la raíz del repositorio junto con app.py
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "01 · Diagnóstico Estatal",
    "02 · Regresión Lineal",
    "03 · Random Forest",
    "04 · Holt-Winters",
    "05 · Mapa de Calor",
])

# ════════════════════════════════════════════════════════
# TAB 1 — Diagnóstico
# ════════════════════════════════════════════════════════
with tab1:
    df_est = df_raw[df_raw['entidad'] == entidad_sel].copy() if 'entidad' in df_raw.columns else df_raw.copy()

    total_inc = int(df_est['incidencia_delictiva'].sum())
    anios_rng = f"{int(df_est['anio'].min())}–{int(df_est['anio'].max())}" if 'anio' in df_est.columns else "—"
    tasa_prom = round(df_est['tasa_100k'].mean(), 1)

    disp_nac = (df_raw.groupby('entidad')['tasa_100k'].mean()
                .reset_index().rename(columns={'tasa_100k': 'Promedio_100k'})
                .sort_values('Promedio_100k', ascending=False))
    max_e  = disp_nac.iloc[0]
    min_e  = disp_nac.iloc[-1]
    brecha = round(max_e['Promedio_100k'] / max(min_e['Promedio_100k'], 0.01), 1)

    # Divider
    st.markdown("""
    <div class="orca-section">
        <div class="orca-section-dot"></div>
        <div class="orca-section-label">Indicadores clave · """ + entidad_sel + """</div>
        <div class="orca-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidencias",    f"{total_inc:,}",   anios_rng)
    c2.metric("Tasa prom /100k hab.", f"{tasa_prom}")
    c3.metric("Estado más crítico",   max_e['entidad'],   f"{max_e['Promedio_100k']:.1f} /100k")
    c4.metric("Brecha máx/mín",       f"{brecha}x")

    st.markdown("""
    <div class="orca-section">
        <div class="orca-section-dot"></div>
        <div class="orca-section-label">Disparidad nacional · tasa /100k hab.</div>
        <div class="orca-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        color_disp = []
        for e in disp_nac['entidad']:
            if e == max_e['entidad']:
                color_disp.append(COLORS['red'])
            elif e == min_e['entidad']:
                color_disp.append(COLORS['green'])
            elif e == entidad_sel:
                color_disp.append(COLORS['accent'])
            else:
                color_disp.append('#1e2a3a')

        fig_disp = go.Figure(go.Bar(
            x=disp_nac['Promedio_100k'],
            y=disp_nac['entidad'],
            orientation='h',
            marker=dict(
                color=color_disp,
                line=dict(width=0),
            ),
            text=disp_nac['Promedio_100k'].round(1),
            textposition='outside',
            textfont=dict(size=8, family="IBM Plex Mono, monospace", color=COLORS['text2']),
        ))
        plotly_layout(fig_disp, f"Incidencia delictiva · Tasa mensual promedio por 100 000 hab. · 2015–2025")
        fig_disp.update_layout(
            height=780,
            yaxis=dict(autorange='reversed', tickfont=dict(size=9)),
            bargap=0.35,
        )
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_b:
        if 'subtipo_delito' in df_est.columns:
            st.markdown("""
            <div class="orca-section">
                <div class="orca-section-dot"></div>
                <div class="orca-section-label">Composición delictiva</div>
                <div class="orca-section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            top10 = (df_est.groupby('subtipo_delito')['incidencia_delictiva']
                     .sum().nlargest(10).reset_index())

            fig_pie = px.pie(
                top10,
                values='incidencia_delictiva',
                names='subtipo_delito',
                hole=0.6,
                color_discrete_sequence=PALETTE,
            )
            plotly_layout(fig_pie, f"Top 10 delitos · {entidad_sel}")
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent',
                textfont=dict(size=8, family="IBM Plex Mono, monospace"),
            )
            fig_pie.update_layout(
                height=360,
                showlegend=True,
                legend=dict(
                    font=dict(size=8, family="IBM Plex Mono, monospace"),
                    orientation='v',
                ),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("""
            <div class="orca-section">
                <div class="orca-section-dot"></div>
                <div class="orca-section-label">Tabla de frecuencias</div>
                <div class="orca-section-line"></div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                top10.rename(columns={'subtipo_delito': 'Delito', 'incidencia_delictiva': 'Casos'})
                .set_index('Delito'),
                use_container_width=True
            )

    gua_row = disp_nac[disp_nac['entidad'] == 'Guanajuato']
    gua_val = gua_row['Promedio_100k'].values[0] if not gua_row.empty else 0
    ratio   = gua_val / max(min_e['Promedio_100k'], 0.01)

    st.markdown(f"""
    <div class="orca-card">
    <strong>Análisis de contraste regional</strong><br/>
    Mediante tasas normalizadas por 100 000 habitantes se cuantifica una brecha crítica entre
    <span class="hl">{max_e['entidad']}</span> (máximo nacional: {max_e['Promedio_100k']:.1f})
    y <span class="hl">{min_e['entidad']}</span> (mínimo: {min_e['Promedio_100k']:.1f}).
    Guanajuato registra tasa promedio de <span class="hl">{gua_val:.1f}</span>,
    representando una intensidad delictiva <span class="hl">{ratio:.1f}×</span> mayor al estado más seguro.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 2 — Regresión Lineal
# ════════════════════════════════════════════════════════
with tab2:
    with st.spinner("Entrenando modelos de regresión lineal..."):
        df_fut = horizonte_futuro(df_hi, periodos_fc)
        resultados_lr   = []
        predicciones_lr = []

        for estado in sorted(df_hi['entidad'].dropna().unique()):
            data_e = (df_hi[df_hi['entidad'] == estado]
                      .groupby(['fecha', 'mes_num', 'mes'])['tasa_100k']
                      .mean().reset_index().sort_values('fecha'))
            if len(data_e) < 12: continue

            X = data_e[['mes_num', 'mes']].values
            y = data_e['tasa_100k'].values
            m = LinearRegression().fit(X, y)

            resultados_lr.append({
                'Estado':           estado,
                'Tendencia (β₁)':   round(m.coef_[0], 4),
                'Tasa Prom /100k':  round(y.mean(), 2),
                'R²':               round(r2_score(y, m.predict(X)), 4),
            })
            y_fut = np.maximum(m.predict(df_fut[['mes_num', 'mes']].values), 0)
            for j, val in enumerate(y_fut):
                predicciones_lr.append({'Estado': estado, 'Fecha': df_fut.iloc[j]['fecha'], 'Tasa_Predicha': val})

    df_res_lr  = pd.DataFrame(resultados_lr).sort_values('Tasa Prom /100k', ascending=False)
    df_pred_lr = pd.DataFrame(predicciones_lr)

    st.markdown("""
    <div class="orca-section">
        <div class="orca-section-dot"></div>
        <div class="orca-section-label">Métricas del modelo</div>
        <div class="orca-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("R² promedio",        f"{df_res_lr['R²'].mean():.4f}")
    c2.metric("Mejor R²",           f"{df_res_lr['R²'].max():.4f}",
              df_res_lr.loc[df_res_lr['R²'].idxmax(), 'Estado'])
    c3.metric("Estados modelados",  str(len(df_res_lr)))

    st.markdown("""
    <div class="orca-section">
        <div class="orca-section-dot"></div>
        <div class="orca-section-label">Pronóstico comparativo de tasas</div>
        <div class="orca-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    estados_validos = [e for e in estados_contraste if e in df_hi['entidad'].unique()][:6]

    fig_lr = go.Figure()
    for i, estado in enumerate(estados_validos):
        color = PALETTE[i % len(PALETTE)]
        data_e = (df_hi[df_hi['entidad'] == estado]
                  .groupby(['fecha', 'mes_num', 'mes'])['tasa_100k']
                  .mean().reset_index().sort_values('fecha'))
        df_p   = df_pred_lr[df_pred_lr['Estado'] == estado].sort_values('Fecha')

        fig_lr.add_trace(go.Scatter(
            x=data_e['fecha'], y=data_e['tasa_100k'],
            name=estado, mode='lines',
            line=dict(color=color, width=1),
            opacity=0.3, showlegend=False,
        ))
        x_fc = pd.concat([pd.Series([data_e['fecha'].iloc[-1]]), df_p['Fecha']])
        y_fc = pd.concat([pd.Series([data_e['tasa_100k'].iloc[-1]]), df_p['Tasa_Predicha']])
        fig_lr.add_trace(go.Scatter(
            x=x_fc, y=y_fc, name=estado,
            mode='lines',
            line=dict(color=color, width=2, dash='dot'),
        ))

    if not df_pred_lr.empty:
        fig_lr.add_vrect(
            x0=df_pred_lr['Fecha'].min(), x1=df_pred_lr['Fecha'].max(),
            fillcolor="rgba(31,111,235,0.04)", line_width=0,
            annotation_text=f"HORIZONTE · {periodos_fc}M",
            annotation_position="top left",
            annotation_font=dict(color=COLORS['text'], size=8, family="IBM Plex Mono, monospace"),
        )

    plotly_layout(fig_lr, f"Regresión lineal · Pronóstico {periodos_fc} meses · Contraste estatal")
    fig_lr.update_layout(height=420, xaxis_title="Fecha", yaxis_title="Tasa /100k hab.")
    st.plotly_chart(fig_lr, use_container_width=True)

    st.markdown("""
    <div class="orca-section">
        <div class="orca-section-dot"></div>
        <div class="orca-section-label">R² y tendencia por entidad</div>
        <div class="orca-section-line"></div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_res_lr.set_index('Estado'), use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 3 — Random Forest
# ════════════════════════════════════════════════════════
with tab3:
    with st.spinner("Entrenando Random Forest — 100 estimadores, profundidad máx. 10..."):
        resultados_rf   = []
        predicciones_rf = []
        modelos_cache   = {}

        for estado in sorted(df_hi['entidad'].dropna().unique()):
            data_e = (df_hi[df_hi['entidad'] == estado]
                      .groupby(['fecha', 'mes_num', 'mes'])['tasa_100k']
                      .mean().reset_index().sort_values('fecha'))
            if len(data_e) < 24: continue

            X = data_e[['mes_num', 'mes']].values
            y = data_e['tasa_100k'].values

            modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            modelo_rf.fit(X, y)
            y_pred = modelo_rf.predict(X)
            imp    = modelo_rf.feature_importances_

            resultados_rf.append({
                'Estado':              estado,
                'R²':                  round(r2_score(y, y_pred), 4),
                'MAE':                 round(mean_absolute_error(y, y_pred), 2),
                'Imp. Tendencia':      round(imp[0], 4),
                'Imp. Estacionalidad': round(imp[1], 4),
                'Tasa Prom /100k':     round(y.mean(), 2),
            })
            modelos_cache[estado] = {'data': data_e}

            df_fut_rf = horizonte_futuro(df_hi, periodos_fc)
            y_fut = modelo_rf.predict(df_fut_rf[['mes_num', 'mes']].values)
            for j, val in enumerate(y_fut):
                predicciones_rf.append({'Estado': estado, 'Fecha': df_fut_rf.iloc[j]['fecha'], 'Tasa_Predicha': val})

    df_res_rf  = pd.DataFrame(resultados_rf).sort_values('R²', ascending=False)
    df_pred_rf = pd.DataFrame(predicciones_rf)

    st.markdown("""
    <div class="orca-section">
        <div class="orca-section-dot"></div>
        <div class="orca-section-label">Métricas del modelo</div>
        <div class="orca-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² promedio",       f"{df_res_rf['R²'].mean():.4f}")
    c2.metric("Mejor R²",          f"{df_res_rf['R²'].max():.4f}", df_res_rf.iloc[0]['Estado'])
    c3.metric("MAE promedio",       f"{df_res_rf['MAE'].mean():.2f}")
    c4.metric("Estados modelados",  str(len(df_res_rf)))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="orca-section">
            <div class="orca-section-dot"></div>
            <div class="orca-section-label">Importancia de variables</div>
            <div class="orca-section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        avg_imp = df_res_rf[['Imp. Tendencia', 'Imp. Estacionalidad']].mean()
        fig_imp = go.Figure(go.Bar(
            x=['Tendencia\n(mes_num)', 'Estacionalidad\n(mes)'],
            y=avg_imp.values,
            marker=dict(
                color=[COLORS['accent'] if v == avg_imp.max() else '#1e2a3a' for v in avg_imp.values],
                line=dict(width=0),
            ),
            text=[f"{v:.3f}" for v in avg_imp.values],
            textposition='outside',
            textfont=dict(size=9, family="IBM Plex Mono, monospace", color=COLORS['text2']),
            width=0.4,
        ))
        plotly_layout(fig_imp, "Importancia relativa de features · Promedio nacional")
        fig_imp.update_layout(height=300, yaxis_title="Importancia")
        st.plotly_chart(fig_imp, use_container_width=True)

        dominante = ("tendencia temporal → cambio estructural"
                     if avg_imp['Imp. Tendencia'] > avg_imp['Imp. Estacionalidad']
                     else "estacionalidad mensual → patrones de calendario")
        st.markdown(f"""
        <div class="orca-card">
        <strong>Interpretación</strong><br/>
        El modelo detecta dominancia de <span class="hl">{dominante}</span>.
        Si la tendencia domina → existe un cambio estructural (inhibición o escalada).
        Si la estacionalidad domina → el delito sigue patrones de calendario.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="orca-section">
            <div class="orca-section-dot"></div>
            <div class="orca-section-label">Análisis predictivo · {entidad_sel}</div>
            <div class="orca-section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        if entidad_sel in modelos_cache:
            data_h = modelos_cache[entidad_sel]['data']
            df_p_e = df_pred_rf[df_pred_rf['Estado'] == entidad_sel].sort_values('Fecha')

            fig_rf = go.Figure()
            fig_rf.add_trace(go.Scatter(
                x=data_h['fecha'], y=data_h['tasa_100k'],
                name='Histórico SESNSP', mode='lines',
                line=dict(color='#1e2a3a', width=1.5),
            ))
            fig_rf.add_trace(go.Scatter(
                x=df_p_e['Fecha'], y=df_p_e['Tasa_Predicha'],
                name=f'Predicción RF · {periodos_fc}m',
                mode='lines+markers',
                line=dict(color=COLORS['red'], width=2, dash='dot'),
                marker=dict(size=4, symbol='square', color=COLORS['red']),
            ))
            plotly_layout(fig_rf, f"Random Forest · {entidad_sel}")
            fig_rf.update_layout(height=320, xaxis_title="Fecha", yaxis_title="Tasa /100k hab.")
            st.plotly_chart(fig_rf, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="orca-card" style="color:#3d4558">
            Sin datos suficientes para {entidad_sel} (mínimo 24 registros requeridos).
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="orca-section">
        <div class="orca-section-dot"></div>
        <div class="orca-section-label">Reporte ejecutivo por entidad</div>
        <div class="orca-section-line"></div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_res_rf.set_index('Estado'), use_container_width=True)

    mejor      = df_res_rf.iloc[0]
    mayor_tend = df_res_rf.sort_values('Imp. Tendencia', ascending=False).iloc[0]
    st.markdown(f"""
    <div class="orca-card">
    <strong>Resumen ejecutivo</strong> ·
    Precisión promedio (R²): <span class="hl">{df_res_rf['R²'].mean():.4f}</span> ·
    Mejor ajuste: <span class="hl">{mejor['Estado']}</span> (R²={mejor['R²']:.2f}) ·
    Mayor tendencia al alza: <span class="hl">{mayor_tend['Estado']}</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 4 — Holt-Winters
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
    <div class="orca-section">
        <div class="orca-section-dot"></div>
        <div class="orca-section-label">Holt-Winters · {entidad_sel} · trend=add · damped_trend=True</div>
        <div class="orca-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    data_hw = (df_hi[df_hi['entidad'] == entidad_sel]
               .groupby(['fecha', 'mes_num', 'mes'])['tasa_100k']
               .mean().reset_index().sort_values('fecha'))

    if len(data_hw) < 12:
        st.markdown(f"""
        <div class="orca-alert">
        ⚠ Datos insuficientes para {entidad_sel} — se requieren mínimo 12 registros temporales.
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            y_hw   = data_hw['tasa_100k'].values.astype(float)
            fechas = data_hw['fecha'].values

            model_hw = ExponentialSmoothing(
                y_hw, trend='add', seasonal=None, damped_trend=True
            ).fit(optimized=True)

            fitted    = model_hw.fittedvalues
            forecast  = model_hw.forecast(periodos_fc)
            ultima_f  = pd.Timestamp(fechas[-1])
            fechas_fc = [ultima_f + pd.DateOffset(months=i+1) for i in range(periodos_fc)]

            r2_hw  = round(r2_score(y_hw, fitted), 4)
            mae_hw = round(mean_absolute_error(y_hw, fitted), 2)

            c1, c2, c3 = st.columns(3)
            c1.metric("R² Holt-Winters",  f"{r2_hw}")
            c2.metric("MAE ajuste",        f"{mae_hw:.2f}")
            c3.metric("Períodos forecast", str(periodos_fc))

            fig_hw = go.Figure()
            fig_hw.add_trace(go.Scatter(
                x=data_hw['fecha'], y=y_hw,
                name='Real (SESNSP)', mode='lines',
                line=dict(color='#1e2a3a', width=1.5),
            ))
            fig_hw.add_trace(go.Scatter(
                x=data_hw['fecha'], y=fitted,
                name='Ajuste HW', mode='lines',
                line=dict(color=COLORS['accent2'], width=1.5, dash='dot'),
                opacity=0.8,
            ))
            fig_hw.add_trace(go.Scatter(
                x=fechas_fc, y=forecast,
                name=f'Pronóstico {periodos_fc}m',
                mode='lines+markers',
                line=dict(color=COLORS['amber'], width=2.5, dash='dot'),
                marker=dict(size=5, symbol='circle', color=COLORS['amber']),
            ))
            fig_hw.add_vrect(
                x0=fechas_fc[0], x1=fechas_fc[-1],
                fillcolor="rgba(217,119,6,0.04)", line_width=0,
                annotation_text=f"HORIZONTE PRONÓSTICO · {periodos_fc}M",
                annotation_font=dict(color=COLORS['text'], size=8, family="IBM Plex Mono, monospace"),
                annotation_position="top left",
            )
            plotly_layout(fig_hw, f"Holt-Winters · {entidad_sel} · Suavizamiento exponencial con tendencia amortiguada")
            fig_hw.update_layout(height=420, xaxis_title="Fecha", yaxis_title="Tasa /100k hab.")
            st.plotly_chart(fig_hw, use_container_width=True)

            col_t, col_c = st.columns([1, 1])

            with col_t:
                st.markdown("""
                <div class="orca-section">
                    <div class="orca-section-dot"></div>
                    <div class="orca-section-label">Valores pronosticados</div>
                    <div class="orca-section-line"></div>
                </div>
                """, unsafe_allow_html=True)
                df_fc_show = pd.DataFrame({
                    'Fecha': [f.strftime('%Y-%m') for f in fechas_fc],
                    'Tasa predicha /100k': forecast.round(2),
                    'Variación': np.diff(np.concatenate([[y_hw[-1]], forecast])).round(2),
                })
                st.dataframe(df_fc_show.set_index('Fecha'), use_container_width=True)

            with col_c:
                st.markdown(f"""
                <div class="orca-section">
                    <div class="orca-section-dot"></div>
                    <div class="orca-section-label">Comparativa de modelos · {entidad_sel}</div>
                    <div class="orca-section-line"></div>
                </div>
                """, unsafe_allow_html=True)

                comp_rows = []
                data_lr = df_res_lr[df_res_lr['Estado'] == entidad_sel]
                if not data_lr.empty:
                    comp_rows.append({'Modelo': 'Regresión Lineal',
                                      'R²': data_lr.iloc[0]['R²'], 'MAE': '—',
                                      'Pronóstico': '✓', 'Tipo': 'Supervisado'})

                data_rf = df_res_rf[df_res_rf['Estado'] == entidad_sel]
                if not data_rf.empty:
                    comp_rows.append({'Modelo': 'Random Forest',
                                      'R²': data_rf.iloc[0]['R²'],
                                      'MAE': data_rf.iloc[0]['MAE'],
                                      'Pronóstico': '✓', 'Tipo': 'Supervisado'})

                comp_rows.append({'Modelo': 'Holt-Winters',
                                  'R²': r2_hw, 'MAE': mae_hw,
                                  'Pronóstico': '✓', 'Tipo': 'Serie temporal'})

                if comp_rows:
                    st.dataframe(pd.DataFrame(comp_rows).set_index('Modelo'), use_container_width=True)

        except Exception as e:
            st.markdown(f"""
            <div class="orca-alert">
            ⚠ Error en modelo Holt-Winters: {e}
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 5 — Mapa de Calor Estatal
# ════════════════════════════════════════════════════════
with tab5:

    geojson_mx, geo_error = cargar_geojson()

    st.markdown("""
    <div class="orca-section">
        <div class="orca-section-dot"></div>
        <div class="orca-section-label">Mapa de calor · Incidencia delictiva por entidad federativa</div>
        <div class="orca-section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    if geo_error:
        st.markdown(f"""
        <div class="orca-alert">
        ⚠ No se pudo cargar el GeoJSON de México: {geo_error}<br/>
        Verifica conectividad o carga el archivo manualmente.
        </div>
        """, unsafe_allow_html=True)
    elif df_raw is None:
        st.markdown("""
        <div class="orca-alert">⚠ Sin datos SESNSP cargados.</div>
        """, unsafe_allow_html=True)
    else:
        # ── Construir DataFrame para el mapa ──────────────────
        if delito_mapa == 'Todos los delitos de alto impacto':
            df_mapa_src = df_hi.copy()
            titulo_delito = "Todos los delitos de alto impacto"
        else:
            df_mapa_src = df_hi[df_hi['tipo_delito'] == delito_mapa].copy() \
                if 'tipo_delito' in df_hi.columns else df_hi.copy()
            titulo_delito = delito_mapa

        # Tasa promedio por estado (todos los años disponibles)
        df_mapa = (df_mapa_src.groupby('entidad')['tasa_100k']
                   .mean().reset_index()
                   .rename(columns={'tasa_100k': 'Tasa_prom'}))
        df_mapa['Tasa_prom'] = df_mapa['Tasa_prom'].round(2)

        # Rank de criticidad
        df_mapa = df_mapa.sort_values('Tasa_prom', ascending=False).reset_index(drop=True)
        df_mapa['Ranking'] = df_mapa.index + 1
        df_mapa['Nivel'] = pd.cut(
            df_mapa['Tasa_prom'],
            bins=[-1, df_mapa['Tasa_prom'].quantile(0.25),
                      df_mapa['Tasa_prom'].quantile(0.50),
                      df_mapa['Tasa_prom'].quantile(0.75),
                      float('inf')],
            labels=['BAJO', 'MODERADO', 'ALTO', 'CRÍTICO']
        )

        # KPIs del mapa
        estado_critico   = df_mapa.iloc[0]
        estado_seguro    = df_mapa.iloc[-1]
        media_nacional   = df_mapa['Tasa_prom'].mean()
        estados_criticos = len(df_mapa[df_mapa['Nivel'] == 'CRÍTICO'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Zona más crítica",    estado_critico['entidad'],
                  f"{estado_critico['Tasa_prom']:.1f} /100k")
        c2.metric("Zona más segura",     estado_seguro['entidad'],
                  f"{estado_seguro['Tasa_prom']:.1f} /100k")
        c3.metric("Media nacional",      f"{media_nacional:.1f} /100k")
        c4.metric("Estados nivel crítico", str(estados_criticos))

        st.markdown("""
        <div class="orca-section">
            <div class="orca-section-dot"></div>
            <div class="orca-section-label">Mapa de calor · Tasa promedio /100k hab. · 2015–2025</div>
            <div class="orca-section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col_map, col_rank = st.columns([3, 1])

        with col_map:
            fig_mapa = px.choropleth(
                df_mapa,
                geojson=geojson_mx,
                locations='entidad',
                featureidkey='properties.entidad_norm',
                color='Tasa_prom',
                hover_name='entidad',
                hover_data={
                    'Tasa_prom':  ':.2f',
                    'Ranking':    True,
                    'Nivel':      True,
                    'entidad':    False,
                },
                color_continuous_scale=[
                    [0.00, '#0d1117'],
                    [0.20, '#0d2137'],
                    [0.40, '#1f6feb'],
                    [0.60, '#d97706'],
                    [0.80, '#c0392b'],
                    [1.00, '#7b1113'],
                ],
                labels={'Tasa_prom': 'Tasa /100k'},
                title='',
            )

            fig_mapa.update_geos(
                fitbounds='locations',
                visible=False,
                bgcolor='#0c0d0f',
            )
            fig_mapa.update_coloraxes(
                colorbar=dict(
                    title=dict(
                        text='TASA /100K',
                        font=dict(family='IBM Plex Mono, monospace', size=9,
                                  color=COLORS['text2']),
                    ),
                    tickfont=dict(family='IBM Plex Mono, monospace', size=8,
                                  color=COLORS['text']),
                    bgcolor='#0e0f12',
                    bordercolor=COLORS['grid'],
                    borderwidth=1,
                    thickness=10,
                    len=0.7,
                    x=1.01,
                )
            )
            fig_mapa.update_traces(
                marker_line_color='#1a1e28',
                marker_line_width=0.8,
                hovertemplate=(
                    "<b style='font-family:IBM Plex Mono'>%{hovertext}</b><br>"
                    "Tasa: %{z:.2f} /100k<br>"
                    "<extra></extra>"
                ),
            )
            plotly_layout(fig_mapa, f"Distribución territorial · {titulo_delito}")
            fig_mapa.update_layout(
                height=520,
                margin=dict(l=0, r=60, t=44, b=0),
                paper_bgcolor='#0c0d0f',
                geo=dict(bgcolor='#0c0d0f'),
            )
            st.plotly_chart(fig_mapa, use_container_width=True)

        with col_rank:
            st.markdown("""
            <div class="orca-section">
                <div class="orca-section-dot"></div>
                <div class="orca-section-label">Ranking nacional</div>
                <div class="orca-section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            # Color por nivel
            nivel_color = {
                'CRÍTICO':  '#c0392b',
                'ALTO':     '#d97706',
                'MODERADO': '#1f6feb',
                'BAJO':     '#2ea043',
            }

            for _, row in df_mapa.iterrows():
                color = nivel_color.get(str(row['Nivel']), COLORS['text2'])
                es_sel = row['entidad'] == entidad_sel
                borde  = f"border-left: 2px solid {color};" if es_sel else f"border-left: 2px solid #1a1e28;"
                fondo  = "background:#111318;" if es_sel else ""
                st.markdown(f"""
                <div style="
                    display:flex; justify-content:space-between; align-items:center;
                    padding:6px 10px; margin-bottom:2px;
                    font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
                    border:1px solid #1a1e28; border-radius:1px;
                    {borde} {fondo}
                ">
                    <span style="color:#2d3448; width:16px">{int(row['Ranking'])}</span>
                    <span style="color:#5a6478; flex:1; margin:0 8px; overflow:hidden;
                                 text-overflow:ellipsis; white-space:nowrap">
                        {row['entidad'][:18]}
                    </span>
                    <span style="color:{color}; text-align:right">
                        {row['Tasa_prom']:.1f}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # ── Gráfica de barras por nivel de criticidad ─────────
        st.markdown("""
        <div class="orca-section">
            <div class="orca-section-dot"></div>
            <div class="orca-section-label">Clasificación por nivel de riesgo operativo</div>
            <div class="orca-section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col_bar, col_info = st.columns([2, 1])

        with col_bar:
            color_bars = [nivel_color.get(str(n), COLORS['accent']) for n in df_mapa['Nivel']]
            highlight  = [COLORS['accent2'] if e == entidad_sel else c
                          for e, c in zip(df_mapa['entidad'], color_bars)]

            fig_bar = go.Figure(go.Bar(
                x=df_mapa['entidad'],
                y=df_mapa['Tasa_prom'],
                marker=dict(color=highlight, line=dict(width=0)),
                text=df_mapa['Tasa_prom'].round(1),
                textposition='outside',
                textfont=dict(size=7, family='IBM Plex Mono, monospace',
                              color=COLORS['text']),
                hovertemplate=(
                    "<b>%{x}</b><br>Tasa: %{y:.2f} /100k<extra></extra>"
                ),
            ))

            # Línea media nacional
            fig_bar.add_hline(
                y=media_nacional,
                line_dash='dot',
                line_color=COLORS['text'],
                line_width=1,
                annotation_text=f"  MEDIA NAC. {media_nacional:.1f}",
                annotation_font=dict(size=8, family='IBM Plex Mono, monospace',
                                     color=COLORS['text']),
                annotation_position='top left',
            )

            # Marca del estado seleccionado
            idx_sel = df_mapa[df_mapa['entidad'] == entidad_sel].index
            if len(idx_sel) > 0:
                tasa_sel = df_mapa.loc[idx_sel[0], 'Tasa_prom']
                fig_bar.add_annotation(
                    x=entidad_sel,
                    y=tasa_sel,
                    text=f"◈ {entidad_sel}",
                    showarrow=True,
                    arrowhead=0,
                    arrowcolor=COLORS['accent2'],
                    font=dict(size=8, family='IBM Plex Mono, monospace',
                              color=COLORS['accent2']),
                    ax=0, ay=-32,
                )

            plotly_layout(fig_bar, f"Tasa /100k por entidad · {titulo_delito} · Ordenado por criticidad")
            fig_bar.update_layout(
                height=320,
                xaxis=dict(tickangle=-45, tickfont=dict(size=7)),
                yaxis_title='Tasa /100k hab.',
                bargap=0.2,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_info:
            st.markdown("""
            <div class="orca-section">
                <div class="orca-section-dot"></div>
                <div class="orca-section-label">Leyenda de niveles</div>
                <div class="orca-section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            for nivel, color in nivel_color.items():
                n_estados = len(df_mapa[df_mapa['Nivel'] == nivel])
                rango_df  = df_mapa[df_mapa['Nivel'] == nivel]['Tasa_prom']
                rango_str = f"{rango_df.min():.1f}–{rango_df.max():.1f}" if not rango_df.empty else "—"
                st.markdown(f"""
                <div style="
                    padding:12px 14px; margin-bottom:8px;
                    border:1px solid #1a1e28; border-left: 2px solid {color};
                    border-radius:1px; background:#0e0f12;
                    font-family:'IBM Plex Mono',monospace;
                ">
                    <div style="color:{color};font-size:0.62rem;letter-spacing:0.14em;font-weight:600">
                        {nivel}
                    </div>
                    <div style="color:#3d4558;font-size:0.58rem;margin-top:4px">
                        {n_estados} entidades · {rango_str} /100k
                    </div>
                </div>
                """, unsafe_allow_html=True)

            nivel_sel_row = df_mapa[df_mapa['entidad'] == entidad_sel]
            if not nivel_sel_row.empty:
                nivel_sel   = str(nivel_sel_row.iloc[0]['Nivel'])
                rank_sel    = int(nivel_sel_row.iloc[0]['Ranking'])
                tasa_sel_v  = nivel_sel_row.iloc[0]['Tasa_prom']
                color_ns    = nivel_color.get(nivel_sel, COLORS['text2'])
                st.markdown(f"""
                <div class="orca-card" style="margin-top:12px">
                <strong>{entidad_sel}</strong><br/>
                Ranking: <span class="hl">#{rank_sel} / 32</span><br/>
                Nivel: <span style="color:{color_ns};font-family:'IBM Plex Mono',monospace">
                    {nivel_sel}
                </span><br/>
                Tasa prom: <span class="hl">{tasa_sel_v:.2f} /100k</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Recomendaciones operativas ──────────────────────────
        st.markdown("""
        <div class="orca-section">
            <div class="orca-section-dot"></div>
            <div class="orca-section-label">Recomendaciones operativas · Uso como página de consulta</div>
            <div class="orca-section-line"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="orca-card">
        <strong>Para maximizar el valor de consulta institucional:</strong><br/><br/>

        <span class="hl">01 · Filtro por delito</span><br/>
        El selector del sidebar permite aislar cualquier delito de alto impacto.
        Para planificación de operativos, filtra por "Robo a transeúnte" o "Homicidio"
        según el objetivo de la estrategia — la distribución territorial cambia significativamente
        por tipo de delito.<br/><br/>

        <span class="hl">02 · Lectura de zonas de riesgo</span><br/>
        Los estados en nivel <span style="color:#c0392b">CRÍTICO</span> requieren
        priorización de recursos. Los de nivel <span style="color:#d97706">ALTO</span>
        son candidatos a intervención preventiva antes de escalar. La brecha entre
        <strong>{estado_critico['entidad']}</strong> ({estado_critico['Tasa_prom']:.1f} /100k)
        y <strong>{estado_seguro['entidad']}</strong> ({estado_seguro['Tasa_prom']:.1f} /100k)
        indica desigualdad estructural, no aleatoria.<br/><br/>

        <span class="hl">03 · Siguiente paso técnico recomendado</span><br/>
        Con datos municipales del SESNSP (disponibles en el mismo portal, CSV separado),
        el mapa puede bajarse a granularidad de municipio — pasando de 32 polígonos a ~2,400.
        Eso habilita planificación táctica de patrullaje por cuadrante, no solo estratégica
        por estado. El código de este tab escala sin modificaciones, solo requiere cambiar
        el GeoJSON y el nivel de agregación.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#1e2330;letter-spacing:0.12em">
        ◈ ORCA · SISTEMA DE INTELIGENCIA DELICTIVA · CLASIFICADO: USO INSTITUCIONAL
    </span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;color:#1e2330;letter-spacing:0.08em">
        UIA LEÓN · INGENIERÍA EN IA · CRISP-ML(Q)
    </span>
</div>
""", unsafe_allow_html=True)
