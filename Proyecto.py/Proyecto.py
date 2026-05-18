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
    page_title="¿Qué tan seguro es México? · Datos abiertos de seguridad pública",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CSS — CIVIC / OPEN DATA AESTHETIC
# Limpio, legible, accesible. Sin jerga militar.
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

.stApp { background: #f8f9fb; color: #1a1d23; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}
section[data-testid="stSidebar"] label {
    color: #6b7280 !important;
    font-size: 0.78rem !important;
    font-weight: 500;
    letter-spacing: 0.01em;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #374151 !important;
    font-size: 0.85rem !important;
}

/* ── Métricas ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 0.72rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #111827 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    font-weight: 500;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 50%, #1e3a8a 100%);
    border-radius: 16px;
    padding: 40px 48px 36px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #93c5fd;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.2;
    color: #ffffff;
    margin: 0 0 10px;
    letter-spacing: -0.03em;
}
.hero-desc {
    font-size: 0.95rem;
    color: #bfdbfe;
    line-height: 1.6;
    max-width: 620px;
    margin: 0 0 20px;
}
.hero-pills {
    display: flex; gap: 8px; flex-wrap: wrap;
}
.hero-pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #e0f2fe;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}

/* ── Sección label ── */
.sec-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9ca3af;
    margin: 28px 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e5e7eb;
}

/* ── Tarjeta de contexto ── */
.ctx-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0 20px;
    font-size: 0.88rem;
    color: #1e40af;
    line-height: 1.65;
}
.ctx-card.warn {
    background: #fff7ed;
    border-color: #fed7aa;
    border-left-color: #f97316;
    color: #9a3412;
}
.ctx-card.danger {
    background: #fef2f2;
    border-color: #fecaca;
    border-left-color: #ef4444;
    color: #991b1b;
}
.ctx-card.ok {
    background: #f0fdf4;
    border-color: #bbf7d0;
    border-left-color: #22c55e;
    color: #166534;
}
.ctx-card strong { font-weight: 600; }
.ctx-num {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    background: rgba(59,130,246,0.12);
    padding: 1px 5px;
    border-radius: 4px;
}

/* ── Tarjeta info blanca ── */
.info-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    font-size: 0.85rem;
    color: #4b5563;
    line-height: 1.7;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.info-card strong { color: #111827; font-weight: 600; }
.info-card .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #1d4ed8;
}

/* ── Alert ── */
.alert-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.85rem;
    color: #7f1d1d;
    line-height: 1.6;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px 12px 0 0;
    border-bottom: 2px solid #e5e7eb;
    padding: 0 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #6b7280 !important;
    font-size: 0.82rem !important;
    font-weight: 500;
    padding: 12px 18px !important;
    border-bottom: 2px solid transparent;
    border-radius: 0 !important;
    background: transparent !important;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: #1d4ed8 !important;
    border-bottom: 2px solid #1d4ed8 !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* ── DataFrames ── */
.stDataFrame { border: 1px solid #e5e7eb !important; border-radius: 10px !important; }
.stDataFrame th {
    background: #f9fafb !important;
    color: #6b7280 !important;
    font-size: 0.72rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stDataFrame td {
    color: #374151 !important;
    font-size: 0.82rem !important;
}

/* ── Fuente / metodología pill ── */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.68rem;
    font-weight: 500;
    color: #6b7280;
    margin-right: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Rank row ── */
.rank-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 12px;
    margin-bottom: 3px;
    border-radius: 8px;
    font-size: 0.78rem;
    border: 1px solid #f3f4f6;
    background: #ffffff;
    transition: background 0.1s;
}
.rank-row.selected {
    background: #eff6ff;
    border-color: #bfdbfe;
}

/* ── Nivel badge ── */
.nivel-badge {
    font-size: 0.62rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.nivel-critico  { background:#fee2e2; color:#dc2626; }
.nivel-alto     { background:#fff7ed; color:#ea580c; }
.nivel-moderado { background:#fefce8; color:#ca8a04; }
.nivel-bajo     { background:#f0fdf4; color:#16a34a; }

hr { border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #f3f4f6; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
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

NOMBRES_AMIGABLES = {
    'Homicidio':                      'Homicidios',
    'Secuestro':                      'Secuestros',
    'Extorsión':                      'Extorsiones',
    'Feminicidio':                    'Feminicidios',
    'Robo de vehículo automotor':     'Robo de autos',
    'Robo a casa habitación':         'Robo a casa',
    'Robo a transeúnte en vía pública': 'Robo en calle',
}

# Colores cívicos (no militares)
C = {
    'blue':   '#1d4ed8',
    'blue2':  '#3b82f6',
    'blue3':  '#93c5fd',
    'green':  '#16a34a',
    'red':    '#dc2626',
    'orange': '#ea580c',
    'amber':  '#ca8a04',
    'gray':   '#6b7280',
    'bg':     '#f8f9fb',
    'white':  '#ffffff',
    'border': '#e5e7eb',
    'grid':   '#f3f4f6',
}

PALETTE = ['#1d4ed8','#16a34a','#dc2626','#ea580c','#7c3aed','#0891b2','#be185d','#ca8a04']

def plotly_layout(fig, title="", subtitle=""):
    label = f"{title}<br><span style='font-size:10px;color:#9ca3af;font-weight:400'>{subtitle}</span>" if subtitle else title
    fig.update_layout(
        paper_bgcolor=C['white'],
        plot_bgcolor=C['white'],
        font=dict(family="Inter, sans-serif", color=C['gray'], size=12),
        title=dict(
            text=label,
            font=dict(color='#111827', size=13, family="Inter, sans-serif"),
            x=0.0, xanchor='left', pad=dict(l=4),
        ),
        margin=dict(l=16, r=16, t=52, b=16),
        legend=dict(
            bgcolor=C['white'],
            bordercolor=C['border'],
            borderwidth=1,
            font=dict(size=11, family="Inter, sans-serif", color=C['gray']),
        ),
        xaxis=dict(
            gridcolor=C['grid'], zerolinecolor=C['grid'],
            linecolor=C['border'],
            tickfont=dict(size=11, color=C['gray']),
        ),
        yaxis=dict(
            gridcolor=C['grid'], zerolinecolor=C['grid'],
            linecolor=C['border'],
            tickfont=dict(size=11, color=C['gray']),
        ),
    )
    return fig

# ─────────────────────────────────────────────────────────
# HELPERS DE CONTEXTO — Lenguaje ciudadano
# ─────────────────────────────────────────────────────────
def nivel_seguridad(tasa, media):
    """Convierte tasa numérica en lenguaje ciudadano."""
    if tasa > media * 1.75:
        return "muy por encima del promedio nacional", "danger"
    elif tasa > media * 1.2:
        return "por encima del promedio nacional", "warn"
    elif tasa < media * 0.6:
        return "muy por debajo del promedio nacional", "ok"
    else:
        return "cercano al promedio nacional", ""

def contexto_tasa(estado, tasa, media, estado_max, tasa_max, estado_min, tasa_min):
    """Genera párrafo explicativo en español llano."""
    nivel_txt, _ = nivel_seguridad(tasa, media)
    veces_min = round(tasa / max(tasa_min, 0.01), 1)
    pct_media = round(abs(tasa - media) / media * 100)
    direccion = "más alto" if tasa > media else "más bajo"

    return (
        f"En <strong>{estado}</strong> se registran en promedio "
        f"<strong>{tasa:.1f} delitos de alto impacto por cada 100,000 habitantes</strong> al mes — "
        f"{nivel_txt}. "
        f"La tasa es un {pct_media}% {direccion} que el promedio nacional ({media:.1f} /100k) "
        f"y {veces_min}× mayor que la del estado más seguro ({estado_min}, con {tasa_min:.1f} /100k)."
    )

def contexto_tendencia(beta, estado):
    """Interpreta la pendiente del modelo lineal en lenguaje ciudadano."""
    if abs(beta) < 0.005:
        return f"La incidencia delictiva en <strong>{estado}</strong> se ha mantenido <strong>relativamente estable</strong> en los últimos años. No hay una tendencia clara al alza ni a la baja."
    elif beta > 0:
        return f"En <strong>{estado}</strong> los delitos de alto impacto muestran una tendencia <strong>al alza</strong>: por cada mes que pasa, la tasa aumenta en promedio <strong>{beta:.3f} puntos por cada 100,000 habitantes</strong>. Esto indica que la situación ha empeorado gradualmente."
    else:
        return f"En <strong>{estado}</strong> los delitos de alto impacto muestran una tendencia <strong>a la baja</strong>: la tasa disminuye en promedio <strong>{abs(beta):.3f} puntos por cada 100,000 habitantes</strong> por mes. Esto indica una mejora sostenida."

def contexto_pronostico(estado, ultimo_val, forecast_vals):
    """Interpreta el pronóstico Holt-Winters en lenguaje ciudadano."""
    ultimo = float(ultimo_val)
    final  = float(forecast_vals[-1])
    cambio = final - ultimo
    pct    = round(abs(cambio) / max(ultimo, 0.01) * 100, 1)

    if abs(cambio) < 0.3:
        return f"El modelo proyecta que la incidencia en <strong>{estado}</strong> <strong>se mantendrá estable</strong> en los próximos meses, sin cambios significativos respecto al nivel actual ({ultimo:.1f} /100k)."
    elif cambio > 0:
        return f"El modelo proyecta un <strong>incremento del {pct}%</strong> en la incidencia de {estado} para los próximos meses, pasando de {ultimo:.1f} a {final:.1f} delitos por cada 100,000 habitantes. Este pronóstico debe tomarse como señal de alerta, no como certeza."
    else:
        return f"El modelo proyecta una <strong>reducción del {pct}%</strong> en la incidencia de {estado} para los próximos meses, pasando de {ultimo:.1f} a {final:.1f} delitos por cada 100,000 habitantes. Una tendencia positiva si se sostiene."

# ─────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1vxqqM0-L1A5yIMs0vJbdIr1_dddEwMnn"
GDRIVE_URL     = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

@st.cache_data(ttl=3600)
def cargar_sesnsp():
    try:
        import io, re
        session  = requests.Session()
        response = session.get(GDRIVE_URL, stream=True, timeout=60)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value; break
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
            if candidate in cols_lower: rename_map[cols_lower[candidate]] = 'entidad'; break
        for candidate in ['tipo de delito', 'tipo_delito', 'delito']:
            if candidate in cols_lower: rename_map[cols_lower[candidate]] = 'tipo_delito'; break
        for candidate in ['subtipo de delito', 'subtipo_delito', 'subtipo']:
            if candidate in cols_lower: rename_map[cols_lower[candidate]] = 'subtipo_delito'; break
        for candidate in ['incidencia_delictiva', 'incidencia', 'total', 'valor']:
            if candidate in cols_lower: rename_map[cols_lower[candidate]] = 'incidencia_delictiva'; break
        for candidate in ['fecha', 'date', 'periodo']:
            if candidate in cols_lower: rename_map[cols_lower[candidate]] = 'fecha'; break

        df = df.rename(columns=rename_map)
        meses_es = ['enero','febrero','marzo','abril','mayo','junio',
                    'julio','agosto','septiembre','octubre','noviembre','diciembre']

        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            df = df.dropna(subset=['fecha'])
            df['anio']    = df['fecha'].dt.year
            df['mes']     = df['fecha'].dt.month
            df['mes_num'] = (df['anio'] - df['anio'].min()) * 12 + df['mes']
            df['incidencia_delictiva'] = pd.to_numeric(df['incidencia_delictiva'], errors='coerce').fillna(0)
            df_long = df.copy()
        else:
            cols_meses = [c for c in df.columns if c.lower() in meses_es]
            if not cols_meses:
                return None, "No se reconoció el formato del CSV."
            id_vars = [c for c in df.columns if c not in cols_meses]
            df_long = df.melt(id_vars=id_vars, value_vars=cols_meses,
                              var_name='mes_nombre', value_name='incidencia_delictiva')
            df_long['incidencia_delictiva'] = pd.to_numeric(df_long['incidencia_delictiva'], errors='coerce').fillna(0)
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
        return None, f"Error al cargar los datos: {e}"


@st.cache_data(ttl=86400)
def cargar_geojson():
    url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        geojson = r.json()
        NOMBRE_MAP = {
            'Aguascalientes': 'Aguascalientes', 'Baja California': 'Baja California',
            'Baja California Sur': 'Baja California Sur', 'Campeche': 'Campeche',
            'Chiapas': 'Chiapas', 'Chihuahua': 'Chihuahua',
            'Coahuila': 'Coahuila de Zaragoza', 'Colima': 'Colima',
            'Distrito Federal': 'Ciudad de México', 'Ciudad de México': 'Ciudad de México',
            'Durango': 'Durango', 'Guanajuato': 'Guanajuato', 'Guerrero': 'Guerrero',
            'Hidalgo': 'Hidalgo', 'Jalisco': 'Jalisco',
            'México': 'México', 'Mexico': 'México',
            'Michoacán': 'Michoacán de Ocampo', 'Morelos': 'Morelos',
            'Nayarit': 'Nayarit', 'Nuevo León': 'Nuevo León', 'Oaxaca': 'Oaxaca',
            'Puebla': 'Puebla', 'Querétaro': 'Querétaro', 'Quintana Roo': 'Quintana Roo',
            'San Luis Potosí': 'San Luis Potosí', 'Sinaloa': 'Sinaloa',
            'Sonora': 'Sonora', 'Tabasco': 'Tabasco', 'Tamaulipas': 'Tamaulipas',
            'Tlaxcala': 'Tlaxcala', 'Veracruz': 'Veracruz de Ignacio de la Llave',
            'Yucatán': 'Yucatán', 'Zacatecas': 'Zacatecas',
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
    st.markdown("### 🔍 Explora los datos")
    st.caption("Filtra la información para tu estado y tipo de delito.")
    st.markdown("---")

    df_raw, error_carga = cargar_sesnsp()

    if df_raw is not None:
        df_hi = df_raw[df_raw['tipo_delito'].isin(DELITOS_ALTO_IMPACTO)].copy() \
            if 'tipo_delito' in df_raw.columns else df_raw.copy()

        entidades = sorted(df_raw['entidad'].dropna().unique()) if 'entidad' in df_raw.columns else []
        idx_gto   = entidades.index('Guanajuato') if 'Guanajuato' in entidades else 0

        entidad_sel = st.selectbox("¿De qué estado quieres ver información?", entidades, index=idx_gto)
        periodos_fc = st.slider("¿Cuántos meses hacia adelante pronosticar?", 1, 12, 6)

        estados_contraste = st.multiselect(
            "Compara con otros estados",
            entidades,
            default=[entidades[i] for i in [0, idx_gto, -1] if i < len(entidades)][:3]
        )

        st.markdown("---")
        delitos_disponibles = ['Todos los delitos de alto impacto'] + DELITOS_ALTO_IMPACTO
        delito_mapa = st.selectbox("Tipo de delito para el mapa", delitos_disponibles, index=0)

        st.markdown("---")
        st.success(f"✓ {len(df_raw):,} registros cargados del SESNSP")
    else:
        entidad_sel       = "Guanajuato"
        periodos_fc       = 6
        estados_contraste = []
        delito_mapa       = 'Todos los delitos de alto impacto'
        df_hi             = None
        st.error("No se pudieron cargar los datos.")

    st.markdown("---")
    st.markdown("""
    **¿De dónde vienen estos datos?**

    Fuente oficial: [SESNSP](https://www.gob.mx/sesnsp) — Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública. Datos 2015–2025.

    Las tasas están normalizadas por cada 100,000 habitantes para poder comparar estados de diferente tamaño.
    """)

# ─────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">📊 Datos Abiertos · Seguridad Pública México</div>
    <div class="hero-title">¿Qué tan seguro<br>es donde vives?</div>
    <div class="hero-desc">
        Consulta, compara y entiende la incidencia delictiva en México con datos oficiales
        del gobierno federal. Sin tecnicismos. Sin intermediarios.
    </div>
    <div class="hero-pills">
        <span class="hero-pill">📁 Fuente: SESNSP</span>
        <span class="hero-pill">📅 2015–2025</span>
        <span class="hero-pill">🗺️ 32 estados</span>
        <span class="hero-pill">🔓 Datos abiertos</span>
        <span class="hero-pill">🤖 Modelos predictivos</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# ERROR
# ─────────────────────────────────────────────────────────
if df_raw is None:
    st.markdown(f"""
    <div class="alert-box">
    ⚠️ <strong>No se pudieron cargar los datos</strong><br/><br/>
    {error_carga}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Tu estado",
    "📈 ¿Va subiendo o bajando?",
    "🌲 Análisis avanzado",
    "🔮 Pronóstico",
    "🗺️ Mapa de México",
])

# ════════════════════════════════════════════════════════
# TAB 1 — Tu estado (Diagnóstico)
# ════════════════════════════════════════════════════════
with tab1:
    df_est = df_raw[df_raw['entidad'] == entidad_sel].copy() if 'entidad' in df_raw.columns else df_raw.copy()

    total_inc = int(df_est['incidencia_delictiva'].sum())
    anios_rng = f"{int(df_est['anio'].min())}–{int(df_est['anio'].max())}" if 'anio' in df_est.columns else "—"
    tasa_prom = round(df_est['tasa_100k'].mean(), 1)

    disp_nac  = (df_raw.groupby('entidad')['tasa_100k'].mean()
                 .reset_index().rename(columns={'tasa_100k': 'Promedio_100k'})
                 .sort_values('Promedio_100k', ascending=False))
    max_e     = disp_nac.iloc[0]
    min_e     = disp_nac.iloc[-1]
    media_nac = round(disp_nac['Promedio_100k'].mean(), 1)
    brecha    = round(max_e['Promedio_100k'] / max(min_e['Promedio_100k'], 0.01), 1)

    # Rango del estado seleccionado
    rank_sel_row = disp_nac[disp_nac['entidad'] == entidad_sel]
    rank_sel = int(rank_sel_row.index[0]) + 1 if not rank_sel_row.empty else "—"

    st.markdown(f'<p class="sec-label">Resumen · {entidad_sel}</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posición nacional",   f"#{rank_sel} de 32",  "a mayor número = más seguro")
    c2.metric("Tasa mensual /100k",  f"{tasa_prom}",        f"Nacional: {media_nac}")
    c3.metric("Estado más afectado", max_e['entidad'],      f"{max_e['Promedio_100k']:.1f} /100k")
    c4.metric("Diferencia máx/mín",  f"{brecha}×",          "entre el peor y el mejor estado")

    # Contextualizador ciudadano
    nivel_txt, nivel_cls = nivel_seguridad(tasa_prom, media_nac)
    ctx_txt = contexto_tasa(
        entidad_sel, tasa_prom, media_nac,
        max_e['entidad'], max_e['Promedio_100k'],
        min_e['entidad'], min_e['Promedio_100k']
    )
    st.markdown(f'<div class="ctx-card {nivel_cls}">🔍 <strong>¿Qué significa esto?</strong><br/>{ctx_txt}</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<p class="sec-label">Comparación entre todos los estados</p>', unsafe_allow_html=True)
        color_bars = []
        for e in disp_nac['entidad']:
            if e == max_e['entidad']:     color_bars.append(C['red'])
            elif e == min_e['entidad']:   color_bars.append(C['green'])
            elif e == entidad_sel:        color_bars.append(C['blue'])
            else:                         color_bars.append('#dbeafe')

        fig_disp = go.Figure(go.Bar(
            x=disp_nac['Promedio_100k'],
            y=disp_nac['entidad'],
            orientation='h',
            marker=dict(color=color_bars, line=dict(width=0)),
            text=disp_nac['Promedio_100k'].round(1),
            textposition='outside',
            textfont=dict(size=9, color=C['gray']),
        ))
        # Línea de media nacional
        fig_disp.add_vline(
            x=media_nac, line_dash='dot', line_color=C['gray'], line_width=1.5,
            annotation_text=f"Promedio nacional: {media_nac}",
            annotation_font=dict(size=10, color=C['gray']),
            annotation_position="top",
        )
        plotly_layout(fig_disp,
                      "Tasa mensual de delitos por cada 100,000 habitantes",
                      "Promedio 2015–2025 · Fuente: SESNSP")
        fig_disp.update_layout(
            height=780,
            yaxis=dict(autorange='reversed', tickfont=dict(size=10)),
            bargap=0.3,
        )
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_b:
        if 'subtipo_delito' in df_est.columns:
            st.markdown(f'<p class="sec-label">¿Qué delitos ocurren más en {entidad_sel}?</p>',
                        unsafe_allow_html=True)
            top10 = (df_est.groupby('subtipo_delito')['incidencia_delictiva']
                     .sum().nlargest(10).reset_index())

            fig_pie = px.pie(
                top10, values='incidencia_delictiva', names='subtipo_delito',
                hole=0.55, color_discrete_sequence=PALETTE,
            )
            plotly_layout(fig_pie, f"Composición delictiva · {entidad_sel}")
            fig_pie.update_traces(
                textposition='inside', textinfo='percent',
                textfont=dict(size=10),
            )
            fig_pie.update_layout(height=340, showlegend=True,
                                  legend=dict(font=dict(size=10), orientation='v'))
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown('<p class="sec-label">Detalle por tipo</p>', unsafe_allow_html=True)
            st.dataframe(
                top10.rename(columns={'subtipo_delito': 'Delito', 'incidencia_delictiva': 'Casos registrados'})
                .set_index('Delito'),
                use_container_width=True
            )

    # Nota de metodología
    st.markdown("---")
    st.markdown(f"""
    <div class="info-card">
    <strong>📌 Nota sobre los datos</strong><br/>
    La <em>tasa por cada 100,000 habitantes</em> permite comparar estados de diferente tamaño de población.
    Un estado con más habitantes puede tener más delitos en número absoluto pero menor tasa que uno más pequeño.
    Los datos provienen del <strong>SESNSP</strong> y corresponden a delitos <em>denunciados ante el Ministerio Público</em>.
    La cifra negra (delitos no denunciados) puede ser 3 o 4 veces mayor, según encuestas del INEGI (ENVIPE).
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 2 — ¿Va subiendo o bajando? (Regresión Lineal)
# ════════════════════════════════════════════════════════
with tab2:
    with st.spinner("Calculando tendencias..."):
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
                'Estado': estado,
                'Tendencia (β₁)': round(m.coef_[0], 4),
                'Tasa Prom /100k': round(y.mean(), 2),
                'R²': round(r2_score(y, m.predict(X)), 4),
            })
            y_fut = np.maximum(m.predict(df_fut[['mes_num', 'mes']].values), 0)
            for j, val in enumerate(y_fut):
                predicciones_lr.append({'Estado': estado, 'Fecha': df_fut.iloc[j]['fecha'], 'Tasa_Predicha': val})

    df_res_lr  = pd.DataFrame(resultados_lr).sort_values('Tasa Prom /100k', ascending=False)
    df_pred_lr = pd.DataFrame(predicciones_lr)

    # Contextualizador de tendencia para el estado seleccionado
    row_sel_lr = df_res_lr[df_res_lr['Estado'] == entidad_sel]
    if not row_sel_lr.empty:
        beta_sel = row_sel_lr.iloc[0]['Tendencia (β₁)']
        ctx_tend = contexto_tendencia(beta_sel, entidad_sel)
        tipo_card = "warn" if beta_sel > 0.01 else ("ok" if beta_sel < -0.01 else "")
        icon = "📈" if beta_sel > 0.01 else ("📉" if beta_sel < -0.01 else "➡️")
        st.markdown(f'<div class="ctx-card {tipo_card}">{icon} <strong>Tendencia en {entidad_sel}</strong><br/>{ctx_tend}</div>',
                    unsafe_allow_html=True)

    st.markdown('<p class="sec-label">Evolución histórica y tendencia · Comparación de estados</p>',
                unsafe_allow_html=True)

    estados_validos = [e for e in estados_contraste if e in df_hi['entidad'].unique()][:6]
    fig_lr = go.Figure()
    for i, estado in enumerate(estados_validos):
        color  = PALETTE[i % len(PALETTE)]
        data_e = (df_hi[df_hi['entidad'] == estado]
                  .groupby(['fecha', 'mes_num', 'mes'])['tasa_100k']
                  .mean().reset_index().sort_values('fecha'))
        df_p   = df_pred_lr[df_pred_lr['Estado'] == estado].sort_values('Fecha')

        fig_lr.add_trace(go.Scatter(
            x=data_e['fecha'], y=data_e['tasa_100k'],
            name=estado, mode='lines',
            line=dict(color=color, width=1.2), opacity=0.4, showlegend=False,
        ))
        x_fc = pd.concat([pd.Series([data_e['fecha'].iloc[-1]]), df_p['Fecha']])
        y_fc = pd.concat([pd.Series([data_e['tasa_100k'].iloc[-1]]), df_p['Tasa_Predicha']])
        fig_lr.add_trace(go.Scatter(
            x=x_fc, y=y_fc, name=estado, mode='lines',
            line=dict(color=color, width=2.5, dash='dash'),
        ))

    if not df_pred_lr.empty:
        fig_lr.add_vrect(
            x0=df_pred_lr['Fecha'].min(), x1=df_pred_lr['Fecha'].max(),
            fillcolor="rgba(29,78,216,0.04)", line_width=0,
            annotation_text=f"  Proyección ({periodos_fc} meses)",
            annotation_font=dict(size=10, color=C['gray']),
            annotation_position="top left",
        )

    plotly_layout(fig_lr,
                  "¿La delincuencia sube o baja?",
                  "Línea sólida = datos históricos · Línea punteada = tendencia proyectada")
    fig_lr.update_layout(height=420, xaxis_title="", yaxis_title="Delitos /100k hab.")
    st.plotly_chart(fig_lr, use_container_width=True)

    # Tabla de tendencias — lenguaje simplificado
    st.markdown('<p class="sec-label">Tabla de tendencias por estado</p>', unsafe_allow_html=True)
    df_tabla_lr = df_res_lr.copy()
    df_tabla_lr['Dirección'] = df_tabla_lr['Tendencia (β₁)'].apply(
        lambda x: '📈 Subiendo' if x > 0.01 else ('📉 Bajando' if x < -0.01 else '➡️ Estable')
    )
    st.dataframe(
        df_tabla_lr[['Estado', 'Tasa Prom /100k', 'Dirección', 'R²']].set_index('Estado'),
        use_container_width=True
    )

    st.markdown("""
    <div class="info-card">
    <strong>📌 ¿Cómo leer esta gráfica?</strong><br/>
    La parte sólida de cada línea muestra los datos reales de 2015 a 2025.
    La parte punteada es la <em>tendencia proyectada</em> si el comportamiento reciente se mantiene.
    Una tendencia al alza no significa que el delito <em>vaya a</em> subir — es una proyección estadística,
    no una predicción garantizada.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 3 — Análisis avanzado (Random Forest)
# ════════════════════════════════════════════════════════
with tab3:
    with st.spinner("Procesando análisis avanzado..."):
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
                'Estado': estado, 'R²': round(r2_score(y, y_pred), 4),
                'MAE': round(mean_absolute_error(y, y_pred), 2),
                'Imp. Tendencia': round(imp[0], 4),
                'Imp. Estacionalidad': round(imp[1], 4),
                'Tasa Prom /100k': round(y.mean(), 2),
            })
            modelos_cache[estado] = {'data': data_e}
            df_fut_rf = horizonte_futuro(df_hi, periodos_fc)
            y_fut = modelo_rf.predict(df_fut_rf[['mes_num', 'mes']].values)
            for j, val in enumerate(y_fut):
                predicciones_rf.append({'Estado': estado, 'Fecha': df_fut_rf.iloc[j]['fecha'], 'Tasa_Predicha': val})

    df_res_rf  = pd.DataFrame(resultados_rf).sort_values('R²', ascending=False)
    df_pred_rf = pd.DataFrame(predicciones_rf)

    # Explicación ciudadana antes de los datos
    avg_imp    = df_res_rf[['Imp. Tendencia', 'Imp. Estacionalidad']].mean()
    dom_factor = "el paso del tiempo" if avg_imp['Imp. Tendencia'] > avg_imp['Imp. Estacionalidad'] else "el mes del año"
    dom_expl   = (
        "Esto significa que la delincuencia cambia principalmente de forma <strong>estructural</strong> año con año — "
        "no tanto por la época del año, sino por cambios más profundos en la sociedad o las instituciones."
        if avg_imp['Imp. Tendencia'] > avg_imp['Imp. Estacionalidad'] else
        "Esto significa que la delincuencia sigue <strong>patrones de calendario</strong> — ciertos meses son "
        "consistentemente más peligrosos que otros, independientemente del año."
    )

    st.markdown(f"""
    <div class="ctx-card">
    🤖 <strong>¿Qué factor influye más en la delincuencia?</strong><br/>
    El análisis de 32 estados revela que el factor más determinante es <strong>{dom_factor}</strong>.
    {dom_expl}
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="sec-label">¿Tendencia o estacionalidad?</p>', unsafe_allow_html=True)
        fig_imp = go.Figure(go.Bar(
            x=['Cambio\naño a año', 'Según el\nmes del año'],
            y=avg_imp.values,
            marker=dict(
                color=[C['blue'] if v == avg_imp.max() else '#dbeafe' for v in avg_imp.values],
                line=dict(width=0),
            ),
            text=[f"{v:.0%}" for v in avg_imp.values],
            textposition='outside',
            textfont=dict(size=12, color=C['gray']),
            width=0.45,
        ))
        plotly_layout(fig_imp,
                      "¿Qué determina más la delincuencia?",
                      "Importancia relativa de cada factor · Promedio nacional")
        fig_imp.update_layout(height=300, yaxis_title="Importancia relativa",
                              yaxis_tickformat='.0%')
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_b:
        st.markdown(f'<p class="sec-label">Evolución en {entidad_sel}</p>', unsafe_allow_html=True)
        if entidad_sel in modelos_cache:
            data_h = modelos_cache[entidad_sel]['data']
            df_p_e = df_pred_rf[df_pred_rf['Estado'] == entidad_sel].sort_values('Fecha')
            fig_rf = go.Figure()
            fig_rf.add_trace(go.Scatter(
                x=data_h['fecha'], y=data_h['tasa_100k'],
                name='Datos reales', mode='lines',
                line=dict(color='#bfdbfe', width=1.5),
            ))
            fig_rf.add_trace(go.Scatter(
                x=df_p_e['Fecha'], y=df_p_e['Tasa_Predicha'],
                name='Proyección', mode='lines+markers',
                line=dict(color=C['red'], width=2.5, dash='dash'),
                marker=dict(size=5, symbol='square', color=C['red']),
            ))
            plotly_layout(fig_rf, f"Datos reales y proyección · {entidad_sel}")
            fig_rf.update_layout(height=300, xaxis_title="", yaxis_title="Delitos /100k hab.")
            st.plotly_chart(fig_rf, use_container_width=True)
        else:
            st.info(f"Datos insuficientes para {entidad_sel} (se requieren al menos 24 meses).")

    st.markdown('<p class="sec-label">Resumen por estado</p>', unsafe_allow_html=True)
    df_tabla_rf = df_res_rf[['Estado', 'Tasa Prom /100k', 'R²', 'MAE']].copy()
    st.dataframe(df_tabla_rf.set_index('Estado'), use_container_width=True)

    mejor      = df_res_rf.iloc[0]
    mayor_tend = df_res_rf.sort_values('Imp. Tendencia', ascending=False).iloc[0]
    st.markdown(f"""
    <div class="info-card">
    <strong>📌 ¿Qué es el análisis avanzado?</strong><br/>
    Este análisis usa un modelo de <em>Bosque Aleatorio</em> (Random Forest) — un algoritmo de inteligencia
    artificial que encuentra patrones complejos en los datos. Aquí se usa para identificar si la delincuencia
    responde más al paso del tiempo o a patrones estacionales. El modelo más preciso es
    <strong>{mejor['Estado']}</strong> con un ajuste del <strong>{mejor['R²']:.0%}</strong>.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 4 — Pronóstico (Holt-Winters)
# ════════════════════════════════════════════════════════
with tab4:
    data_hw = (df_hi[df_hi['entidad'] == entidad_sel]
               .groupby(['fecha', 'mes_num', 'mes'])['tasa_100k']
               .mean().reset_index().sort_values('fecha'))

    if len(data_hw) < 12:
        st.warning(f"No hay suficientes datos para pronosticar {entidad_sel}.")
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

            # Contextualizador ciudadano del pronóstico
            ctx_fc = contexto_pronostico(entidad_sel, y_hw[-1], forecast)
            cambio_pct = (forecast[-1] - y_hw[-1]) / max(y_hw[-1], 0.01) * 100
            fc_cls = "warn" if cambio_pct > 5 else ("ok" if cambio_pct < -5 else "")
            fc_icon = "📈" if cambio_pct > 5 else ("📉" if cambio_pct < -5 else "➡️")

            st.markdown(f'<div class="ctx-card {fc_cls}">{fc_icon} <strong>¿Qué se espera en {entidad_sel}?</strong><br/>{ctx_fc}</div>',
                        unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Tasa actual",         f"{y_hw[-1]:.1f} /100k")
            c2.metric(f"Proyección a {periodos_fc} meses", f"{forecast[-1]:.1f} /100k",
                      f"{cambio_pct:+.1f}%")
            c3.metric("Precisión del modelo", f"{r2_hw:.0%}")

            st.markdown('<p class="sec-label">Evolución histórica y pronóstico</p>', unsafe_allow_html=True)

            fig_hw = go.Figure()
            fig_hw.add_trace(go.Scatter(
                x=data_hw['fecha'], y=y_hw,
                name='Datos reales', mode='lines',
                line=dict(color='#bfdbfe', width=2),
            ))
            fig_hw.add_trace(go.Scatter(
                x=data_hw['fecha'], y=fitted,
                name='Ajuste del modelo', mode='lines',
                line=dict(color=C['blue2'], width=1.5, dash='dot'), opacity=0.7,
            ))
            fig_hw.add_trace(go.Scatter(
                x=fechas_fc, y=forecast,
                name=f'Pronóstico ({periodos_fc} meses)',
                mode='lines+markers',
                line=dict(color=C['orange'], width=3, dash='dash'),
                marker=dict(size=7, symbol='circle', color=C['orange']),
            ))
            fig_hw.add_vrect(
                x0=fechas_fc[0], x1=fechas_fc[-1],
                fillcolor="rgba(234,88,12,0.05)", line_width=0,
                annotation_text=f"  Zona de pronóstico · {periodos_fc} meses",
                annotation_font=dict(size=10, color=C['gray']),
                annotation_position="top left",
            )
            plotly_layout(fig_hw,
                          f"¿Hacia dónde va la delincuencia en {entidad_sel}?",
                          "Datos reales · Ajuste del modelo · Pronóstico")
            fig_hw.update_layout(height=420, xaxis_title="", yaxis_title="Delitos /100k hab.")
            st.plotly_chart(fig_hw, use_container_width=True)

            col_t, col_c = st.columns([1, 1])

            with col_t:
                st.markdown('<p class="sec-label">Valores del pronóstico</p>', unsafe_allow_html=True)
                df_fc_show = pd.DataFrame({
                    'Mes': [f.strftime('%B %Y') for f in fechas_fc],
                    'Delitos proyectados /100k': forecast.round(2),
                    'Variación vs. mes anterior': np.diff(np.concatenate([[y_hw[-1]], forecast])).round(2),
                })
                st.dataframe(df_fc_show.set_index('Mes'), use_container_width=True)

            with col_c:
                st.markdown(f'<p class="sec-label">Comparativa de modelos · {entidad_sel}</p>',
                            unsafe_allow_html=True)
                comp_rows = []
                data_lr = df_res_lr[df_res_lr['Estado'] == entidad_sel]
                if not data_lr.empty:
                    comp_rows.append({'Modelo': 'Tendencia lineal', 'Precisión (R²)': data_lr.iloc[0]['R²'],
                                      'Error promedio (MAE)': '—', 'Tipo': 'Regresión'})
                data_rf = df_res_rf[df_res_rf['Estado'] == entidad_sel]
                if not data_rf.empty:
                    comp_rows.append({'Modelo': 'Bosque Aleatorio', 'Precisión (R²)': data_rf.iloc[0]['R²'],
                                      'Error promedio (MAE)': data_rf.iloc[0]['MAE'], 'Tipo': 'IA'})
                comp_rows.append({'Modelo': 'Suavizamiento Exponencial', 'Precisión (R²)': r2_hw,
                                  'Error promedio (MAE)': mae_hw, 'Tipo': 'Serie de tiempo'})
                if comp_rows:
                    st.dataframe(pd.DataFrame(comp_rows).set_index('Modelo'), use_container_width=True)

            st.markdown("""
            <div class="info-card">
            <strong>📌 ¿Qué es un pronóstico estadístico?</strong><br/>
            Este pronóstico usa <em>Suavizamiento Exponencial de Holt-Winters</em>, una técnica que analiza
            la tendencia reciente de los datos para proyectar valores futuros. <strong>No predice el futuro
            con certeza</strong> — es una estimación basada en patrones históricos.
            Factores como cambios en política de seguridad, variaciones económicas o fenómenos sociales
            pueden hacer que la realidad difiera de la proyección.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error al generar el pronóstico: {e}")


# ════════════════════════════════════════════════════════
# TAB 5 — Mapa de México
# ════════════════════════════════════════════════════════
with tab5:
    geojson_mx, geo_error = cargar_geojson()

    if geo_error:
        st.markdown(f'<div class="alert-box">⚠️ No se pudo cargar el mapa: {geo_error}</div>',
                    unsafe_allow_html=True)
    elif df_raw is None:
        st.markdown('<div class="alert-box">⚠️ Sin datos cargados.</div>', unsafe_allow_html=True)
    else:
        if delito_mapa == 'Todos los delitos de alto impacto':
            df_mapa_src  = df_hi.copy()
            titulo_delito = "todos los delitos de alto impacto"
        else:
            df_mapa_src  = df_hi[df_hi['tipo_delito'] == delito_mapa].copy() \
                if 'tipo_delito' in df_hi.columns else df_hi.copy()
            titulo_delito = NOMBRES_AMIGABLES.get(delito_mapa, delito_mapa).lower()

        df_mapa = (df_mapa_src.groupby('entidad')['tasa_100k']
                   .mean().reset_index().rename(columns={'tasa_100k': 'Tasa_prom'}))
        df_mapa['Tasa_prom'] = df_mapa['Tasa_prom'].round(2)
        df_mapa = df_mapa.sort_values('Tasa_prom', ascending=False).reset_index(drop=True)
        df_mapa['Ranking'] = df_mapa.index + 1

        q1, q2, q3 = (df_mapa['Tasa_prom'].quantile(0.25),
                      df_mapa['Tasa_prom'].quantile(0.50),
                      df_mapa['Tasa_prom'].quantile(0.75))

        def clasif_nivel(t):
            if t > q3:   return 'Zona crítica'
            elif t > q2: return 'Zona de atención'
            elif t > q1: return 'Zona moderada'
            else:        return 'Zona segura'

        df_mapa['Nivel'] = df_mapa['Tasa_prom'].apply(clasif_nivel)

        estado_critico = df_mapa.iloc[0]
        estado_seguro  = df_mapa.iloc[-1]
        media_nac_m    = df_mapa['Tasa_prom'].mean()
        n_criticos     = len(df_mapa[df_mapa['Nivel'] == 'Zona crítica'])

        # Contexto ciudadano del mapa
        st.markdown(f"""
        <div class="ctx-card">
        🗺️ <strong>¿Cómo leer el mapa?</strong><br/>
        El mapa muestra la tasa promedio de <strong>{titulo_delito}</strong> por cada 100,000 habitantes en cada estado.
        Los colores más oscuros (rojo) indican mayor incidencia; los más claros (azul) indican menor incidencia.
        El estado más afectado es <strong>{estado_critico['entidad']}</strong>
        ({estado_critico['Tasa_prom']:.1f} /100k) y el menos afectado es
        <strong>{estado_seguro['entidad']}</strong> ({estado_seguro['Tasa_prom']:.1f} /100k).
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estado más afectado",   estado_critico['entidad'],   f"{estado_critico['Tasa_prom']:.1f} /100k")
        c2.metric("Estado más seguro",     estado_seguro['entidad'],    f"{estado_seguro['Tasa_prom']:.1f} /100k")
        c3.metric("Promedio nacional",     f"{media_nac_m:.1f} /100k")
        c4.metric("Estados en zona crítica", f"{n_criticos} de 32")

        col_map, col_rank = st.columns([3, 1])

        NIVEL_COLORS = {
            'Zona crítica':     '#dc2626',
            'Zona de atención': '#ea580c',
            'Zona moderada':    '#ca8a04',
            'Zona segura':      '#16a34a',
        }

        with col_map:
            st.markdown('<p class="sec-label">Mapa de calor por estado</p>', unsafe_allow_html=True)
            fig_mapa = px.choropleth(
                df_mapa,
                geojson=geojson_mx,
                locations='entidad',
                featureidkey='properties.entidad_norm',
                color='Tasa_prom',
                hover_name='entidad',
                hover_data={'Tasa_prom': ':.2f', 'Ranking': True, 'Nivel': True, 'entidad': False},
                color_continuous_scale=[
                    [0.00, '#dbeafe'],
                    [0.25, '#93c5fd'],
                    [0.50, '#f59e0b'],
                    [0.75, '#ef4444'],
                    [1.00, '#7f1d1d'],
                ],
                labels={'Tasa_prom': 'Tasa /100k'},
            )
            fig_mapa.update_geos(fitbounds='locations', visible=False, bgcolor=C['bg'])
            fig_mapa.update_coloraxes(
                colorbar=dict(
                    title=dict(text='Tasa /100k', font=dict(size=11, color=C['gray'])),
                    tickfont=dict(size=10, color=C['gray']),
                    bgcolor=C['white'],
                    bordercolor=C['border'],
                    borderwidth=1,
                    thickness=12, len=0.7, x=1.01,
                )
            )
            fig_mapa.update_traces(
                marker_line_color=C['border'],
                marker_line_width=0.8,
            )
            fig_mapa.update_layout(
                height=500,
                paper_bgcolor=C['bg'],
                geo=dict(bgcolor=C['bg']),
                margin=dict(l=0, r=60, t=10, b=0),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_mapa, use_container_width=True)

        with col_rank:
            st.markdown('<p class="sec-label">Ranking de estados</p>', unsafe_allow_html=True)
            NIVEL_BADGE = {
                'Zona crítica':     'nivel-critico',
                'Zona de atención': 'nivel-alto',
                'Zona moderada':    'nivel-moderado',
                'Zona segura':      'nivel-bajo',
            }
            for _, row in df_mapa.iterrows():
                badge_cls  = NIVEL_BADGE.get(str(row['Nivel']), '')
                is_sel     = row['entidad'] == entidad_sel
                row_cls    = 'rank-row selected' if is_sel else 'rank-row'
                nombre_c   = row['entidad'][:20]
                st.markdown(f"""
                <div class="{row_cls}">
                    <span style="color:#9ca3af;width:20px;font-size:0.7rem">{int(row['Ranking'])}</span>
                    <span style="flex:1;margin:0 8px;color:#374151;font-size:0.75rem;
                                 overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
                        {nombre_c}
                    </span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 color:#111827;font-weight:600">
                        {row['Tasa_prom']:.1f}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # Gráfica de barras ciudadana
        st.markdown(f'<p class="sec-label">Comparación entre todos los estados · {titulo_delito.capitalize()}</p>',
                    unsafe_allow_html=True)

        col_bar, col_leyenda = st.columns([2, 1])

        with col_bar:
            color_bars_m = []
            for _, row in df_mapa.iterrows():
                if row['entidad'] == entidad_sel:
                    color_bars_m.append(C['blue'])
                else:
                    color_bars_m.append(NIVEL_COLORS.get(str(row['Nivel']), C['gray']))

            fig_bar = go.Figure(go.Bar(
                x=df_mapa['entidad'],
                y=df_mapa['Tasa_prom'],
                marker=dict(color=color_bars_m, line=dict(width=0)),
                text=df_mapa['Tasa_prom'].round(1),
                textposition='outside',
                textfont=dict(size=8, color=C['gray']),
                hovertemplate="<b>%{x}</b><br>%{y:.2f} delitos /100k<extra></extra>",
            ))
            fig_bar.add_hline(
                y=media_nac_m, line_dash='dot', line_color=C['gray'], line_width=1.5,
                annotation_text=f"  Promedio nacional: {media_nac_m:.1f}",
                annotation_font=dict(size=10, color=C['gray']),
                annotation_position="top left",
            )
            if entidad_sel in df_mapa['entidad'].values:
                tasa_s = df_mapa.loc[df_mapa['entidad'] == entidad_sel, 'Tasa_prom'].values[0]
                fig_bar.add_annotation(
                    x=entidad_sel, y=tasa_s,
                    text=f"📍 {entidad_sel}",
                    showarrow=True, arrowhead=2, arrowcolor=C['blue'],
                    font=dict(size=10, color=C['blue']), ax=0, ay=-36,
                )
            plotly_layout(fig_bar,
                          f"Tasa de {titulo_delito} por estado",
                          "Delitos por cada 100,000 habitantes · Promedio 2015–2025")
            fig_bar.update_layout(height=340, xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                                  yaxis_title='Delitos /100k hab.', bargap=0.2)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_leyenda:
            st.markdown('<p class="sec-label">¿Qué significa cada zona?</p>', unsafe_allow_html=True)
            zonas = [
                ('Zona crítica',     'nivel-critico',  'Tasa por encima del percentil 75. Requiere atención urgente.'),
                ('Zona de atención', 'nivel-alto',     'Tasa entre percentil 50 y 75. Por encima del promedio.'),
                ('Zona moderada',    'nivel-moderado', 'Tasa entre percentil 25 y 50. Cercana al promedio.'),
                ('Zona segura',      'nivel-bajo',     'Tasa por debajo del percentil 25. Mejor que la mayoría.'),
            ]
            for nombre, badge_cls, desc in zonas:
                n_est = len(df_mapa[df_mapa['Nivel'] == nombre])
                st.markdown(f"""
                <div class="info-card" style="padding:12px 16px;margin-bottom:8px">
                    <span class="nivel-badge {badge_cls}">{nombre}</span>
                    <div style="margin-top:8px;font-size:0.78rem;color:#6b7280">{desc}</div>
                    <div style="margin-top:4px;font-size:0.72rem;color:#9ca3af">{n_est} estados en esta categoría</div>
                </div>
                """, unsafe_allow_html=True)

            # Ficha del estado seleccionado
            row_sel_m = df_mapa[df_mapa['entidad'] == entidad_sel]
            if not row_sel_m.empty:
                r        = row_sel_m.iloc[0]
                badge_c  = NIVEL_BADGE.get(str(r['Nivel']), '')
                st.markdown(f"""
                <div class="ctx-card" style="margin-top:8px">
                <strong>📍 {entidad_sel}</strong><br/>
                Posición: <strong>#{int(r['Ranking'])} de 32</strong><br/>
                Tasa: <strong>{r['Tasa_prom']:.2f} /100k</strong><br/>
                Clasificación: <span class="nivel-badge {badge_c}">{r['Nivel']}</span>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("""
    **Fuentes de datos**

    <span class="source-pill">📁 SESNSP</span>
    <span class="source-pill">📊 INEGI</span>

    Datos oficiales del Secretariado Ejecutivo del Sistema Nacional de Seguridad Pública.
    """, unsafe_allow_html=True)

with col_f2:
    st.markdown("""
    **Metodología**

    Tasas normalizadas por cada 100,000 habitantes.
    Modelos: Regresión Lineal, Random Forest, Holt-Winters (CRISP-ML(Q)).
    """)

with col_f3:
    st.markdown("""
    **Sobre este portal**

    Herramienta de datos abiertos para ciudadanos.
    Desarrollado en UIA León · Ingeniería en IA.
    Los datos son públicos y de libre consulta.
    """)
