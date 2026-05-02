import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
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
    page_title="Observatorio Delictivo · México",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #0f1626 !important;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p { color: #94a3b8 !important; font-size: 0.82rem; }

[data-testid="metric-container"] {
    background: #111827; border: 1px solid #1e3a5f;
    border-radius: 12px; padding: 14px 18px;
}
[data-testid="metric-container"] label {
    color: #64748b !important; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: .08em;
    font-family: 'Space Mono', monospace;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important; font-family: 'Space Mono', monospace;
}

.obs-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f172a 100%);
    border: 1px solid #1e3a5f; border-radius: 16px;
    padding: 28px 36px; margin-bottom: 24px;
}
.obs-header h1 { font-family: 'Space Mono', monospace; font-size: 1.5rem; color: #f1f5f9; margin: 0 0 6px; }
.obs-header p  { color: #64748b; margin: 0; font-size: .88rem; }

.badge {
    display: inline-block; background: #0ea5e920; border: 1px solid #0ea5e950;
    color: #38bdf8; font-family: 'Space Mono', monospace; font-size: .62rem;
    padding: 3px 10px; border-radius: 20px; margin-right: 6px; letter-spacing: .1em;
}

.section-title {
    font-family: 'Space Mono', monospace; font-size: .7rem; letter-spacing: .18em;
    text-transform: uppercase; color: #38bdf8; border-left: 3px solid #38bdf8;
    padding-left: 10px; margin: 22px 0 14px;
}

.model-card {
    background: #111827; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px; font-size: .83rem; color: #94a3b8; line-height: 1.6;
}
.model-card strong { color: #e2e8f0; }

.alert-box {
    background: #1a0f0f; border: 1px solid #7f1d1d; border-radius: 8px;
    padding: 14px 18px; font-size: .83rem; color: #fca5a5; line-height: 1.6;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0f1626; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab"] { color: #64748b; border-radius: 8px; font-size: .82rem; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #38bdf8 !important; }

hr { border-color: #1e2d4a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTES (de tus scripts originales)
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

# Plotly dark template base
PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    'accent':  '#38bdf8',
    'accent2': '#818cf8',
    'green':   '#34d399',
    'red':     '#f87171',
    'pink':    '#f472b6',
    'orange':  '#fb923c',
    'bg':      '#111827',
    'grid':    '#1e2d4a',
    'text':    '#94a3b8',
}

def plotly_layout(fig, title=""):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(family="DM Sans, sans-serif", color=COLORS['text'], size=12),
        title=dict(text=title, font=dict(color="#e2e8f0", size=14), x=0.01),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="#0f1626", bordercolor=COLORS['grid'], borderwidth=1),
        xaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid']),
    )
    return fig

# ─────────────────────────────────────────────────────────
# CARGA DE DATOS — lee el ZIP del repo directamente
# ─────────────────────────────────────────────────────────
PATH_ZIP = "INM_estatal_dic25.csv.zip"
PATH_CSV = "INM_estatal_dic25.csv"

@st.cache_data
def cargar_sesnsp():
    """
    Intenta cargar desde ZIP (recomendado para GitHub) o CSV plano.
    Limpia columnas y calcula tasa_100k + variables temporales.
    """
    path = PATH_ZIP if os.path.exists(PATH_ZIP) else PATH_CSV if os.path.exists(PATH_CSV) else None
    if path is None:
        return None, "No se encontró el archivo de datos SESNSP en el repositorio."

    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()

        # ── Normalizar nombres de columnas ──────────────────
        # El CSV del SESNSP puede venir con diferentes nombres según la versión
        rename_map = {}
        cols_lower = {c.lower(): c for c in df.columns}

        # Columna entidad
        for candidate in ['entidad', 'estado', 'entidad federativa', 'nom_ent']:
            if candidate in cols_lower:
                rename_map[cols_lower[candidate]] = 'entidad'
                break

        # Columna tipo_delito
        for candidate in ['tipo de delito', 'tipo_delito', 'delito']:
            if candidate in cols_lower:
                rename_map[cols_lower[candidate]] = 'tipo_delito'
                break

        # Columna subtipo_delito
        for candidate in ['subtipo de delito', 'subtipo_delito', 'subtipo']:
            if candidate in cols_lower:
                rename_map[cols_lower[candidate]] = 'subtipo_delito'
                break

        # Columna bien jurídico
        for candidate in ['bien jurídico afectado', 'bien juridico afectado', 'bien_juridico']:
            if candidate in cols_lower:
                rename_map[cols_lower[candidate]] = 'bien_juridico'
                break

        df = df.rename(columns=rename_map)

        # ── Detectar columnas de meses ──────────────────────
        meses_es = ['enero','febrero','marzo','abril','mayo','junio',
                    'julio','agosto','septiembre','octubre','noviembre','diciembre']
        cols_meses = [c for c in df.columns if c.lower() in meses_es]

        if not cols_meses:
            return None, "No se encontraron columnas de meses en el CSV. Verifica el formato."

        # ── Convertir formato ancho → largo ─────────────────
        id_vars = [c for c in df.columns if c not in cols_meses]
        df_long = df.melt(id_vars=id_vars, value_vars=cols_meses,
                          var_name='mes_nombre', value_name='incidencia_delictiva')
        df_long['incidencia_delictiva'] = pd.to_numeric(
            df_long['incidencia_delictiva'], errors='coerce').fillna(0)

        # Mapeo mes → número
        mes_num_map = {m: i+1 for i, m in enumerate(meses_es)}
        df_long['mes'] = df_long['mes_nombre'].str.lower().map(mes_num_map)

        # Columna año
        anio_col = next((c for c in df_long.columns if 'año' in c.lower() or 'anio' in c.lower()), None)
        if anio_col:
            df_long = df_long.rename(columns={anio_col: 'anio'})
        else:
            df_long['anio'] = 2024  # fallback

        df_long['anio'] = pd.to_numeric(df_long['anio'], errors='coerce')
        df_long = df_long.dropna(subset=['anio', 'mes'])
        df_long['anio'] = df_long['anio'].astype(int)
        df_long['mes']  = df_long['mes'].astype(int)

        # Fecha y mes_num para modelos
        df_long['fecha']   = pd.to_datetime(dict(year=df_long['anio'], month=df_long['mes'], day=1))
        df_long['mes_num'] = (df_long['anio'] - df_long['anio'].min()) * 12 + df_long['mes']

        # Tasa por 100k
        def tasa(row):
            pob = POBLACION_ESTADOS.get(row.get('entidad', ''), 1_000_000)
            return (row['incidencia_delictiva'] / pob) * 100_000 if pob > 0 else 0

        df_long['tasa_100k'] = df_long.apply(tasa, axis=1)

        return df_long, None

    except Exception as e:
        return None, f"Error al procesar el archivo: {e}"


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
    st.markdown("### 🔭 Observatorio")
    st.markdown("---")

    df_raw, error_carga = cargar_sesnsp()

    if df_raw is not None:
        # Filtrar solo delitos de alto impacto para los modelos
        df_hi = df_raw[df_raw['tipo_delito'].isin(DELITOS_ALTO_IMPACTO)].copy() \
            if 'tipo_delito' in df_raw.columns else df_raw.copy()

        entidades = sorted(df_raw['entidad'].dropna().unique()) if 'entidad' in df_raw.columns else []
        idx_gto   = entidades.index('Guanajuato') if 'Guanajuato' in entidades else 0

        entidad_sel = st.selectbox("Estado de análisis", entidades, index=idx_gto)
        periodos_fc = st.slider("Períodos a pronosticar (HW)", 1, 12, 6)

        estados_contraste = st.multiselect(
            "Estados para gráficas de contraste",
            entidades,
            default=[entidades[i] for i in [0, idx_gto, -1] if i < len(entidades)][:3]
        )
    else:
        entidad_sel      = "Guanajuato"
        periodos_fc      = 6
        estados_contraste = []
        df_hi            = None

    st.markdown("---")
    st.caption("CRISP-ML(Q) · Iberoamericana León")

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="obs-header">
  <h1>🔭 Observatorio Delictivo · México</h1>
  <p>Análisis CRISP-ML(Q) · Incidencia estatal 2015–2025 · SESNSP</p>
  <br/>
  <span class="badge">SESNSP</span>
  <span class="badge">REG. LINEAL</span>
  <span class="badge">RANDOM FOREST</span>
  <span class="badge">HOLT-WINTERS</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# ERROR DE CARGA
# ─────────────────────────────────────────────────────────
if df_raw is None:
    st.markdown(f"""
    <div class="alert-box">
    ❌ <strong>Archivo no encontrado</strong><br/><br/>
    {error_carga}<br/><br/>
    Asegúrate de subir <code>INM_estatal_dic25.csv.zip</code> (o el .csv) 
    a la raíz de tu repositorio de GitHub junto con este <code>app.py</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Diagnóstico Estatal",
    "📐 Regresión Lineal",
    "🌲 Random Forest",
    "〰️ Holt-Winters",
])

# ════════════════════════════════════════════════════════
# TAB 1 — Diagnóstico  (Proyecto.py §4 + §6)
# ════════════════════════════════════════════════════════
with tab1:
    df_est = df_raw[df_raw['entidad'] == entidad_sel].copy() if 'entidad' in df_raw.columns else df_raw.copy()

    # KPIs
    total_inc = int(df_est['incidencia_delictiva'].sum())
    anios_rng = f"{int(df_est['anio'].min())}–{int(df_est['anio'].max())}" if 'anio' in df_est.columns else "—"
    tasa_prom = round(df_est['tasa_100k'].mean(), 1)

    # Disparidad nacional
    disp_nac = (df_raw.groupby('entidad')['tasa_100k'].mean()
                .reset_index().rename(columns={'tasa_100k': 'Promedio_100k'})
                .sort_values('Promedio_100k', ascending=False))
    max_e  = disp_nac.iloc[0]
    min_e  = disp_nac.iloc[-1]
    brecha = round(max_e['Promedio_100k'] / max(min_e['Promedio_100k'], 0.01), 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidencias",    f"{total_inc:,}",   anios_rng)
    c2.metric("Tasa prom /100k hab.", f"{tasa_prom}")
    c3.metric("Estado más crítico",   max_e['entidad'],   f"{max_e['Promedio_100k']:.1f} /100k")
    c4.metric("Brecha máx/mín",       f"{brecha}x")

    col_a, col_b = st.columns([2, 1])

    # Gráfica disparidad nacional horizontal
    with col_a:
        st.markdown('<p class="section-title">Disparidad nacional · Tasa promedio /100k hab.</p>', unsafe_allow_html=True)
        color_disp = [COLORS['red'] if e == max_e['entidad']
                      else COLORS['green'] if e == min_e['entidad']
                      else COLORS['accent']
                      for e in disp_nac['entidad']]
        fig_disp = go.Figure(go.Bar(
            x=disp_nac['Promedio_100k'],
            y=disp_nac['entidad'],
            orientation='h',
            marker_color=color_disp,
            text=disp_nac['Promedio_100k'].round(1),
            textposition='outside',
            textfont=dict(size=9, color=COLORS['text']),
        ))
        plotly_layout(fig_disp, "Incidencia delictiva · Tasa mensual promedio por 100k hab. (2015–2025)")
        fig_disp.update_layout(height=750, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_disp, use_container_width=True)

    # Top 10 delitos del estado + pie
    with col_b:
        if 'subtipo_delito' in df_est.columns:
            st.markdown(f'<p class="section-title">Top 10 delitos · {entidad_sel}</p>', unsafe_allow_html=True)
            top10 = (df_est.groupby('subtipo_delito')['incidencia_delictiva']
                     .sum().nlargest(10).reset_index())
            fig_pie = px.pie(top10, values='incidencia_delictiva', names='subtipo_delito',
                             hole=0.45, color_discrete_sequence=px.colors.sequential.Plasma_r)
            plotly_layout(fig_pie, f"Composición delictiva · {entidad_sel}")
            fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                                  textfont_size=9)
            fig_pie.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown(f'<p class="section-title">Tabla top 10</p>', unsafe_allow_html=True)
            st.dataframe(
                top10.rename(columns={'subtipo_delito': 'Delito', 'incidencia_delictiva': 'Casos'})
                .set_index('Delito'),
                use_container_width=True
            )

    # Análisis de contraste (≈ Proyecto.py §6)
    gua_row = disp_nac[disp_nac['entidad'] == 'Guanajuato']
    gua_val = gua_row['Promedio_100k'].values[0] if not gua_row.empty else 0
    ratio   = gua_val / max(min_e['Promedio_100k'], 0.01)
    st.markdown(f"""
    <div class="model-card">
    <strong>Análisis de contraste regional</strong><br/>
    Al analizar la disparidad regional mediante tasas normalizadas por cada 100,000 habitantes,
    se observa una brecha crítica entre el máximo nacional (<strong>{max_e['entidad']}</strong>
    con {max_e['Promedio_100k']:.1f}) y el mínimo (<strong>{min_e['entidad']}</strong>
    con {min_e['Promedio_100k']:.1f}).<br/>
    Guanajuato se posiciona con una tasa promedio de <strong>{gua_val:.1f}</strong>, lo que representa
    una intensidad delictiva <strong>{ratio:.1f}x</strong> mayor que el estado más seguro.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — Regresión Lineal  (Proyecto.py §3 + §5)
# ════════════════════════════════════════════════════════
with tab2:
    with st.spinner("Entrenando Regresión Lineal..."):
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

    c1, c2, c3 = st.columns(3)
    c1.metric("R² promedio",       f"{df_res_lr['R²'].mean():.4f}")
    c2.metric("Mejor R²",          f"{df_res_lr['R²'].max():.4f}",
              df_res_lr.loc[df_res_lr['R²'].idxmax(), 'Estado'])
    c3.metric("Estados modelados",  str(len(df_res_lr)))

    st.markdown('<p class="section-title">Pronóstico de tasas · Análisis de contrastes</p>', unsafe_allow_html=True)

    palette_contraste = [COLORS['red'], COLORS['orange'], COLORS['green'],
                         COLORS['accent'], COLORS['accent2'], COLORS['pink']]
    estados_validos = [e for e in estados_contraste if e in df_hi['entidad'].unique()][:6]

    fig_lr = go.Figure()
    for i, estado in enumerate(estados_validos):
        data_e = (df_hi[df_hi['entidad'] == estado]
                  .groupby(['fecha', 'mes_num', 'mes'])['tasa_100k']
                  .mean().reset_index().sort_values('fecha'))
        df_p   = df_pred_lr[df_pred_lr['Estado'] == estado].sort_values('Fecha')

        # Histórico (tenue)
        fig_lr.add_trace(go.Scatter(
            x=data_e['fecha'], y=data_e['tasa_100k'],
            name=estado, mode='lines',
            line=dict(color=palette_contraste[i], width=1),
            opacity=0.35, showlegend=False,
        ))
        # Pronóstico (sólido)
        x_fc = pd.concat([pd.Series([data_e['fecha'].iloc[-1]]), df_p['Fecha']])
        y_fc = pd.concat([pd.Series([data_e['tasa_100k'].iloc[-1]]), df_p['Tasa_Predicha']])
        fig_lr.add_trace(go.Scatter(
            x=x_fc, y=y_fc, name=f"Predicción {estado}",
            mode='lines', line=dict(color=palette_contraste[i], width=2.5, dash='dash'),
        ))

    # Banda del horizonte
    if not df_pred_lr.empty:
        fig_lr.add_vrect(
            x0=df_pred_lr['Fecha'].min(), x1=df_pred_lr['Fecha'].max(),
            fillcolor="rgba(148,163,184,0.05)", line_width=0,
            annotation_text="Horizonte predicción", annotation_position="top left",
            annotation_font_color=COLORS['text'],
        )

    plotly_layout(fig_lr, f"Regresión Lineal · Pronóstico {periodos_fc} meses · Contrastes estatales")
    fig_lr.update_layout(height=420, xaxis_title="Fecha", yaxis_title="Tasa /100k hab.")
    st.plotly_chart(fig_lr, use_container_width=True)

    st.markdown('<p class="section-title">R² y tendencia por estado</p>', unsafe_allow_html=True)
    st.dataframe(df_res_lr.set_index('Estado'), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — Random Forest  (XGBoost.py)
# ════════════════════════════════════════════════════════
with tab3:
    with st.spinner("Entrenando Random Forest (100 árboles, max_depth=10)..."):
        resultados_rf   = []
        predicciones_rf = []
        modelos_cache   = {}

        for estado in sorted(df_hi['entidad'].dropna().unique()):
            data_e = (df_hi[df_hi['entidad'] == estado]
                      .groupby(['fecha', 'mes_num', 'mes'])['tasa_100k']
                      .mean().reset_index().sort_values('fecha'))
            if len(data_e) < 24: continue  # igual que XGBoost.py

            X = data_e[['mes_num', 'mes']].values
            y = data_e['tasa_100k'].values

            # Parámetros exactos de XGBoost.py
            modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            modelo_rf.fit(X, y)
            y_pred = modelo_rf.predict(X)
            imp    = modelo_rf.feature_importances_

            resultados_rf.append({
                'Estado':                     estado,
                'R²':                         round(r2_score(y, y_pred), 4),
                'MAE':                        round(mean_absolute_error(y, y_pred), 2),
                'Imp. Tendencia':             round(imp[0], 4),
                'Imp. Estacionalidad':        round(imp[1], 4),
                'Tasa Prom /100k':            round(y.mean(), 2),
            })
            modelos_cache[estado] = {'data': data_e}

            df_fut_rf = horizonte_futuro(df_hi, periodos_fc)
            y_fut = modelo_rf.predict(df_fut_rf[['mes_num', 'mes']].values)
            for j, val in enumerate(y_fut):
                predicciones_rf.append({'Estado': estado, 'Fecha': df_fut_rf.iloc[j]['fecha'], 'Tasa_Predicha': val})

    df_res_rf  = pd.DataFrame(resultados_rf).sort_values('R²', ascending=False)
    df_pred_rf = pd.DataFrame(predicciones_rf)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² promedio",      f"{df_res_rf['R²'].mean():.4f}")
    c2.metric("Mejor R²",         f"{df_res_rf['R²'].max():.4f}", df_res_rf.iloc[0]['Estado'])
    c3.metric("MAE promedio",      f"{df_res_rf['MAE'].mean():.2f}")
    c4.metric("Estados modelados", str(len(df_res_rf)))

    col_a, col_b = st.columns(2)

    # Gráfica importancia
    with col_a:
        st.markdown('<p class="section-title">Importancia de variables (promedio nacional)</p>', unsafe_allow_html=True)
        avg_imp = df_res_rf[['Imp. Tendencia', 'Imp. Estacionalidad']].mean()
        fig_imp = go.Figure(go.Bar(
            x=['Tendencia\n(mes_num)', 'Estacionalidad\n(mes)'],
            y=avg_imp.values,
            marker_color=[COLORS['accent'] if v == avg_imp.max() else COLORS['accent2']
                          for v in avg_imp.values],
            text=[f"{v:.3f}" for v in avg_imp.values],
            textposition='outside', textfont=dict(color=COLORS['text']),
            width=0.45,
        ))
        plotly_layout(fig_imp, "¿Qué influye más en la incidencia delictiva?")
        fig_imp.update_layout(height=300, yaxis_title="Importancia relativa")
        st.plotly_chart(fig_imp, use_container_width=True)

        dominante = ("la tendencia temporal → cambio estructural"
                     if avg_imp['Imp. Tendencia'] > avg_imp['Imp. Estacionalidad']
                     else "la estacionalidad mensual → patrones de calendario")
        st.markdown(f"""
        <div class="model-card">
        <strong>Interpretación para tu hipótesis</strong><br/>
        El modelo detecta que domina <strong>{dominante}</strong>.
        Si la tendencia domina → existe un cambio estructural (posible inhibición o escalada).<br/>
        Si la estacionalidad domina → el delito sigue patrones de calendario (meses específicos).
        </div>
        """, unsafe_allow_html=True)

    # Gráfica estado detallado
    with col_b:
        st.markdown(f'<p class="section-title">Análisis predictivo · {entidad_sel}</p>', unsafe_allow_html=True)
        if entidad_sel in modelos_cache:
            data_h = modelos_cache[entidad_sel]['data']
            df_p_e = df_pred_rf[df_pred_rf['Estado'] == entidad_sel].sort_values('Fecha')

            fig_rf = go.Figure()
            fig_rf.add_trace(go.Scatter(
                x=data_h['fecha'], y=data_h['tasa_100k'],
                name='Histórico (SESNSP)', mode='lines',
                line=dict(color=COLORS['accent2'], width=1.5), opacity=0.8,
            ))
            fig_rf.add_trace(go.Scatter(
                x=df_p_e['Fecha'], y=df_p_e['Tasa_Predicha'],
                name=f'Predicción RF · {periodos_fc}m',
                mode='lines+markers',
                line=dict(color=COLORS['red'], width=2.5, dash='dash'),
                marker=dict(size=5, symbol='square'),
            ))
            plotly_layout(fig_rf, f"Random Forest · {entidad_sel}")
            fig_rf.update_layout(height=320, xaxis_title="Fecha", yaxis_title="Tasa /100k hab.")
            st.plotly_chart(fig_rf, use_container_width=True)
        else:
            st.info(f"No hay suficientes datos para {entidad_sel} (mínimo 24 registros).")

    st.markdown('<p class="section-title">Reporte ejecutivo por estado</p>', unsafe_allow_html=True)
    st.dataframe(df_res_rf.set_index('Estado'), use_container_width=True)

    mejor      = df_res_rf.iloc[0]
    mayor_tend = df_res_rf.sort_values('Imp. Tendencia', ascending=False).iloc[0]
    st.markdown(f"""
    <div class="model-card">
    <strong>Resumen ejecutivo</strong><br/>
    Precisión promedio (R²): <strong>{df_res_rf['R²'].mean():.4f}</strong> ·
    Estado con mejor ajuste: <strong>{mejor['Estado']}</strong> (R²={mejor['R²']:.2f}) ·
    Mayor tendencia al alza: <strong>{mayor_tend['Estado']}</strong>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 4 — Holt-Winters  (Series_tiempo_HW.py)
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<p class="section-title">Holt-Winters · {entidad_sel} (trend=add, damped_trend=True)</p>', unsafe_allow_html=True)

    data_hw = (df_hi[df_hi['entidad'] == entidad_sel]
               .groupby(['fecha', 'mes_num', 'mes'])['tasa_100k']
               .mean().reset_index().sort_values('fecha'))

    if len(data_hw) < 12:
        st.warning(f"Pocos datos para {entidad_sel}.")
    else:
        try:
            y_hw   = data_hw['tasa_100k'].values.astype(float)
            fechas = data_hw['fecha'].values

            model_hw = ExponentialSmoothing(
                y_hw, trend='add', seasonal=None, damped_trend=True
            ).fit(optimized=True)

            fitted   = model_hw.fittedvalues
            forecast = model_hw.forecast(periodos_fc)
            ultima_f  = pd.Timestamp(fechas[-1])
            fechas_fc = [ultima_f + pd.DateOffset(months=i+1) for i in range(periodos_fc)]

            r2_hw  = round(r2_score(y_hw, fitted), 4)
            mae_hw = round(mean_absolute_error(y_hw, fitted), 2)

            c1, c2, c3 = st.columns(3)
            c1.metric("R² Holt-Winters",   f"{r2_hw}")
            c2.metric("MAE ajuste",         f"{mae_hw:.2f}")
            c3.metric("Períodos pronost.",  str(periodos_fc))

            # Gráfica HW
            fig_hw = go.Figure()
            fig_hw.add_trace(go.Scatter(
                x=data_hw['fecha'], y=y_hw,
                name='Real (SESNSP)', mode='lines',
                line=dict(color=COLORS['accent2'], width=1.8), opacity=0.85,
            ))
            fig_hw.add_trace(go.Scatter(
                x=data_hw['fecha'], y=fitted,
                name='Ajuste HW', mode='lines',
                line=dict(color=COLORS['accent'], width=1.5, dash='dot'),
            ))
            fig_hw.add_trace(go.Scatter(
                x=fechas_fc, y=forecast,
                name=f'Pronóstico {periodos_fc}m',
                mode='lines+markers',
                line=dict(color=COLORS['pink'], width=2.5, dash='dash'),
                marker=dict(size=6, symbol='circle'),
            ))
            # Banda de pronóstico
            fig_hw.add_vrect(
                x0=fechas_fc[0], x1=fechas_fc[-1],
                fillcolor="rgba(244,114,182,0.06)", line_width=0,
                annotation_text="Horizonte pronóstico",
                annotation_font_color=COLORS['text'],
            )
            plotly_layout(fig_hw, f"Holt-Winters · {entidad_sel}")
            fig_hw.update_layout(height=400, xaxis_title="Fecha", yaxis_title="Tasa /100k hab.")
            st.plotly_chart(fig_hw, use_container_width=True)

            # Tabla de valores pronosticados
            col_t, col_c = st.columns([1, 1])
            with col_t:
                st.markdown("**Valores pronosticados:**")
                df_fc_show = pd.DataFrame({
                    'Fecha': [f.strftime('%Y-%m') for f in fechas_fc],
                    'Tasa predicha /100k': forecast.round(2),
                    'Variación': np.diff(np.concatenate([[y_hw[-1]], forecast])).round(2),
                })
                st.dataframe(df_fc_show.set_index('Fecha'), use_container_width=True)

            # Comparativa modelos del estado seleccionado
            with col_c:
                st.markdown(f"**Comparativa de modelos · {entidad_sel}:**")
                comp_rows = []

                data_lr = df_res_lr[df_res_lr['Estado'] == entidad_sel]
                if not data_lr.empty:
                    comp_rows.append({'Modelo': 'Regresión Lineal',
                                      'R²': data_lr.iloc[0]['R²'], 'MAE': '—',
                                      'Pronóstico temporal': '✅', 'Tipo': 'Supervisado'})

                data_rf = df_res_rf[df_res_rf['Estado'] == entidad_sel]
                if not data_rf.empty:
                    comp_rows.append({'Modelo': 'Random Forest',
                                      'R²': data_rf.iloc[0]['R²'],
                                      'MAE': data_rf.iloc[0]['MAE'],
                                      'Pronóstico temporal': '✅', 'Tipo': 'Supervisado'})

                comp_rows.append({'Modelo': 'Holt-Winters',
                                  'R²': r2_hw, 'MAE': mae_hw,
                                  'Pronóstico temporal': '✅', 'Tipo': 'Serie de tiempo'})

                if comp_rows:
                    st.dataframe(pd.DataFrame(comp_rows).set_index('Modelo'), use_container_width=True)

        except Exception as e:
            st.error(f"Error Holt-Winters: {e}")

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#334155;font-size:.72rem;font-family:Space Mono,monospace;">'
    'Observatorio Delictivo · CRISP-ML(Q) · Ingeniería en Inteligencia Artificial · Iberoamericana León'
    '</p>',
    unsafe_allow_html=True
)
