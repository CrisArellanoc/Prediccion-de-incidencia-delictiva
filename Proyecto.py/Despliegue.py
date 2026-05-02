import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# 1. Obtenemos la dirección de la carpeta actual
folder_path = os.path.dirname(__CSV__)

# 2. Cargamos los archivos directamente usando sus nombres exactos
# Asegúrate de que los nombres coincidan con los que tienes en GitHub
path_dataset = os.path.join(folder_path, 'dataset_maestro_percepcion_2024.csv')
path_predicciones = os.path.join(folder_path, 'predicciones_2026.csv')

df = pd.read_csv(path_dataset)
df_pred = pd.read_csv(path_predicciones)

st.write("Datos cargados automáticamente desde el repositorio.")
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
section[data-testid="stSidebar"] { background: #0f1626 !important; border-right: 1px solid #1e2d4a; }
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color: #94a3b8 !important; font-size: 0.82rem; }
[data-testid="metric-container"] { background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 14px 18px; }
[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.72rem; text-transform: uppercase; letter-spacing:.08em; font-family:'Space Mono',monospace; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #38bdf8 !important; font-family:'Space Mono',monospace; }
.obs-header { background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#0f172a 100%); border:1px solid #1e3a5f; border-radius:16px; padding:28px 36px; margin-bottom:24px; }
.obs-header h1 { font-family:'Space Mono',monospace; font-size:1.5rem; color:#f1f5f9; margin:0 0 6px; }
.obs-header p  { color:#64748b; margin:0; font-size:.88rem; }
.badge { display:inline-block; background:#0ea5e920; border:1px solid #0ea5e950; color:#38bdf8; font-family:'Space Mono',monospace; font-size:.62rem; padding:3px 10px; border-radius:20px; margin-right:6px; letter-spacing:.1em; }
.section-title { font-family:'Space Mono',monospace; font-size:.7rem; letter-spacing:.18em; text-transform:uppercase; color:#38bdf8; border-left:3px solid #38bdf8; padding-left:10px; margin:22px 0 14px; }
.model-card { background:#111827; border:1px solid #1e3a5f; border-radius:10px; padding:14px 18px; margin-bottom:10px; font-size:.83rem; color:#94a3b8; line-height:1.6; }
.model-card strong { color:#e2e8f0; }
.info-box { background:#0f2744; border:1px solid #1e3a5f; border-radius:8px; padding:12px 16px; font-size:.82rem; color:#94a3b8; line-height:1.6; }
.stTabs [data-baseweb="tab-list"] { background:#0f1626; border-radius:10px; padding:4px; gap:4px; border:1px solid #1e2d4a; }
.stTabs [data-baseweb="tab"] { color:#64748b; border-radius:8px; font-size:.82rem; padding:8px 18px; }
.stTabs [aria-selected="true"] { background:#1e3a5f !important; color:#38bdf8 !important; }
.stButton>button { background:linear-gradient(135deg,#0ea5e9,#0284c7); color:white; border:none; border-radius:8px; font-family:'Space Mono',monospace; font-size:.76rem; padding:10px 24px; }
hr { border-color:#1e2d4a; }
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
    'Robo de vehículo automotor', 'Robo a casa habitación', 'Robo a transeúnte en vía pública'
]

# ─────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────
DARK_BG  = "#0a0e1a"
CARD_BG  = "#111827"
ACCENT   = "#38bdf8"
ACCENT2  = "#818cf8"
GREEN    = "#34d399"
RED      = "#f87171"
PINK     = "#f472b6"
GRID_CLR = "#1e2d4a"
TEXT_CLR = "#94a3b8"

def apply_dark(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD_BG)
    ax.figure.patch.set_facecolor(CARD_BG)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_CLR)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.set_title(title, color="#e2e8f0", fontsize=10, pad=10, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis='y', color=GRID_CLR, linewidth=0.6, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

# ─────────────────────────────────────────────────────────
# FUNCIONES DE CARGA (con @st.cache_data para performance)
# ─────────────────────────────────────────────────────────
@st.cache_data
def cargar_sesnsp(file):
    df = pd.read_csv(file)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df_f = df[df['tipo_delito'].isin(DELITOS_ALTO_IMPACTO)].copy()
    df_a = df_f.groupby(['entidad', 'fecha'])['incidencia_delictiva'].sum().reset_index()

    def tasa(row):
        pob = POBLACION_ESTADOS.get(row['entidad'], 1_000_000)
        return (row['incidencia_delictiva'] / pob) * 100_000

    df_a['tasa_100k'] = df_a.apply(tasa, axis=1)
    df_a['anio']      = df_a['fecha'].dt.year
    df_a['mes']       = df_a['fecha'].dt.month
    df_a['mes_num']   = (df_a['anio'] - df_a['anio'].min()) * 12 + df_a['mes']
    return df_a


@st.cache_data
def cargar_envipe(vic1_file, vic2_file):
    df1 = pd.read_csv(vic1_file, low_memory=False)
    df2 = pd.read_csv(vic2_file, low_memory=False)
    llaves = ['ID_VIV', 'ID_HOG', 'ID_PER']
    df = pd.merge(df1, df2, on=llaves, how='inner', suffixes=('', '_drop'))
    df = df[[c for c in df.columns if not c.endswith('_drop')]]
    df['FAC_ELE'] = pd.to_numeric(df['FAC_ELE'], errors='coerce')
    df['AP4_1']   = pd.to_numeric(df['AP4_1'],   errors='coerce')
    df['es_inseguro'] = np.where(df['AP4_1'] == 2, 1, 0)
    cols_d = [c for c in df.columns if c.startswith('AP4_2_') and '99' not in c]
    for c in cols_d:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['es_victima'] = df[cols_d].apply(lambda x: 1 if 1 in x.values else 0, axis=1)
    return df


def calcular_indicadores_envipe(df_full):
    def _calc(group):
        total = group['FAC_ELE'].sum()
        if total == 0: return None
        return pd.Series({
            'Percepcion_Inseguridad':  round((group['es_inseguro'] * group['FAC_ELE']).sum() / total * 100, 2),
            'Tasa_Victimizacion_100k': round((group['es_victima']  * group['FAC_ELE']).sum() / total * 100_000, 0),
            'Poblacion_Estatal': int(total)
        })
    return df_full.groupby('NOM_ENT').apply(_calc).reset_index().sort_values('Percepcion_Inseguridad', ascending=False)


def horizonte_futuro(df_ag, periodos):
    ultima_fecha = df_ag['fecha'].max()
    mes_num_max  = df_ag['mes_num'].max()
    return pd.DataFrame([{
        'fecha':   ultima_fecha + pd.DateOffset(months=i),
        'mes_num': mes_num_max + i,
        'mes':    (ultima_fecha + pd.DateOffset(months=i)).month
    } for i in range(1, periodos + 1)])

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔭 Observatorio Delictivo")
    st.markdown("---")

    st.markdown("**Fuente SESNSP**")
    sesnsp_file = st.file_uploader("INM_estatal_dic25.csv", type="csv", key="sesnsp")

    st.markdown("**Fuente ENVIPE 2024**")
    vic1_file = st.file_uploader("vic1_envipe2024.csv", type="csv", key="vic1")
    vic2_file = st.file_uploader("vic2_envipe2024.csv", type="csv", key="vic2")

    st.markdown("---")
    st.markdown("**Parámetros**")
    periodos_fc = st.slider("Períodos a pronosticar", 1, 12, 6)

    estados_disponibles = list(POBLACION_ESTADOS.keys())
    estados_contraste = st.multiselect(
        "Estados para gráficas de contrastes",
        estados_disponibles,
        default=["Guanajuato", "Ciudad de México", "Jalisco"]
    )
    estado_detalle = st.selectbox(
        "Estado para análisis detallado (RF + HW)",
        estados_disponibles,
        index=estados_disponibles.index("Guanajuato")
    )

    st.markdown("---")
    st.caption("CRISP-ML(Q) · Iberoamericana León")

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="obs-header">
  <h1>🔭 Observatorio Delictivo · México</h1>
  <p>Análisis CRISP-ML(Q) · Incidencia estatal 2015–2025 · Percepción ENVIPE 2024</p>
  <br/>
  <span class="badge">SESNSP</span>
  <span class="badge">ENVIPE</span>
  <span class="badge">REG. LINEAL</span>
  <span class="badge">RANDOM FOREST</span>
  <span class="badge">HOLT-WINTERS</span>
</div>
""", unsafe_allow_html=True)

sesnsp_ok = sesnsp_file is not None
envipe_ok = vic1_file is not None and vic2_file is not None

if not sesnsp_ok and not envipe_ok:
    st.markdown("""
    <div class="info-box">
    📂 <strong>Carga tus archivos en la barra lateral para comenzar.</strong><br/><br/>
    • <code>INM_estatal_dic25.csv</code> — datos SESNSP (Proyecto.py + XGBoost.py)<br/>
    • <code>conjunto_de_datos_tper_vic1_envipe2024.csv</code> + <code>vic2...</code> — microdatos ENVIPE (Series_tiempo_HW.py)
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Disparidad Estatal",
    "📐 Regresión Lineal",
    "🌲 Random Forest",
    "〰️ Holt-Winters · ENVIPE",
])

# ════════════════════════════════════════════════════════
# TAB 1 — Disparidad Estatal  (Proyecto.py §4 + §6)
# ════════════════════════════════════════════════════════
with tab1:
    if not sesnsp_ok:
        st.warning("Carga el CSV de SESNSP para ver este análisis.")
    else:
        with st.spinner("Procesando datos SESNSP..."):
            df_ag = cargar_sesnsp(sesnsp_file)

        df_disp = (df_ag.groupby('entidad')['tasa_100k']
                   .mean().reset_index()
                   .rename(columns={'entidad': 'Estado', 'tasa_100k': 'Promedio_Historico'})
                   .sort_values('Promedio_Historico', ascending=False))

        max_e   = df_disp.iloc[0]
        min_e   = df_disp.iloc[-1]
        gua_row = df_disp[df_disp['Estado'] == 'Guanajuato']
        gua_val = gua_row['Promedio_Historico'].values[0] if not gua_row.empty else 0
        brecha  = max_e['Promedio_Historico'] / max(min_e['Promedio_Historico'], 0.01)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estado más crítico",   max_e['Estado'],  f"{max_e['Promedio_Historico']:.1f} /100k")
        c2.metric("Estado más seguro",    min_e['Estado'],  f"{min_e['Promedio_Historico']:.1f} /100k")
        c3.metric("Guanajuato",           f"{gua_val:.1f} /100k")
        c4.metric("Brecha máx/mín",       f"{brecha:.1f}x")

        st.markdown('<p class="section-title">Disparidad de incidencia · Tasa mensual promedio /100k hab.</p>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 11))
        colores_bar = [
            RED    if e == max_e['Estado'] else
            GREEN  if e == min_e['Estado'] else
            ACCENT + "99"
            for e in df_disp['Estado']
        ]
        bars = ax.barh(df_disp['Estado'], df_disp['Promedio_Historico'], color=colores_bar, height=0.65)
        for bar in bars:
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}", va='center', color=TEXT_CLR, fontsize=7.5, fontweight='bold')
        apply_dark(ax, "Disparidad de Incidencia Delictiva (2015–2025)\nTasa Mensual Promedio por cada 100,000 hab.",
                   "Delitos /100k hab. (promedio mensual)", "")
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=7)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        ratio = gua_val / max(min_e['Promedio_Historico'], 0.01)
        st.markdown(f"""
        <div class="model-card">
        <strong>Análisis de contraste regional</strong><br/>
        Al analizar la disparidad regional mediante tasas normalizadas por cada 100,000 habitantes,
        se observa una brecha crítica entre el máximo nacional (<strong>{max_e['Estado']}</strong>
        con {max_e['Promedio_Historico']:.1f}) y el mínimo (<strong>{min_e['Estado']}</strong>
        con {min_e['Promedio_Historico']:.1f}).<br/>
        Guanajuato se posiciona con una tasa promedio de <strong>{gua_val:.1f}</strong>, lo que representa
        una intensidad delictiva <strong>{ratio:.1f}x</strong> mayor que el estado más seguro.
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — Regresión Lineal  (Proyecto.py §3 + §5)
# ════════════════════════════════════════════════════════
with tab2:
    if not sesnsp_ok:
        st.warning("Carga el CSV de SESNSP para ver este análisis.")
    else:
        with st.spinner("Entrenando Regresión Lineal por estado..."):
            df_ag = cargar_sesnsp(sesnsp_file)
            df_fut = horizonte_futuro(df_ag, periodos_fc)

            resultados_lr   = []
            predicciones_lr = []

            for estado in sorted(df_ag['entidad'].unique()):
                data_e = df_ag[df_ag['entidad'] == estado].sort_values('fecha')
                if len(data_e) < 12: continue

                X = data_e[['mes_num', 'mes']]
                y = data_e['tasa_100k']
                m = LinearRegression().fit(X, y)

                resultados_lr.append({
                    'Estado':             estado,
                    'Tendencia':          m.coef_[0],
                    'Promedio_Historico': y.mean(),
                    'R2':                 r2_score(y, m.predict(X))
                })

                y_fut = np.maximum(m.predict(df_fut[['mes_num', 'mes']]), 0)
                for j, val in enumerate(y_fut):
                    predicciones_lr.append({
                        'Estado': estado,
                        'Fecha':  df_fut.iloc[j]['fecha'],
                        'Tasa_Predicha': val
                    })

        df_res_lr  = pd.DataFrame(resultados_lr).sort_values('Promedio_Historico', ascending=False)
        df_pred_lr = pd.DataFrame(predicciones_lr)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² promedio",      f"{df_res_lr['R2'].mean():.4f}")
        c2.metric("Mejor R²",         f"{df_res_lr['R2'].max():.4f}",
                  df_res_lr.loc[df_res_lr['R2'].idxmax(), 'Estado'])
        c3.metric("Estados modelados", str(len(df_res_lr)))

        st.markdown('<p class="section-title">Pronóstico de tasas · Análisis de contrastes (§5)</p>', unsafe_allow_html=True)

        # Gráfica §5 — pronóstico contrastes
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        palette_contraste = [RED, "#f39c12", GREEN, ACCENT, ACCENT2, PINK]
        estados_validos = [e for e in estados_contraste if e in df_ag['entidad'].unique()][:6]

        for i, estado in enumerate(estados_validos):
            df_h = df_ag[df_ag['entidad'] == estado].sort_values('fecha')
            ax2.plot(df_h['fecha'], df_h['tasa_100k'],
                     color=palette_contraste[i], lw=1.2, alpha=0.35)

            df_p = df_pred_lr[df_pred_lr['Estado'] == estado].sort_values('Fecha')
            x_tot = pd.concat([pd.Series([df_h['fecha'].iloc[-1]]), df_p['Fecha']])
            y_tot = pd.concat([pd.Series([df_h['tasa_100k'].iloc[-1]]), df_p['Tasa_Predicha']])
            ax2.plot(x_tot, y_tot, color=palette_contraste[i], lw=2.5, ls='--', label=estado)

        if not df_pred_lr.empty:
            ax2.axvspan(df_pred_lr['Fecha'].min(), df_pred_lr['Fecha'].max(),
                        color='#94a3b815', label='Horizonte predicción')

        apply_dark(ax2, f"Pronóstico Regresión Lineal · {periodos_fc} meses", "Fecha", "Tasa /100k hab.")
        ax2.legend(fontsize=7.5, labelcolor=TEXT_CLR, facecolor=DARK_BG,
                   edgecolor=GRID_CLR, bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown('<p class="section-title">R² y tendencia por estado</p>', unsafe_allow_html=True)
        st.dataframe(
            df_res_lr[['Estado', 'R2', 'Tendencia', 'Promedio_Historico']]
            .rename(columns={'Tendencia': 'Tendencia (β₁)', 'Promedio_Historico': 'Tasa Prom /100k'})
            .set_index('Estado').round(4),
            use_container_width=True
        )

# ════════════════════════════════════════════════════════
# TAB 3 — Random Forest  (XGBoost.py)
# ════════════════════════════════════════════════════════
with tab3:
    if not sesnsp_ok:
        st.warning("Carga el CSV de SESNSP para ver este análisis.")
    else:
        with st.spinner("Entrenando Random Forest (100 árboles, max_depth=10)..."):
            df_ag = cargar_sesnsp(sesnsp_file)
            df_fut_rf = horizonte_futuro(df_ag, periodos_fc)

            resultados_rf   = []
            predicciones_rf = []
            modelos_cache   = {}

            for estado in sorted(df_ag['entidad'].unique()):
                data_e = df_ag[df_ag['entidad'] == estado].sort_values('fecha')
                if len(data_e) < 24: continue  # igual que XGBoost.py

                X = data_e[['mes_num', 'mes']]
                y = data_e['tasa_100k']

                # Parámetros exactos de XGBoost.py
                modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                modelo_rf.fit(X, y)
                y_pred = modelo_rf.predict(X)
                imp    = modelo_rf.feature_importances_

                resultados_rf.append({
                    'Estado':                    estado,
                    'R2_Score':                  r2_score(y, y_pred),
                    'MAE':                       mean_absolute_error(y, y_pred),
                    'Importancia_Tendencia':     imp[0],
                    'Importancia_Estacionalidad':imp[1],
                    'Promedio_Historico':         y.mean(),
                })
                modelos_cache[estado] = {'model': modelo_rf, 'data': data_e}

                y_fut = modelo_rf.predict(df_fut_rf[['mes_num', 'mes']])
                for j, val in enumerate(y_fut):
                    predicciones_rf.append({
                        'Estado': estado,
                        'Fecha':  df_fut_rf.iloc[j]['fecha'],
                        'Tasa_Predicha': val
                    })

        df_res_rf  = pd.DataFrame(resultados_rf).sort_values('R2_Score', ascending=False)
        df_pred_rf = pd.DataFrame(predicciones_rf)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² promedio",       f"{df_res_rf['R2_Score'].mean():.4f}")
        c2.metric("Mejor R²",          f"{df_res_rf['R2_Score'].max():.4f}",
                  df_res_rf.iloc[0]['Estado'])
        c3.metric("MAE promedio",       f"{df_res_rf['MAE'].mean():.2f}")
        c4.metric("Estados modelados",  str(len(df_res_rf)))

        col_a, col_b = st.columns(2)

        # Gráfica 1 — Importancia de variables (XGBoost.py §4 gráfica 1)
        with col_a:
            st.markdown('<p class="section-title">Importancia de variables (promedio nacional)</p>', unsafe_allow_html=True)
            avg_imp = df_res_rf[['Importancia_Tendencia', 'Importancia_Estacionalidad']].mean()
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            labels_imp   = ['Tendencia\n(mes_num)', 'Estacionalidad\n(mes)']
            colors_imp   = [ACCENT if v == avg_imp.max() else ACCENT2 for v in avg_imp.values]
            ax3.bar(labels_imp, avg_imp.values, color=colors_imp, width=0.45)
            for i, v in enumerate(avg_imp.values):
                ax3.text(i, v + 0.005, f"{v:.3f}", ha='center', color=TEXT_CLR, fontsize=9, fontweight='bold')
            apply_dark(ax3, "¿Qué influye más en la incidencia delictiva?\nTendencia Temporal vs Estacionalidad Mensual",
                       "", "Importancia relativa")
            st.pyplot(fig3, use_container_width=True)
            plt.close()

            dominante = ("la tendencia temporal → cambio estructural (posible inhibición o escalada)"
                         if avg_imp['Importancia_Tendencia'] > avg_imp['Importancia_Estacionalidad']
                         else "la estacionalidad mensual → el delito sigue patrones de calendario")
            st.markdown(f"""
            <div class="model-card">
            <strong>Interpretación para tu hipótesis</strong><br/>
            El modelo detecta que domina <strong>{dominante}</strong>.
            </div>
            """, unsafe_allow_html=True)

        # Gráfica 2 — Estado detallado (XGBoost.py §4 gráfica 2)
        with col_b:
            st.markdown(f'<p class="section-title">Análisis predictivo · {estado_detalle}</p>', unsafe_allow_html=True)
            if estado_detalle not in modelos_cache:
                st.info(f"No hay suficientes datos para {estado_detalle} (mínimo 24 registros).")
            else:
                data_h  = modelos_cache[estado_detalle]['data']
                df_p_e  = df_pred_rf[df_pred_rf['Estado'] == estado_detalle].sort_values('Fecha')
                fig4, ax4 = plt.subplots(figsize=(5, 3.5))
                ax4.plot(data_h['fecha'], data_h['tasa_100k'],
                         color=ACCENT2, lw=1.5, alpha=0.7, label='Datos históricos (SESNSP)')
                ax4.plot(df_p_e['Fecha'], df_p_e['Tasa_Predicha'],
                         color=RED, lw=2.5, ls='--', marker='s', ms=4,
                         label=f'Predicción RF · {periodos_fc}m')
                apply_dark(ax4, f"Análisis Predictivo Random Forest · {estado_detalle}", "Fecha", "Tasa /100k hab.")
                ax4.legend(fontsize=7, labelcolor=TEXT_CLR, facecolor=DARK_BG, edgecolor=GRID_CLR)
                plt.tight_layout()
                st.pyplot(fig4, use_container_width=True)
                plt.close()

        # Tabla reporte ejecutivo (XGBoost.py §5)
        st.markdown('<p class="section-title">Reporte ejecutivo por estado</p>', unsafe_allow_html=True)
        tabla_rf = df_res_rf[['Estado', 'R2_Score', 'MAE',
                               'Importancia_Tendencia', 'Importancia_Estacionalidad',
                               'Promedio_Historico']].copy()
        tabla_rf.columns = ['Estado', 'R²', 'MAE', 'Imp. Tendencia', 'Imp. Estacionalidad', 'Tasa Prom /100k']
        st.dataframe(tabla_rf.set_index('Estado').round(4), use_container_width=True)

        mejor = df_res_rf.iloc[0]
        mayor_tend = df_res_rf.sort_values('Importancia_Tendencia', ascending=False).iloc[0]
        st.markdown(f"""
        <div class="model-card">
        <strong>Resumen ejecutivo</strong><br/>
        Precisión promedio (R²): <strong>{df_res_rf['R2_Score'].mean():.4f}</strong> ·
        Estado con mejor ajuste: <strong>{mejor['Estado']}</strong> (R²={mejor['R2_Score']:.2f}) ·
        Mayor tendencia al alza detectada: <strong>{mayor_tend['Estado']}</strong>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 4 — Holt-Winters + ENVIPE  (Series_tiempo_HW.py)
# ════════════════════════════════════════════════════════
with tab4:
    col_hw, col_env = st.columns([1, 1])

    # ── Holt-Winters sobre SESNSP ──────────────────────
    with col_hw:
        st.markdown(f'<p class="section-title">Holt-Winters · {estado_detalle}</p>', unsafe_allow_html=True)
        if not sesnsp_ok:
            st.warning("Carga el CSV de SESNSP.")
        else:
            df_ag   = cargar_sesnsp(sesnsp_file)
            data_hw = df_ag[df_ag['entidad'] == estado_detalle].sort_values('fecha')

            if len(data_hw) < 12:
                st.warning(f"Pocos datos para {estado_detalle}.")
            else:
                try:
                    y_hw   = data_hw['tasa_100k'].values.astype(float)
                    fechas = data_hw['fecha'].values

                    # Mismo config que Series_tiempo_HW.py implica: trend=add, damped
                    model_hw = ExponentialSmoothing(
                        y_hw, trend='add', seasonal=None, damped_trend=True
                    ).fit(optimized=True)

                    fitted   = model_hw.fittedvalues
                    forecast = model_hw.forecast(periodos_fc)
                    ultima_f  = pd.Timestamp(fechas[-1])
                    fechas_fc = [ultima_f + pd.DateOffset(months=i + 1) for i in range(periodos_fc)]

                    r2_hw  = round(r2_score(y_hw, fitted), 4)
                    mae_hw = round(mean_absolute_error(y_hw, fitted), 2)

                    c1, c2 = st.columns(2)
                    c1.metric("R² Holt-Winters", f"{r2_hw}")
                    c2.metric("MAE ajuste",       f"{mae_hw:.2f}")

                    fig5, ax5 = plt.subplots(figsize=(6, 4))
                    ax5.plot(fechas, y_hw,    color=ACCENT2, lw=1.5, alpha=0.8, label='Real (SESNSP)')
                    ax5.plot(fechas, fitted,  color=ACCENT,  lw=1.5, ls='--',   label='Ajuste HW')
                    ax5.plot(fechas_fc, forecast, color=PINK, lw=2, ls='--',
                             marker='o', ms=4, label=f'Pronóstico {periodos_fc}m')
                    ax5.axvspan(fechas_fc[0], fechas_fc[-1], color='#f472b610')
                    apply_dark(ax5, f"Holt-Winters · {estado_detalle}\n(trend=add, damped_trend=True)",
                               "Fecha", "Tasa /100k hab.")
                    ax5.legend(fontsize=7, labelcolor=TEXT_CLR, facecolor=DARK_BG, edgecolor=GRID_CLR)
                    plt.tight_layout()
                    st.pyplot(fig5, use_container_width=True)
                    plt.close()

                    st.markdown("**Valores pronosticados:**")
                    df_fc_show = pd.DataFrame({
                        'Fecha': [f.strftime('%Y-%m') for f in fechas_fc],
                        'Tasa predicha /100k': forecast.round(2)
                    })
                    st.dataframe(df_fc_show.set_index('Fecha'), use_container_width=True)

                except Exception as e:
                    st.error(f"Error Holt-Winters: {e}")

    # ── ENVIPE 2024 ────────────────────────────────────
    with col_env:
        st.markdown('<p class="section-title">ENVIPE 2024 · Percepción de inseguridad</p>', unsafe_allow_html=True)
        if not envipe_ok:
            st.info("Carga los dos archivos CSV de ENVIPE (vic1 y vic2) para ver este análisis.")
        else:
            with st.spinner("Procesando microdatos ENVIPE con factor de expansión FAC_ELE..."):
                try:
                    df_full = cargar_envipe(vic1_file, vic2_file)
                    df_est  = calcular_indicadores_envipe(df_full)
                    prom_nac = (
                        (df_full['es_inseguro'] * df_full['FAC_ELE']).sum()
                        / df_full['FAC_ELE'].sum() * 100
                    )

                    c1, c2 = st.columns(2)
                    c1.metric("Promedio nacional", f"{prom_nac:.1f}%")
                    c2.metric("Estado más inseguro",
                              df_est.iloc[0]['NOM_ENT'],
                              f"{df_est.iloc[0]['Percepcion_Inseguridad']:.1f}%")

                    # Gráfica percepción (≈ Series_tiempo_HW.py §4)
                    fig6, ax6 = plt.subplots(figsize=(6, 9))
                    paleta_env = [RED if x > prom_nac else ACCENT
                                  for x in df_est['Percepcion_Inseguridad']]
                    ax6.barh(df_est['NOM_ENT'], df_est['Percepcion_Inseguridad'],
                             color=paleta_env, height=0.65,
                             edgecolor=DARK_BG, linewidth=0.4)
                    ax6.axvline(prom_nac, color=RED, ls='--', lw=1.5,
                                label=f'Promedio nacional: {prom_nac:.1f}%')
                    for i, (_, row) in enumerate(df_est.iterrows()):
                        ax6.text(row['Percepcion_Inseguridad'] + 0.3, i,
                                 f"{row['Percepcion_Inseguridad']:.1f}%",
                                 va='center', color=TEXT_CLR, fontsize=6.5, fontweight='bold')
                    apply_dark(ax6,
                               "ENVIPE 2024: Percepción de Inseguridad\npor Entidad Federativa (18+ años)",
                               "% ciudadanos que se sienten inseguros", "")
                    ax6.invert_yaxis()
                    ax6.tick_params(axis='y', labelsize=6.5)
                    ax6.legend(fontsize=8, labelcolor=TEXT_CLR, facecolor=DARK_BG, edgecolor=GRID_CLR)
                    plt.tight_layout()
                    st.pyplot(fig6, use_container_width=True)
                    plt.close()

                    # Scatter percepción vs victimización (análisis adicional)
                    st.markdown('<p class="section-title">Percepción vs Victimización real</p>', unsafe_allow_html=True)
                    fig7, ax7 = plt.subplots(figsize=(6, 4))
                    ax7.scatter(df_est['Percepcion_Inseguridad'],
                                df_est['Tasa_Victimizacion_100k'],
                                color=ACCENT, alpha=0.75, s=50,
                                edgecolors=DARK_BG, linewidths=0.5)
                    for _, row in df_est.iterrows():
                        ax7.annotate(row['NOM_ENT'][:8],
                                     (row['Percepcion_Inseguridad'], row['Tasa_Victimizacion_100k']),
                                     fontsize=5.5, color=TEXT_CLR,
                                     xytext=(2, 2), textcoords='offset points')
                    ax7.axvline(prom_nac, color=RED, ls='--', lw=1, alpha=0.5)
                    apply_dark(ax7, "Percepción de inseguridad vs Victimización real",
                               "Percepción de inseguridad (%)", "Victimización /100k hab.")
                    plt.tight_layout()
                    st.pyplot(fig7, use_container_width=True)
                    plt.close()

                except Exception as e:
                    st.error(f"Error al procesar ENVIPE: {e}")

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
