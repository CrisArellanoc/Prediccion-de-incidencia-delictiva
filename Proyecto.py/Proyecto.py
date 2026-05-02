import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configuración de estética y alertas
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# =============================================================================
# 1. CONFIGURACIÓN Y DATOS POBLACIONALES
# =============================================================================
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

RUTA_CSV = '/Users/cristobalarellano/Desktop/Tercer semestre/Análisis de datos/Proyecto/INM_estatal_dic25.csv'

# =============================================================================
# 2. CARGA Y PREPARACIÓN (DATA ENGINEERING)
# =============================================================================
try:
    df = pd.read_csv(RUTA_CSV)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df_filtrado = df[df['tipo_delito'].isin(DELITOS_ALTO_IMPACTO)].copy()
    
    df_agrupado = df_filtrado.groupby(['entidad', 'fecha'])['incidencia_delictiva'].sum().reset_index()
    
    def calcular_tasa(row):
        pob = POBLACION_ESTADOS.get(row['entidad'], 1000000)
        return (row['incidencia_delictiva'] / pob) * 100000

    df_agrupado['tasa_100k'] = df_agrupado.apply(calcular_tasa, axis=1)
    df_agrupado['anio'] = df_agrupado['fecha'].dt.year
    df_agrupado['mes']  = df_agrupado['fecha'].dt.month
    df_agrupado['mes_num'] = (df_agrupado['anio'] - df_agrupado['anio'].min()) * 12 + df_agrupado['mes']

except FileNotFoundError:
    print("Error: No se encontró el CSV en la ruta especificada.")
    exit()

# =============================================================================
# 3. MODELADO Y PRONÓSTICO 2026
# =============================================================================
resultados = []
predicciones_fut = []

ultima_fecha = df_agrupado['fecha'].max()
mes_num_max = df_agrupado['mes_num'].max()
df_futuro = pd.DataFrame([{
    'fecha': ultima_fecha + pd.DateOffset(months=i),
    'mes_num': mes_num_max + i,
    'mes': (ultima_fecha + pd.DateOffset(months=i)).month
} for i in range(1, 7)])

for estado in sorted(df_agrupado['entidad'].unique()):
    data_est = df_agrupado[df_agrupado['entidad'] == estado].sort_values('fecha')
    if len(data_est) < 12: continue

    X = data_est[['mes_num', 'mes']]
    y = data_est['tasa_100k']
    modelo = LinearRegression().fit(X, y)
    
    resultados.append({
        'Estado': estado,
        'Tendencia': modelo.coef_[0],
        'Promedio_Historico': y.mean(),
        'R2': r2_score(y, modelo.predict(X))
    })

    y_fut = np.maximum(modelo.predict(df_futuro[['mes_num', 'mes']]), 0)
    for j, val in enumerate(y_fut):
        predicciones_fut.append({'Estado': estado, 'Fecha': df_futuro.iloc[j]['fecha'], 'Tasa_Predicha': val})

df_res = pd.DataFrame(resultados).sort_values('Promedio_Historico', ascending=False)
df_pred = pd.DataFrame(predicciones_fut)

# =============================================================================
# 4. GRÁFICA DE DISPARIDAD ESTATAL (MÁXIMOS VS MÍNIMOS)
# =============================================================================
print("Generando gráfica de disparidad regional...")
plt.figure(figsize=(12, 12))
bar_plot = sns.barplot(data=df_res, x='Promedio_Historico', y='Estado', palette='mako')

for i in bar_plot.containers:
    bar_plot.bar_label(i, padding=3, fmt='%.1f', fontsize=10, fontweight='bold')

plt.title('Disparidad de Incidencia Delictiva (2015-2025)\nTasa Mensual Promedio por cada 100,000 habitantes', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Delitos por cada 100k hab. (Promedio Mensual)', fontsize=12)
plt.ylabel('Entidad Federativa', fontsize=12)
plt.tight_layout()
plt.savefig('3_4_disparidad_estatal.png', dpi=300)

# =============================================================================
# 5. GRÁFICA DE PREDICCIÓN 2026 (ESTADOS CLAVE)
# =============================================================================
print("Generando gráfica predictiva 2026...")
plt.figure(figsize=(15, 8))
# Seleccionamos el top, el bottom y Guanajuato para el contraste
estados_plot = [df_res['Estado'].iloc[0], 'Guanajuato', df_res['Estado'].iloc[-1]]
colores = ['#e74c3c', '#f39c12', '#27ae60']

for i, estado in enumerate(estados_plot):
    df_h = df_agrupado[df_agrupado['entidad'] == estado].sort_values('fecha')
    plt.plot(df_h['fecha'], df_h['tasa_100k'], color=colores[i], linewidth=1.5, alpha=0.3)
    
    df_p = df_pred[df_pred['Estado'] == estado].sort_values('Fecha')
    x_total = pd.concat([pd.Series(df_h['fecha'].iloc[-1]), df_p['Fecha']])
    y_total = pd.concat([pd.Series(df_h['tasa_100k'].iloc[-1]), df_p['Tasa_Predicha']])
    
    plt.plot(x_total, y_total, color=colores[i], linewidth=3, linestyle='--', label=f'Predicción {estado} 2026')

plt.axvspan(df_pred['Fecha'].min(), df_pred['Fecha'].max(), color='gray', alpha=0.1, label='Horizonte de Predicción')
plt.title('Pronóstico de Tasas Delictivas 1S-2026: Análisis de Contrastes', fontsize=16, fontweight='bold')
plt.ylabel('Tasa de Delitos / 100k hab.')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('3_5_prediccion_2026.png', dpi=300)

# =============================================================================
# 6. ANÁLISIS DE RESULTADOS PARA EL DOCUMENTO
# =============================================================================
max_est = df_res.iloc[0]
min_est = df_res.iloc[-1]
gua_est = df_res[df_res['Estado'] == 'Guanajuato'].iloc[0]

print("\n" + "="*60)
print("PÁRRAFO DE ANÁLISIS SUGERIDO PARA EL CAPÍTULO 3:")
print("="*60)
print(f"Al analizar la disparidad regional mediante tasas normalizadas por cada 100,000 habitantes,")
print(f"se observa una brecha crítica entre el máximo nacional ({max_est['Estado']} con {max_est['Promedio_Historico']:.1f})")
print(f"y el mínimo ({min_est['Estado']} con {min_est['Promedio_Historico']:.1f}).")
print(f"Guanajuato se posiciona con una tasa promedio de {gua_est['Promedio_Historico']:.1f}, lo que representa")
print(f"una intensidad delictiva {(gua_est['Promedio_Historico']/min_est['Promedio_Historico']):.1f} veces mayor que el estado más seguro.")
print("="*60)

print("\n✓ Proceso Completado exitosamente.")