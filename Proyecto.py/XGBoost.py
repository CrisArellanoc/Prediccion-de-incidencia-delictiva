import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Configuración de estética
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# =============================================================================
# 1. CONSTANTES Y DICCIONARIOS (POBLACIÓN 2020-2025 aprox)
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
# 2. CARGA Y PREPROCESAMIENTO
# =============================================================================
try:
    df = pd.read_csv(RUTA_CSV)
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Filtrado por delitos de alto impacto
    df_filtrado = df[df['tipo_delito'].isin(DELITOS_ALTO_IMPACTO)].copy()
    
    # Agrupación por Estado y Fecha
    df_agrupado = df_filtrado.groupby(['entidad', 'fecha'])['incidencia_delictiva'].sum().reset_index()
    
    # Normalización por Tasa 100k habitantes
    def calcular_tasa(row):
        pob = POBLACION_ESTADOS.get(row['entidad'], 1000000)
        return (row['incidencia_delictiva'] / pob) * 100000

    df_agrupado['tasa_100k'] = df_agrupado.apply(calcular_tasa, axis=1)
    
    # Ingeniería de Variables (Features) para el modelo
    df_agrupado['anio'] = df_agrupado['fecha'].dt.year
    df_agrupado['mes']  = df_agrupado['fecha'].dt.month
    # Creamos un índice numérico de meses para capturar la tendencia temporal
    df_agrupado['mes_num'] = (df_agrupado['anio'] - df_agrupado['anio'].min()) * 12 + df_agrupado['mes']

except Exception as e:
    print(f"Error al procesar el archivo: {e}")
    exit()

# =============================================================================
# 3. MODELADO CON RANDOM FOREST Y PREDICCIÓN 2026
# =============================================================================
resultados = []
predicciones_fut = []

# Preparar horizonte de predicción (Primeros 6 meses de 2026)
ultima_fecha = df_agrupado['fecha'].max()
mes_num_max = df_agrupado['mes_num'].max()
df_futuro = pd.DataFrame([{
    'fecha': ultima_fecha + pd.DateOffset(months=i),
    'mes_num': mes_num_max + i,
    'mes': (ultima_fecha + pd.DateOffset(months=i)).month
} for i in range(1, 7)])

# Iterar por cada estado para un modelo personalizado
for estado in sorted(df_agrupado['entidad'].unique()):
    data_est = df_agrupado[df_agrupado['entidad'] == estado].sort_values('fecha')
    if len(data_est) < 24: continue  # Necesitamos datos suficientes para los árboles

    X = data_est[['mes_num', 'mes']]
    y = data_est['tasa_100k']
    
    # Definición del Random Forest
    # n_estimators: 100 árboles para estabilidad
    # max_depth: limitado a 10 para evitar que el modelo solo "memorice" (overfitting)
    modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    modelo_rf.fit(X, y)
    
    # Evaluación
    y_pred = modelo_rf.predict(X)
    r2 = r2_score(y, y_pred)
    importancias = modelo_rf.feature_importances_

    resultados.append({
        'Estado': estado,
        'R2_Score': r2,
        'Importancia_Tendencia': importancias[0],
        'Importancia_Estacionalidad': importancias[1],
        'Promedio_Historico': y.mean()
    })

    # Predicción a futuro
    y_fut = modelo_rf.predict(df_futuro[['mes_num', 'mes']])
    for j, val in enumerate(y_fut):
        predicciones_fut.append({
            'Estado': estado, 
            'Fecha': df_futuro.iloc[j]['fecha'], 
            'Tasa_Predicha': val
        })

df_res = pd.DataFrame(resultados).sort_values('R2_Score', ascending=False)
df_pred = pd.DataFrame(predicciones_fut)

# =============================================================================
# 4. VISUALIZACIÓN DE RESULTADOS
# =============================================================================
# Gráfica 1: Importancia de Variables (¿Qué explica más el delito?)
plt.figure(figsize=(10, 5))
avg_imp = df_res[['Importancia_Tendencia', 'Importancia_Estacionalidad']].mean()
sns.barplot(x=avg_imp.index, y=avg_imp.values, palette='magma')
plt.title('Random Forest: ¿Qué influye más en la Incidencia Delictiva?\n(Tendencia Temporal vs Estacionalidad Mensual)', fontsize=14)
plt.ylabel('Importancia Relativa')
plt.tight_layout()
plt.savefig('rf_importancia_variables.png')

# Gráfica 2: Comparativa de Predicción para Guanajuato (Estado de interés)
plt.figure(figsize=(14, 7))
estado_target = 'Guanajuato'
df_h = df_agrupado[df_agrupado['entidad'] == estado_target]
df_p = df_pred[df_pred['Estado'] == estado_target]

plt.plot(df_h['fecha'], df_h['tasa_100k'], label='Datos Históricos (SESNSP)', color='#2c3e50', alpha=0.6)
plt.plot(df_p['Fecha'], df_p['Tasa_Predicha'], label='Predicción Random Forest 2026', color='#e74c3c', lw=3, ls='--')

plt.title(f'Análisis Predictivo Random Forest: {estado_target}', fontsize=16, fontweight='bold')
plt.ylabel('Tasa de Delitos / 100k hab.')
plt.legend()
plt.tight_layout()
plt.savefig('rf_prediccion_guanajuato.png')

# =============================================================================
# 5. RESUMEN EJECUTIVO
# =============================================================================
print("\n" + "="*60)
print("REPORTE DE MODELO RANDOM FOREST")
print("="*60)
print(f"Precisión promedio del modelo (R2 Score): {df_res['R2_Score'].mean():.4f}")
print(f"Estado con mejor ajuste: {df_res.iloc[0]['Estado']} (R2: {df_res.iloc[0]['R2_Score']:.2f})")
print(f"Estado con mayor tendencia al alza detectada: {df_res.sort_values('Importancia_Tendencia', ascending=False).iloc[0]['Estado']}")
print("-" * 60)
print("Interpretación para tu hipótesis:")
print("Si la 'Importancia_Estacionalidad' es alta, el delito sigue patrones de calendario.")
print("Si la 'Importancia_Tendencia' es alta, hay un cambio estructural (posible Inhibición).")
print("="*60)