import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os


# Configuración de estética y alertas
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# =============================================================================
# 1. RUTAS Y CARGA DE DATOS
# =============================================================================

print("=" * 60)
print("INICIANDO PROCESAMIENTO ENVIPE 2024 - PROYECTO IBERO")
print("=" * 60)

# Ajusta estas rutas a la ubicación real en tu Mac
path_base = r'/Users/cristobalarellano/Desktop/'
FILE_VIC1 = os.path.join(path_base, 'conjunto_de_datos_tper_vic1_envipe2024.csv')
FILE_VIC2 = os.path.join(path_base, 'conjunto_de_datos_tper_vic2_envipe2024.csv')

try:
    print("Cargando microdatos de victimización...")
    df_vic1 = pd.read_csv(FILE_VIC1, low_memory=False)
    df_vic2 = pd.read_csv(FILE_VIC2, low_memory=False)
    
    # Unión por llaves únicas de vivienda, hogar y persona
    llaves = ['ID_VIV', 'ID_HOG', 'ID_PER']
    df_full = pd.merge(df_vic1, df_vic2, on=llaves, how='inner', suffixes=('', '_drop'))
    
    # Limpieza de duplicados tras el merge
    df_full = df_full[[c for c in df_full.columns if not c.endswith('_drop')]]
    print(f"✓ Éxito: {len(df_full):,} registros vinculados.")

except FileNotFoundError as e:
    print(f"X Error: No se encontraron los archivos. Detalle: {e}")
    exit()

# =============================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# =============================================================================

# Convertir variables críticas a numérico
df_full['FAC_ELE'] = pd.to_numeric(df_full['FAC_ELE'], errors='coerce')
df_full['AP4_1']   = pd.to_numeric(df_full['AP4_1'], errors='coerce')

# A. Definir Percepción (1=Seguro, 2=Inseguro en ENVIPE)
df_full['es_inseguro'] = np.where(df_full['AP4_1'] == 2, 1, 0)

# B. Definir Victimización (Revisión de delitos AP4_2_01 al 13)
cols_delitos = [c for c in df_full.columns if c.startswith('AP4_2_') and '99' not in c]
for col in cols_delitos:
    df_full[col] = pd.to_numeric(df_full[col], errors='coerce')

df_full['es_victima'] = df_full[cols_delitos].apply(
    lambda x: 1 if 1 in x.values else 0, axis=1
)

# =============================================================================
# 3. PROCESAMIENTO ESTADÍSTICO PONDERADO
# =============================================================================

def calcular_indicadores(group):
    total_pob = group['FAC_ELE'].sum()
    if total_pob == 0: return None
    
    # Aplicación del factor de expansión (Población representada)
    inseguridad_relativa = (group['es_inseguro'] * group['FAC_ELE']).sum() / total_pob * 100
    tasa_victimizacion = (group['es_victima'] * group['FAC_ELE']).sum() / total_pob * 100_000
    
    return pd.Series({
        'Percepcion_Inseguridad': round(inseguridad_relativa, 2),
        'Tasa_Victimizacion_100k': round(tasa_victimizacion, 0),
        'Poblacion_Estatal': int(total_pob)
    })

print("Calculando métricas ponderadas por entidad federativa...")
df_estados = df_full.groupby('NOM_ENT').apply(calcular_indicadores).reset_index()
df_estados = df_estados.sort_values('Percepcion_Inseguridad', ascending=False)

# =============================================================================
# 4. VISUALIZACIÓN DE ALTA RESOLUCIÓN
# =============================================================================

# Cálculo de promedio nacional ponderado para la línea de referencia
promedio_nac = (df_full['es_inseguro'] * df_full['FAC_ELE']).sum() / df_full['FAC_ELE'].sum() * 100

plt.figure(figsize=(14, 12))

# Paleta condicional: Rojo para estados arriba del promedio, azul para los de abajo
paleta = ['#d9534f' if x > promedio_nac else '#5bc0de' for x in df_estados['Percepcion_Inseguridad']]

bar = sns.barplot(
    data=df_estados, 
    x='Percepcion_Inseguridad', 
    y='NOM_ENT', 
    palette=paleta,
    edgecolor='0.2'
)

# Línea de Promedio Nacional
plt.axvline(promedio_nac, color='#c9302c', linestyle='--', linewidth=2, 
            label=f'Promedio Nacional: {promedio_nac:.1f}%')

# Etiquetas de datos
for i in bar.containers:
    bar.bar_label(i, padding=5, fmt='%.1f%%', fontsize=10, fontweight='bold')

# Formato de gráfica
plt.title('ENVIPE 2024: Percepción de Inseguridad por Entidad Federativa\n(Población de 18 años y más)', 
          fontsize=18, fontweight='bold', pad=25)
plt.xlabel('Porcentaje de ciudadanos que se sienten inseguros en su estado', fontsize=12)
plt.ylabel('Estado', fontsize=12)
plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()

# Exportación de resultados
plt.savefig('grafica_percepcion_2024.png', dpi=300)
df_estados.to_csv('dataset_maestro_percepcion_2024.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("DATOS LISTOS PARA EL CAPÍTULO 3")
print(f"1. Imagen: 'grafica_percepcion_2024.png'")
print(f"2. CSV Consolidado: 'dataset_maestro_percepcion_2024.csv'")
print("=" * 60)

# Top 5 para revisión rápida
print("\nTop 5 Estados con Mayor Percepción de Inseguridad:")
print(df_estados[['NOM_ENT', 'Percepcion_Inseguridad']].head())