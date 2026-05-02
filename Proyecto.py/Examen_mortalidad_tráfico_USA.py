# =============================================================================
# SEGUNDO EXAMEN PARCIAL (PRÁCTICO)
# Alumno: Cristóbal Arellano Carranza
# Matrícula: 194700-2
# =============================================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =============================================================================
# 1. LOS ARCHIVOS DE DATOS SIN PROCESAR Y SU FORMATO
# =============================================================================
import os
current_dir = os.getcwd()
print(current_dir)
file_list = os.listdir(current_dir)
print(file_list)

# =============================================================================
# 2. LEER Y OBTENER UNA DESCRIPCIÓN GENERAL DE LOS DATOS
# =============================================================================
car_acc = pd.read_csv('/Users/cristobalarellano/Desktop/road-accidents.csv', comment='#', sep='|')

rows_and_cols = car_acc.shape
print(f"There are {rows_and_cols[0]} rows and {rows_and_cols[1]} columns.")

car_acc_information = car_acc.info()
print(car_acc_information)

print(car_acc.tail())

# =============================================================================
# 3. CREAR UN RESUMEN TEXTUAL Y GRÁFICO DE LOS DATOS
# =============================================================================

sum_stat_car = car_acc.describe()
print(sum_stat_car)

sns.pairplot(car_acc)
plt.show()

# =============================================================================
# 4. CUANTIFICAR LA ASOCIACIÓN DE CARACTERÍSTICAS Y ACCIDENTES
# =============================================================================
corr_columns = car_acc.corr(numeric_only=True)
print(corr_columns)

# =============================================================================
# 5. AJUSTAR UNA REGRESIÓN LINEAL MÚLTIPLE
# =============================================================================
features = car_acc[['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time']]
target   = car_acc['drvr_fatl_col_bmiles']

reg = linear_model.LinearRegression()
reg.fit(features, target)

fit_coef = reg.coef_
print(fit_coef)

# =============================================================================
# 6. REALIZAR PCA EN DATOS ESTANDARIZADOS
# =============================================================================
scaler          = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA()
pca.fit(features_scaled)

plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal component #')
plt.ylabel('Proportion of variance explained')
plt.show()

# Varianza acumulada de los dos primeros componentes
two_first_comp_var_exp = pca.explained_variance_ratio_[:2].cumsum()[-1]
print(f"Varianza explicada por los 2 primeros componentes: {two_first_comp_var_exp:.2%}")

# =============================================================================
# 7. VISUALIZAR LOS DOS PRIMEROS COMPONENTES PRINCIPALES
# =============================================================================
pca     = PCA(n_components=2)
p_comps = pca.fit_transform(features_scaled)

p_comp1 = p_comps[:, 0]
p_comp2 = p_comps[:, 1]

plt.scatter(p_comp1, p_comp2)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# =============================================================================
# 8. ENCONTRAR GRUPOS DE ESTADOS SIMILARES EN LOS DATOS
# =============================================================================
ks       = range(1, 10)
inertias = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=8)
    km.fit(features_scaled)
    inertias.append(km.inertia_)

plt.plot(ks, inertias, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.show()

# =============================================================================
# 9. K-MEANS PARA VISUALIZAR GRUPOS EN EL DIAGRAMA DE DISPERSIÓN PCA
# =============================================================================
km = KMeans(n_clusters=3, random_state=8)
km.fit(features_scaled)

plt.scatter(p_comp1, p_comp2, c=km.labels_)
plt.title('Clusters de estados en espacio PCA')
plt.show()

# =============================================================================
# 10. VISUALIZAR LAS DIFERENCIAS DE CARACTERÍSTICAS ENTRE LOS GRUPOS
# =============================================================================
car_acc['cluster'] = km.labels_

melt_car = pd.melt(
    car_acc,
    id_vars='cluster',
    value_vars=['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time'],
    var_name='measurement',
    value_name='percent'
)

sns.violinplot(x='percent', y='measurement', hue='cluster', data=melt_car)
plt.show()

# =============================================================================
# 11. CALCULAR EL NÚMERO DE ACCIDENTES DENTRO DE CADA GRUPO
# =============================================================================
miles_driven  = pd.read_csv('/Users/cristobalarellano/Desktop/miles-driven.csv', sep='|')
car_acc_miles = car_acc.merge(miles_driven, on='state')

# (Accidentes por mil millones de millas * Millas anuales en millones) / 1000
car_acc_miles['num_drvr_fatl_col'] = (
    car_acc_miles['drvr_fatl_col_bmiles'] * car_acc_miles['million_miles_annually']
) / 1000

sns.barplot(
    x='cluster',
    y='num_drvr_fatl_col',
    data=car_acc_miles,
    estimator=np.sum,
    errorbar=None
)
plt.title('Total de accidentes fatales por cluster')
plt.show()

count_mean_sum = car_acc_miles.groupby('cluster')['num_drvr_fatl_col'].agg(['count', 'mean', 'sum'])
print(count_mean_sum)

# =============================================================================
# 12. TOMAR UNA DECISIÓN
# =============================================================================

cluster_num = 2
print(f"Propuesta de política pública: Enfocar recursos en el clúster {cluster_num}")