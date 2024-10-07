import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

eo_ev_list = [
    {'Host Element': 'Fe', 'Dopant Element': 'V', 'EO (eV)': -7.11, 'EC (eV)': -6.82},
    {'Host Element': 'Fe', 'Dopant Element': 'Fe', 'EO (eV)': -5.86, 'EC (eV)': -6.51},
    {'Host Element': 'Fe', 'Dopant Element': 'Co', 'EO (eV)': -5.79, 'EC (eV)': -6.78},
    {'Host Element': 'Fe', 'Dopant Element': 'Ni', 'EO (eV)': -5.28, 'EC (eV)': -6.03},
    {'Host Element': 'Fe', 'Dopant Element': 'Cu', 'EO (eV)': -4.98, 'EC (eV)': -5.22},
    {'Host Element': 'Fe', 'Dopant Element': 'Y', 'EO (eV)': -6.41, 'EC (eV)': -6.01},
    {'Host Element': 'Fe', 'Dopant Element': 'Zr', 'EO (eV)': -6.79, 'EC (eV)': -6.25},
    {'Host Element': 'Fe', 'Dopant Element': 'Nb', 'EO (eV)': -6.91, 'EC (eV)': -6.64},
    {'Host Element': 'Fe', 'Dopant Element': 'Mo', 'EO (eV)': -6.72, 'EC (eV)': -7.13},
    {'Host Element': 'Fe', 'Dopant Element': 'Ru', 'EO (eV)': -5.54, 'EC (eV)': -7.03},
    {'Host Element': 'Fe', 'Dopant Element': 'Rh', 'EO (eV)': -5.01, 'EC (eV)': -6.28},
    {'Host Element': 'Fe', 'Dopant Element': 'Pd', 'EO (eV)': -4.66, 'EC (eV)': -5.34},
    {'Host Element': 'Fe', 'Dopant Element': 'Ag', 'EO (eV)': -4.87, 'EC (eV)': -5.24},
    {'Host Element': 'Fe', 'Dopant Element': 'W', 'EO (eV)': -7.52, 'EC (eV)': -7.21},
    {'Host Element': 'Fe', 'Dopant Element': 'Re', 'EO (eV)': -6.95, 'EC (eV)': -7.48}
]

eo_ev_data = pd.DataFrame(eo_ev_list)

encoder_host = OneHotEncoder(sparse_output=False)
encoder_dopant = OneHotEncoder(sparse_output=False)

encoded_host = encoder_host.fit_transform(eo_ev_data[['Host Element']])
encoded_dopant = encoder_dopant.fit_transform(eo_ev_data[['Dopant Element']])

encoded_host_df = pd.DataFrame(encoded_host, columns=encoder_host.get_feature_names_out(['Host Element']))
encoded_dopant_df = pd.DataFrame(encoded_dopant, columns=encoder_dopant.get_feature_names_out(['Dopant Element']))

eo_ev_data_encoded = pd.concat([encoded_host_df, encoded_dopant_df, eo_ev_data[['EO (eV)', 'EC (eV)']]], axis=1)

X = eo_ev_data_encoded.drop(columns=['EO (eV)', 'EC (eV)'])
y_EO = eo_ev_data_encoded['EO (eV)']
y_EC = eo_ev_data_encoded['EC (eV)']

def calculate_mse(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    return mse_train, mse_test

X_train, X_test, y_EO_train, y_EO_test = train_test_split(X, y_EO, test_size=0.2, random_state=42)
_, _, y_EC_train, y_EC_test = train_test_split(X, y_EC, test_size=0.2, random_state=42)

param_grid_knn = {
    'n_neighbors': [6], 
    'weights': ['uniform', 'distance']  
}

param_grid_gbr = {
    'n_estimators': [100], 
    'learning_rate': [0.1,],  
    'max_depth': [3],  
    'random_state': [42]
}

knn_model = KNeighborsRegressor()
gbr_model = GradientBoostingRegressor()

grid_search_knn_EO = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='neg_mean_squared_error')
grid_search_gbr_EO = GridSearchCV(gbr_model, param_grid_gbr, cv=5, scoring='neg_mean_squared_error')

grid_search_knn_EO.fit(X_train, y_EO_train)
grid_search_gbr_EO.fit(X_train, y_EO_train)

grid_search_knn_EC = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='neg_mean_squared_error')
grid_search_gbr_EC = GridSearchCV(gbr_model, param_grid_gbr, cv=5, scoring='neg_mean_squared_error')

grid_search_knn_EC.fit(X_train, y_EC_train)
grid_search_gbr_EC.fit(X_train, y_EC_train)

mse_EO_knn_train, mse_EO_knn_test = calculate_mse(grid_search_knn_EO.best_estimator_, X_train, X_test, y_EO_train, y_EO_test)
mse_EO_gbr_train, mse_EO_gbr_test = calculate_mse(grid_search_gbr_EO.best_estimator_, X_train, X_test, y_EO_train, y_EO_test)

mse_EC_knn_train, mse_EC_knn_test = calculate_mse(grid_search_knn_EC.best_estimator_, X_train, X_test, y_EC_train, y_EC_test)
mse_EC_gbr_train, mse_EC_gbr_test = calculate_mse(grid_search_gbr_EC.best_estimator_, X_train, X_test, y_EC_train, y_EC_test)

print(f'MSE EO (KNN) Train: {mse_EO_knn_train:.2f}')
print(f'MSE EO (KNN) Test: {mse_EO_knn_test:.2f}')
print(f'MSE EO (GBR) Train: {mse_EO_gbr_train:.2f}')
print(f'MSE EO (GBR) Test: {mse_EO_gbr_test:.2f}')
print(f'MSE EC (KNN) Train: {mse_EC_knn_train:.2f}')
print(f'MSE EC (KNN) Test: {mse_EC_knn_test:.2f}')
print(f'MSE EC (GBR) Train: {mse_EC_gbr_train:.2f}')
print(f'MSE EC (GBR) Test: {mse_EC_gbr_test:.2f}')

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

labels = ['KNN Train', 'KNN Test', 'GBR Train', 'GBR Test']
eo_values = [mse_EO_knn_train, mse_EO_knn_test, mse_EO_gbr_train, mse_EO_gbr_test]
ec_values = [mse_EC_knn_train, mse_EC_knn_test, mse_EC_gbr_train, mse_EC_gbr_test]

bars1 = ax[0].bar(labels, eo_values, color=['blue', 'lightblue', 'green', 'lightgreen'])
ax[0].set_xlabel('Model')
ax[0].set_ylabel('Root Mean Squared Error')
ax[0].set_title('Eo (eV) Model Comparison')
ax[0].set_ylim(0, max(max(eo_values), max(ec_values)) + 0.1) 
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

for bar in bars1:
    yval = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

bars2 = ax[1].bar(labels, ec_values, color=['blue', 'lightblue', 'green', 'lightgreen'])
ax[1].set_xlabel('Model')
ax[1].set_ylabel('Root Mean Squared Error')
ax[1].set_title('Ec (eV) Model Comparison')
ax[1].set_ylim(0, max(max(eo_values), max(ec_values)) + 0.1)  
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

for bar in bars2:
    yval = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()

data = """
Dopant Element,Atomic Number,Atomic Mass (amu),Group,Electronegativity,Melting Point,Boiling Point,Heat of Fusion,Ionisation Energy,Number of d electrons,d-band filling,Surface energy,Wigner-Seitz Radius,d-band centre
Sc,21,44.956,3,1.36,1814,3109,14.1,633.09,1,0.2,1.275,3.43,1.85
Ti,22,47.867,4,1.54,1943,3560,15.5,658.81,2,0.3,2.046,3.05,1.50
V,23,50.942,5,1.63,2183,3680,20.9,650.91,3,0.4,2.586,2.82,1.06
Fe,26,55.845,8,1.83,1811,3134,11.7,762.47,6,0.7,2.446,2.66,-0.92
Co,27,58.933,9,1.88,1768,3200,16.2,760.40,7,0.8,2.536,2.62,-1.17
Ni,28,58.693,10,1.91,1728,3186,17.2,737.13,8,0.9,2.415,2.60,-1.29
Cu,29,63.546,11,1.90,1358,2833,13.3,745.48,10,1.0,1.808,2.67,-2.67
Y,39,88.906,3,1.22,1795,3618,11.4,599.88,1,0.2,1.125,3.76,2.21
Zr,40,91.224,4,1.33,2127,4679,16.9,640.07,2,0.3,1.955,3.35,1.95
Nb,41,92.906,5,1.60,2750,5014,26.4,652.13,4,0.4,2.678,3.07,1.41
Mo,42,95.950,6,2.16,2895,4912,32.0,684.32,5,0.5,2.954,2.99,-0.60
Ru,44,101.070,8,2.20,2606,4420,24.0,710.18,7,0.7,3.047,2.79,-1.41
Rh,45,102.906,9,2.28,2236,3968,21.5,719.68,8,0.8,2.680,2.81,-1.73
Pd,46,106.420,10,2.20,1828,3236,17.6,804.39,10,0.9,2.027,2.87,-1.83
Ag,47,107.868,11,1.93,1235,2435,11.3,731.00,10,1.0,1.248,3.01,-4.30
W,74,184.340,6,1.70,3687,5828,35.4,758.76,4,0.5,3.470,2.95,0.77
Re,75,186.207,7,1.90,3458,5863,33.2,755.82,5,0.6,3.613,2.87,-0.51
Ir,77,192.217,9,2.20,2719,4701,26.1,865.19,7,0.8,3.024,2.84,-2.11
Pt,78,195.084,10,2.20,2041,4098,19.6,864.39,9,0.9,2.482,2.90,-2.25
Au,79,196.967,11,2.40,1337,3109,13.2,890.13,10,1.0,1.503,3.00,-3.56
"""

from io import StringIO
df = pd.read_csv(StringIO(data))


X = df.drop(columns=['Dopant Element'])
y = df['Electronegativity']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsRegressor(n_neighbors=6, weights='uniform', algorithm='auto')

knn.fit(X_train, y_train)

y_train_pred_knn = knn.predict(X_train)
y_test_pred_knn = knn.predict(X_test)

mse_EO_knn_train = mean_squared_error(y_train, y_train_pred_knn)
mse_EO_knn_test = mean_squared_error(y_test, y_test_pred_knn)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

gbr.fit(X_train, y_train)

y_train_pred_gbr = gbr.predict(X_train)
y_test_pred_gbr = gbr.predict(X_test)


mse_EO_gbr_train = mean_squared_error(y_train, y_train_pred_gbr)
mse_EO_gbr_test = mean_squared_error(y_test, y_test_pred_gbr)

df['EO_KNN'] = pd.Series(y_test_pred_knn, index=X_test.index)
df['EC_GBR'] = pd.Series(y_test_pred_gbr, index=X_test.index)

corr_matrix = df.drop(columns=['Dopant Element']).corr()

plt.figure(figsize=(9, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Dopant Element Properties with EO and EC')
plt.show()