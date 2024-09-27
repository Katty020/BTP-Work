import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Embedded data from EoEv.csv
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

# Convert the list of dictionaries to DataFrame
eo_ev_data = pd.DataFrame(eo_ev_list)

# One-hot encode the categorical columns
encoder_host = OneHotEncoder(sparse_output=False)
encoder_dopant = OneHotEncoder(sparse_output=False)

encoded_host = encoder_host.fit_transform(eo_ev_data[['Host Element']])
encoded_dopant = encoder_dopant.fit_transform(eo_ev_data[['Dopant Element']])

# Create DataFrames from the encoded data
encoded_host_df = pd.DataFrame(encoded_host, columns=encoder_host.get_feature_names_out(['Host Element']))
encoded_dopant_df = pd.DataFrame(encoded_dopant, columns=encoder_dopant.get_feature_names_out(['Dopant Element']))

# Concatenate the encoded columns with the original DataFrame
eo_ev_data_encoded = pd.concat([encoded_host_df, encoded_dopant_df, eo_ev_data[['EO (eV)', 'EC (eV)']]], axis=1)

# Split the data into features and target variables
X = eo_ev_data_encoded.drop(columns=['EO (eV)', 'EC (eV)'])
y_EO = eo_ev_data_encoded['EO (eV)']
y_EC = eo_ev_data_encoded['EC (eV)']

# Function to calculate MSE for a given model and target variable
def calculate_mse(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Split the data into training and testing sets
X_train, X_test, y_EO_train, y_EO_test = train_test_split(X, y_EO, test_size=0.2, random_state=42)
_, _, y_EC_train, y_EC_test = train_test_split(X, y_EC, test_size=0.2, random_state=42)

# Initialize the models
knn_model = KNeighborsRegressor(n_neighbors=5, weights='uniform')
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)

# Calculate MSE for EO
mse_EO_knn = calculate_mse(knn_model, X_train, X_test, y_EO_train, y_EO_test)
mse_EO_gbr = calculate_mse(gbr_model, X_train, X_test, y_EO_train, y_EO_test)

# Calculate MSE for EC
mse_EC_knn = calculate_mse(knn_model, X_train, X_test, y_EC_train, y_EC_test)
mse_EC_gbr = calculate_mse(gbr_model, X_train, X_test, y_EC_train, y_EC_test)

print(f'MSE EO (KNN): {mse_EO_knn:.2f}')
print(f'MSE EO (GBR): {mse_EO_gbr:.2f}')
print(f'MSE EC (KNN): {mse_EC_knn:.2f}')
print(f'MSE EC (GBR): {mse_EC_gbr:.2f}')

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

labels = ['KNN', 'GBR']
eo_values = [mse_EO_knn, mse_EO_gbr]
ec_values = [mse_EC_knn, mse_EC_gbr]

# Plot EO (eV) Model Comparison
bars1 = ax[0].bar(labels, eo_values, color=['blue', 'green'])
ax[0].set_xlabel('Model')
ax[0].set_ylabel('Mean Squared Error')
ax[0].set_title('EO (eV) Model Comparison')
ax[0].set_ylim(0, max(max(eo_values), max(ec_values)) + 0.1)  # Set y-axis limit
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

for bar in bars1:
    yval = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

# Plot EC (eV) Model Comparison
bars2 = ax[1].bar(labels, ec_values, color=['blue', 'green'])
ax[1].set_xlabel('Model')
ax[1].set_ylabel('Mean Squared Error')
ax[1].set_title('EC (eV) Model Comparison')
ax[1].set_ylim(0, max(max(eo_values), max(ec_values)) + 0.1)  # Set y-axis limit
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Add annotations
for bar in bars2:
    yval = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()