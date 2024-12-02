import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import OneHotEncoder

# Data Setup
eo_ev_list = [
    {'Host Element': 'Fe', 'Dopant Element': 'Sc', 'EO (eV)': -6.63, 'EC (eV)': -5.99},
    {'Host Element': 'Fe', 'Dopant Element': 'Ti', 'EO (eV)': -6.99, 'EC (eV)': -6.21},
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
    {'Host Element': 'Fe', 'Dopant Element': 'Re', 'EO (eV)': -6.95, 'EC (eV)': -7.48},
    {'Host Element': 'Fe', 'Dopant Element': 'Ir', 'EO (eV)': -5.08, 'EC (eV)': -6.66},
    {'Host Element': 'Fe', 'Dopant Element': 'Pt', 'EO (eV)': -4.59, 'EC (eV)': -5.68},
    {'Host Element': 'Fe', 'Dopant Element': 'Au', 'EO (eV)': -4.37, 'EC (eV)': -5.09}
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

X_train, X_test, y_EO_train, y_EO_test = train_test_split(X, y_EO, test_size=0.2, random_state=42)
_, _, y_EC_train, y_EC_test = train_test_split(X, y_EC, test_size=0.2, random_state=42)

# Define hyperparameter grids
knn_params = {'n_neighbors': [3, 5, 8, 9], 'weights': ['uniform', 'distance']}
gbr_params = {
    'n_estimators': [100, 115, 120, 130, 150],
    'learning_rate': [0.1, 0.2, 0.5],
    'max_depth': [3, 4]
}

# Results DataFrame
results = []

# KNN Performance
for n_neighbors in knn_params['n_neighbors']:
    for weight in knn_params['weights']:
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weight)
        
        # EO (eV)
        start_time = time.time()
        knn.fit(X_train, y_EO_train)
        end_time = time.time()
        
        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)
        train_rmse = mean_squared_error(y_EO_train, y_pred_train, squared=False)
        test_rmse = mean_squared_error(y_EO_test, y_pred_test, squared=False)
        
        results.append({'Model': 'KNN', 'Target': 'EO (eV)', 'Param': f'n_neighbors={n_neighbors}, weight={weight}', 
                        'Train RMSE': train_rmse, 'Test RMSE': test_rmse, 
                        'Time Taken': end_time - start_time})
        
        # EC (eV)
        start_time = time.time()
        knn.fit(X_train, y_EC_train)
        end_time = time.time()
        
        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)
        train_rmse = mean_squared_error(y_EC_train, y_pred_train, squared=False)
        test_rmse = mean_squared_error(y_EC_test, y_pred_test, squared=False)
        
        results.append({'Model': 'KNN', 'Target': 'EC (eV)', 'Param': f'n_neighbors={n_neighbors}, weight={weight}', 
                        'Train RMSE': train_rmse, 'Test RMSE': test_rmse, 
                        'Time Taken': end_time - start_time})

# GBR Performance
for n_estimators in gbr_params['n_estimators']:
    for learning_rate in gbr_params['learning_rate']:
        for max_depth in gbr_params['max_depth']:
            gbr = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
            
            # EO (eV)
            start_time = time.time()
            gbr.fit(X_train, y_EO_train)
            end_time = time.time()
            
            y_pred_train = gbr.predict(X_train)
            y_pred_test = gbr.predict(X_test)
            train_rmse = mean_squared_error(y_EO_train, y_pred_train, squared=False)
            test_rmse = mean_squared_error(y_EO_test, y_pred_test, squared=False)
            
            results.append({'Model': 'GBR', 'Target': 'EO (eV)', 'Param': f'n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}', 
                            'Train RMSE': train_rmse, 'Test RMSE': test_rmse, 
                            'Time Taken': end_time - start_time})
            
            # EC (eV)
            start_time = time.time()
            gbr.fit(X_train, y_EC_train)
            end_time = time.time()
            
            y_pred_train = gbr.predict(X_train)
            y_pred_test = gbr.predict(X_test)
            train_rmse = mean_squared_error(y_EC_train, y_pred_train, squared=False)
            test_rmse = mean_squared_error(y_EC_test, y_pred_test, squared=False)
            
            results.append({'Model': 'GBR', 'Target': 'EC (eV)', 'Param': f'n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}', 
                            'Train RMSE': train_rmse, 'Test RMSE': test_rmse, 
                                                        'Time Taken': end_time - start_time})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate overall percentage error
results_df['Percentage Error'] = ((results_df['Test RMSE'] - results_df['Train RMSE']).abs() / results_df['Train RMSE']) * 100

# Save results to CSV
results_df.to_csv('model_performance_results.csv', index=False)

# Visualization
# Generate Time Taken Bar Graph
for model in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model]
    plt.figure(figsize=(12, 8))
    plt.barh(subset['Param'], subset['Time Taken'], color='orange', label='Time Taken')
    plt.xlabel('Time (s)')
    plt.ylabel('Parameters')
    plt.title(f'{model} Time Taken Across Parameters')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model}_time_taken.png')
    plt.show()

# Generate RMSE Bar Graph
for model in results_df['Model'].unique():
    for target in results_df['Target'].unique():
        subset = results_df[(results_df['Model'] == model) & (results_df['Target'] == target)]
        plt.figure(figsize=(12, 8))
        plt.barh(subset['Param'], subset['Test RMSE'], color='skyblue', label='Test RMSE')
        plt.barh(subset['Param'], subset['Train RMSE'], color='lightgreen', label='Train RMSE', alpha=0.7)
        plt.xlabel('RMSE')
        plt.ylabel('Parameters')
        plt.title(f'{model} {target} RMSE Across Parameters')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model}_{target}_rmse.png')
        plt.show()

# Generate Percentage Error Bar Graph
for model in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model]
    plt.figure(figsize=(12, 8))
    plt.barh(subset['Param'], subset['Percentage Error'], color='red', label='Percentage Error')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Parameters')
    plt.title(f'{model} Percentage Error Across Parameters')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model}_percentage_error.png')
    plt.show()

