import pandas as pd

data = {
    "Model": ["KNN"] * 8 + ["GBR"] * 24,
    "Param": [
        "n_neighbors=3, weight=uniform", "n_neighbors=3, weight=distance",
        "n_neighbors=5, weight=uniform", "n_neighbors=5, weight=distance",
        "n_neighbors=8, weight=uniform", "n_neighbors=8, weight=distance",
        "n_neighbors=9, weight=uniform", "n_neighbors=9, weight=distance",
        "n_estimators=100, learning_rate=0.1, max_depth=3",
        "n_estimators=100, learning_rate=0.1, max_depth=4",
        "n_estimators=100, learning_rate=0.2, max_depth=3",
        "n_estimators=100, learning_rate=0.2, max_depth=4",
        "n_estimators=100, learning_rate=0.5, max_depth=3",
        "n_estimators=100, learning_rate=0.5, max_depth=4",
        "n_estimators=115, learning_rate=0.1, max_depth=3",
        "n_estimators=115, learning_rate=0.1, max_depth=4",
        "n_estimators=115, learning_rate=0.2, max_depth=3",
        "n_estimators=115, learning_rate=0.2, max_depth=4",
        "n_estimators=115, learning_rate=0.5, max_depth=3",
        "n_estimators=115, learning_rate=0.5, max_depth=4",
        "n_estimators=120, learning_rate=0.1, max_depth=3",
        "n_estimators=120, learning_rate=0.1, max_depth=4",
        "n_estimators=120, learning_rate=0.2, max_depth=3",
        "n_estimators=120, learning_rate=0.2, max_depth=4",
        "n_estimators=120, learning_rate=0.5, max_depth=3",
        "n_estimators=120, learning_rate=0.5, max_depth=4",
        "n_estimators=130, learning_rate=0.1, max_depth=3",
        "n_estimators=130, learning_rate=0.1, max_depth=4",
        "n_estimators=130, learning_rate=0.2, max_depth=3",
        "n_estimators=130, learning_rate=0.2, max_depth=4",
        "n_estimators=130, learning_rate=0.5, max_depth=3",
        "n_estimators=130, learning_rate=0.5, max_depth=4",
        "n_estimators=150, learning_rate=0.1, max_depth=3",
        "n_estimators=150, learning_rate=0.1, max_depth=4",
        "n_estimators=150, learning_rate=0.2, max_depth=3",
        "n_estimators=150, learning_rate=0.2, max_depth=4",
        "n_estimators=150, learning_rate=0.5, max_depth=3",
        "n_estimators=150, learning_rate=0.5, max_depth=4",
    ],
    "Train RMSE": [
        0.6803, 0.0, 0.7778, 0.0, 0.8674, 0.0, 0.8871, 0.0,
        0.0583, 0.0196, 0.0038, 0.0004, 0.0, 0.0,
        0.0404, 0.0116, 0.0018, 0.0001, 0.0, 0.0,
        0.0358, 0.0097, 0.0014, 0.0001, 0.0, 0.0,
        0.0281, 0.0068, 0.0008, 0.0, 0.0, 0.0,
        0.0173, 0.0034, 0.0003, 0.0, 0.0, 0.0
    ],
    "Test RMSE": [
        1.1378, 1.1378, 1.3095, 1.3095, 1.1515, 1.1515, 1.0830, 1.0830,
        1.2815, 1.2881, 1.3035, 1.2414, 1.2393, 1.2311,
        1.2788, 1.2882, 1.3047, 1.2414, 1.2393, 1.2311,
        1.2755, 1.2879, 1.3050, 1.2414, 1.2393, 1.2311,
        1.2765, 1.2875, 1.3051, 1.2414, 1.2393, 1.2311,
        1.2785, 1.2880, 1.3053, 1.2414, 1.2393, 1.2311
    ],
    "Time Taken": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0038,
        0.033, 0.041, 0.059, 0.067, 0.052, 0.067,
        0.064, 0.065, 0.041, 0.071, 0.083, 0.052,
        0.049, 0.084, 0.056, 0.088, 0.053, 0.050,
        0.068, 0.050, 0.057, 0.080, 0.057, 0.049,
        0.089, 0.100, 0.096, 0.082, 0.074, 0.087
    ]
}

df = pd.DataFrame(data)

file_name = 'model_performance.xlsx'
df.to_excel(file_name, index=False)

print(f"Data saved to {file_name}")
