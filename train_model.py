import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

# Mock training data (load %, temp °C)
data = pd.DataFrame({
    'load': [20, 30, 80, 90, 25, 35, 85, 95],
    'temp': [25, 30, 70, 75, 22, 32, 68, 72],
    'anomaly': [0, 0, 0, 0, 0, 0, 0, 0]  # 1=anomaly
})

# Train model
model = IsolationForest(contamination=0.1)
model.fit(data[['load', 'temp']])

# Save model
with open('switch_model.pkl', 'wb') as f:
    pickle.dump(model, f)