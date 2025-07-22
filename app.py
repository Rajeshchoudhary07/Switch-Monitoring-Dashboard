from flask import Flask, render_template, jsonify
import pandas as pd
import random
import pickle
from datetime import datetime

app = Flask(__name__)

# Load pretrained anomaly detection model
with open('switch_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Mock switch data
def generate_switch_data():
    return {
        "switch_id": random.randint(1, 100),
        "status": random.choice(["ON", "OFF", "ERROR"]),
        "load": round(random.uniform(0, 100), 2),
        "temp": round(random.uniform(20, 80), 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    data = generate_switch_data()
    # Predict anomaly (0=normal, 1=anomaly)
    anomaly = model.predict([[data['load'], data['temp']]])[0]
    data['anomaly'] = bool(anomaly)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)