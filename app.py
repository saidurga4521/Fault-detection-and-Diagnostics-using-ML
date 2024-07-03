import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and the scaler
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('sc.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the expected feature names in the order they were fitted
# Ensure this matches the order of features expected by StandardScaler
feature_names = ['belt_speed', 'motor_temperature', 'motor_current', 'drum_temperature', 'belt_tension']
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract features from JSON request
    belt_speed = float(data['belt_speed'])
    motor_temperature = float(data['motor_temperature'])
    motor_current = float(data['motor_current'])
    drum_temperature = float(data['drum_temperature'])
    belt_tension = float(data['belt_tension'])

    # Create input data array in the expected order
    input_data = np.array([[belt_speed, motor_temperature, motor_current, drum_temperature, belt_tension]])
    
    # Transform input data using the loaded scaler
    # Ensure feature_names are used for columns parameter
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    status_prediction = model.predict(scaled_input_data)
    
    # Map the numerical prediction to the corresponding status
    status_mapping = {0: 'broken belt', 1: 'skid', 2: 'off track', 3: 'high temperature', 4: 'mistracking', 5: 'normal', 6: 'slip', 7: 'break'}
    status = status_mapping[status_prediction[0]]
    
    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(debug=True)

