from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Function to prepare the input data
def prepare_input(data):
    # Define the feature names
    feature_names = ['SHOT_DISTANCE', 'TIME_REMAINING', 'ACTION_TYPE_ENCODED', 'HTM_ENCODED', 'VTM_ENCODED', 
                     'SHOT_TYPE_3PT Field Goal', 'SHOT_ZONE_BASIC_Backcourt', 'SHOT_ZONE_BASIC_In The Paint (Non-RA)',
                     'SHOT_ZONE_BASIC_Left Corner 3', 'SHOT_ZONE_BASIC_Mid-Range', 'SHOT_ZONE_BASIC_Restricted Area',
                     'SHOT_ZONE_BASIC_Right Corner 3', 'SHOT_ZONE_AREA_Center(C)', 'SHOT_ZONE_AREA_Left Side Center(LC)',
                     'SHOT_ZONE_AREA_Left Side(L)', 'SHOT_ZONE_AREA_Right Side Center(RC)', 'SHOT_ZONE_AREA_Right Side(R)',
                     'SHOT_ZONE_RANGE_24+ ft.', 'SHOT_ZONE_RANGE_8-16 ft.', 'SHOT_ZONE_RANGE_Back Court Shot', 
                     'SHOT_ZONE_RANGE_Less Than 8 ft.', 'PERIOD_2', 'PERIOD_3', 'PERIOD_4', 'PERIOD_5', 'PERIOD_6', 'PERIOD_7']

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([data], columns=feature_names)    
    return input_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    shot_distance = float(request.form['shot_distance'])
    time_remaining = float(request.form['time_remaining'])
    action_type_encoded = int(request.form['action_type_encoded'])
    htm_encoded = int(request.form['htm_encoded'])
    vtm_encoded = int(request.form['vtm_encoded'])
    
    # Shot type and zone inputs
    shot_type = int(request.form['shot_type'])
    shot_zone = int(request.form['shot_zone'])
    shot_range = int(request.form['shot_range'])
    
    # Period input
    period = int(request.form['period'])
    
    # Create input feature vector
    input_data = [shot_distance, time_remaining, action_type_encoded, htm_encoded, vtm_encoded, 
                  shot_type == 1, shot_zone == 1, shot_zone == 2, shot_zone == 3, shot_zone == 4, shot_zone == 5, 
                  shot_zone == 6, shot_zone == 7, shot_zone == 8, shot_zone == 9, shot_zone == 10, shot_zone == 11,
                  shot_range == 1, shot_range == 2, shot_range == 3, shot_range == 4, 
                  period == 2, period == 3, period == 4, period == 5, period == 6, period == 7]
    
    # Prepare the input data
    input_df = prepare_input(input_data)
    
    # Predict the probability
    prediction = model.predict_proba(input_df)
    
    # Get the probability of making the shot
    shot_made_prob = prediction[0][1]
    
    return render_template('index.html', prediction_text=f'Predicted Probability of Making Shot: {shot_made_prob:.4f}')

if __name__ == "__main__":
    app.run(debug=True)
