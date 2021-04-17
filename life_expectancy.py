import pandas as pd
from flask import Flask, jsonify, request
import joblib 

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    input_data = req['data']
    input_data_df = pd.DataFrame.from_dict(input_data)

    model = joblib.load('life_expect.pkl')
    #scale_obj = joblib.load('scale.pkl')
    #input_data_scaled = scale_obj.transform(input_data_df)
    #print(input_data_scaled)
    
    prediction = model.predict(input_data_df)
    prediction = round(prediction[0]*100,2)
    
    return jsonify({'Life_expectancy is':prediction})


@app.route('/')
def home():
    return "Life expectancy prediction"

       
if __name__=='__main__':
    app.run(host='0.0.0.0', port='4000')