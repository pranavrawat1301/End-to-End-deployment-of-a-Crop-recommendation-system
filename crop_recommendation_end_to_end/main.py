import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

#from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


#ann = pickle.load(open('ann.pkl','rb'))

ann = load_model('ann.h5')
sc = pickle.load(open('standard_scalar.pkl','rb'))
ct = pickle.load(open('column_transformer.pkl','rb'))




@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods = ['POST'])
def predict():
    
    Nitrogen = ''
    Phosphorus = ''
    Pottasium = ''
    Name_of_the_city = ''
    Rainfall = ''
    ph = ''
    
    if request.method=='POST':
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Pottasium = float(request.form['Pottasium'])
        Name_of_the_city = str(request.form['Name_of_the_city'])
        Rainfall = float(request.form['Rainfall'])
        ph = float(request.form['ph'])
    import requests 
    city_name = Name_of_the_city
    weather_data = requests.get('http://api.weatherstack.com/current?access_key=282e3892595fa9b15764697842c943c2&query={}'.format(city_name))
    #observation_time = ''
    temperature = ''
    #weather_description = ''
    humidity = ''
    #observation_time = weather_data.json()['current']['observation_time']
    temperature = weather_data.json()['current']['temperature']
    #weather_description = weather_data.json()['current']['weather_descriptions']
    humidity = weather_data.json()['current']['humidity']
    
    #(classifier.predict(sc.transform([[91,43,44,21.87974,83.00027,6.5111,203.9355]])))

    prediction = ann.predict(sc.transform([[Nitrogen,Phosphorus,Pottasium,temperature,humidity,ph,Rainfall]])) 
    prediction = ct.named_transformers_['encoder'].inverse_transform(prediction)
    print(prediction)
    
    return render_template('home.html', prediction_text="The recommended crop for these factors is {}".format(prediction[0]))
    


'''int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction[0])
    #prediction[0] is not compulsary as it just shows that if the prediction were to be a multi-valued array the first value is to be chosen
    #output = round(prediction[0], 2) for rounding off 
    return render_template('home.html', prediction_text="AQI for Jaipur {}".format(prediction[0]))'''

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = ann.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run()
    