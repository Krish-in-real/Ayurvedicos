import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np
from sklearn import svm



app = Flask(__name__)
##load the model
model_disease = pickle.load(open('model.pkl','rb'))  # type: ignore
model_medicine = 0##medicine data set


def processSymptoms(inp):
    li = list(pd.read_csv('Symptoms.csv')['Symptoms'])
    for  i in range(0,len(li)):
        if li[i] in inp['Symptoms']:
            li[i] = 1
        else:
            li[i] = 0
    return np.array([li]).reshape(1, -1);
    
    
# def processDisease(input):
#     ##to shape the data
#     pass
    
@app.route('/')
def home():
    return render_template('home.html')


###implementing the model here
@app.route('/predict_disease',methods=['POST'])
def predict_disease():
    data = request.json['data']
    data = processSymptoms(dict(data))
    output = model_disease.predict(data)
    df = pd.read_csv('Disease.csv')
    result = df['disease'][int(output[0])-1]
    return jsonify(result)
    
    
###returning the symptoms for render
@app.route('/symptoms',methods=['get'])
def get_symptoms():
    with open('Symptoms.json', 'r') as json_file:
        json_data = json.load(json_file)
    return json_data


# @app.route('/predict_medicine',methods=['POST'])
# def predict_medicine():
#     # data = request.json['data']
#     # ##process the data and take the necessory one then pass it to the model
#     # data = processDisease(dict(data));
#     # print(data)
#     # output = model_medicine.predict(data)
#     # print(output[0])
#     # return jsonify(output[0])
#     pass
    
if __name__ == "__main__":
    app.run(debug=True)