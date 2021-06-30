from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import os
import joblib

def load_model(model_file):
    load_model = joblib.load(open(os.path.join(model_file),'rb'))
    return load_model

def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key

app = Flask(__name__)

#Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    df = pd.read_csv('data/clean_hepatitis_dataset.csv')
    return render_template('dataset.html',df_table = df)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        steroid = request.form['steroid']
        antivirals = request.form['antivirals']
        fatigue = request.form['fatigue']
        spiders = request.form['spiders']
        ascites = request.form['ascites']
        varices = request.form['varices']
        bilirubin = request.form['bilirubin']
        alk_phosphate = request.form['alk_phosphate']
        sgot = request.form['sgot']
        albumin = request.form['albumin']
        protime = request.form['protime']
        histology = request.form['histology']
        
        pretty_data = {"age":age,"sex":sex,"steroid":steroid,"antivirals":antivirals,"fatigue":fatigue,
        "spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,
        "sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
        
        single_data = [age,sex,steroid,antivirals,fatigue,spiders,ascites,varices,
        bilirubin,alk_phosphate,sgot,albumin,protime,histology]

        print(single_data)
        print(len(single_data))


        encoded_values = [float(int(i)) for i in single_data]
        model = load_model('models/logistic_regression_hepB_model.pkl')
        prediction = model.predict(np.array(encoded_values).reshape(1,-1))
        print(predict)
        prediction_label = {"Die":1,"Live":2}
        final_result = get_key(prediction[0],prediction_label)
        pred_prob = model.predict_proba(np.array(encoded_values).reshape(1,-1))
        pred_probalility_score = {"Die":pred_prob[0][0]*100,"Live":pred_prob[0][1]*100}

    return render_template('predict.html', single_data=single_data,pretty_data=pretty_data
    ,encoded_values=encoded_values,prediction=prediction,pred_probalility_score=pred_probalility_score,
    final_result=final_result)


 

if __name__ == '__main__':
    app.run(debug=True)
