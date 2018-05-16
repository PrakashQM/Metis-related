import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request
import pickle
from prakash_helper import convert_json_DF,raw_final_DF
from sklearn.ensemble import GradientBoostingClassifier


with open('final_model.pkl', 'rb') as f:
        classifier = pickle.load(f)
DF_default = pd.read_pickle('default_DF.pkl')


app=Flask(__name__)
@app.route('/api',methods=['POST','GET'])
def make_predict():
    data = request.json 
    #X_test_raw = pd.read_json(data)
    X_test_raw = pd.DataFrame.from_dict(data[0],orient='index')
    X_test = raw_final_DF(X_test_raw.T,DF_default)
    predict_me = classifier.predict(X_test)
    output = ('% Probability of sucess {}'.format(round(100*classifier.predict_proba(X_test)[0][1],2)))
    return jsonify(results=output)

if __name__=='__main__':
   app.run(port=8000,debug=True)


    
