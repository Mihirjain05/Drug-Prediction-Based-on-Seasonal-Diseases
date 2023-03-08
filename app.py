# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:03:02 2023

@author: ASUS
"""

from flask import Flask, render_template, request
import re
import pandas as pd
import copy
import pickle
import joblib
#import Final 
from my_functions import map_age, map_drug_name, map_duration, map_season

model = pickle.load(open('DT.pkl','rb'))
impute = joblib.load('imputation_encoding')


#define flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        clean_data = pd.read_csv(f)
        #1st stage of preprocessing 
        clean_data.drop_duplicates()
        clean_data = clean_data.dropna(subset=['condition','duration_of_medication','age_range','gender'])
        clean_data.drop(['upvotes'],axis=1,inplace=True)
        clean_data['review_date'] = clean_data['review_date'].str.replace('-', '/')
        clean_data['review_date'] = pd.to_datetime(clean_data['review_date'])
        clean_data['year'] = clean_data['review_date'].dt.year
        clean_data['month'] = clean_data['review_date'].dt.month
        clean_data['day'] = clean_data['review_date'].dt.day
        condition = clean_data['condition']
        
        
	  #feature engineering
	  
        
        clean_data['season'] = clean_data['month'].apply(map_season)

        clean_data['age_group'] = clean_data['age_range'].apply(map_age)

        #clean_data['drug_name'] = clean_data['drug_name'].apply(map_drug_name)  

        clean_data['medication_duration'] = clean_data['duration_of_medication'].apply(map_duration)
        clean_data.drop(['age_range', 'duration_of_medication', 'review_date' , 'year', 'month', 'day'], axis=1, inplace=True)
        clean_data.dropna()
        
        clean_data1 = impute.transform(clean_data)
        prediction = pd.DataFrame(model.predict(clean_data1), columns = ['Drug_Name'])
        final = pd.concat([condition, prediction], axis = 1)
        #prediction.to_sql('result', con = engine, if_exists='replace', chunksize=1000, index = False)
        
        return render_template('new.html', Y = final.to_html(justify = 'center'))
    
if __name__ == '__main__':
    app.run(debug=True)
        
        
   