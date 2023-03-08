#!/usr/bin/env python
# coding: utf-8

# # Drug prediction based on seasonal diseases

#  # Objective 
#  
#  - Maximize profits,
#  
#  - Minimize inventory costs 
# 
# # Constraints 
# 
# - Non-availabilty of raw data
# - Quality assurance
# - Data security
#         

# In[2]:


# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from feature_engine.outliers import Winsorizer
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from termcolor import colored

import joblib
import pickle


# In[3]:


import os
os.getcwd()

os.chdir(r'C:\Users\ASUS\Desktop\Project 93')


# In[4]:


#pip install mysql-connector-python
import mysql.connector
from mysql.connector import Error


# # Creating SQL connection with Python

# In[5]:


#pip install sqlalchemy
#pip install pymysql
from sqlalchemy import create_engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="987654", # passwrd
                               db="proj93")) #database


# In[6]:


df = pd.read_csv(r"C:\Users\ASUS\Desktop\Project 93\drug_.csv")
df.to_sql('drug', con = engine, if_exists = 'replace', chunksize = 1000, index= False)


# In[7]:


sql = "SELECT * FROM proj93.drug;"


# In[8]:


data = pd.read_sql_query(sql, engine)


# # Data Preprocessing and Feature Engineering

# In[9]:


data.shape


# In[10]:


data.info()


# In[11]:


data.head()


# In[12]:


duplicate = data.duplicated()
duplicate


# In[13]:


sum(duplicate)

data.drop_duplicates()


# # Missing Values

# In[14]:


missing=data.isna().sum().sort_values(ascending=False)
print(colored("Number of Missing Values\n\n",'blue',attrs=['bold']),missing)


# In[15]:


missing_p=((data.isna().sum()/data.count())*100).sort_values(ascending=False)
missing_p= missing_p[missing_p>0]
print(colored("Percentage of Missing Values\n\n",'blue',attrs=['bold']),missing_p)


# In[16]:


data = data.dropna(subset=['drug_name','condition','duration_of_medication','age_range','gender'])
data


# In[17]:


data.shape


# In[18]:


# Dropping the upvote column

data.drop(['upvotes'],axis=1,inplace=True)


# In[19]:


# Checking the number of unique values in each columns

print(colored('Number of Unique Values:\n\n','blue',attrs=['bold']),df.nunique())


# # Conversion of review date column for better understanding

# In[20]:


data['review_date'] = data['review_date'].str.replace('-', '/')#.str.replace('/', '')


# In[21]:


data['review_date'] = pd.to_datetime(data['review_date'])
data.info()


# # Split date column into year, month, and day columns

# In[22]:


data['year'] = data['review_date'].dt.year
data['month'] = data['review_date'].dt.month
data['day'] = data['review_date'].dt.day


# # Creating custom function for Feature Engineering

# In[23]:


def map_season(month):
    if month in [11, 12, 1]:
        return 'Winter'
    elif month in [2, 3]:
        return 'Spring'
    elif month in [4, 5, 6]:
        return 'Summer'
    elif month in [7, 8]:
        return 'Monsoon'
    else:
        return 'Autumn'
    
    
def map_age(age_range):
    if age_range in ['  0-2', '  3-6', '  7-12', '  13-18', '  19-24']:
        return 'Young'
    elif age_range in ['  25-34', '  35-44', '  45-54']:
        return 'Adult'
    else:
        return 'Senior'
    
    
def map_duration(duration_of_medication):
    if duration_of_medication in ['less than 1 month', '1 to 6 months', '6 months to less than 1 year']:
        return 'Short term'
    elif duration_of_medication in ['1 to less than 2 years', '2 to less than 5 years']:
        return 'Medium term'
    else:
        return 'Long term'
    

def map_drug_name(drug_name):
    if drug_name in ['Crestor oral','simvastatin oral','pravastatin oral','Lipitor oral','rosuvastatin oral']:
        return 'Statin'
    elif drug_name in ['Warfarin oral','Coumadin oral']:
        return 'Anticoagulant'
    elif drug_name in ['Cipro oral','doxycycline hyclate oral','Flagyl oral',
                  'metronidazole oral',
                  'ciprofloxacin oral',
                  'Augmentin oral',
                  'Levaquin oral',
                  'levofloxacin oral',
                  'cephalexin oral',
                  'clarithromycin oral',
                  'Biaxin oral',
                  'amoxicillin oral',
                  'Keflex oral',
                  'clindamycin HCl oral',
                  'Macrobid oral',
                  'azithromycin oral',
                  'cefuroxime axetil oral',
                  'amoxicillin-potassium clavulanate oral',
                  'Bactrim DS oral',
                  'Xifaxan oral',
                  'Rocephin injection',
                  'penicillin V potassium oral',
                  'vancomycin oral',
                  'Cleocin HCl oral',
                  'erythromycin oral',
                  'Rocephin intravenous',
                  'cefadroxil oral']:
        return 'Antibiotic'
    elif drug_name in ['Tamiflu oral',
                 'valacyclovir oral',
                 'Valtrex oral',
                 'acyclovir oral']:
        return 'Antiviral'
    elif drug_name in ['Celebrex oral',
             'ibuprofen oral',
             'nabumetone oral',
             'meloxicam oral',
             'celecoxib oral']:
        return 'NSAID'
    elif drug_name in ['prednisone oral',
                      'methylprednisolone oral',
                      'clobetasol topical',
                      'mometasone topical',
                      'halobetasol propionate topical',
                      'Olux-E topical',
                      'Elocon topical']:
        return 'Corticosteroid'
    elif drug_name in ['Flexeril oral',
                       'cyclobenzaprine oral',
                       'baclofen oral',
                       'methocarbamol oral',
                       'dicyclomine oral']:
        return 'Muscle_Relaxant'
    elif drug_name in ['Tessalon Perles oral',
                   'benzonatate oral',
                   'Tussionex oral',
                   'Tussionex Pennkinetic ER oral',
                   'Hycodan (with homatropine) oral',
                   'Hydromet oral',
                   'dextromethorphan HBr oral']:
        return 'Antitussive'
    elif drug_name in ['terbinafine HCI oral',
                  'fluconazole oral',
                  'Lamisil oral',
                  'Diflucan oral',
                  'ketoconazole topical',
                  'itraconazole oral']:
        return 'Antifungal'
    elif drug_name in ['Effexor XR oral',
                      'bupropion HCl oral',
                      'Lexapro oral',
                      'venlafaxine oral',
                      'sertraline oral',
                      'fluoxetine oral',
                      'trazodone oral',
                      'Prozac oral',
                      'citalopram oral',
                      'amitriptyline oral',
                      'Cymbalta oral',
                      'escitalopram oxalate oral',
                      'duloxetine oral',
                      'Desyrel oral',
                      'Celexa oral',
                      'nortriptyline oral',
                      'Vibramycin oral']:
        return 'Antidepressant'
    elif drug_name in ['methadone oral',
                        'oxycodone oral',
                        'Percocet oral',
                        'morphine oral',
                        'OxyContin oral',
                        'Dilaudid oral',
                        'hydrocodone-acetaminophen oral',
                        'tramadol oral']:
        return 'Opioid_analgesic'
    elif drug_name in ['amlodipine oral',
                        'losartan oral']:
        return 'Antihypertensive'
    elif drug_name in ['Byetta subcutaneous',
                    'Trulicity subcutaneous',
                    'Actos oral',
                    'glipizide oral',
                    'metformin oral']:
        return 'Antidiabetic'
    elif drug_name in ['lorazepam oral',
                      'Ativan oral',
                      'clonazepam oral',
                      'alprazolam oral',
                      'buspirone oral']:
        return 'Benzodiazepine'
    elif drug_name in ['allopurinol oral']:
        return 'Antigout_Medication'
    elif drug_name in ['Cialis oral', 'Levitra oral']:
        return 'Erectile_Dysfunction_Medication'
    elif drug_name in ['atenolol oral',
                    'metoprolol succinate oral',
                    'carvedilol oral']:
        return 'Beta_blocker'
    elif drug_name in ['topiramate oral',
                      'Neurontin oral',
                      'Depakote oral',
                      'gabapentin oral',
                      'Lyrica oral',
                      'lamotrigine oral']:
        return 'Anticonvulsant'
    elif drug_name in ['Benadryl oral',
                     'hydroxyzine HCl oral',
                     'promethazine oral',
                     'diphenhydramine oral',
                     'loratadine oral',
                     'Astelin nasal',
                     'Allegra-D 12 Hour oral',
                     'fexofenadine oral',
                     'levocetirizine oral',
                     'Phenergan injection',
                     'chlorpheniramine oral',
                     'Clarinex oral']:
        return 'Antihistamine'
    elif drug_name in ['Emgality Pen subcutaneous']:
        return 'CGRP_Inhibitor'
    elif drug_name in ['omeprazole oral', 'pantoprazole oral', 'Dexilant oral']:
        return 'Antacid'
    elif drug_name in ['methylphenidate HCl oral',
                     'Concerta oral',
                     'Adderall XR oral']:
        return 'CNS_stimulant'
    elif drug_name in ['estradiol oral']:
        return 'Estrogen_Hormone'
    elif drug_name in ['montelukast oral']:
        return 'Leukotriene_receptor_antagonist'
    elif drug_name in ['zolpidem oral',
                 'Ambien oral',
                 'Fiorinal oral']:
        return 'Sedatives'
    elif drug_name in ['Lomotil oral',
                     'diphenoxylate-atropine oral']:
        return 'Antidiarrheal'
    elif drug_name in ['Abilify oral', 'quetiapine oral']:
        return 'Antipsychotic'
    elif drug_name in ['acetaminophen oral']:
        return 'Antipyretic'
    elif drug_name in ['Keppra oral', 'Dilantin Kapseal oral']:
        return 'Antiepileptic'
    elif drug_name in ['Keppra oral', 'Dilantin Kapseal oral']:
        return 'Antiepileptic'
    elif drug_name in ['Detrol LA oral']:
        return 'Antipasmodic'
    elif drug_name in ['phentermine oral']:
        return 'Anorectic'
    elif drug_name in ['levothyroxine oral', 'Armour Thyroid oral']:
        return 'Thyroid_Hormone_Replacement'
    elif drug_name in ['hydrochlorothiazide oral', 'furosemide oral', 'spironolactone oral']:
        return 'Diuretic'
    elif drug_name in ['lisinopril oral']:
        return 'Ace_Inhibitor'
    elif drug_name in ['Namenda oral']:
        return 'NMDA_receptor_antagonist'
    elif drug_name in ['tamsulosin oral']:
        return 'Alpha_1_blocker'
    elif drug_name in ['phenylephrine oral']:
        return 'Nasal_Decongestant'
    elif drug_name in ['Lotronex oral']:
        return 'Serotonin_Antagonists'
    elif drug_name in ['vardenafil oral']:
        return 'PDE5_Inhibitor'
    elif drug_name in ['Elidel topical', 'Protopic topical']:
        return 'Topical_immunomodulator'
    elif drug_name in ['guaifenesin oral']:
        return 'Expectorant'
    elif drug_name in ['Malarone oral']:
        return 'Antimalarial'


# # Using the custom functions to map features for generation of new features

# In[24]:


data['season'] = data['month'].apply(map_season)

data['age_group'] = data['age_range'].apply(map_age)

data['drug_name'] = data['drug_name'].apply(map_drug_name)  

data['medication_duration'] = data['duration_of_medication'].apply(map_duration)


# ## Droping the original column post feature engineering

# In[25]:


data.drop(['age_range', 'duration_of_medication', 'review_date' , 'year', 'month', 'day'], axis=1, inplace=True)


# # Exploratary Data Analysis

# ## Measure of central tendency

# In[26]:


data.mean()


# In[27]:


data.median()


# In[28]:


data.mode()


# ## Measure of dispersion

# In[29]:


data.var()


# In[30]:


data.std()


# ## Skewness

# In[31]:


data.skew()


# - Measures of asymmetry in the distribution
# - Negative skewness implies mass of the distribution is concentrated on the right.

# ## Kurtosis

# In[32]:


data.kurt()


# - A measure of the peakness of the distribution
# - For symetric distribution negative kurtosis implies wider peak and thinner tail

# # Unique drug counts

# In[33]:


data_dn = data.drug_name.value_counts()
data_dn.head(50)


# # Unique condition count

# In[34]:


data_cond = data.condition.value_counts()
data_cond.head(50)


# # Unique numbers of season

# In[35]:


data.season.value_counts()


# # Unique number of condition

# In[36]:


data.condition.value_counts()


# # Graphical Representation

# In[37]:


plt.figure(1, figsize=(10, 5))
sns.countplot(x=data["age_group"])
plt.xticks(rotation = 45)
plt.show()


# In[38]:


plt.figure( figsize=(10, 5))
sns.displot(data, x="age_group", hue="gender")
plt.xticks(rotation = 45)
plt.show()


# In[58]:


freq_table_age = data['age_group'].value_counts()
freq_table_gender = data['gender'].value_counts()
freq_table_duration = data['medication_duration'].value_counts()
freq_table_date = data['season'].value_counts()
freq_table_condition = data['condition'].value_counts()
freq_table_drug=data['drug_name'].value_counts()


# In[59]:


plt.figure( figsize=(10, 5))
freq_table_condition.iloc[:15].plot(kind='bar')
plt.xlabel('Condition')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Condition',fontweight = 'bold', fontsize = 14, fontfamily = 'sans-serif')
plt.xticks(ticks=range(0, 15), labels=freq_table_condition.index[:15], rotation=90)
plt.show()


# In[60]:


plt.figure( figsize=(10, 5))
freq_table_duration.plot(kind='bar')
plt.xlabel('Duration_of_Medication')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Duration of Medication',fontweight = 'bold', fontsize = 14, fontfamily = 'sans-serif')
plt.xticks( rotation=90)
plt.show()


# In[62]:


plt.figure( figsize=(10, 5))
freq_table_drug.plot(kind='bar')
plt.xlabel('Duration_of_Medication')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Duration of Medication',fontweight = 'bold', fontsize = 14, fontfamily = 'sans-serif')
plt.xticks( rotation=90)
plt.show()


# In[63]:


plt.figure( figsize=(15, 7))
freq_table_age.plot(kind='pie', autopct='%1.1f%%')
plt.axis('equal')
plt.title('Pie Chart of Age Range',fontweight = 'bold', fontsize = 14, fontfamily = 'sans-serif')
plt.show()


# In[64]:


plt.figure( figsize=(15, 7))
freq_table_gender.plot(kind='pie', autopct='%1.1f%%')
plt.axis('equal')
plt.title('Pie Chart of Gender',fontweight = 'bold', fontsize = 14, fontfamily = 'sans-serif')
plt.show()


# In[65]:


plt.figure( figsize=(15, 7))
freq_table_duration.plot(kind='pie', autopct='%1.1f%%')
plt.axis('equal')
plt.title('Pie Chart of Duration of Medication',fontweight = 'bold', fontsize = 14, fontfamily = 'sans-serif')
plt.show()


# In[66]:


plt.figure( figsize=(15, 7))
freq_table_condition.iloc[:15].plot(kind='pie', autopct='%1.1f%%')
plt.axis('equal')
plt.title('Pie Chart of Top 15 Conditions',fontweight = 'bold', fontsize = 14, fontfamily = 'sans-serif')
plt.show()


# In[67]:


data.dropna(inplace=True)
data.info()


# # Data split into Input and Output variable

# In[68]:


X = data.drop('drug_name', axis = 1) # Predictors 
y = data['drug_name'] # Target 


# #### Separating Numeric and Non-Numeric columns

# In[40]:


numeric_features = X.select_dtypes(exclude = ['object']).columns


# In[41]:


numeric_features


# In[42]:


categorical_features = X.select_dtypes(include=['object']).columns


# In[43]:


categorical_features


# #### Imputation to handle missing values and Encoding - Ordinal Encoder to convert Categorical data to Numeric values

# In[44]:


num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])
encoding_pipeline = Pipeline([('Ordinal', OrdinalEncoder())])


# In[45]:


preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features),('categorical', encoding_pipeline, categorical_features)])


# In[46]:


imputation_encoding = preprocessor.fit(X)


# In[47]:


joblib.dump(imputation_encoding, 'imputation_encoding')


# In[48]:


num_data = imputation_encoding.transform(X)
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
num_data = pd.DataFrame(num_data, columns = num_cols+cat_cols)
num_data


# # Spliting the data into Train and Test

# In[49]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(num_data, y, test_size = 0.2, random_state = 0) 


# # Decision Tree

# In[50]:


model = DT(criterion = 'entropy')
model.fit(X_train, Y_train)


# ### Prediction for train data

# In[51]:


pred_train1 = model.predict(X_train)
acc_train1 = accuracy_score(Y_train, pred_train1) 
print(acc_train1)


# ### Prediction for test data

# In[52]:


preds = model.predict(X_test)
preds


# In[53]:


pd.crosstab(Y_test, preds, rownames = ['Actual'], colnames= ['Predictions']) 


# In[54]:


acc_test1 = accuracy_score(Y_test, preds) 
print(accuracy_score(Y_test, preds))


# # Train accuracy - 97%
# 
# # Test accuracy - 85%
# 
# # The difference between trainning and test accuracy is 12% in order to over come this we will use cross validation.

# ## Cross Validation using grid search

# In[55]:


param_grid = { 'criterion':['gini','entropy'], 'max_depth': np.arange(3, 50)}


# In[56]:


dtree_model = DT()


# In[57]:


dtree_gscv = GridSearchCV(dtree_model, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1, refit=True)


# In[58]:


y = pd.DataFrame(y)
y


# In[59]:


#Train
dtree_gscv.fit(num_data, y)


# In[60]:


# The best set of parameter values
dtree_gscv.best_params_


# In[61]:


# Model with best parameter values
DT_best = dtree_gscv.best_estimator_
DT_best


# In[62]:


# Accuracy
pred_train2 = DT_best.predict(X_train)
acc_train2 = accuracy_score(Y_train, pred_train2) 
print(acc_train2)


# In[63]:


# Prediction on Test Data
preds1 = DT_best.predict(X_test)
preds1
pd.crosstab(Y_test, preds1, rownames = ['Actual'], colnames= ['Predictions']) 


# In[64]:


# Accuracy
acc_test2 = accuracy_score(Y_test, preds1) 
print(accuracy_score(Y_test, preds1)) 


# # Training accuracy after cross validation is 92%
# 
# # Testing accuracy after cross validation is 93%
# 
# # Considering the model perfomance we will save this DT Model.

# In[65]:


from sklearn.metrics import classification_report


# In[66]:


# calculate precision, recall, and F1-score
print(classification_report(Y_test, preds1))


# In[67]:


pickle.dump(DT_best, open('DT.pkl', 'wb'))


# # Random Forest Classifier

# In[68]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)


# In[69]:


rf_clf.fit(X_train,Y_train)
accuracy = rf_clf.score(X_train,Y_train)
accuracy


# In[70]:


preds2 = rf_clf.predict(X_test)
pd.crosstab(Y_test, preds2, rownames = ['Actual'], colnames= ['Predictions'])


# In[71]:


# Accuracy
print(accuracy_score(Y_test, preds2))


# ## IT is an overfit model.

# # Bagging

# In[72]:


from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(base_estimator = clftree, n_estimators = 10, bootstrap = True, n_jobs = -1, random_state = 0)
bag_clf.fit(X_train, Y_train)


# In[73]:


from sklearn.metrics import accuracy_score, confusion_matrix
pred = bag_clf.predict(X_test)
confusion_matrix_bag_clf_test = confusion_matrix(Y_test,pred )
accuracy_score_bag_clf_test = accuracy_score(Y_test, pred)
accuracy_score_bag_clf_test


# In[74]:


pred_train = bag_clf.predict(X_train)
confusion_matrix_bag_clf_train = confusion_matrix(Y_train,pred_train )
accuracy_score_bag_clf_train = accuracy_score(Y_train,pred_train)
accuracy_score_bag_clf_train


# # Bagging is also an overfit model.

# # Stacking

# In[75]:


from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# In[76]:


# Base estimators
estimators = [('rf', RandomForestClassifier(n_estimators = 10, random_state = 42)),
              ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state = 42)))]


# In[77]:


# Meta Model stacked on top of base estimators

clf = StackingClassifier(estimators = estimators, final_estimator = LogisticRegression())


# In[78]:


# Fit the model on traing data
stacking = clf.fit(X_train, Y_train)


# In[79]:


# Accuracy
stacking.score(X_test, Y_test)


# In[80]:


# Accuracy
stacking.score(X_train, Y_train)


# ## Stacking is also an overfit model.

# # XGBoost

# In[81]:


from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# In[82]:


le = LabelEncoder()


# In[83]:


y1 = le.fit_transform(y)


# In[84]:


y1_train, y1_test = train_test_split(y1, random_state=42, test_size=0.2)
y1_train


# In[85]:


xgb_clf = xgb.XGBClassifier(max_depth = 5, n_estimators = 10000, 
                            learning_rate = 0.3, n_jobs = -1)


# In[86]:


xgb_clf1 = xgb_clf.fit(X_train, y1_train)


# In[87]:


xgb_pred = xgb_clf1.predict(X_test)


# In[88]:


# Evaluation on Testing Data
print(confusion_matrix(y1_test, xgb_pred))


# In[89]:


accuracy_score(y1_test, xgb_pred)


# In[90]:


xgb.plot_importance(xgb_clf)


# # Thank You
