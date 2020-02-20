#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all libraries needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import pickle

#Create a class
class absenteeism_model():
    
        def __init__(self, model_file, scaler_file):
        #read the model and scaler files which have been saved in the folder
            with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
            
            #take a data file (*.csv) and preprocess it in the same way as in the study
        def load_and_clean_data(self, data_file):
               
            #import the data
            df = pd.read_csv(data_file, delimiter=',')
            #store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            #drop the 'ID' column
            df = df.drop(['ID'], axis=1)
                
            #Adding a column with 'NaN' to preserve code created during the ground work stage
                
            df['Absenteeism Time in Hours'] = 'NaN'
                
            # create a separate dataframe, containing dummy values for all the available reasons
                
            reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
                
            #split reason_columns into 4 types
            reason_type_1 = reason_columns.loc[:,0:14].max(axis=1)
            reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
            reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
            reason_type_4 = reason_columns.loc[:,22:].max(axis=1)
                
            df = df.drop(['Reason for Absence'], axis=1)
                
            #concatenate df and the 4 types of reason for absence
            df =pd.concat([df, reason_type_1,reason_type_2,reason_type_3,reason_type_4], axis=1)
                
            #assign names to the 4 reason variables
            column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                               'Daily Work Load Average', 'Body Mass Index', 'Education',
                               'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_type_1', 
                                'Reason_type_2','Reason_type_3', 'Reason_type_4']
            df.columns = column_names
                
            #reorder the columns in df
            column_names_reordered = ['Reason_type_1', 'Reason_type_2','Reason_type_3', 'Reason_type_4','Date', 
                                          'Transportation Expense', 'Distance to Work', 'Age',
                                           'Daily Work Load Average', 'Body Mass Index', 'Education',
                                           'Children', 'Pets', 'Absenteeism Time in Hours']
            df = df[column_names_reordered]
                
            #Convert 'Date' column to timestamp
            df['Date'] = pd.to_datetime(df.Date, format= '%d/%m/%Y')
                
            #Months
            list_months = []
            for i in range(df.shape[0]):
                list_months.append(df['Date'][i].month)
            df['Months'] = list_months
                
            #day of the week
            list_dayofweek = []
            for i in range(df.shape[0]):
                list_dayofweek.append(df['Date'][i].weekday())
            df['Weekday'] = list_dayofweek
                
            #drop the Date column
            df= df.drop(['Date'], axis=1)
                
            #Reorder Columns
            newcol = ['Reason_type_1', 'Reason_type_2', 'Reason_type_3', 'Reason_type_4', 'Months',
                           'Weekday','Transportation Expense', 'Distance to Work', 'Age',
                           'Daily Work Load Average', 'Body Mass Index', 'Education',
                           'Children', 'Pets', 'Absenteeism Time in Hours']
            df = df[newcol]
                
            #Regrouping Education
            df['Education'] = df['Education'].map({1:0, 2:1,3:1,4:1})
                
            #replace missing value with 0
                
            df = df.fillna(value=0)
                
            #drop the original absenteeism time
            df = df.drop(['Absenteeism Time in Hours'], axis=1)
            df = df.drop(['Distance to Work','Weekday', 'Daily Work Load Average'], axis=1)
                
            self.preprocessed_data = df.copy()
                
                
            #scaling
            cols_to_scale = ['Months', 'Transportation Expense',
                   'Age', 'Body Mass Index', 'Children', 'Pets']
            cols_noscale = ['Reason_type_1', 'Reason_type_2', 'Reason_type_3', 'Reason_type_4','Education']
            columns_to_scale = df[cols_to_scale]
            noscale_columns = df[cols_noscale]
                
            absenteeism_scalar = StandardScaler()
            absenteeism_scalar.fit(columns_to_scale)
            scaled_inputs = absenteeism_scalar.transform(columns_to_scale)
                
            noscale_columns = pd.DataFrame(noscale_columns, columns=cols_noscale)
            scaled_inputs = pd.DataFrame(scaled_inputs, columns = cols_to_scale)
                
            total_inputs = pd.concat([noscale_columns, scaled_inputs], axis=1)
            new_cols = ['Reason_type_1', 'Reason_type_2', 'Reason_type_3', 'Reason_type_4',
                            'Months', 'Transportation Expense', 'Age','Body Mass Index', 'Education', 'Children', 'Pets']
            total_inputs = total_inputs[new_cols]
                
            self.data = total_inputs
                
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
                
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
                
        #predict the outputs and the probabilities and
        #add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
                self.preprocessed_data['ID'] = self.df_with_predictions['ID']
                return self.preprocessed_data