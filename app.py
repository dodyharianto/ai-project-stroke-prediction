import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_model():
    model_name = 'lgbm_baseline.pkl'
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
        
    return model

def tidy_columns(col_list):
      return [col.title().replace('_', ' ') for col in col_list]

def generate_dummy_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df.columns = tidy_columns(df.columns)
    df.drop(['Id', 'Stroke'], axis = 1, inplace = True)
    
    return df.head(1000)

def main():
    st.title('Stroke Predictor Website')
    st.markdown('<h2 style> Welcome! </h2>', unsafe_allow_html = True)
    
    stroke_image = Image.open('stroke.png')
    st.image(stroke_image)
    
    st.markdown('<h4 style> Predict stroke outcome based on general information in seconds. </h4>', unsafe_allow_html = True)
    
    st.info('All data is available at: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset')
    
    gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    age = st.sidebar.number_input('Age (years)', value = 30, min_value = 0, max_value = 100)
    hypertension = st.sidebar.radio('Hypertension', (1, 0))
    heart_disease = st.sidebar.radio('Heart Disease', (1, 0))
    ever_married = st.sidebar.radio('Ever Married', ('Yes', 'No'))
    work_type = st.sidebar.multiselect('Work Type', options = ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'], default = 'Never_worked')
    residence_type = st.sidebar.radio('Residence Type', ('Rural', 'Urban'), default = 'Urban')
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', min_value = 50.0, max_value = 300.0)
    bmi = st.sidebar.slider('Body Mass Index', min_value = 5.0, max_value = 70.0)
    smoking_status = st.sidebar.multiselect('Smoking Status', options = ['formerly smoked', 'never smoked', 'smokes', 'Unknown'], default = 'never smoked')
    
    features = {
        'Gender': gender,
        'Age': age,
        'Hypertension': hypertension,
        'Heart Disease': heart_disease,
        'Ever Married': ever_married,
        'Work Type': work_type,
        'Residence Type': residence_type,
        'Avg Glucose Level': avg_glucose_level,
        'Bmi': bmi,
        'Smoking Status': smoking_status
    }

    features_df = pd.DataFrame(features)
    sample_df = generate_dummy_data()
    data = [features_df, sample_df]
    combined_df = pd.concat(data).reset_index().drop('index', axis = 1)
    
    combined_df['Ever Married'].replace({'Yes': 1, 'No': 0}, inplace = True)
    combined_df['Gender'].replace({'Male': 1, 'Female': 0}, inplace = True)
    
    multiple_categories_columns = ['Work Type', 'Residence Type', 'Smoking Status']
    dummies = pd.get_dummies(combined_df[multiple_categories_columns])
    combined_df.drop(multiple_categories_columns, axis = 1, inplace = True)
    
    print(f'Features df shape: {features_df.shape}')
    print(f'Dummies shape: {dummies.shape}')
    
    preprocessed_features_df = combined_df.join(dummies)
    final_df = preprocessed_features_df.iloc[[0]] # get the inputted data
    print(f'Final df shape: {final_df.shape}')
    
    if st.button('Predict') == True:
        model = load_model()
        outcome = model.predict(final_df)
        st.write('Outcome: ', 'Stroke' if outcome == 1 else 'Normal')

if __name__ == '__main__':
    main()    
