import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import xgboost as xgb


client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key=os.environ.get("ChurnPred")
)

def load_model(filename):
    with open(filename,'rb') as file:
      return pickle.load(file)

xgboost_model = load_model('xgboost_model.pkl')

random_forest_model = load_model('rf_model.pkl')

decision_tree_model = load_model('dt_model.pkl')

naive_bayes_model = load_model('nb_model.pkl')

svm_model = load_model('svm_model.pkl')

knn_model = load_model('knn_model.pkl')

voting_classifier_model = load_model('voting_classifier.pkl')

xgb_model = load_model('xgb_model.pkl')

xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')

def prepare_input(credit_score, age, tenure, balance, num_products, has_credit_card, estimated_salary, location, gender, CLV, Tenure_Age_Ratio, Age_Group_MiddleAge, AgeGroup_Senior, AgeGroup_Elderly,is_active_member):


  input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': int(has_credit_card),
    'EstimatedSalary': estimated_salary,
    'Geography_France': 1 if location == 'France' else 0,
    'Geography_Germany': 1 if location == 'Germany' else 0,
    'Geography_Spain': 1 if location == 'Spain' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'Gender_Female': 1 if gender == 'Female' else 0,
    'CLV': 0,
    'TenureAgeRatio': 0,
    'AgeGroup_MiddleAge': 0,
    'AgeGroup_Senior': 0,
    'AgeGroup_Elderly': 0, 
    'IsActiveMember': int(is_active_member)
  }

  input_df = pd.DataFrame([input_dict])
  input_df = pd.get_dummies(input_df, drop_first=True)
  return input_df, input_dict



def make_predictions(input_df, input_dict):  
  
    probabilities = {
      'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
      
      'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
      'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],}

    avg_probability = np.mean(list(probabilities.values()))

    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
      st.write(f"{model}{prob}")
    st.write(f"Average Probability: {avg_probability}")

    return avg_probability

def explain_prediction(probability, input_dict, surname):
  prompt = f"""You are an expert data scientist at a bank, where you specialized in interpreting and explaining predictions of machine learning models.
  Your machine learning model has preidcted a customer name {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting churn:

                    Features | Importance
                NumofPrducts | 0.323888
              IsActiveMember | 0.164146
                Age          | 0.109550
            Geography_Germany| 0.091373
            Geography_France | 0.046463
            Geography_Spain  | 0.036855
                Credit_Score | 0.035005
            Estimated_Salary | 0.032655
                HasCrCard    | 0.031940
                Tenure       | 0.030054
                Gender_Male  | 0.000000


            {pd.set_option('display.max_columns', None)}

            Here are summary statistics for churned customers:
            {df[df['Exited'] == 1 ].describe()}

            Here are summary statistics for non-churned customers:
            {df[df['Exited'] == 0 ].describe()}


    - If the customer has over 40% risk of churning, generate a three sentence explanation of why they are at risk of churning. 
    - If the customer has less than 40% risk of churning, generate a three sentence explanation of why they are a lesser risk of churning. 
    - Your explanation should be based on the customer's informations, the statistics of churned and non-churned customers, and the features' importances. 

    """

  print("EXPLANATION REPORT", prompt)


  raw_response = client.chat.completions.create(model="llama-3.2-3b-preview", 
    messages=[{
      "role":"user",
      "content": prompt
    }
  ]
)


  return raw_response.choices[0].message.content

st.title("Customer Churn Prediction App")
df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split("-")[0])

  print("Selected Customer ID", selected_customer_id)

  selected_surname = selected_customer_option.split("-")[1]
  print("Surname",selected_surname)

  selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]
  print("Selected Customer", selected_customer)

  col1, col2 = st.columns(2)
  df['EstimatedSalary'] = pd.to_numeric(df['EstimatedSalary'], errors='coerce')

with col1:

  credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=int(selected_customer['CreditScore']))

  location = st.selectbox(
    "Location", ["Spain", "France", "Germany"],
    index=["Spain","France","Germany"].index(
      selected_customer['Geography']))

  gender = st.radio("Gender", ["Male", "Female"],
                   index=0 if selected_customer['Gender'] == 'Male'
                    else 1)
  
  age = st.number_input(
    "Age", 
    min_value=18,
    max_value=100,
    value=int(selected_customer['Age']))

  tenure = st.number_input(
    "Tenure (years)",
    min_value=0,
    max_value=50,
    value=int(selected_customer['Tenure']))
  
  with col2:

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=float(selected_customer['Balance']))
    
    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=10,
        value=int(selected_customer['NumOfProducts']))
    
    has_credit_card = st.checkbox(
        "Has Credit Card",
        value=bool(selected_customer['HasCrCard']))

    is_active_member = st.checkbox(
        "Is Active Member",
        value=bool(selected_customer['IsActiveMember']))

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer['EstimatedSalary']) if pd.notnull(selected_customer['EstimatedSalary']) else 0.0)
    
    CLV = 0

    Tenure_Age_Ratio = selected_customer['Tenure'] / selected_customer['Age'] if selected_customer['Age'] != 0 else 0
    
    Age_Group_MiddleAge = 1 if 35 <= selected_customer['Age'] < 50 else 0
    
    AgeGroup_Senior = 1 if 50 <= selected_customer['Age'] < 65 else 0
    
    AgeGroup_Elderly = 1 if selected_customer['Age'] >= 65 else 0


  

  input_df, input_dict = prepare_input(credit_score, age, tenure, balance, num_products, has_credit_card, estimated_salary, location, gender, CLV, Tenure_Age_Ratio, Age_Group_MiddleAge, AgeGroup_Senior, AgeGroup_Elderly,is_active_member)
  avg_probability = make_predictions(input_df, input_dict)
  
  explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

  st.markdown('---')

  st.subheader('Explanation of Prediction')

  st.markdown(explanation)

def generate_email(probability, input_dict, surname, explanation):
  prompt: f"""You are a manager at HS Bank. You are responsible for seeing that customers are retained at the bank and continue to have a long and pleasant experience with HS Bank. 
  
  You notice that a customer by the name {surname} has a {round(probability * 100, 1)}% probability of churning. 
  
  Here is the customer's information{input_dict}
  
  Here is some explanation on why they churned{explain_prediction}
  
  Generate an email for the customer to stay with the bank, in bullet point format, or with a list of incentives that they would have for not churning(based on their information). Do not include information on their churning prediction or the model used to collect this information."""

  raw_response = client.chat.completions.create(
    model="llama-3.2-3b-preview", messages=[{
    "role":"user",
    "content": "prompt"
}])

  print("\n\nEMAIL PROMPT", prompt)


  return raw_response.choices[0].message.content






