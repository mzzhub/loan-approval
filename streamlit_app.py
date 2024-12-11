import streamlit as st

st.title('💵 Loan Approval Check 🏦')

st.write('Fill the details...')

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/mzzhub/loan-approval/refs/heads/master/loan_data.csv")
df = df.drop(["loan_percent_income", "cb_person_cred_hist_length", "previous_loan_defaults_on_file"], axis = 1)

object_columns = df.select_dtypes(include='object').columns

df["loan_intent"] = df["loan_intent"].replace({"DEBTCONSOLIDATION" : "Debt consolidation", "HOMEIMPROVEMENT" : "Home improvement"})

for col in object_columns:
    df[col] = df[col].apply(lambda x : x.title())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

x = df.drop("loan_status", axis = 1)
y = df["loan_status"]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(x)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 11, weights = 'distance')
knn.fit(x, y)

# user inputs
age = int(st.slider("Age", 20, 114, 30))
gender = str(st.radio("Gender", ("Male", "Female")))
education = str(st.selectbox("Education", ("Bachelor", "Associate", "High School", "Master", "Doctorate")))
income = int(st.number_input("Annual Income", 8000, 7200000, 50000, 500))
emp_exp = int(st.number_input("Employment Experience in Years", 0, 125, 5))
home = str(st.radio("House Ownership", ("Rent", "Mortgage", "Own", "Other")))
amount = int(st.number_input("Requried Loan Amount", 500, 35000, 10000, 500))
purpose = str(st.selectbox("Purpose", ("Education", "Medical", "Venture", "Personal", "Debt Consolidation", "Home Improvement")))
rate = float(st.number_input("Intrest Rate", 5.50, 20.00, 6.00, 0.01))
score = int(st.number_input("Current Credit Score", 390, 850, 400))

user_input = {
    'person_age': [age],
    'person_gender': [gender],
    'person_education': [education],
    'person_income': [income],
    'person_emp_exp': [emp_exp],
    'person_home_ownership': [home],
    'loan_amnt': [amount],
    'loan_intent': [purpose],
    'loan_int_rate': [rate],
    'credit_score': [score]
}

input_df = pd.DataFrame(user_input)

st.write(input_df)
st.write(input_df.select_dtypes("object"))

