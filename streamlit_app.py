import streamlit as st

st.title('💵 Loan Eligibility Check 🏦')

st.write('Fill the details...')

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/mzzhub/loan-approval/refs/heads/master/loan_data.csv")
df = df.drop(["loan_percent_income", "cb_person_cred_hist_length", "previous_loan_defaults_on_file"], axis = 1)

object_columns = df.select_dtypes(include='object').columns

df["loan_intent"] = df["loan_intent"].replace({"DEBTCONSOLIDATION" : "Debt consolidation", "HOMEIMPROVEMENT" : "Home improvement"})

for col in object_columns:
    df[col] = df[col].apply(lambda x : x.title())

# replace object columns
df["person_gender"] = df["person_gender"].replace({"Male" : 0, "Female" : 1})
df["person_education"] = df["person_education"].replace({"Bachelor" : 0, "Associate" : 1, "High School" : 2, "Master" : 3, "Doctorate" : 4})
df["person_home_ownership"] = df["person_home_ownership"].replace({"Rent" : 0, "Mortgage" : 1, "Own" : 2, "Other" : 3})
df["loan_intent"] = df["loan_intent"].replace({"Education" :0, "Medical" : 1, "Venture" : 2, "Personal" : 3, "Debt Consolidation" : 4, "Home Improvement" : 5})

x = df.drop("loan_status", axis = 1)
y = df["loan_status"]

# from sklearn.preprocessing import StandardScaler
# ss01 = StandardScaler()
# x = ss01.fit_transform(x)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 11, weights = 'distance')
knn.fit(x, y)

# user inputs
age = int(st.slider("**Age**", 20, 114, 30))
gender = str(st.radio("**Gender**", ("Male", "Female")))
education = str(st.selectbox("**Education**", ("Bachelor", "Associate", "High School", "Master", "Doctorate")))
income = int(st.number_input("**Annual Income**", 8000, 7200000, 50000, 500))
emp_exp = int(st.number_input("**Employment Experience in Years**", 0, 125, 5))
home = str(st.radio("**House Ownership**", ("Rent", "Mortgage", "Own", "Other")))
amount = int(st.number_input("**Requried Loan Amount**", 500, 35000, 10000, 500))
purpose = str(st.selectbox("**Purpose**", ("Education", "Medical", "Venture", "Personal", "Debt Consolidation", "Home Improvement")))
rate = float(st.number_input("**Intrest Rate**", 5.50, 20.00, 6.00, 0.01))
score = int(st.number_input("**Current Credit Score**", 390, 850, 400))

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

input_df["person_gender"] = input_df["person_gender"].replace({"Male" : 0, "Female" : 1})
input_df["person_education"] = input_df["person_education"].replace({"Bachelor" : 0, "Associate" : 1, "High School" : 2, "Master" : 3, "Doctorate" : 4})
input_df["person_home_ownership"] = input_df["person_home_ownership"].replace({"Rent" : 0, "Mortgage" : 1, "Own" : 2, "Other" : 3})
input_df["loan_intent"] = input_df["loan_intent"].replace({"Education" :0, "Medical" : 1, "Venture" : 2, "Personal" : 3, "Debt Consolidation" : 4, "Home Improvement" : 5})

row_array = input_df.iloc[0].to_numpy().reshape(1, -1)

pred = knn.predict(row_array)

prob = knn.predict_proba(row_array)
prob_df = pd.DataFrame(prob, columns = ["Eligible", "Ineligible"])
prob_df = prob_df * 100

st.subheader("Prediction")

# st.write(prob_df)
import numpy as np
possible_output = np.array(["Eligible", "Ineligible"])

# st.success(possible_output[pred][0])

if pred[0] == 0:  # Assuming 0 corresponds to "Eligible"
    st.success(possible_output[pred][0])  # Green background
else:  # Assuming 1 corresponds to "Ineligible"
    st.error(possible_output[pred][0])  # Red background

st.dataframe(prob_df, column_config = {
                                        "Eligible" : st.column_config.ProgressColumn(
                                                                                        "Eligible",
                                                                                        format = "%.1f%%",
                                                                                        width = "medium",
                                                                                        min_value = 0,
                                                                                        max_value = 100
                                                                                    
                                                                                    ),
                                        "Ineligible" : st.column_config.ProgressColumn(
                                                                                        "Ineligible",
                                                                                        format = "%.1f%%",
                                                                                        width = "medium",
                                                                                        min_value = 0,
                                                                                        max_value = 100
                                                                                    )
                                    }, hide_index = True)



chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.area_chart(chart_data)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the Random Forest model
rf_gscv = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],       # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],      # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],      # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],        # Minimum samples required to be at a leaf node
    'bootstrap': [True, False]            # Whether bootstrap samples are used
}

# Define the GridSearchCV
gscv = GridSearchCV(estimator=rf_gscv, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the grid search to the data
gscv.fit(x, y)

# Print the best parameters and best score
st.write("Best Parameters:", gscv.best_params_)
st.write("Best Accuracy:", gscv.best_score_)
