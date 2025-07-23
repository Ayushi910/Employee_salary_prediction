import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders (2).pkl")  # Optional for decoding if needed

# Manual categories based on training
manual_classes = {
    'workclass': ['Private', 'Local-gov', 'Self-emp-not-inc', 'Federal-gov', 'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
    'education': ['11th', 'HS-grad', 'Assoc-acdm', 'Some-college', '10th', 'Prof-school', '7th-8th', 'Bachelors', 'Masters', 'Doctorate', '5th-6th', 'Assoc-voc', '9th', '12th', '1st-4th', 'Preschool'],
    'marital-status': ['Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced', 'Separated', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical', 'Exec-managerial', 'Tech-support', 'Sales', 'Priv-house-serv', 'Transport-moving', 'Handlers-cleaners', 'Armed-Forces'],
    'relationship': ['Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife', 'Other-relative'],
    'race': ['Black', 'White', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo'],
    'gender': ['Male', 'Female'],
    'native-country': ['United-States', '?', 'Peru', 'Guatemala', 'Mexico', 'Dominican-Republic', 'Ireland', 'Germany', 'Philippines', 'Thailand', 'Haiti', 'El-Salvador', 'Puerto-Rico', 'Vietnam', 'South', 'Columbia', 'Japan', 'India', 'Cambodia', 'Poland', 'Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal', 'China', 'Nicaragua', 'Honduras', 'Iran', 'Scotland', 'Jamaica', 'Ecuador', 'Yugoslavia', 'Hungary', 'Hong', 'Greece', 'Trinadad&Tobago', 'Outlying-US(Guam-USVI-etc)', 'France', 'Holand-Netherlands']
}

# Main title and instructions
st.title("💼 Income Level Prediction App")
st.write("Use the sidebar to input your information and predict whether the income is >50K or <=50K.")

# Sidebar inputs
st.sidebar.header("🔧 Input Parameters")

user_input = {}
for col in manual_classes:
    user_input[col] = st.sidebar.selectbox(f"{col.replace('-', ' ').capitalize()}", manual_classes[col])

# Numeric inputs in sidebar
age = st.sidebar.number_input("Age", min_value=17, max_value=100, step=1)
hours_per_week = st.sidebar.number_input("Hours per Week", min_value=1, max_value=100, step=1)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, step=1)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=10000, step=1)
education_num = st.sidebar.number_input("Education Number (education-num)", min_value=1, max_value=20, step=1)

# Prediction button in main area
if st.button("🚀 Predict Income Level"):
    try:
        input_data = {}

        # Encode categorical inputs
        for col in manual_classes:
            selected_val = user_input[col]
            input_data[col] = manual_classes[col].index(selected_val)

        # Assign numerical values
        input_data["age"] = age
        input_data["capital-gain"] = capital_gain
        input_data["capital-loss"] = capital_loss
        input_data["hours-per-week"] = hours_per_week
        input_data["educational-num"] = education_num

        # Prepare DataFrame
        input_df = pd.DataFrame([{
            "age": input_data["age"],
            "workclass": input_data["workclass"],
            "educational-num": input_data["educational-num"],
            "marital-status": input_data["marital-status"],
            "occupation": input_data["occupation"],
            "relationship": input_data["relationship"],
            "race": input_data["race"],
            "gender": input_data["gender"],
            "capital-gain": input_data["capital-gain"],
            "capital-loss": input_data["capital-loss"],
            "hours-per-week": input_data["hours-per-week"],
            "native-country": input_data["native-country"]
        }])

        # Predict
        prediction = model.predict(input_df)
        predicted_label = ">50K" if prediction[0] == 1 else "<=50K"

        # Show result
        st.success(f"🎯 Predicted Income: **{predicted_label}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
