import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

model = joblib.load('random_forest_model.joblib')

scaler = joblib.load('scaler.joblib')

le_state = joblib.load('le_state.joblib')
le_port = joblib.load('le_port.joblib')
le_commodity = joblib.load('le_commodity.joblib')

export_21 = pd.read_csv('exports.csv')

st.set_page_config(page_title="Export Quantity Predictor")

st.title("Export Quantity Predictor")

states = sorted(export_21['state_name'].unique())
years = sorted(export_21['year'].unique())

year = st.selectbox("Select Year", years)
state_name = st.selectbox("Select State", states)

@st.cache_data
def get_ports(state):
    return sorted(export_21[export_21['state_name'] == state]['port_of_export'].unique())

ports = get_ports(state_name)
port_of_export = st.selectbox("Select Port of Export", ports)

@st.cache_data
def get_commodities(state, port):
    return sorted(export_21[(export_21['state_name'] == state) & 
                            (export_21['port_of_export'] == port)]['general_principal_commodity_category'].unique())

commodities = get_commodities(state_name, port_of_export)
commodity_category = st.selectbox("Select Commodity Category", commodities)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'year': [year],
        'state_name': [state_name],
        'port_of_export': [port_of_export],
        'general_principal_commodity_category': [commodity_category]
    })

    try:
        input_data['state_name'] = le_state.transform([state_name])[0] if state_name in le_state.classes_ else -1
        input_data['port_of_export'] = le_port.transform([port_of_export])[0] if port_of_export in le_port.classes_ else -1
        input_data['general_principal_commodity_category'] = le_commodity.transform([commodity_category])[0] if commodity_category in le_commodity.classes_ else -1
    except Exception as e:
        st.error(f"Error during encoding: {e}")
        st.stop()

    input_scaled = scaler.transform(input_data)

    predicted_quantity = model.predict(input_scaled)[0]

    st.success(f"Predicted Quantity: {predicted_quantity:.2f}")

st.info("""
This app predicts the quantity of exports based on the year, state, port of export, and commodity category.
Select the options from the dropdowns and click 'Predict' to get the estimated export quantity.
""")

st.markdown("---")
st.markdown("Created by Jasbir Singh")