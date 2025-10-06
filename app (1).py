import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib 
from statsmodels.tsa.holtwinters import ExponentialSmoothing 

st.set_page_config(layout="wide")

st.title("South African Crime Analysis Dashboard")


try:
    df = pd.read_csv('SACrimeStats_v2.csv')
    df2 = pd.read_csv('ProPopulation.csv')
except FileNotFoundError:
    st.error("Make sure 'SACrimeStats_v2.csv' and 'ProPopulation.csv' are in the same directory.")
    st.stop()

# --- Data Preparation (as done in the notebook) ---
crime = df.melt(
   id_vars = ["Province", "Station", "Category"],
    var_name = "Year",
    value_name =  "CrimeCount"
)
crime["Year"] = crime["Year"].str.split("-").str[0].astype(int)
crimeDF = crime.merge(df2, on="Province", how="left")
crimeDF["CrimeRate_per100k"] = (
    (crimeDF["CrimeCount"] / crimeDF["Population"]) * 100000
)
crimeDF = crimeDF.drop_duplicates()
masterDF = crimeDF.copy()

# --- Load the trained model and other necessary objects ---

try:

    threshold = masterDF["CrimeCount"].quantile(0.75)
    masterDF["Hotspot"] = (masterDF["CrimeCount"] >= threshold).astype(int)

    # Recreate encoded features for prediction if needed
    masterDF_encoded = pd.get_dummies(masterDF[["Province", "Category"]], drop_first=True)
    X = pd.concat([masterDF_encoded, masterDF[["CrimeCount", "Year", "Population", "CrimeRate_per100k"]]], axis=1)

    national_crime_ts = masterDF.groupby("Year")["CrimeCount"].sum().reset_index()
    national_crime_ts.columns = ["Year", "CrimeCount"]
    national_crime_ts = national_crime_ts.sort_values(by="Year")
    ts_model = ExponentialSmoothing(national_crime_ts["CrimeCount"], trend="add", seasonal=None)
    ts_fit = ts_model.fit()

except FileNotFoundError:
    st.warning("Model file not found. Some features might be unavailable. Please train and save the model in the notebook first.")

# --- Dashboard Layout ---

st.sidebar.header("Settings")
selected_province = st.sidebar.selectbox("Select Province", ["All"] + list(masterDF["Province"].unique()))
selected_category = st.sidebar.selectbox("Select Crime Category", ["All"] + list(masterDF["Category"].unique()))
forecast_steps = st.sidebar.slider("Years to Forecast", 1, 10, 5)

# --- Filtering Data ---
filtered_df = masterDF.copy()
if selected_province != "All":
    filtered_df = filtered_df[filtered_df["Province"] == selected_province]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

# --- Display Key Metrics ---
st.header("Key Crime Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Crime Count (Filtered)", int(filtered_df["CrimeCount"].sum()))
col2.metric("Average Crime Rate (Filtered)", f"{filtered_df['CrimeRate_per100k'].mean():.2f} per 100k")
col3.metric("Number of Hotspot Locations (Filtered)", int(filtered_df["Hotspot"].sum()))


# --- Visualizations ---

st.header("Crime Trends Over Time")
crime_trend = filtered_df.groupby("Year")["CrimeCount"].sum().reset_index()
fig_trend = px.line(crime_trend, x="Year", y="CrimeCount", title="Crime Count Over Time")
st.plotly_chart(fig_trend, use_container_width=True)

st.header("Crime Distribution by Category (Filtered)")
crime_by_category = filtered_df.groupby("Category")["CrimeCount"].sum().reset_index()
fig_category = px.bar(crime_by_category, x="Category", y="CrimeCount", title="Total Crime Count by Category")
st.plotly_chart(fig_category, use_container_width=True)

st.header("Total Crime Count by Province")
crime_by_province_plot = masterDF.groupby("Province")["CrimeCount"].sum().reset_index()
fig_province = px.bar(crime_by_province_plot, x="Province", y="CrimeCount", title="Total Crime Count by Province")
st.plotly_chart(fig_province, use_container_width=True)


# --- Hotspot Analysis ---
st.header("Crime Hotspot Locations")
st.write("Locations identified as potential crime hotspots (Top 25% of crime counts).")
hotspot_locations = filtered_df[filtered_df["Hotspot"] == 1]
if not hotspot_locations.empty:
    st.dataframe(hotspot_locations[["Province", "Station", "Category", "CrimeCount", "Year"]])
else:
    st.write("No hotspot locations found for the selected filters.")

# --- Time Series Forecast ---
st.header("National Crime Forecast")
if 'ts_fit' in locals(): # Check if the time series model was fitted
    last_year = national_crime_ts["Year"].max()
    forecast_years = list(range(last_year + 1, last_year + forecast_steps + 1))
    forecast_values = ts_fit.forecast(steps=forecast_steps)

    forecast_df = pd.DataFrame({
        "Year": forecast_years,
        "ForecastedCrimes": forecast_values
    })

    st.write(f"National Crime Forecast for the next {forecast_steps} years:")
    st.dataframe(forecast_df)

    # Plotting the forecast
    fig_forecast = px.line(national_crime_ts, x="Year", y="CrimeCount", title="National Crime Count and Forecast")
    fig_forecast.add_scatter(x=forecast_df["Year"], y=forecast_df["ForecastedCrimes"], mode='lines', name='Forecast')
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("Time series model not available. Please ensure the model is trained and loaded correctly.")
