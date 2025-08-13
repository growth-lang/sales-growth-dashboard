import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import altair as alt
from io import BytesIO

st.set_page_config(page_title="Sales Growth Dashboard", layout="wide")

st.title("📈 Sales Growth & Forecasting Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Raw Data", data.head())

    # Date formatting
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    # Metric calculations
    total_mrr = data['mrr'].sum()
    avg_conversion = (data['work_orders'].sum() / data['leads'].sum()) * 100

    col1, col2 = st.columns(2)
    col1.metric("Total MRR", f"${total_mrr:,.0f}")
    col2.metric("Avg. Conversion Rate", f"{avg_conversion:.2f}%")

    # Weekly trends
    weekly = data.groupby(pd.Grouper(key='date', freq='W')).sum().reset_index()
    chart = alt.Chart(weekly).mark_line().encode(
        x='date:T',
        y='mrr:Q',
        tooltip=['date:T', 'mrr:Q']
    ).properties(title="Weekly MRR Trend")
    st.altair_chart(chart, use_container_width=True)

    # Forecast
    st.subheader("📊 4-Week MRR Forecast")
    df_prophet = data[['date', 'mrr']].rename(columns={'date': 'ds', 'mrr': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=28)
    forecast = model.predict(future)

    forecast_chart = alt.Chart(forecast).mark_line().encode(
        x='ds:T',
        y='yhat:Q'
    ).properties(title="MRR Forecast (Prophet)")
    st.altair_chart(forecast_chart, use_container_width=True)

    # Bottleneck insights
    st.subheader("🔍 AI Insights (Simplified)")
    if avg_conversion < 20:
        st.warning("Conversion rate is low — review lead qualification process.")
    if weekly['mrr'].iloc[-1] < weekly['mrr'].iloc[-2]:
        st.warning("MRR has dropped this week — investigate possible causes.")

    # CSV Export
    st.download_button(
        label="Download Forecast CSV",
        data=forecast.to_csv(index=False),
        file_name="forecast.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file with columns: date, leads, work_orders, mrr.")
