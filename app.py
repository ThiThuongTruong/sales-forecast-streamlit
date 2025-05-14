import streamlit as st
import pandas as pd
import numpy as np
import joblib
from itertools import product
from datetime import timedelta
import altair as alt

st.set_page_config(page_title="Sales Forecast - BigMart", layout="wide")
st.title("📊 Sales Forecast - BigMart Store - Project by Thuong and Doan")

uploaded_file = st.file_uploader("📁 Upload sales CSV file (BigMart format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    forecast_days = st.selectbox("⏳ Select forecast duration", [30, 60], index=0)
    future_dates = pd.date_range(start=df["Date"].max() + timedelta(days=1), periods=forecast_days)
    sku_list = df["SKU"].unique()
    store_list = df["Store_ID"].unique()
    future_df = pd.DataFrame(list(product(sku_list, store_list, future_dates)), columns=["SKU", "Store_ID", "Date"])
    future_df["Month"] = future_df["Date"].dt.month
    future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek
    future_df["Weekday"] = future_df["Date"].dt.day_name()

    sku_info = df.drop_duplicates("SKU")[["SKU", "Product_Type", "Promotion", "Item_MRP"]]
    store_info = df.drop_duplicates("Store_ID")[["Store_ID", "Store_Type", "Region", "Season", "Holiday"]]
    future_df = future_df.merge(sku_info, on="SKU", how="left")
    future_df = future_df.merge(store_info, on="Store_ID", how="left")

    try:
        model = joblib.load("xgb_forecasting_model.pkl")
    except:
        st.error("❌ Model file 'xgb_forecasting_model.pkl' not found.")
        st.stop()

    feature_cols = [col for col in df.columns if col not in ["Sales_Quantity", "Date"]]
    X_future = future_df[feature_cols]
    preds = model.predict(X_future)
    future_df["Predicted_Sales"] = np.round(preds).astype(int)

    st.success("✅ Forecast completed!")

    # Download
    st.download_button("⬇️ Download Forecast CSV", future_df[["Date", "SKU", "Store_ID", "Predicted_Sales"]].to_csv(index=False), file_name="forecast.csv")

    # Table preview
    st.dataframe(future_df[["Date", "SKU", "Store_ID", "Predicted_Sales"]])

    # Chart: Total predicted sales by date
    chart_total = future_df.groupby("Date")["Predicted_Sales"].sum().reset_index()
    total_chart = alt.Chart(chart_total).mark_line(point=True).encode(
        x=alt.X("Date:T"),
        y=alt.Y("Predicted_Sales:Q", scale=alt.Scale(zero=False))
    ).properties(title="📈 Total Predicted Sales Over Time")
    st.altair_chart(total_chart, use_container_width=True)

    # Chart: by SKU
    st.subheader("📦 Forecast by SKU")
    sku_options = sorted(future_df["SKU"].unique())
    selected_sku = st.selectbox("Select SKU", sku_options)
    sku_data = future_df[future_df["SKU"] == selected_sku].groupby("Date")["Predicted_Sales"].sum().reset_index()
    sku_chart = alt.Chart(sku_data).mark_line(point=True).encode(
        x=alt.X("Date:T"),
        y=alt.Y("Predicted_Sales:Q", scale=alt.Scale(zero=False))
    ).properties(title=f"📦 Predicted Sales for SKU: {selected_sku}")
    st.altair_chart(sku_chart, use_container_width=True)

    # Chart: by Store
    st.subheader("🏬 Forecast by Store")
    store_options = sorted(future_df["Store_ID"].unique())
    selected_store = st.selectbox("Select Store", store_options)
    store_data = future_df[future_df["Store_ID"] == selected_store].groupby("Date")["Predicted_Sales"].sum().reset_index()
    store_chart = alt.Chart(store_data).mark_line(point=True).encode(
        x=alt.X("Date:T"),
        y=alt.Y("Predicted_Sales:Q", scale=alt.Scale(zero=False))
    ).properties(title=f"🏬 Predicted Sales for Store: {selected_store}")
    st.altair_chart(store_chart, use_container_width=True)

else:
    st.info("📂 Please upload a CSV file to begin.")
