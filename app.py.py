import streamlit as st
import pandas as pd
import numpy as np
import joblib
from itertools import product
from datetime import timedelta

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("Sales Forecast - BigMart Store")

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload sales CSV file (BigMart format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Step 2: Choose forecast duration
    forecast_days = st.selectbox("Select forecast duration", options=[30, 60], index=0)

    # Step 3: Generate future dates
    future_dates = pd.date_range(start=df["Date"].max() + timedelta(days=1), periods=forecast_days)
    sku_list = df["SKU"].unique()
    store_list = df["Store_ID"].unique()
    future_df = pd.DataFrame(list(product(sku_list, store_list, future_dates)), columns=["SKU", "Store_ID", "Date"])

    # Step 4: Add time features
    future_df["Month"] = future_df["Date"].dt.month
    future_df["DayOfWeek"] = future_df["Date"].dt.dayofweek
    future_df["Weekday"] = future_df["Date"].dt.day_name()

    # Step 5: Merge static info
    sku_info = df.drop_duplicates("SKU")[["SKU", "Product_Type", "Promotion", "Item_MRP"]]
    store_info = df.drop_duplicates("Store_ID")[["Store_ID", "Store_Type", "Region", "Season", "Holiday"]]
    future_df = future_df.merge(sku_info, on="SKU", how="left")
    future_df = future_df.merge(store_info, on="Store_ID", how="left")

    # Step 6: Load trained model
    try:
        model = joblib.load("xgb_forecasting_model.pkl")
    except:
        st.error("❌ Model file not found: xgb_forecasting_model.pkl")
        st.stop()

    # Step 7: Prepare features
    feature_cols = [col for col in df.columns if col not in ["Sales_Quantity", "Date"]]
    X_future = future_df[feature_cols]

    # Step 8: Predict
    preds = model.predict(X_future)
    future_df["Predicted_Sales"] = np.round(preds).astype(int)

    # Step 9: Output
    st.success(f"✅ Forecast complete for {forecast_days} days!")
    filename = f"sales_forecast_{forecast_days}days.csv"
    st.download_button("Download Forecast CSV",
                       data=future_df[["Date", "SKU", "Store_ID", "Predicted_Sales"]].to_csv(index=False).encode("utf-8"),
                       file_name=filename,
                       mime="text/csv")

    st.dataframe(future_df[["Date", "SKU", "Store_ID", "Predicted_Sales"]])

else:
    st.info("Please upload a CSV file to begin.")
