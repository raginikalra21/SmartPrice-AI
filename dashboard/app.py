import streamlit as st
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="SmartPrice AI", layout="wide")

st.title("💡 SmartPrice AI")
st.subheader("Electronic Product Price Prediction & Listing Simulator")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("../dataset/cleaned_electronics_data.csv")

# ---------------- CLEAN NUMERIC FIELDS ----------------
df["discounted_price"] = df["discounted_price"].replace("₹", "", regex=True).replace(",", "", regex=True).astype(float)
df["actual_price"] = df["actual_price"].replace("₹", "", regex=True).replace(",", "", regex=True).astype(float)
df["discount_percentage"] = df["discount_percentage"].replace("%", "", regex=True).astype(float)
df["rating_count"] = df["rating_count"].replace(",", "", regex=True).astype(float)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

df = df.dropna()

# ---------------- FEATURE ENGINEERING ----------------
# Extract brand from product name (first word)
df["brand"] = df["product_name"].str.split().str[0]

features = ["category", "brand", "actual_price", "discount_percentage", "rating", "rating_count"]
target = "discounted_price"

df = df[features + [target]]

# ---------------- ENCODING ----------------
le_category = LabelEncoder()
le_brand = LabelEncoder()

df["category"] = le_category.fit_transform(df["category"])
df["brand"] = le_brand.fit_transform(df["brand"])

X = df[features]
y = df[target]

# ---------------- TRAIN MODEL ----------------
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X, y)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📦 Product Listing Details")

category_name = st.sidebar.selectbox(
    "Category",
    sorted(le_category.classes_)
)

brand_name = st.sidebar.selectbox(
    "Brand",
    sorted(le_brand.classes_)
)

actual_price = st.sidebar.number_input("Actual Price (₹)", min_value=100.0, step=100.0)
discount_percentage = st.sidebar.slider("Discount (%)", 0, 90, 10)
rating = st.sidebar.slider("Rating", 1.0, 5.0, 4.0)
rating_count = st.sidebar.number_input("Rating Count", min_value=1, step=10)

# ---------------- ENCODE INPUT ----------------
category_encoded = le_category.transform([category_name])[0]
brand_encoded = le_brand.transform([brand_name])[0]

input_data = np.array([[ 
    category_encoded, 
    brand_encoded, 
    actual_price, 
    discount_percentage, 
    rating, 
    rating_count
]])

# ---------------- PREDICTION ----------------
prediction = model.predict(input_data)[0]

st.success(f"💰 Predicted Selling Price: ₹ {round(prediction, 2)}")

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📊 Key Pricing Factors (Model Insight)")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

# ---------------- BUSINESS INSIGHT ----------------
st.info(
    "🧠 Insight: This AI model analyzes historical e-commerce pricing trends to recommend an optimal selling price. "
    "Adjust discount, brand, and ratings to simulate market impact in real time."
)
