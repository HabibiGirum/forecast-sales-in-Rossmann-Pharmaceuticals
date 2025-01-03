import streamlit as st
import pandas as pd
from scripts.task_1 import EDA

# Streamlit UI
st.title("Retail Data Analysis Dashboard 📊")

# Sidebar for Dataset Upload
st.sidebar.title("Upload Your Datasets")

uploaded_train = st.sidebar.file_uploader("Upload Train Dataset (CSV)", type="csv")
uploaded_test = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type="csv")
uploaded_store = st.sidebar.file_uploader("Upload Store Dataset (CSV)", type="csv")

# Load Data
@st.cache_data
def load_default_data():
    sample_submission = pd.read_csv('data/sample_submission.csv')
    store = pd.read_csv('data/store.csv')
    test = pd.read_csv('data/test.csv')
    train = pd.read_csv('data/train.csv')
    return sample_submission, store, test, train

def load_uploaded_data(train_file, test_file, store_file):
    train = pd.read_csv(train_file) if train_file else pd.read_csv('data/train.csv')
    test = pd.read_csv(test_file) if test_file else pd.read_csv('data/test.csv')
    store = pd.read_csv(store_file) if store_file else pd.read_csv('data/store.csv')
    return train, test, store

# Load datasets based on user upload
if uploaded_train and uploaded_test and uploaded_store:
    train, test, store = load_uploaded_data(uploaded_train, uploaded_test, uploaded_store)
    st.sidebar.success("Custom datasets loaded successfully!")
else:
    sample_submission, store, test, train = load_default_data()
    st.sidebar.info("Using default datasets.")

# Initialize EDA
eda = EDA(train, test, store)
eda.merge_data()
eda.preprocess_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose Section",
    ["Data Overview", "Data Integrity", "Visualizations"]
)

# # Section 1: Data Overview
# if section == "Data Overview":
#     st.header("Data Overview")
#     st.write("### Train Data Sample:")
#     st.write(train.head())
#     st.write("### Store Data Sample:")
#     st.write(store.head())
#     st.write("### Test Data Sample:")
#     st.write(test.head())

# # Section 2: Data Integrity
# if section == "Data Integrity":
#     st.header("Data Integrity Check")
#     null_values, unique_values = eda.get_data_integrity()
    
#     st.subheader("Null Values in Train Data")
#     st.write(null_values)
    
#     st.subheader("Unique Values in Train Data")
#     st.write(unique_values)

# # Section 3: Visualizations
# if section == "Visualizations":
#     st.header("Data Visualizations")

#     # Sales Distribution
#     st.subheader("Sales Distribution")
#     st.pyplot(eda.plot_sales_distribution())

#     # Assortment Distribution
#     st.subheader("Assortment Distribution")
#     st.pyplot(eda.plot_assortment_distribution())

#     # Sales vs Customers
#     st.subheader("Sales vs Customers Scatter Plot")
#     st.pyplot(eda.plot_sales_vs_customers())

#     # Monthly Sales Trend
#     st.subheader("Monthly Sales Trend")
#     st.pyplot(eda.plot_monthly_sales_trend())

# Footer
st.sidebar.write("---")
st.sidebar.write("Developed by Habtamu Girum")
