import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Car Data EDA Dashboard", layout="wide")

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cars.csv")
    return df

df = load_data()

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("📌 Dashboard Menu")
menu = st.sidebar.radio(
    "Go to Section",
    ["Introduction", "Car Data Analysis", "EDA Dashboard", "Conclusion"]
)

st.sidebar.markdown("---")

# Filter Options
st.sidebar.subheader("🔍 Filter Options")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

selected_col = st.sidebar.selectbox("Select Column to Filter (Optional)", ["None"] + cat_cols)

if selected_col != "None":
    unique_values = df[selected_col].dropna().unique().tolist()
    selected_value = st.sidebar.selectbox(f"Select {selected_col}", unique_values)
    filtered_df = df[df[selected_col] == selected_value]
else:
    filtered_df = df.copy()

# ---------------------- INTRODUCTION ----------------------
if menu == "Introduction":
    st.title("🚗 Car Data Exploratory Data Analysis (EDA) Dashboard")
    st.markdown("## ✅ Introduction")

    st.write("""
    Welcome to the **Car Data EDA Dashboard**!  
    This project is designed to perform **complete Exploratory Data Analysis (EDA)** on a car dataset using **Python + Streamlit**.

    ###  Main Objective
    The main aim of this project is to:
    - Understand the overall structure of the dataset  
    - Identify important patterns and trends  
    - Detect missing values and duplicate records  
    - Study the statistical summary of numeric features  
    - Analyze relationships between variables using visualizations  

    ###  Why EDA is Important?
    Exploratory Data Analysis helps us to:
    ✅ clean and prepare data for Machine Learning  
    ✅ understand feature distributions  
    ✅ identify outliers and unusual values  
    ✅ compare categories like brands, models, fuel types, etc.  
    ✅ discover correlations between different car attributes  

    ###  Tools & Libraries Used
    This dashboard is created using:
    - **Streamlit** (Web App / Dashboard)
    - **Pandas** (Data Handling)
    - **NumPy** (Numeric Operations)
    - **Matplotlib & Seaborn** (Data Visualization)

    ###  What You Can Do in This Dashboard?
    You can explore the dataset using the following sections:

    ✅ **Car Data Analysis**  
    - Data types  
    - Missing values  
    - Duplicate values  
    - Summary statistics  
    - Correlation matrix  

    ✅ **EDA Dashboard**  
    - Histogram  
    - Boxplot  
    - Bar chart  
    - Scatter plot  
    - Interactive filtering from sidebar  

    ###  Tip
    Use the **sidebar filters** to explore a specific category and perform analysis on filtered data.
    """)

    st.markdown("---")
    st.markdown("###  Dataset Preview (Top 10 Rows)")
    st.dataframe(filtered_df.head(10))

    st.markdown("###  Dataset Size")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", filtered_df.shape[0])
    col2.metric("Total Columns", filtered_df.shape[1])
    col3.metric("Numeric Columns", len(numeric_cols))

# ---------------------- CAR DATA ANALYSIS ----------------------
elif menu == "Car Data Analysis":
    st.title("📊 Car Data Analysis")

    st.subheader("✅ Dataset Information")
    st.write("### Columns in Dataset:")
    st.write(filtered_df.columns)

    st.subheader("✅ Data Types")
    st.dataframe(filtered_df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}))

    st.subheader("✅ Missing Values")
    missing = filtered_df.isnull().sum()
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["Column", "Missing Values"]

    if missing_df.empty:
        st.success("✅ No missing values found!")
    else:
        st.warning("⚠️ Missing values found!")
        st.dataframe(missing_df)

    st.subheader("✅ Duplicate Values")
    duplicates = filtered_df.duplicated().sum()
    if duplicates == 0:
        st.success("✅ No duplicate values found!")
    else:
        st.warning(f" Duplicate rows found: {duplicates}")

    st.subheader("✅ Statistical Summary")
    st.dataframe(filtered_df.describe())

    st.subheader("✅ Correlation Matrix")
    if len(numeric_cols) > 1:
        corr = filtered_df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation.")

# ---------------------- EDA DASHBOARD ----------------------
elif menu == "EDA Dashboard":
    st.title("📈 EDA Dashboard (Visualizations)")

    st.markdown("### ✅ Select Features for Analysis")

    col1, col2 = st.columns(2)

    with col1:
        num_col = st.selectbox("Select Numeric Column", numeric_cols)

    with col2:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Boxplot", "Bar Chart", "Scatter Plot"]
        )

    st.markdown("---")

    # Histogram
    if chart_type == "Histogram":
        st.subheader(f"📌 Histogram of {num_col}")
        fig, ax = plt.subplots()
        ax.hist(filtered_df[num_col].dropna(), bins=20)
        ax.set_xlabel(num_col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Boxplot
    elif chart_type == "Boxplot":
        st.subheader(f"📌 Boxplot of {num_col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=filtered_df[num_col], ax=ax)
        st.pyplot(fig)

    # Bar Chart
    elif chart_type == "Bar Chart":
        if len(cat_cols) > 0:
            cat_col = st.selectbox("Select Categorical Column", cat_cols)
            st.subheader(f"📌 Bar Chart: {cat_col}")

            value_counts = filtered_df[cat_col].value_counts().head(10)

            fig, ax = plt.subplots()
            ax.bar(value_counts.index.astype(str), value_counts.values)
            ax.set_xlabel(cat_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No categorical columns found for bar chart!")

    # Scatter Plot
    elif chart_type == "Scatter Plot":
        if len(numeric_cols) > 1:
            x_col = st.selectbox("Select X axis", numeric_cols, index=0)
            y_col = st.selectbox("Select Y axis", numeric_cols, index=1)

            st.subheader(f"📌 Scatter Plot: {x_col} vs {y_col}")

            fig, ax = plt.subplots()
            ax.scatter(filtered_df[x_col], filtered_df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for scatter plot!")

    st.markdown("---")
    st.subheader("✅ Top 10 Rows After Filter")
    st.dataframe(filtered_df.head(10))

# ---------------------- CONCLUSION ----------------------
elif menu == "Conclusion":
    st.title("✅ Conclusion")
    st.markdown("## 📌 Final Summary & Insights")

    st.write("""
    This **Car Data EDA Dashboard** successfully demonstrates a complete exploratory analysis process.
    The purpose of this project was to **understand the dataset clearly** and create a **data-driven dashboard** for analysis.

    ### ✅ Work Completed in This Project
    ✔ Loaded and explored the dataset  
    ✔ Checked the dataset shape, columns, and data types  
    ✔ Identified missing values and duplicates (if any)  
    ✔ Generated statistical summary using descriptive statistics  
    ✔ Built a correlation matrix to understand relationships between numeric variables  
    ✔ Created multiple visualizations for deeper understanding  
    ✔ Added interactive filters to explore specific categories easily  

    ### 📊 Key Learning Outcomes
    After performing EDA, we can:
    ✅ decide which columns/features are useful  
    ✅ detect outliers and handle them properly  
    ✅ understand which features are correlated  
    ✅ plan data cleaning and preprocessing for Machine Learning  

    ### 🚀 Next Step (Future Improvements)
    In future, we can improve this project by adding:
    - ✅ Advanced feature engineering  
    - ✅ Handling missing values using imputation methods  
    - ✅ Machine Learning models (Price Prediction / Mileage Prediction)  
    - ✅ Interactive Plotly charts for better UI  
    - ✅ Export filtered dataset option  

    ### 🎉 Final Message
    This dashboard provides a **clear, visual, and interactive** way to analyze car datasets and is a strong base for any **Data Science / ML project**.
    """)

    st.success("🎉 Thank You for Using the Car Data EDA Dashboard!")
