import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title with custom styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            color: #2c3e50;
        }
    </style>
    <h1 class='title'>Interactive Data Analysis Dashboard</h1>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Settings")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your data", type=["csv"])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # Display basic information
    col1, col2 = st.columns(2)
    with col1:
        st.header("📊 Data Overview")
        st.write("Shape of the dataset:", df.shape)
        st.write("Columns:", ", ".join(df.columns))
    with col2:
        st.header("📈 Quick Stats")
        st.write("Missing values:", df.isnull().sum().sum())
        st.write("Numeric columns:", len(df.select_dtypes(include=['float64', 'int64']).columns))
    
    # Data Preview with toggle
    if st.checkbox("Show Raw Data Preview"):
        st.dataframe(df.head())
    
    # Visualization Section
    st.header("🎨 Data Visualization")
    
    # Visualization Type Selector
    viz_type = st.selectbox(
        "Choose Visualization Type",
        ["Distribution Plots", "Correlation Analysis", "Scatter Plots", "Box Plots", "Time Series Analysis"]
    )
    
    # Get numeric and datetime columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    
    if viz_type == "Distribution Plots":
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Select column for histogram", numeric_cols)
            fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            selected_col_kde = st.selectbox("Select column for KDE plot", numeric_cols)
            fig = px.violin(df, y=selected_col_kde, box=True, title=f'Distribution of {selected_col_kde}')
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Analysis":
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix,
                          title="Correlation Heatmap",
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            st.subheader("Top 5 Correlations")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i,j]
                    })
            corr_df = pd.DataFrame(corr_pairs)
            st.dataframe(corr_df.nlargest(5, 'Correlation'))
    
    elif viz_type == "Scatter Plots":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X axis", numeric_cols)
        with col2:
            y_col = st.selectbox("Select Y axis", numeric_cols)
        
        color_col = st.selectbox("Select color variable (optional)", ["None"] + list(df.columns))
        if color_col == "None":
            fig = px.scatter(df, x=x_col, y=y_col, title=f'{x_col} vs {y_col}')
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f'{x_col} vs {y_col} by {color_col}')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plots":
        numeric_col = st.selectbox("Select numeric column", numeric_cols)
        category_cols = df.select_dtypes(include=['object']).columns
        if len(category_cols) > 0:
            category_col = st.selectbox("Select category column", category_cols)
            fig = px.box(df, x=category_col, y=numeric_col, title=f'Box Plot of {numeric_col} by {category_col}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns found for box plots")
    
    elif viz_type == "Time Series Analysis":
        # Check if any datetime columns exist
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            # Try to convert object columns to datetime
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols = date_cols.append(pd.Index([col]))
                except:
                    continue
        
        if len(date_cols) > 0:
            date_col = st.selectbox("Select date column", date_cols)
            value_col = st.selectbox("Select value column", numeric_cols)
            
            # Resample data by selected frequency
            freq = st.selectbox("Select time frequency", ["Day", "Week", "Month", "Year"])
            freq_map = {"Day": "D", "Week": "W", "Month": "M", "Year": "Y"}
            
            df_time = df.set_index(date_col)
            df_time = df_time[value_col].resample(freq_map[freq]).mean()
            
            fig = px.line(df_time, title=f'Time Series of {value_col} by {freq}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No datetime columns found for time series analysis")
    
    # Download processed data
    if st.button("Download Processed Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a CSV file to begin analysis")
    
# Add footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #f0f2f6;
        }
    </style>
    <div class='footer'>
        Made with ❤️ using Streamlit
    </div>
""", unsafe_allow_html=True) 