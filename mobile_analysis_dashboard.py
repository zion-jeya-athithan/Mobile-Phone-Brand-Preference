import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# Set page config
st.set_page_config(page_title="Mobile Model Analysis Dashboard", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    return pd.read_excel('modified_dataset.xlsx')

# Load the dataset
df = load_data()

# Sidebar
st.sidebar.header("Dashboard Controls")
analysis_type = st.sidebar.radio("Choose Analysis Type", ["Dataset Summary", "Clustering", "Association Rules"])

# Main content
st.title("Mobile Model Analysis Dashboard")

if analysis_type == "Dataset Summary":
    st.header("Dataset Summary")
    
    # Display basic information about the dataset
    st.subheader("Dataset Info")
    st.write(f"Number of records: {len(df)}")
    st.write(f"Number of features: {len(df.columns)}")
    
    # Display column names and types
    st.subheader("Column Information")
    st.write(df.dtypes)
    
    # Display statistical summary of numerical columns
    st.subheader("Statistical Summary")
    st.write(df.describe())
    
    # Display first few rows of the dataset
    st.subheader("Sample Data")
    st.write(df.head())

elif analysis_type == "Clustering":
    st.header("Clustering Analysis")

    # Cluster creation
    if 'Cluster' not in df.columns:
        st.write("Creating clusters based on available features.")
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_columns) == 0:
            st.error("No numerical columns found for clustering. Please provide numerical features or a 'Cluster' column.")
        else:
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(df[numeric_columns])
            kmeans = KMeans(n_clusters=2, random_state=42)
            df['Cluster'] = kmeans.fit_predict(normalized_features)
            st.write(f"Clusters created based on the following features: {', '.join(numeric_columns)}")
    else:
        st.write("Using existing 'Cluster' column.")

    # Ensure we have exactly two clusters
    df['Cluster'] = df['Cluster'].astype(int) % 2

    # Separate data for Cluster 0 and Cluster 1
    cluster_0_data = df[df['Cluster'] == 0]
    cluster_1_data = df[df['Cluster'] == 1]

    # Count occurrences of each mobile brand for both clusters
    cluster_0_counts = cluster_0_data['Mobile Model'].value_counts().reset_index()
    cluster_0_counts.columns = ['Mobile Model', 'Count']
    cluster_1_counts = cluster_1_data['Mobile Model'].value_counts().reset_index()
    cluster_1_counts.columns = ['Mobile Model', 'Count']

    # Create two columns for the plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster 1")
        fig_cluster_0 = px.bar(
            cluster_0_counts,
            x='Mobile Model',
            y='Count',
            title='Count of Mobile Models in Cluster 1',
            labels={'Mobile Model': 'Mobile Model', 'Count': 'Count of Instances'},
            hover_data={'Count': True},
            text='Count'
        )
        fig_cluster_0.update_traces(marker_color='blue', textposition='outside')
        st.plotly_chart(fig_cluster_0, use_container_width=True)

    with col2:
        st.subheader("Cluster 2")
        fig_cluster_1 = px.bar(
            cluster_1_counts,
            x='Mobile Model',
            y='Count',
            title='Count of Mobile Models in Cluster 2',
            labels={'Mobile Model': 'Mobile Model', 'Count': 'Count of Instances'},
            hover_data={'Count': True},
            text='Count'
        )
        fig_cluster_1.update_traces(marker_color='yellow', textposition='outside')
        st.plotly_chart(fig_cluster_1, use_container_width=True)

    # Print summary of clusters
    st.subheader("Cluster Distribution")
    st.write(df['Cluster'].value_counts())

elif analysis_type == "Association Rules":
    st.header("Association Rules Analysis")

    # One-hot encode the 'Mobile Model' and 'Preferred Mobile Brand' columns
    df_onehot = pd.get_dummies(df[['Mobile Model', 'Preferred Mobile Brand']])

    # Sidebar controls for Apriori algorithm
    min_support = st.sidebar.slider("Minimum Support", 0.01, 0.05)
    min_threshold = st.sidebar.slider("Minimum Threshold (Lift)", 1.0, 1.30)

    # Apply the Apriori algorithm to get frequent itemsets
    frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)

    # Generate the association rules based on lift and confidence
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)

    # Check if rules were generated
    if not rules.empty:
        # Convert frozenset to string for easier plotting
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

        # Create a scatter plot using Plotly for the generated rules
        fig = px.scatter(
            rules,
            x='support',
            y='lift',
            color='confidence',
            size='confidence',
            hover_name='antecedents',
            hover_data=['consequents', 'support', 'lift'],
            title='Association Rules: Support vs. Lift',
            labels={'support': 'Support', 'lift': 'Lift'},
            color_continuous_scale=px.colors.sequential.Viridis
        )

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title='Support',
            yaxis_title='Lift',
            legend_title='Confidence',
            height=600,
        )

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

        # Display summary metrics for accuracy
        st.subheader("Summary of Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        st.warning("No association rules generated. Try adjusting the minimum support and lift thresholds.")

# Add some information about the dataset
st.sidebar.subheader("Dataset Information")
st.sidebar.write(f"Number of records: {len(df)}")
st.sidebar.write(f"Number of features: {len(df.columns)}")