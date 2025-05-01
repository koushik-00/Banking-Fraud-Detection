
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🔍 Banking Fraud Detection Dashboard")

# Upload CSV
# uploaded_file = st.file_uploader("Upload Cleaned CSV File", type=["csv"])
uploaded_file = 'cleaned_data.csv'

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Basic preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()
    df['fraud_label_str'] = df['fraud_label'].map({0: 'Legit', 1: 'Fraud'})

    # Show Data
    st.subheader("🧾 Dataset Preview")
    st.dataframe(df.head())

    # Fraud Summary
    st.subheader("📊 Fraud Distribution")
    fraud_counts = df['fraud_label_str'].value_counts().reset_index()
    fraud_counts.columns = ['Label', 'Count']
    fig = px.pie(fraud_counts, names='Label', values='Count', title='Fraud vs Legit Transactions')
    st.plotly_chart(fig)

    # Fraud by Hour
    st.subheader("⏰ Fraud Rate by Hour")
    fraud_by_hour = df.groupby('hour')['fraud_label'].mean().reset_index()
    fig2 = px.bar(fraud_by_hour, x='hour', y='fraud_label', labels={'fraud_label': 'Fraud Rate'}, title='Fraud Rate by Hour')
    st.plotly_chart(fig2)

    # Fraud by Day
    st.subheader("📅 Fraud Rate by Day of Week")
    fraud_by_day = df.groupby('day')['fraud_label'].mean().reset_index()
    fig3 = px.bar(fraud_by_day, x='day', y='fraud_label', labels={'fraud_label': 'Fraud Rate'}, title='Fraud Rate by Day')
    st.plotly_chart(fig3)

    # Categorical Features
    st.subheader("📈 Fraud Rate by Categorical Features")
    cat_feature = st.selectbox("Select Category", ['device_type', 'card_type'])
    fraud_cat = df.groupby(cat_feature)['fraud_label'].mean().sort_values(ascending=False).reset_index()
    fig4 = px.bar(fraud_cat, x=cat_feature, y='fraud_label', labels={'fraud_label': 'Fraud Rate'}, title=f'Fraud Rate by {cat_feature.title()}')
    st.plotly_chart(fig4)


    # Box
    st.title("Fraud Detection Dashboard")
    

    # Create Seaborn boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='fraud_label_str', y='transaction_amount', ax=ax)
    ax.set_title("Transaction Amount by Fraud Label")

    # Show plot in Streamlit
    st.pyplot(fig)

    st.markdown("We didnt like this particular box plot. Too many points are overlapping and we also have extreme outliners. So we figured out that maybe violin + box plot might help solve this issue.")

    # Violin
    st.title("Fraud Detection Dashboard")
    # st.markdown("### Violin Plot: Transaction Amount Distribution by Fraud Label")

    # Create violin plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=df,
        x='fraud_label_str',
        y='transaction_amount',
        palette=['skyblue', 'salmon'],
        inner='box',
        ax=ax
    )
    ax.set_title("Transaction Amount Distribution by Fraud Label")
    ax.set_xlabel("Fraud Label")
    ax.set_ylabel("Transaction Amount")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Display plot in Streamlit
    st.pyplot(fig)

    # st.markdown('')


    # Correlation Heatmap
    st.subheader("🔍 Feature Correlation Heatmap")
    num_cols = df.select_dtypes(include='number')
    corr = num_cols.corr()
    fig5, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig5)

    #hist
    # df = pd.read_csv("your_dataset.csv")  # Replace with your actual file path

    # Streamlit title and description
    # st.title("Fraud Detection Dashboard")
    # st.markdown("### Hourly Fraud Trend")
    # st.markdown("This histogram shows the distribution of fraudulent and non-fraudulent transactions across the hours of the day.")

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='hour', hue='fraud_label_str', bins=24, multiple='stack', ax=ax)
    ax.set_title("Transaction Hour vs Fraud")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Transactions")

    # Display in Streamlit
    st.pyplot(fig)

    # Sankey diagram
    st.subheader("🔗 Sankey Diagram: Device → Card → Fraud")
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()
    df['fraud_label_str'] = df['fraud_label'].map({0: 'Legit', 1: 'Fraud'})

    # Create Sankey data
    device_types = list(df['device_type'].unique())
    card_types = list(df['card_type'].unique())
    fraud_types = ['Legit', 'Fraud']

    labels = device_types + card_types + fraud_types

    source = []
    target = []
    value = []

    # Device → Card
    for d in device_types:
        for c in card_types:
            count = len(df[(df['device_type'] == d) & (df['card_type'] == c)])
            if count > 0:
                source.append(labels.index(d))
                target.append(labels.index(c))
                value.append(count)

    # Card → Fraud
    for c in card_types:
        for f in fraud_types:
            count = len(df[(df['card_type'] == c) & (df['fraud_label_str'] == f)])
            if count > 0:
                source.append(labels.index(c))
                target.append(labels.index(f))
                value.append(count)

    # Build Sankey diagram
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    sankey_fig.update_layout(title_text="Flow from Device → Card → Fraud", font_size=12)
    st.plotly_chart(sankey_fig, use_container_width=True)

    st.title("Risk Score Analysis")
# st.markdown("### Top Features Correlated with Risk Score")
# st.markdown("This chart displays the features most strongly associated with the `risk_score` in descending order of correlation.")

    # Calculate correlations
    numeric = df.select_dtypes(include=np.number)
    correlation = numeric.corr()['risk_score'].sort_values(ascending=False)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=correlation.values[1:10],
        y=correlation.index[1:10],
        palette='viridis',
        ax=ax
    )
    ax.set_title("Top Features Correlated with Risk Score")
    ax.set_xlabel("Correlation")
    plt.tight_layout()

    # Display plot
    st.pyplot(fig)


    # st.title("PCA Visualization of Transaction Behavior")
    # # st.markdown("""
    # # This 2D PCA projection visualizes high-dimensional transaction behavior data,
    # # colored by fraud labels to show how patterns separate or overlap in reduced space.
    # # """)

    # # Define features to include in PCA
    # features = ['transaction_amount', 'account_balance', 'daily_transaction_count',
    #             'avg_transaction_amount_7d', 'transaction_distance', 'risk_score']

    # # Drop rows with missing values in these features
    # df_clean = df.dropna(subset=features + ['fraud_label_str']).copy()

    # # Scale the features
    # scaled = StandardScaler().fit_transform(df_clean[features])

    # # Perform PCA
    # pca = PCA(n_components=2)
    # components = pca.fit_transform(scaled)

    # # Add PCA results back to DataFrame
    # df_clean['PC1'] = components[:, 0]
    # df_clean['PC2'] = components[:, 1]

    # # Plot using Seaborn
    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.scatterplot(data=df_clean, x='PC1', y='PC2', hue='fraud_label_str', alpha=0.5, ax=ax)
    # ax.set_title("PCA Projection of Transaction Behavior")
    # ax.grid(True)

    # # Display in Streamlit
    # st.pyplot(fig)

    st.title("Time-Based Fraud Heatmap")
    # st.markdown("""
    # This heatmap shows the number of fraud cases by **day of the week** and **hour of the day**.
    # It helps visualize temporal patterns in fraudulent behavior.
    # """)

    # Group and pivot data
    time_df = df.groupby(['day', 'hour'])['fraud_label'].sum().reset_index()
    time_pivot = time_df.pivot(index='day', columns='hour', values='fraud_label')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(time_pivot, cmap='YlOrRd', linewidths=0.5, annot=True, fmt='.0f', ax=ax)
    ax.set_title("Fraud Count by Day and Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")

    # Display in Streamlit
    st.pyplot(fig)

    # Boxplot for Amount
    # st.subheader("💵 Transaction Amount Distribution")
    # fig6 = px.box(df, x='fraud_label_str', y='amount', title='Amount Distribution by Fraud Label')
    # st.plotly_chart(fig6)

else:
    st.info("Please upload a cleaned CSV file to begin.")
