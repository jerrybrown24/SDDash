import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("student_dropout_dataset.csv")
    return df

df = load_data()

# Title
st.title("🎓 Student Dropout Analysis Dashboard")

st.markdown("""
Welcome to the all-in-one **Student Dropout Insights** dashboard. 
Here, Directors, Deans, HRs and stakeholders can explore macro and micro level data insights 
to understand dropout trends, risks, and improvement areas.
""")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Demographics",
    "Academic Performance",
    "Engagement",
    "Dropout Analysis"
])

# Overview Tab
with tab1:
    st.header("📊 Dataset Overview")
    st.markdown("Here's a snapshot of the dataset used for analysis.")
    st.dataframe(df.head(20))

    st.markdown("#### Distribution of Dropouts")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='dropout_flag', palette='Set2', ax=ax1)
    ax1.set_xticklabels(['Not Dropped Out', 'Dropped Out'])
    st.pyplot(fig1)

    st.markdown("Dropout Score vs Dropout Flag")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x='dropout_score', hue='dropout_flag', kde=True, ax=ax2)
    st.pyplot(fig2)

# Demographics Tab
with tab2:
    st.header("👥 Demographics")
    gender = st.selectbox("Filter by Gender", ['All'] + df['gender'].unique().tolist())
    region = st.selectbox("Filter by Region", ['All'] + df['region'].astype(str).unique().tolist())

    filtered_df = df.copy()
    if gender != 'All':
        filtered_df = filtered_df[filtered_df['gender'] == gender]
    if region != 'All':
        filtered_df = filtered_df[filtered_df['region'] == int(region)]

    st.markdown("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['age'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("### Dropout Flag by Gender")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='gender', hue='dropout_flag', palette='pastel', ax=ax)
    st.pyplot(fig)

# Academic Tab
with tab3:
    st.header("📚 Academic Performance")

    st.markdown("### GPA Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='GPA', ax=ax)
    ax.set_xticklabels(['Not Dropped Out', 'Dropped Out'])
    st.pyplot(fig)

    st.markdown("### High School Score vs Dropout Score")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='high_school_score', y='dropout_score', hue='dropout_flag', ax=ax)
    st.pyplot(fig)

    st.markdown("### Failures Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['failures'], kde=True, bins=10, ax=ax)
    st.pyplot(fig)

# Engagement Tab
with tab4:
    st.header("📈 Engagement Metrics")

    st.markdown("### Attendance Rate vs Dropout")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='attendance_rate', ax=ax)
    ax.set_xticklabels(['Not Dropped Out', 'Dropped Out'])
    st.pyplot(fig)

    st.markdown("### Assignment Submission Rate")
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x='assignment_submission_rate', hue='dropout_flag', ax=ax, fill=True)
    st.pyplot(fig)

    st.markdown("### Peer Interaction Score by Dropout")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='peer_interaction_score', ax=ax)
    st.pyplot(fig)

# Dropout Analysis Tab
with tab5:
    st.header("🚨 Dropout Risk Insights")

    st.markdown("### Dropout Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['dropout_score'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=True, ax=ax)
    st.pyplot(fig)

    st.markdown("### Dropout by Socioeconomic Status")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='socioeconomic_status', hue='dropout_flag', palette='muted', ax=ax)
    st.pyplot(fig)

    st.markdown("### Commute Time Impact")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='commute_time_minutes', ax=ax)
    st.pyplot(fig)
