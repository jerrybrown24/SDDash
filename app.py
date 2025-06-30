import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Data load and preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv('student_dropout_dataset.csv')
    region_map = {1: 'Rural', 2: 'Suburban', 3: 'Urban'}
    df['region'] = df['region'].map(region_map)
    df['dropout_flag'] = df['dropout_flag'].map({1: 'Will Dropout', 0: 'Will Not Dropout'})
    return df

df = load_data()

# Sidebar Prediction
st.sidebar.header("📋 Student Dropout Prediction")
# Dropdown options
ages = sorted(df['age'].unique())
genders = sorted(df['gender'].unique())
socio_status = sorted(df['socioeconomic_status'].unique())
regions = ['Rural', 'Suburban', 'Urban']

# User Inputs
age = st.sidebar.selectbox("Age", ages)
gender = st.sidebar.selectbox("Gender", genders)
ses = st.sidebar.selectbox("Socioeconomic Status", socio_status)
region = st.sidebar.selectbox("Region", regions)
hsscore = st.sidebar.slider("High School Score", int(df['high_school_score'].min()), int(df['high_school_score'].max()))
gpa = st.sidebar.slider("GPA", float(df['GPA'].min()), float(df['GPA'].max()))
credits = st.sidebar.slider("Credits Completed", int(df['credits_completed'].min()), int(df['credits_completed'].max()))
failures = st.sidebar.slider("Failures", int(df['failures'].min()), int(df['failures'].max()))
attendance = st.sidebar.slider("Attendance Rate", 0.0, 100.0, float(df['attendance_rate'].mean()))
submission = st.sidebar.slider("Assignment Submission Rate", 0.0, 100.0, float(df['assignment_submission_rate'].mean()))
commute = st.sidebar.slider("Commute Time (minutes)", 0.0, 120.0, float(df['commute_time_minutes'].mean()))
peer_score = st.sidebar.slider("Peer Interaction Score", 0.0, 100.0, float(df['peer_interaction_score'].mean()))
duration = st.sidebar.slider("Program Duration (months)", int(df['program_duration_months'].min()), int(df['program_duration_months'].max()))

# Overview Tab
st.title("🎓 Student Dropout Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Demographics", "Academics", "Engagement", "Dropout Analysis", "ML Prediction"
])

with tab1:
    st.header("📊 Dataset Overview with Interactive Filters")
    # Sliders
    min_age, max_age = st.slider("Select Age Range", int(df['age'].min()), int(df['age'].max()), (18, 30))
    min_gpa, max_gpa = st.slider("Select GPA Range", float(df['GPA'].min()), float(df['GPA'].max()), (float(df['GPA'].min()), float(df['GPA'].max())))
    selected_region = st.selectbox("Select Region", ['All'] + regions)

    filtered = df[(df['age'] >= min_age) & (df['age'] <= max_age) & (df['GPA'] >= min_gpa) & (df['GPA'] <= max_gpa)]
    if selected_region != 'All':
        filtered = filtered[filtered['region'] == selected_region]

    st.dataframe(filtered.head(20))

    st.markdown("**Distribution of Dropouts**")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered, x='dropout_flag', palette='Set2', ax=ax1)
    st.pyplot(fig1)

    st.markdown("**Dropout Score Distribution**")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=filtered, x='dropout_score', hue='dropout_flag', kde=True, ax=ax2)
    st.pyplot(fig2)

with tab2:
    st.header("👥 Demographics")
    st.markdown("Explore how demographic factors relate to dropout rates.")
    st.markdown("**Dropout Rate by Gender**")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='gender', hue='dropout_flag', ax=ax)
    st.pyplot(fig)

    st.markdown("**Dropout Rate by Socioeconomic Status**")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='socioeconomic_status', hue='dropout_flag', ax=ax)
    st.pyplot(fig)

    st.markdown("**Dropout Rate by Region**")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='region', hue='dropout_flag', ax=ax)
    st.pyplot(fig)

with tab3:
    st.header("📚 Academic Performance")
    st.markdown("**GPA Distribution by Dropout Flag**")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='GPA', ax=ax)
    st.pyplot(fig)

    st.markdown("**High School Score vs Dropout Score**")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='high_school_score', y='dropout_score', hue='dropout_flag', ax=ax)
    st.pyplot(fig)

    st.markdown("**Failures Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df['failures'], kde=True, bins=10, ax=ax)
    st.pyplot(fig)

with tab4:
    st.header("📈 Engagement Metrics")
    st.markdown("**Attendance Rate by Dropout**")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='attendance_rate', ax=ax)
    st.pyplot(fig)

    st.markdown("**Assignment Submission Rate by Dropout**")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='assignment_submission_rate', ax=ax)
    st.pyplot(fig)

    st.markdown("**Peer Interaction by Dropout**")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='peer_interaction_score', ax=ax)
    st.pyplot(fig)

    st.markdown("**Commute Time by Dropout**")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='commute_time_minutes', ax=ax)
    st.pyplot(fig)

with tab5:
    st.header("🚨 Dropout Risk Analysis")
    st.markdown("**Dropout Score Distribution**")
    fig, ax = plt.subplots()
    sns.histplot(df['dropout_score'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), cmap='coolwarm', annot=True, ax=ax)
    st.pyplot(fig)

    st.markdown("**Dropout by Program Duration**")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='dropout_flag', y='program_duration_months', ax=ax)
    st.pyplot(fig)

# ========== ML MODELS ==========
with tab6:
    st.header("🔮 Predict Dropout & Dropout Score with ML")

    # Preprocessing for ML
    model_df = pd.read_csv('student_dropout_dataset.csv')
    X = model_df.drop(['dropout_flag', 'dropout_score'], axis=1)
    y_class = model_df['dropout_flag']
    y_reg = model_df['dropout_score']

    # Encoding
    X['gender'] = LabelEncoder().fit_transform(X['gender'])
    # region: 1, 2, 3 already mapped
    # All other features are numeric

    # Split data
    X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    _, _, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Classification Model with Hyperparameter Tuning
    clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }
    search_clf = RandomizedSearchCV(clf, params, n_iter=10, cv=3, scoring='f1', n_jobs=-1)
    search_clf.fit(X_train, y_train_c)
    best_clf = search_clf.best_estimator_

    # Regression Model with Hyperparameter Tuning
    regr = RandomForestRegressor(random_state=42)
    params_reg = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }
    search_reg = RandomizedSearchCV(regr, params_reg, n_iter=10, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    search_reg.fit(X_train, y_train_r)
    best_regr = search_reg.best_estimator_

    # Model Metrics
    y_pred_c = best_clf.predict(X_test)
    y_prob_c = best_clf.predict_proba(X_test)[:,1]
    y_pred_r = best_regr.predict(X_test)
    f1 = f1_score(y_test_c, y_pred_c)
    roc = roc_auc_score(y_test_c, y_prob_c)

    st.write(f"**Best Classification F1 Score:** {f1:.3f}")
    st.write(f"**ROC-AUC:** {roc:.3f}")

    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test_c, y_pred_c)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Will Not Dropout', 'Will Dropout'])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

    # --- Prediction Section
    st.subheader("Predict Dropout For New Student:")
    # Map sidebar inputs to ML model features
    region_val = {'Rural': 1, 'Suburban': 2, 'Urban': 3}[region]
    gender_val = LabelEncoder().fit(['Male','Female']).transform([gender])[0]
    sample_input = [[
        age, gender_val, ses, region_val, hsscore, gpa, credits, failures,
        attendance, submission, commute, peer_score, duration
    ]]

    pred_dropout = best_clf.predict(sample_input)[0]
    prob_dropout = best_clf.predict_proba(sample_input)[0][1]
    pred_score = best_regr.predict(sample_input)[0]

    if pred_dropout == 1:
        st.error(f"⚠️ This student is likely to **DROP OUT**! (Prob: {prob_dropout:.2f})")
    else:
        st.success(f"✅ This student is **NOT likely to drop out**. (Prob: {1-prob_dropout:.2f})")

    st.info(f"**Predicted Dropout Score:** {pred_score:.2f}")

    st.caption("Classification by RandomForest (best params found with RandomizedSearchCV). Regression for dropout_score.")

# Footer
st.caption("Dashboard developed for Director, Dean, HR & Stakeholders | All analysis and models are for demonstration/insight purposes.")

