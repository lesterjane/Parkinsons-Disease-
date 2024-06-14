import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom CSS for background image
page_bg_img = '''
<style>
.stApp {
background-image: url("https://altoida.com/wp-content/uploads/2022/09/Atypical-Parkinsonism-syndrome-refers-to-any-conditions-that-involve-the-types-of-movement-problems-seen-in-Parkinsons-disease.-scaled.jpg");
background-size: cover;
background-repeat: no-repeat;
background-attachment: fixed;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the dataset
data = pd.read_csv('parkinsons_data.csv')

# Preprocessing
numeric_data = data.select_dtypes(include='number')
X = numeric_data.drop('status', axis=1)
y = numeric_data['status']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Define the prediction function
def predict_parkinsons(mdvp_hz_avg, mdvp_hz_max, mdvp_hz_min, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe):
    input_data = np.array([[mdvp_hz_avg, mdvp_hz_max, mdvp_hz_min, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
    input_data = scaler.transform(input_data)
    prediction_proba = rf_classifier.predict_proba(input_data)
    return prediction_proba

# Sidebar navigation
selected = st.sidebar.radio("Navigation", ["Home", "Data Info", "Prediction", "Visualization"])

if selected == "Home":
    st.title("Detection and Diagnosis of Parkinson's Disease")
    st.write("""
        ## About the App

        Parkinson's disease (PD) is the most prevalent neurodegenerative movement disorder. It is characterized by the deterioration of dopaminergic neurons, resulting in severe motor symptoms and cognitive impairment in patients.
        This web application leverages a machine learning algorithm to predict whether a user has Parkinson's disease or is healthy. Utilizing a Random Forest Classifier, the app analyzes various feature values and generates a probabilistic output for potential pathological scenarios in the future.

        ## How to use

        1. Enter your personal and medical information in the input fields present in the Prediction tab.
        2. Click on the “Predict” button to see your prediction result.
        3. You can also view the data distribution and feature importance graphs by clicking on the Visualization tab.

        ## Disclaimer

        This web application is for educational and research purposes only. It is not intended to provide medical advice or diagnosis. Please consult your doctor before using this app or making any decisions based on its results.
    """)


elif selected == "Data Info":
    st.title("Data Info")
    st.write("View Data")
    if st.checkbox("View data"):
        st.write(data)

    st.title("Columns Description:")
    if st.checkbox("View Summary"):
        st.write(data.describe())
        
    if st.checkbox("Column Names"):
        st.write(data.columns.tolist())
        
    if st.checkbox("Column data types"):
        st.write(data.dtypes)
        
    if st.checkbox("Column Data"):
        st.write(data)

    if st.checkbox("Total of Healthy and Parkinson"):
        total_count = len(data)
        healthy_count = (data['status'] == 0).sum()
        parkinson_count = (data['status'] == 1).sum()

        st.write("The total dataset has", total_count, "data points.")
        st.write("Healthy Count:", healthy_count)
        st.write("Parkinson Count:", parkinson_count)
        st.write("Out of these, ", healthy_count, "are classified as healthy and", parkinson_count, "are classified as having Parkinson's disease.")

    if st.checkbox("Features MDVP"):
        mdvp_features_info = {
            "MDVP (Hz)": "This feature represents the average vocal fundamental frequency in Hertz. It indicates the average pitch of the voice.",
            "MDVP (Hz)": "This feature represents the maximum vocal fundamental frequency in Hertz. It indicates the highest pitch of the voice.",
            "MDVP (Hz)": "This feature represents the minimum vocal fundamental frequency in Hertz. It indicates the lowest pitch of the voice.",
            "MDVP (%)": "Jitter is a measure of the variation in frequency of vocal fold vibrations. This feature represents the percentage of jitter in the voice signal.",
            "MDVP (Abs)": "Jitter can also be measured in absolute terms. This feature represents the absolute jitter in the voice signal.",
            "MDVP:RAP": "RAP stands for Relative Amplitude Perturbation. It is another measure of variations in vocal fold vibrations.",
            "MDVP:PPQ": "PPQ stands for Five-Point Period Perturbation Quotient. It is another measure of perturbations in vocal fold vibrations.",
            "Jitter:DDP": "Jitter DDP (Jitter) is the average absolute difference of differences between consecutive jitter cycles.",
            "MDVP:Shimmer": "Shimmer is a measure of the variation in amplitude of vocal fold vibrations. This feature represents shimmer in the voice signal.",
            "MDVP (dB)": "Shimmer can also be measured in decibels. This feature represents shimmer measured in decibels.",
            "Shimmer:APQ3": "APQ stands for Amplitude Perturbation Quotient. APQ3 represents the amplitude perturbation quotient measured in three segments.",
            "Shimmer:APQ5": "Similarly, APQ5 represents the amplitude perturbation quotient measured in five segments.",
            "MDVP:APQ": "MDVP represents the absolute perturbation quotient.",
            "Shimmer:DDA": "DDA stands for Difference in Amplitude. This feature represents the average absolute difference between consecutive differences in amplitude.",
            "NHR": "NHR stands for Noise-to-Harmonics Ratio. It measures the ratio of noise to harmonics in the voice signal.",
            "HNR": "HNR stands for Harmonics-to-Noise Ratio. It measures the ratio of harmonics to noise in the voice signal.",
            "Status": "This is the target variable indicating the presence (1) or absence (0) of Parkinson's disease.",
            "RPDE": "RPDE stands for Recurrence Period Density Entropy. It is a measure of the complexity of vocal fold vibrations.",
            "DFA": "DFA stands for Detrended Fluctuation Analysis. It is a measure of the fractal scaling properties of vocal fold vibrations.",
            "Spread1": "Spread1 is a feature representing the spread of principal component 1 in a dataset.",
            "Spread2": "Spread2 is a feature representing the spread of principal component 2 in a dataset.",
            "D2": "D2 represents the dimensionality of principal component 2.",
            "PPE": "PPE stands for Pitch Period Entropy. It is a measure of the entropy of pitch periods in the voice signal."
        }
        for feature, description in mdvp_features_info.items():
            st.write(f"{feature}: {description}")

            


elif selected == "Prediction":
    st.title("Prediction")
    st.header("Enter your medical information:")
    mdvp_hz_avg = st.number_input("MDVP: Average vocal fundamental frequency (Hz)", min_value=100.0, max_value=300.0)
    mdvp_hz_max = st.number_input("MDVP: Maximum vocal fundamental frequency (Hz)", min_value=100.0, max_value=600.0)
    mdvp_hz_min = st.number_input("MDVP: Minimum vocal fundamental frequency (Hz)", min_value=100.0, max_value=300.0)
    jitter_percent = st.number_input("MDVP: Jitter (%)", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    jitter_abs = st.number_input("MDVP: Jitter (Abs)", min_value=0.0, max_value=0.1, step=0.00001, format="%.5f")
    rap = st.number_input("MDVP: RAP", min_value=0.0, max_value=0.15, step=0.001, format="%.3f")
    ppq = st.number_input("MDVP: PPQ", min_value=0.0, max_value=0.1, step=0.001, format="%.3f")
    ddp = st.number_input("Jitter: DDP", min_value=0.0, max_value=0.1, step=0.001, format="%.3f")
    shimmer = st.number_input("MDVP: Shimmer", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    shimmer_db = st.number_input("MDVP: Shimmer (dB)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    apq3 = st.number_input("Shimmer: APQ3", min_value=0.0, max_value=0.1, step=0.001, format="%.3f")
    apq5 = st.number_input("Shimmer: APQ5", min_value=0.0, max_value=0.1, step=0.001, format="%.3f")
    apq = st.number_input("MDVP: APQ", min_value=0.0, max_value=0.1, step=0.001, format="%.3f")
    dda = st.number_input("Shimmer: DDA", min_value=0.0, max_value=0.1, step=0.001, format="%.3f")
    nhr = st.number_input("NHR", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, step=0.1, format="%.1f")
    rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    spread1 = st.number_input("spread1", min_value=-10.0, max_value=10.0, step=0.1, format="%.1f")
    spread2 = st.number_input("spread2", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    d2 = st.number_input("D2", min_value=0.0, max_value=5.0, step=0.1, format="%.1f")
    ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")

    if st.button("Predict"):
        prediction_proba = predict_parkinsons(mdvp_hz_avg, mdvp_hz_max, mdvp_hz_min, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe)
        
        # Check if the probability of Parkinson's disease (class 1) is greater than 0.5
        if prediction_proba[0][1] > 0.5:
            prediction = 'Parkinson\'s'
        else:
            prediction = 'Healthy'
        
        st.write(f"Prediction: {prediction}")
        st.write(f"Probability of Parkinson's Disease: {prediction_proba[0][1]:.2f}")

elif selected == "Visualization":
    st.title("Data Visualization")
    
    st.write("## Distribution of Health Status")
    colors = ['#008080', '#800080']  
    plt.figure(figsize = (10, 5))
    sns.countplot(x='status', data=data, palette=colors)
    plt.title('Distribution of Health Status')
    plt.xlabel("Status (0: Healthy, 1: With Parkinson's)")
    plt.ylabel('Count')
    st.pyplot(plt)
    
    st.write("## Correlation Matrix")
    # Correlation matrix
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(15, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    st.write("## Relationships between variables")
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x='MDVP:Fhi(Hz)', y='MDVP:Flo(Hz)', data=data)
    plt.title('Relationship between Maximum and Minimum Vocal Fundamental Frequency')
    plt.xlabel('Maximum Vocal Fundamental Frequency (MDVP:Fhi(Hz))')
    plt.ylabel('Minimum Vocal Fundamental Frequency (MDVP:Flo(Hz))')
    st.pyplot(plt)
