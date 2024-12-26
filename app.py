import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load pickle files (TF-IDF vectorizer and SVM model)
with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load stopwords and stemmer
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Function for preprocessing text
def preprocess_text(text):
    # Remove mentions, hashtags, extra whitespace
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus karakter khusus dan angka
    text = re.sub(r'[^\w\s]', '', text) # Menghapus karakter non-alphanumeric dan whitespace
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus whitespace ekstra

    # Case folding
    text = text.lower()

    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])

    # Stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text

#UI
st.title("Sentiment Analysis Application")
st.write("Enter text or upload a CSV file for sentiment analysis (positive or negative).")

# Tab navigation: Single Text Input or File Upload
option = st.radio("Select input :", ("Text", "Upload CSV File"))

#Single Text Input
if option == "Text":
    user_input = st.text_area("Enter Text", placeholder="Type your text here...")

    if st.button("Analyze"):
        if user_input:
            # Preprocess text
            preprocessed_text = preprocess_text(user_input)

            # Extract features using TF-IDF
            features = tfidf_vectorizer.transform([preprocessed_text])

            # Predict sentiment
            prediction = svm_model.predict(features)[0]

            # Display result
            if prediction == 'positif':
                st.success("Result: Positive Sentiment")
            elif prediction == 'negatif':
                st.error("Result: Negative Sentiment")
        else:
            st.warning("Please enter some text to analyze!")

#File Upload
#File Upload
elif option == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Check file size (limit 50 MB)
            if uploaded_file.size > 50 * 1024 * 1024:
                st.error("File is too large! Please upload a file smaller than 50 MB.")
            else:
                # Load CSV file
                df = pd.read_csv(uploaded_file)
                st.write("Uploaded data:")
                st.dataframe(df.head())

                # Validate column
                text_column = st.selectbox("Select text column:", df.columns)
                if st.button("Analyze CSV Sentiment"):
                    # Preprocess all texts
                    df['preprocessed_text'] = df[text_column].astype(str).apply(preprocess_text)

                    # Extract features using TF-IDF
                    features = tfidf_vectorizer.transform(df['preprocessed_text'])

                    # Predict sentiments
                    df['sentiment'] = svm_model.predict(features)

                    # Display results
                    st.success("Analysis complete! Here are the results:")
                    st.dataframe(df[[text_column, 'sentiment']])

                    # Visualize results with a pie chart
                    sentiment_counts = df['sentiment'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures pie chart is circular.
                    st.pyplot(fig)

                    # Download results as CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"An error occurred: {e}")
