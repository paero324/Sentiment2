import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import emoji
import time
import io
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.chart import BarChart, Reference
import tempfile

# --- 1. Setup & Configuration ---
# Download required NLTK data
nltk_data_dir = "./resources/nltk_data_dir/"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("punkt", download_dir=nltk_data_dir)

# Initialize session state
if 'sentiment_history' not in st.session_state:
    st.session_state.sentiment_history = []
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

# --- 2. Helper Functions ---

def preprocess_text(text):
    """Clean and preprocess text."""
    if isinstance(text, str):
        text = text.lower()
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = tokenizer.tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    return ''

def create_sentiment_gauge(score):
    """Create a gauge chart for sentiment visualization."""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': get_sentiment_color(score)},
            'steps': [
                {'range': [0, 0.33], 'color': "#FF4B4B"},
                {'range': [0.33, 0.66], 'color': "#FFA500"},
                {'range': [0.66, 1], 'color': "#00FF00"}
            ]
        }
    ))

def get_sentiment_color(score):
    """Return color based on sentiment score."""
    if score >= 0.66:
        return "#00FF00"
    elif score <= 0.33:
        return "#FF4B4B"
    return "#FFA500"

def create_word_cloud(text_data):
    """Generate word cloud from text data."""
    # Filter kata kosong
    if not text_data or len(text_data.strip()) == 0:
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def get_common_words(text_data, n=20):
    """Get most common words and their frequencies."""
    words = tokenizer.tokenize(text_data.lower())
    word_freq = Counter(words).most_common(n)
    return pd.DataFrame(word_freq, columns=['Word', 'Frequency'])

def export_to_excel(data, analysis_results):
    """Export analysis results to Excel with charts."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Sentiment Analysis Results"

    # Write data
    headers = ["Timestamp", "Text", "Sentiment", "Confidence"]
    ws.append(headers)

    for result in analysis_results:
        ws.append([
            result['timestamp'],
            result['text'],
            result['sentiment'],
            result['confidence']
        ])

    # Create charts
    chart = BarChart()
    chart.title = "Sentiment Distribution"

    # Add data to chart
    sentiments = pd.DataFrame(analysis_results)['sentiment'].value_counts()
    # Note: simplified chart logic for export
    ws.add_chart(chart, "H2")

    # Save to bytes buffer
    excel_buffer = io.BytesIO()
    wb.save(excel_buffer)
    return excel_buffer

def get_important_features(vectorizer, model, n_top_features=10):
    """Get most important features for each class."""
    feature_names = vectorizer.get_feature_names_out()

    # Get feature importance for each class
    importance = model.feature_log_prob_
    top_features = {}

    for i, label in enumerate(model.classes_):
        # Get indices of top features
        top_indices = importance[i].argsort()[-n_top_features:][::-1]
        top_features[label] = {
            'words': [feature_names[j] for j in top_indices],
            'importance': importance[i][top_indices]
        }

    return top_features

def train_model(data):
    """Train sentiment analysis model."""
    try:
        # Preprocess the data
        data['processed_review'] = data['review'].apply(preprocess_text)
        data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            data['processed_review'], 
            data['sentiment'], 
            test_size=0.2, 
            random_state=42
        )

        # Vectorize the text
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train the model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # Get important features
        important_features = get_important_features(vectorizer, model)

        # Store model and vectorizer in session state
        st.session_state.trained_model = model
        st.session_state.vectorizer = vectorizer

        return accuracy, report, cm, important_features

    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None

def analyze_text_with_model(text):
    """Analyze text using trained model."""
    if st.session_state.trained_model is None or st.session_state.vectorizer is None:
        return None

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Vectorize the text
    text_vec = st.session_state.vectorizer.transform([processed_text])

    # Get prediction probability
    proba = st.session_state.trained_model.predict_proba(text_vec)[0]

    # Get prediction
    sentiment = st.session_state.trained_model.predict(text_vec)[0]

    return {
        'sentiment': sentiment,
        'confidence': max(proba),
        'probabilities': {
            'negative': proba[0],
            'positive': proba[1]
        }
    }

def load_and_preprocess_dataset(uploaded_file):
    """Load and preprocess the uploaded dataset."""
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Pastikan kolom 'review' dan 'rating' ada
        if 'review' not in data.columns or 'rating' not in data.columns:
            st.error("Dataset harus memiliki kolom 'review' dan 'rating'.")
            return None

        # Basic data cleaning
        data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')
        data['review'] = data['review'].fillna('').astype(str).apply(preprocess_text)
        data['review_length'] = data['review'].apply(lambda x: len(x.split()))

        return data
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        return None

def extract_ngrams(text, n=2):
    tokens = text.split()
    return list(nltk.ngrams(tokens, n))

# --- 3. Page Configuration ---
st.set_page_config(
    page_title="Enhanced Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# --- 4. Sidebar ---
with st.sidebar:
    st.title("📊 Sentiment Analysis")
    st.markdown("### Upload Dataset and Train Model")

    uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])

    if uploaded_file:
        data = load_and_preprocess_dataset(uploaded_file)
        if data is not None:
            st.session_state.dataset = data
            st.success("Dataset loaded successfully!")

            if st.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    accuracy, report, cm, important_features = train_model(data)
                    if accuracy:
                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")

# --- 5. Main Content ---
st.title("📊 Enhanced Sentiment Analysis")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Realtime Analysis", "Dataset Analysis"])

with tab1:
    st.markdown("### Enter text for real-time analysis")
    text_input = st.text_area(
        "Enter your text here:",
        height=150,
        key="text_input",
        help="Type or paste your text here for sentiment analysis"
    )

    if text_input:
        if st.session_state.trained_model is None:
            st.warning("Please upload a dataset and train the model first.")
        else:
            with st.spinner('Analyzing sentiment...'):
                # Perform sentiment analysis
                analysis_result = analyze_text_with_model(text_input)

                if analysis_result:
                    # Store result in history
                    st.session_state.sentiment_history.append({
                        'timestamp': datetime.now(),
                        'text': text_input,
                        'sentiment': analysis_result['sentiment'],
                        'confidence': analysis_result['confidence']
                    })

                    # Display results in columns
                    col1, col2 = st.columns(2)
                    col3, col4 = st.columns(2, vertical_alignment='center')

                    with col1:
                        st.subheader("Sentiment Analysis Results")
                        st.markdown(f"""
                            **Sentiment:** {analysis_result['sentiment']}  
                            **Confidence:** {analysis_result['confidence']:.2f}  
                            **Probabilities:**
                            - Positive: {analysis_result['probabilities']['positive']:.2f}
                            - Negative: {analysis_result['probabilities']['negative']:.2f}
                        """)

                    with col2:
                        # Common words
                        st.subheader("Most Common Words")
                        common_words_df = get_common_words(text_input)
                        st.dataframe(common_words_df, hide_index=True)
                    
                    with col3:
                        st.subheader("Confidence Gauge")
                        gauge_chart = create_sentiment_gauge(analysis_result['confidence'])
                        st.plotly_chart(gauge_chart, use_container_width=True)
                    
                    with col4:
                        st.subheader("Word Cloud")
                        word_cloud_fig = create_word_cloud(text_input)
                        if word_cloud_fig:
                            st.pyplot(word_cloud_fig)
                        else:
                            st.info("Not enough text to generate word cloud.")

                    # History section
                    if st.session_state.sentiment_history:
                        st.subheader("Analysis History")
                        history_df = pd.DataFrame(st.session_state.sentiment_history)
                        fig = px.line(history_df, x='timestamp', y='confidence',
                                        title="Confidence History",
                                        labels={'confidence': 'Confidence Score', 'timestamp': 'Time'})
                        st.plotly_chart(fig, use_container_width=True)

                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            current_analysis = pd.DataFrame([st.session_state.sentiment_history[-1]])
                            st.download_button(
                                label="⬇️ Download Current Analysis",
                                data=current_analysis.to_csv(index=False),
                                file_name="current_sentiment_analysis.csv",
                                mime="text/csv"
                            )
                        with col2:
                            history_df = pd.DataFrame(st.session_state.sentiment_history)
                            st.download_button(
                                label="⬇️ Download Complete History",
                                data=history_df.to_csv(index=False),
                                file_name="sentiment_analysis_history.csv",
                                mime="text/csv"
                            )

with tab2:
    if st.session_state.dataset is not None:
        data = st.session_state.dataset

        st.subheader("Search and Filter")
        search_term = st.text_input("Search in reviews:", "")
        min_rating = st.slider("Minimum Rating", min_value=1, max_value=5, value=1)

        # Filter data
        filtered_data = data[
            (data['review'].str.contains(search_term, case=False, na=False)) &
            (data['rating'] >= min_rating)
        ]

        st.subheader(f"Dataset Overview (Showing {len(filtered_data)} rows)")
        st.dataframe(filtered_data, use_container_width=True)

        if st.session_state.trained_model is not None:
            st.subheader("Model Performance")
            # Melatih ulang untuk mendapatkan metrik (atau bisa disimpan di session state sebelumnya)
            # Disini kita panggil lagi train_model untuk mendapatkan report & CM display
            # Catatan: Untuk efisiensi, sebaiknya hasil ini disimpan saat training di sidebar
            with st.spinner("Calculating metrics..."):
                accuracy, report, cm, important_features = train_model(data)
            
            if accuracy:
                st.success("✅ Model metrics ready!")
                st.metric("Model Accuracy", f"{accuracy:.2%}")
                
                with st.expander("📝 Classification Report"):
                    c1, c2 = st.columns([1, 0.8], border=False)

                    with c1:
                        st.subheader(":material/summarize: Report")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df)
                    
                    with c2:
                        st.subheader("📊 Precision, Recall, F1-Score")
                        st.caption("This chart displays the performance of a classification model across three key metrics")
                        st.bar_chart(report_df[['precision', 'recall', 'f1-score']].drop('accuracy', errors="ignore"))
                    
                    st.subheader("📉 Confusion Matrix")
                    st.caption("The confusion matrix displays the predicted results of a classification model compared to the actual labels.")
                    fig = px.imshow(cm, 
                                  labels=dict(x="Predicted", y="Actual"),
                                  x=['Negative', 'Positive'],
                                  y=['Negative', 'Positive'],
                                  title="Confusion Matrix",
                                  text_auto=True)
                    st.plotly_chart(fig)

                with st.container():
                    st.header("🔍 Data Insights")
                    st.divider()
                    col1, col2 = st.columns([2,1], gap='small', vertical_alignment="center")
                    col3, col4 = st.columns([1,1])
                    col5, col6 = st.columns([2,1], vertical_alignment='center')
                    col7, col8 = st.columns([1,0.8], vertical_alignment="center")
                    
                    with col1:
                        st.subheader("📏 Review Length Distribution")
                        fig_length = px.histogram(data, x='review_length',
                                                    nbins=50, title="Distribution of Review Text Length",
                                                    labels={"review_length" : "Text Length (Words)"},
                                                    color_discrete_sequence=["blue"])
                        st.plotly_chart(fig_length)
            
                    with col2:
                        st.caption("Overview of the review length in dataset.")
                        st.write(data['review_length'].describe())
                
                    with col3:
                        sentiment_counts = data['sentiment'].value_counts()
                        fig_sentiment = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                                        labels={'x': 'Sentiment', 'y': 'Count'},
                                        title='Sentiment Distribution', color=sentiment_counts.index,
                                        color_discrete_sequence=['green', 'red'])
                        st.plotly_chart(fig_sentiment)
            
                    with col4:
                        fig_boxplt = px.box(data, x='sentiment', y='review_length', color="sentiment",
                                    color_discrete_sequence=["green", "red"])
                        st.plotly_chart(fig_boxplt)

                    with col5:
                        st.subheader("📌 Top 20 Most Frequent Words")
                        all_words = ' '.join(data["review"]).split()
                        common_words = Counter(all_words).most_common(20)
                        words, counts = zip(*common_words)

                        fig_words = px.bar(x=counts, y=words, orientation='h', 
                                    title="Top 20 Most Frequent Words",
                                    labels={'x': 'Frequency', 'y': 'Words'},
                                    color_discrete_sequence=['blue'])
                        fig_words.update_yaxes(categoryorder='total ascending')
                
                        st.plotly_chart(fig_words)

                
                    with col6:
                        word_freq_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
                        st.dataframe(word_freq_df, hide_index=True)

                    with col7:
                        st.subheader("📌 Top 10 Most Frequent Bigrams")
                        all_bigrams = []
                        for review in data['review']:
                            all_bigrams.extend(extract_ngrams(review, n=2))
                
                        bigram_freq = Counter(all_bigrams)
                        top_10_bigrams = bigram_freq.most_common(10)
                        
                        fig_bigrams = px.bar(top_10_bigrams, 
                                            y=[str(bigram[0]) for bigram in top_10_bigrams], 
                                            x=[bigram[1] for bigram in top_10_bigrams], 
                                            orientation='h',
                                            title="Top 10 Most Frequent Bigrams",
                                            labels={"x" : "Frequency", "y" : "Bigrams"},
                                            color_discrete_sequence=['purple'])
                        
                        fig_bigrams.update_yaxes(categoryorder='total ascending')
                        st.plotly_chart(fig_bigrams)

                    with col8:
                        st.subheader("Bigrams Data")
                        top_10_bigrams_df = pd.DataFrame(top_10_bigrams, columns=['Bigram', 'Frequency'])
                        top_10_bigrams_df['Bigram'] = top_10_bigrams_df['Bigram'].astype(str)
                        st.dataframe(top_10_bigrams_df, hide_index=True)

    else:
        st.info("Upload dataset pada sidebar untuk melihat analisis data.")
