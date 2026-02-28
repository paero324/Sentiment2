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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import re
import emoji
import time
import io
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.chart import BarChart, Reference
import tempfile
import seaborn as sns
from PIL import Image as PILImage
import base64

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
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'Naive Bayes'
if 'vectorizer_type' not in st.session_state:
    st.session_state.vectorizer_type = 'Count Vectorizer'
if 'model_comparison' not in st.session_state:
    st.session_state.model_comparison = None

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

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
    data_refs = Reference(ws, min_col=3, min_row=2, max_row=len(analysis_results)+1)
    chart.add_data(data_refs)

    # Add chart to worksheet
    ws.add_chart(chart, "H2")

    # Save to bytes buffer
    excel_buffer = io.BytesIO()
    wb.save(excel_buffer)
    return excel_buffer

def get_important_features(vectorizer, model, n_top_features=10):
    """Get most important features for each class."""
    feature_names = vectorizer.get_feature_names_out()

    # Get feature importance for each class
    if hasattr(model, 'feature_log_prob_'):  # Naive Bayes
        importance = model.feature_log_prob_
    elif hasattr(model, 'coef_'):  # Logistic Regression, SVM, Linear models
        importance = np.abs(model.coef_)
    else:  # Tree-based models
        importance = model.feature_importances_.reshape(1, -1)
    
    top_features = {}

    for i, label in enumerate(model.classes_):
        # Get indices of top features
        if importance.ndim > 1 and importance.shape[0] > 1:
            top_indices = importance[i].argsort()[-n_top_features:][::-1]
        else:
            top_indices = importance[0].argsort()[-n_top_features:][::-1]
        
        top_features[label] = {
            'words': [feature_names[j] for j in top_indices],
            'importance': importance[i][top_indices] if importance.ndim > 1 and importance.shape[0] > 1 else importance[0][top_indices]
        }

    return top_features

def train_model(data, model_type='Naive Bayes', vectorizer_type='Count Vectorizer', tune_hyperparams=False):
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
        if vectorizer_type == 'Count Vectorizer':
            vectorizer = CountVectorizer()
        else:  # TF-IDF
            vectorizer = TfidfVectorizer()
            
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Select model
        if model_type == 'Naive Bayes':
            model = MultinomialNB()
            if tune_hyperparams:
                param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0]}
                model = GridSearchCV(model, param_grid, cv=5)
        elif model_type == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000)
            if tune_hyperparams:
                param_grid = {'C': [0.1, 1, 10, 100]}
                model = GridSearchCV(model, param_grid, cv=5)
        elif model_type == 'SVM':
            model = SVC(probability=True)
            if tune_hyperparams:
                param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                model = GridSearchCV(model, param_grid, cv=5)
        elif model_type == 'Random Forest':
            model = RandomForestClassifier()
            if tune_hyperparams:
                param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
                model = GridSearchCV(model, param_grid, cv=5)
        else:
            model = MultinomialNB()

        # Train the model
        model.fit(X_train_vec, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec)[:, 1]  # Probability of positive class
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test.map({'negative': 0, 'positive': 1}), y_proba)
        roc_auc = auc(fpr, tpr)

        # Get important features
        important_features = get_important_features(vectorizer, model)

        # Store model and vectorizer in session state
        st.session_state.trained_model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.model_type = model_type
        st.session_state.vectorizer_type = vectorizer_type

        return accuracy, report, cm, important_features, (fpr, tpr, roc_auc)

    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

def compare_models(data):
    """Compare different models on the dataset."""
    models = ['Naive Bayes', 'Logistic Regression', 'SVM', 'Random Forest']
    vectorizers = ['Count Vectorizer', 'TF-IDF']
    
    results = []
    
    for model_type in models:
        for vectorizer_type in vectorizers:
            accuracy, report, cm, _, _ = train_model(data, model_type, vectorizer_type)
            if accuracy:
                results.append({
                    'Model': model_type,
                    'Vectorizer': vectorizer_type,
                    'Accuracy': accuracy,
                    'Precision': report['weighted avg']['precision'],
                    'Recall': report['weighted avg']['recall'],
                    'F1-Score': report['weighted avg']['f1-score']
                })
    
    return pd.DataFrame(results)

def analyze_text_with_model(text):
    """Analyze text using trained model."""
    if st.session_state.trained_model is None or st.session_state.vectorizer is None:
        st.error("Please upload and train the model first.")
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

        # Basic data cleaning
        data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')
        data['review'] = data['review'].fillna('').apply(preprocess_text)
        data['review_length'] = data['review'].apply(lambda x: len(x.split()))
        
        return data
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        return None

def extract_ngrams(text, n=2):
    tokens = text.split()
    return list(nltk.ngrams(tokens, n))

def img_to_bytes(img_path):
    img_bytes = io.BytesIO()
    image = PILImage.open(img_path)
    image.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

# Page configuration
st.set_page_config(
    page_title="Enhanced Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem 0.5rem 0 0;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo or image
    try:
        logo = img_to_bytes("logo.png")
        st.image(logo, width=200)
    except:
        st.image("https://via.placeholder.com/200x100?text=Sentiment+Analysis", width=200)
    
    st.title("📊 Sentiment Analysis")
    st.markdown("### Upload Dataset and Train Model")

    uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])

    if uploaded_file:
        data = load_and_preprocess_dataset(uploaded_file)
        if data is not None:
            st.session_state.dataset = data
            st.success("Dataset loaded successfully!")
            
            # Show dataset preview
            with st.expander("Preview Dataset"):
                st.dataframe(data.head())
            
            # Model selection
            model_type = st.selectbox(
                "Select Model:",
                ["Naive Bayes", "Logistic Regression", "SVM", "Random Forest"]
            )
            
            # Vectorizer selection
            vectorizer_type = st.selectbox(
                "Select Vectorizer:",
                ["Count Vectorizer", "TF-IDF"]
            )
            
            # Hyperparameter tuning option
            tune_hyperparams = st.checkbox("Tune Hyperparameters (Slower but more accurate)")
            
            if st.button("Train Model", type="primary"):
                with st.spinner("Training model..."):
                    accuracy, report, cm, important_features, roc_data = train_model(data, model_type, vectorizer_type, tune_hyperparams)
                    if accuracy:
                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                        st.session_state.model_type = model_type
                        st.session_state.vectorizer_type = vectorizer_type
            
            # Model comparison
            if st.button("Compare All Models"):
                with st.spinner("Comparing models..."):
                    comparison_df = compare_models(data)
                    st.session_state.model_comparison = comparison_df
                    st.success("Model comparison complete!")

# Main content
st.markdown('<h1 class="main-header">📊 Enhanced Sentiment Analysis</h1>', unsafe_allow_html=True)

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Realtime Analysis", "Dataset Analysis", "Model Comparison"])

with tab1:
    st.markdown('<h2 class="sub-header">Enter text for real-time analysis</h2>', unsafe_allow_html=True)
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
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.subheader("Sentiment Analysis Results")
                        sentiment_emoji = "😊" if analysis_result['sentiment'] == 'positive' else "😔"
                        st.markdown(f"""
                            **Sentiment:** {sentiment_emoji} {analysis_result['sentiment']}  
                            **Confidence:** {analysis_result['confidence']:.2f}  
                            **Probabilities:**
                            - Positive: {analysis_result['probabilities']['positive']:.2f}
                            - Negative: {analysis_result['probabilities']['negative']:.2f}
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        # Common words
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.subheader("Most Common Words")
                        common_words_df = get_common_words(text_input)
                        st.dataframe(common_words_df, hide_index=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.subheader("Confidence Gauge")
                        gauge_chart = create_sentiment_gauge(analysis_result['confidence'])
                        st.plotly_chart(gauge_chart, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.subheader("Word Cloud")
                        word_cloud_fig = create_word_cloud(text_input)
                        st.pyplot(word_cloud_fig)
                        st.markdown('</div>', unsafe_allow_html=True)

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
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("Search in reviews:", "")
        with col2:
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
            accuracy, report, cm, important_features, roc_data = train_model(
                data, st.session_state.model_type, st.session_state.vectorizer_type
            )
            st.success("✅ Model training complete!")
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Model", st.session_state.model_type)
            col2.metric("Vectorizer", st.session_state.vectorizer_type)
            col3.metric("Accuracy", f"{accuracy:.2%}")
            col4.metric("AUC", f"{roc_data[2]:.2f}" if roc_data else "N/A")
            
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
                st.caption("The confusion matrix displays the predicted results of a classification model compared to the actual labels. This matrix helps identify the strengths and weaknesses of the model in predicting classes.")
                fig = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual"),
                              x=['Negative', 'Positive'],
                              y=['Negative', 'Positive'],
                              title="Confusion Matrix")
                st.plotly_chart(fig)
                
                # ROC Curve
                if roc_data:
                    st.subheader("📈 ROC Curve")
                    st.caption("The ROC curve shows the trade-off between the true positive rate and false positive rate.")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=roc_data[0], y=roc_data[1],
                        mode='lines',
                        name=f'ROC Curve (AUC = {roc_data[2]:.2f})',
                        line=dict(color='darkorange', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(color='navy', width=2, dash='dash')
                    ))
                    fig.update_layout(
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        title='Receiver Operating Characteristic (ROC) Curve'
                    )
                    st.plotly_chart(fig)

            with st.expander("🔍 Important Features"):
                if important_features:
                    for sentiment, features in important_features.items():
                        st.subheader(f"Top Features for {sentiment.capitalize()} Class")
                        features_df = pd.DataFrame({
                            'Word': features['words'],
                            'Importance': features['importance']
                        })
                        st.dataframe(features_df, hide_index=True)
                        
                        # Plot top features
                        fig = px.bar(
                            features_df.head(10), 
                            x='Importance', 
                            y='Word',
                            orientation='h',
                            title=f"Top 10 Important Features for {sentiment.capitalize()} Class",
                            color_discrete_sequence=['green' if sentiment == 'positive' else 'red']
                        )
                        fig.update_yaxes(categoryorder='total ascending')
                        st.plotly_chart(fig, use_container_width=True)

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
                st.caption("Overview of the review length in dataset, including key statistics such as count, mean, and distribution of numerical values.")
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
                fig_bigrams = px.bar(top_10_bigrams, y=[str(bigram[0]) for bigram in top_10_bigrams], 
                                x=[bigram[1] for bigram in top_10_bigrams], orientation='h',
                                title="Top 10 Most Frequent Bigrams",
                                labels={"x" : "Frequency",
                                     "y" : "Bigrams"},
                                color_discrete_sequence=["blue"])
                fig_bigrams.update_yaxes(categoryorder='total ascending')
    
                st.plotly_chart(fig_bigrams)
            
            with col8:
                bigram_freq_df = pd.DataFrame(top_10_bigrams, columns=["Bigram", "Frequency"])
                st.dataframe(bigram_freq_df, hide_index=True)

        # Export buttons
        col1, col2 = st.columns(2)
        with col1:
            # Export dataset
            st.download_button(
                label="⬇️ Download Dataset (CSV)",
                data=filtered_data.to_csv(index=False),
                file_name="sentiment_analysis_dataset.csv",
                mime="text/csv"
            )

        with col2:
            # Export Excel report with charts
            if st.session_state.sentiment_history:
                excel_buffer = export_to_excel(
                    filtered_data,
                    st.session_state.sentiment_history
                )
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=excel_buffer.getvalue(),
                    file_name="sentiment_analysis_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    else:
        st.info("❌ Please upload a dataset to see the analysis.")

with tab3:
    if st.session_state.model_comparison is not None:
        st.subheader("Model Comparison Results")
        comparison_df = st.session_state.model_comparison
        
        # Display comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualize comparison
        fig = px.bar(comparison_df, x='Model', y='Accuracy', color='Vectorizer',
                     barmode='group', title="Model Accuracy Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(comparison_df, x='Model', y='F1-Score', color='Vectorizer',
                     barmode='group', title="Model F1-Score Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for comprehensive comparison
        fig = go.Figure()
        
        for model in comparison_df['Model'].unique():
            model_data = comparison_df[comparison_df['Model'] == model]
            if len(model_data) > 1:  # If model has both vectorizers
                for _, row in model_data.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                        theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        fill='toself',
                        name=f"{model} ({row['Vectorizer']})"
                    ))
            else:  # If model has only one vectorizer
                row = model_data.iloc[0]
                fig.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    fill='toself',
                    name=f"{model} ({row['Vectorizer']})"
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Model Performance Radar Chart"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download comparison results
        st.download_button(
            label="⬇️ Download Model Comparison Results",
            data=comparison_df.to_csv(index=False),
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )
    else:
        st.info("❌ Please compare models in the sidebar to see the results.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, Scikit-learn, and Plotly</p>
    </div>
    """, unsafe_allow_html=True)
