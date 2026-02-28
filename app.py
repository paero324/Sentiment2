import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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
from PIL import Image as PILImage
import base64
from datetime import datetime

# Try to import plotly with error handling
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("""
    ⚠️ **Missing Required Library**
    
    The `plotly` library is required for this application. Please install it by running:
    
    ```bash
    pip install plotly
    ```
    
    Or add it to your `requirements.txt` file.
    """)

# Download required NLTK data
nltk_data_dir = "./resources/nltk_data_dir/"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)

nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
try:
    nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
except:
    st.warning("Could not download NLTK data. Using existing data if available.")

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
try:
    stop_words = set(stopwords.words('indonesian'))
except:
    st.warning("Could not load Indonesian stopwords. Using empty set.")
    stop_words = set()
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
    if not PLOTLY_AVAILABLE:
        # Fallback to matplotlib if plotly is not available
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#FF4B4B', '#FFA500', '#00FF00']
        color_idx = 0 if score <= 0.33 else (1 if score <= 0.66 else 2)
        
        # Create a simple bar chart as fallback
        ax.barh(0, score, color=colors[color_idx], height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_title(f'Sentiment Score: {score:.2f}')
        ax.text(score + 0.02, 0, f'{score:.2f}', va='center')
        return fig
    
    fig = go.Figure(go.Indicator(
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
    fig.update_layout(height=300)
    return fig

def get_sentiment_color(score):
    """Return color based on sentiment score."""
    if score >= 0.66:
        return "#00FF00"
    elif score <= 0.33:
        return "#FF4B4B"
    return "#FFA500"

def create_word_cloud(text_data):
    """Generate word cloud from text data."""
    if not text_data or not isinstance(text_data, str):
        # Return empty figure if no valid text
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No valid text for word cloud", 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def get_common_words(text_data, n=20):
    """Get most common words and their frequencies."""
    if not text_data or not isinstance(text_data, str):
        return pd.DataFrame(columns=['Word', 'Frequency'])
        
    words = tokenizer.tokenize(text_data.lower())
    word_freq = Counter(words).most_common(n)
    return pd.DataFrame(word_freq, columns=['Word', 'Frequency'])

def export_to_excel(data, analysis_results):
    """Export analysis results to Excel with charts."""
    try:
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
    except Exception as e:
        st.error(f"Error exporting to Excel: {str(e)}")
        return io.BytesIO()

def get_important_features(vectorizer, model, n_top_features=10):
    """Get most important features for each class."""
    try:
        feature_names = vectorizer.get_feature_names_out()

        # Get feature importance for each class
        if hasattr(model, 'feature_log_prob_'):  # Naive Bayes
            importance = model.feature_log_prob_
        elif hasattr(model, 'coef_'):  # Logistic Regression, SVM, Linear models
            importance = np.abs(model.coef_)
        elif hasattr(model, 'feature_importances_'):  # Tree-based models
            importance = model.feature_importances_.reshape(1, -1)
        else:
            return {}  # Model doesn't have feature importance
        
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
    except Exception as e:
        st.error(f"Error extracting important features: {str(e)}")
        return {}

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
        
        # Get probability for the positive class
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_vec)
            # Find the index of the positive class
            positive_idx = list(model.classes_).index('positive') if 'positive' in model.classes_ else 0
            y_proba_positive = y_proba[:, positive_idx]
        else:
            # For models without predict_proba (like some SVM configurations)
            decision_scores = model.decision_function(X_test_vec)
            if decision_scores.ndim > 1:
                y_proba_positive = decision_scores[:, 1]
            else:
                y_proba_positive = decision_scores
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve
        y_test_binary = y_test.map({'negative': 0, 'positive': 1})
        fpr, tpr, _ = roc_curve(y_test_binary, y_proba_positive)
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
            if accuracy is not None:
                results.append({
                    'Model': model_type,
                    'Vectorizer': vectorizer_type,
                    'Accuracy': accuracy,
                    'Precision': report['weighted avg']['precision'],
                    'Recall': report['weighted avg']['recall'],
                    'F1-Score': report['weighted avg']['f1-score']
                })
    
    if results:
        return pd.DataFrame(results)
    else:
        st.error("Could not compare models. Please check your data.")
        return pd.DataFrame()

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
    if hasattr(st.session_state.trained_model, 'predict_proba'):
        proba = st.session_state.trained_model.predict_proba(text_vec)[0]
    else:
        # For models without predict_proba
        decision = st.session_state.trained_model.decision_function(text_vec)
        if decision.ndim > 1:
            proba = decision[0]
        else:
            proba = np.array([1-decision[0], decision[0]])  # Rough approximation

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

        # Check if required columns exist
        if 'review' not in data.columns or 'rating' not in data.columns:
            st.error("Dataset must contain 'review' and 'rating' columns.")
            return None

        # Basic data cleaning
        data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')
        data['review'] = data['review'].fillna('').apply(preprocess_text)
        data['review_length'] = data['review'].apply(lambda x: len(x.split()))
        
        return data
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")
        return None

def extract_ngrams(text, n=2):
    if not text or not isinstance(text, str):
        return []
    tokens = text.split()
    return list(nltk.ngrams(tokens, n))

def img_to_bytes(img_path):
    try:
        img_bytes = io.BytesIO()
        image = PILImage.open(img_path)
        image.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    except:
        # Return a placeholder image if the logo doesn't exist
        placeholder = io.BytesIO()
        plt.figure(figsize=(2, 1))
        plt.text(0.5, 0.5, "Logo", horizontalalignment='center')
        plt.axis('off')
        plt.savefig(placeholder, format='PNG')
        plt.close()
        return placeholder.getvalue()

def create_matplotlib_bar_chart(data, title, x_label, y_label, color='blue'):
    """Create a bar chart using matplotlib as fallback."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(data.index, data.values, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

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
                    if accuracy is not None:
                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
                        st.session_state.model_type = model_type
                        st.session_state.vectorizer_type = vectorizer_type
            
            # Model comparison
            if st.button("Compare All Models"):
                with st.spinner("Comparing models..."):
                    comparison_df = compare_models(data)
                    if not comparison_df.empty:
                        st.session_state.model_comparison = comparison_df
                        st.success("Model comparison complete!")
                    else:
                        st.error("Model comparison failed. Please check your data.")

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
                        if PLOTLY_AVAILABLE:
                            st.plotly_chart(gauge_chart, use_container_width=True)
                        else:
                            st.pyplot(gauge_chart)
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
                        
                        if PLOTLY_AVAILABLE:
                            fig = px.line(history_df, x='timestamp', y='confidence',
                                            title="Confidence History",
                                            labels={'confidence': 'Confidence Score', 'timestamp': 'Time'})
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback to matplotlib
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(history_df['timestamp'], history_df['confidence'], marker='o')
                            ax.set_title("Confidence History")
                            ax.set_xlabel("Time")
                            ax.set_ylabel("Confidence Score")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)

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
            if accuracy is not None:
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
                        if PLOTLY_AVAILABLE:
                            st.bar_chart(report_df[['precision', 'recall', 'f1-score']].drop('accuracy', errors="ignore"))
                        else:
                            # Fallback to matplotlib
                            metrics_df = report_df[['precision', 'recall', 'f1-score']].drop('accuracy', errors="ignore")
                            fig = metrics_df.plot(kind='bar', figsize=(8, 5))
                            plt.title("Precision, Recall, F1-Score")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    st.subheader("📉 Confusion Matrix")
                    st.caption("The confusion matrix displays the predicted results of a classification model compared to the actual labels. This matrix helps identify the strengths and weaknesses of the model in predicting classes.")
                    if PLOTLY_AVAILABLE:
                        fig = px.imshow(cm, 
                                      labels=dict(x="Predicted", y="Actual"),
                                      x=['Negative', 'Positive'],
                                      y=['Negative', 'Positive'],
                                      title="Confusion Matrix")
                        st.plotly_chart(fig)
                    else:
                        # Fallback to matplotlib
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                   xticklabels=['Negative', 'Positive'],
                                   yticklabels=['Negative', 'Positive'], ax=ax)
                        ax.set_title("Confusion Matrix")
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)
                    
                    # ROC Curve
                    if roc_data:
                        st.subheader("📈 ROC Curve")
                        st.caption("The ROC curve shows the trade-off between the true positive rate and false positive rate.")
                        if PLOTLY_AVAILABLE:
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
                        else:
                            # Fallback to matplotlib
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.plot(roc_data[0], roc_data[1], color='darkorange', lw=2,
                                   label=f'ROC curve (area = {roc_data[2]:.2f})')
                            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax.set_xlim([0.0, 1.0])
                            ax.set_ylim([0.0, 1.05])
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                            ax.legend(loc="lower right")
                            st.pyplot(fig)

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
                            if PLOTLY_AVAILABLE:
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
                            else:
                                # Fallback to matplotlib
                                fig, ax = plt.subplots(figsize=(10, 6))
                                top_features = features_df.head(10)
                                color = 'green' if sentiment == 'positive' else 'red'
                                ax.barh(top_features['Word'], top_features['Importance'], color=color)
                                ax.set_title(f"Top 10 Important Features for {sentiment.capitalize()} Class")
                                ax.set_xlabel("Importance")
                                ax.invert_yaxis()
                                plt.tight_layout()
                                st.pyplot(fig)

        with st.container():
            st.header("🔍 Data Insights")
            st.divider()
            col1, col2 = st.columns([2,1], gap='small', vertical_alignment="center")
            col3, col4 = st.columns([1,1])
            col5, col6 = st.columns([2,1], vertical_alignment='center')
            col7, col8 = st.columns([1,0.8], vertical_alignment="center")
            
            with col1:
                st.subheader("📏 Review Length Distribution")
                if PLOTLY_AVAILABLE:
                    fig_length = px.histogram(data, x='review_length',
                                                nbins=50, title="Distribution of Review Text Length",
                                                labels={"review_length" : "Text Length (Words)"},
                                                color_discrete_sequence=["blue"])
                    st.plotly_chart(fig_length)
                else:
                    # Fallback to matplotlib
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(data['review_length'], bins=50, color='blue', alpha=0.7)
                    ax.set_title("Distribution of Review Text Length")
                    ax.set_xlabel("Text Length (Words)")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
    
            with col2:
                st.caption("Overview of the review length in dataset, including key statistics such as count, mean, and distribution of numerical values.")
                st.write(data['review_length'].describe())
        
            with col3:
                sentiment_counts = data['sentiment'].value_counts()
                if PLOTLY_AVAILABLE:
                    fig_sentiment = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values,
                                    labels={'x': 'Sentiment', 'y': 'Count'},
                                    title='Sentiment Distribution', color=sentiment_counts.index,
                                    color_discrete_sequence=['green', 'red'])
                    st.plotly_chart(fig_sentiment)
                else:
                    # Fallback to matplotlib
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['green', 'red']
                    ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
                    ax.set_title('Sentiment Distribution')
                    ax.set_xlabel('Sentiment')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
    
            with col4:
                if PLOTLY_AVAILABLE:
                    fig_boxplt = px.box(data, x='sentiment', y='review_length', color="sentiment",
                                color_discrete_sequence=["green", "red"])
                    st.plotly_chart(fig_boxplt)
                else:
                    # Fallback to matplotlib
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sentiments = data['sentiment'].unique()
                    for i, sentiment in enumerate(sentiments):
                        subset = data[data['sentiment'] == sentiment]
                        color = 'green' if sentiment == 'positive' else 'red'
                        ax.boxplot(subset['review_length'], positions=[i], labels=[sentiment], 
                                  patch_artist=True, boxprops=dict(facecolor=color, alpha=0.7))
                    ax.set_title('Review Length by Sentiment')
                    ax.set_xlabel('Sentiment')
                    ax.set_ylabel('Review Length')
                    st.pyplot(fig)

            with col5:
                st.subheader("📌 Top 20 Most Frequent Words")
                all_words = ' '.join(data["review"]).split()
                common_words = Counter(all_words).most_common(20)
                words, counts = zip(*common_words)

                if PLOTLY_AVAILABLE:
                    fig_words = px.bar(x=counts, y=words, orientation='h', 
                                title="Top 20 Most Frequent Words",
                                labels={'x': 'Frequency', 'y': 'Words'},
                                color_discrete_sequence=['blue'])
                    fig_words.update_yaxes(categoryorder='total ascending')
                    st.plotly_chart(fig_words)
                else:
                    # Fallback to matplotlib
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.barh(words, counts, color='blue')
                    ax.set_title("Top 20 Most Frequent Words")
                    ax.set_xlabel("Frequency")
                    ax.set_ylabel("Words")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)

        
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
                
                if PLOTLY_AVAILABLE:
                    fig_bigrams = px.bar(top_10_bigrams, y=[str(bigram[0]) for bigram in top_10_bigrams], 
                                    x=[bigram[1] for bigram in top_10_bigrams], orientation='h',
                                    title="Top 10 Most Frequent Bigrams",
                                    labels={"x" : "Frequency",
                                         "y" : "Bigrams"},
                                    color_discrete_sequence=["blue"])
                    fig_bigrams.update_yaxes(categoryorder='total ascending')
                    st.plotly_chart(fig_bigrams)
                else:
                    # Fallback to matplotlib
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bigram_labels = [str(bigram[0]) for bigram in top_10_bigrams]
                    bigram_counts = [bigram[1] for bigram in top_10_bigrams]
                    ax.barh(bigram_labels, bigram_counts, color='blue')
                    ax.set_title("Top 10 Most Frequent Bigrams")
                    ax.set_xlabel("Frequency")
                    ax.set_ylabel("Bigrams")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
            
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
    if st.session_state.model_comparison is not None and not st.session_state.model_comparison.empty:
        st.subheader("Model Comparison Results")
        comparison_df = st.session_state.model_comparison
        
        # Display comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualize comparison
        if PLOTLY_AVAILABLE:
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
        else:
            # Fallback to matplotlib
            # Accuracy comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            models = comparison_df['Model'].unique()
            vectorizers = comparison_df['Vectorizer'].unique()
            x = np.arange(len(models))
            width = 0.35
            
            for i, vec in enumerate(vectorizers):
                values = comparison_df[comparison_df['Vectorizer'] == vec]['Accuracy'].values
                ax.bar(x + i*width, values, width, label=vec)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(models)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # F1-Score comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            for i, vec in enumerate(vectorizers):
                values = comparison_df[comparison_df['Vectorizer'] == vec]['F1-Score'].values
                ax.bar(x + i*width, values, width, label=vec)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('F1-Score')
            ax.set_title('Model F1-Score Comparison')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(models)
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
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
        <p>Built with ❤️ using Streamlit, Scikit-learn, and Matplotlib</p>
    </div>
    """, unsafe_allow_html=True)
