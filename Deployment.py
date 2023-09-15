import streamlit as st
import nltk
import joblib

# Download NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Initialize NLTK's WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Function to lemmatize and provide POS tags
def lemmatize_with_pos(text):
    # Get POS tags for the words
    pos_tags = pos_tag(text)
    # Lemmatize using POS tags
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemmatized_words

# Function to map POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    tag = tag[0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)  # Default to noun if not found

# Function for text preprocessing
def preprocess(article):
    stop_words = set(stopwords.words('english'))
    preprocess_document = word_tokenize(article.lower())
    preprocess_document = [token for token in preprocess_document if token.isalnum() and token not in stop_words]
    preprocess_document = lemmatize_with_pos(preprocess_document)
    return ' '.join(preprocess_document)

# Load machine learning model and count vectorizer
model = joblib.load('custom_svm_model(final).pkl')
count_vectorizer = joblib.load('vectorizer(final).pkl')

# Set page title and background color
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered", initial_sidebar_state="collapsed")

# Stylish header
st.title("Fake News Detection")
st.markdown("**Enter an article to predict whether it's fake or true.**")

# Text input area
article = st.text_area("Article Text", "", height=200)

# Predict button
if st.button("Predict", key="predict_button"):
    st.info("Predicting...")

    # Preprocess the article
    processed_article = preprocess(article)
    
    # Vectorize the preprocessed article
    new_article_vectorized_count = count_vectorizer.transform([processed_article])

    # Make the prediction
    prediction = model.predict(new_article_vectorized_count)

    # Display the prediction
    if prediction[0] == 1:
        st.success("📢 The article is predicted as fake.")
    else:
        st.success("📰 The article is predicted as true.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ by Your Name")

# Add a classy background color
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)
