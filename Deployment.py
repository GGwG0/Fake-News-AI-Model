#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st

import nltk
import numpy 
import joblib
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

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

def preprocess(article):
    stop_word = set(stopwords.words('english'))
    processed_data = []  # Initialize an empty list
    preprocess_document = word_tokenize(article)
    preprocess_document = [token.lower() for token in preprocess_document if token.isalnum()]
    preprocess_document = [token for token in preprocess_document if token not in stop_word]
    preprocess_document = lemmatize_with_pos(preprocess_document)
    return preprocess_document



# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

print("Test our fake news detection now!")
article = st.text_area("Enter your article:")


# print("\nThere are 2 types of training method:")
# print("1. SVM")
# print("2. Naive Bayes")
# training = input("Choose one of the training method: ")

print("\nIt is detecting ...")

# if training == "1":    
model = joblib.load('custom_svm_model(final).pkl')
#     select_k_best = model.named_steps['feature_selection'] 
count_vectorizer = joblib.load('vectorizer(final).pkl')
# else:
#     model = joblib.load('nb_countvectorizer_model(30_testsize).pkl')
#     count_vectorizer = joblib.load('countvectorizer.pkl')



#Apply the preprocessing function to the 'text' column
if st.button("Predict"): 
    
  processed_article = preprocess(article)
  processed_articles = ' '.join(processed_article)
  new_article_vectorized_count = count_vectorizer.transform([processed_articles])
  prediction = model.predict(new_article_vectorized_count)

  if prediction[0] == 1:
    print("The article is predicted as fake.")
  else:
    print("The article is predicted as true.")
    
#20848 - svm cannot, naive can
#16405 - fake
#14 (true) - svm and bayes cannot


# nb cannot, svm can
#The 2014 IRS filings for the Clinton Foundation have been released and the numbers are absolutely sickening. Out of $91.3 million spent in 2014, the organization s IRS filings show that only $5.2 million of that went to actual charity.See the documents here: ViewIt all gets very ugly when you see the numbers broken down.>$34.8 million the foundation spent on salaries, compensation and employee benefits.>$50.4 million recorded as  other expenses >$851K was marked as  professional fundraising expenses. Despite an additional $30 million in 2014, the Clinton Foundation spent 40% less on charitable grants in 2014 than in 2013. Even as it slashed charitable spending, the foundation increased the amount spent on salaries, employee benefits and compensation by $5 million in 2014. The foundation also spent $5 million more  other expenses  in 2014.It is worth noting that Sean Davis at The Federalist stated,  the bulk of the charitable work lauded by the Clinton Foundation s boosters   the distribution of drugs to impoverished people in developing countries   is no longer even performed by the Clinton Foundation. Those activities were spun off in 2010 and are now managed by the Clinton Health Access Initiative, a completely separate non-profit organization. It is absolutely abhorrent that while she claims to do so much for charity one day, then turns around and lauds her wealth over the heads of hard working Americans the next she is continuously stashing donated funds for her own personal purposes.We all know that politicians aren t going to be honest but this is a crime, literally, and yet it is just another day in the lives of the Clintons.Check out this Haiti citizens interview about the Clintons:H/T [ WZ ]

