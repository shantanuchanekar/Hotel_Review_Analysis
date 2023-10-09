#!/usr/bin/env python
# coding: utf-8

# In[1]:

import nltk
nltk.download('wordnet')
import pandas as pd
import warnings
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from rake_nltk import Rake
import pickle
import streamlit as st
import numpy as np
from nltk.stem import PorterStemmer,WordNetLemmatizer
nltk.download('punkt')
nltk.download('vader_lexicon')
from wordcloud import WordCloud, STOPWORDS
# Warnings ignore 
warnings.filterwarnings(action='ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
 background-image:
    linear-gradient(90deg, rgba(62,62,62,0.3925945378151261) 26%, rgba(0,0,0,0.6138830532212884) 100%),
    url("https://images.pexels.com/photos/260922/pexels-photo-260922.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# loading the trained model
pickle_in = open("model.pkl", 'rb')
model = pickle.load(pickle_in)

pickle_in = open("vectorizer.pkl", 'rb')
vectorizer = pickle.load(pickle_in)

# Title of the application
st.header("𝐏𝐫𝐞𝐝𝐢𝐜𝐭 𝐑𝐚𝐭𝐢𝐧𝐠𝐬 𝐟𝐨𝐫 𝐇𝐨𝐭𝐞𝐥 𝐑𝐞𝐯𝐢𝐞𝐰𝐬")
st.subheader("𝐄𝐧𝐭𝐞𝐫 𝐓𝐡𝐞 𝐑𝐞𝐯𝐢𝐞𝐰 𝐓𝐨 𝐀𝐧𝐚𝐥𝐲𝐳𝐞")

input_text = st.text_area("𝐓𝐲𝐩𝐞 𝐫𝐞𝐯𝐢𝐞𝐰 𝐡𝐞𝐫𝐞", height=150)

option = st.sidebar.selectbox('Menu bar',['Sentiment Analysis','Keywords'])
st.set_option('deprecation.showfileUploaderEncoding', False)
if option == "Sentiment Analysis":
    
    
    
    if st.button("Predict sentiment"):
       
        wordnet=WordNetLemmatizer()
        text=re.sub('[^A-za-z0-9]',' ',input_text)
        text=text.lower()
        text=text.split(' ')
        text = [wordnet.lemmatize(word) for word in text if word not in (stopwords.words('english'))]
        text = ' '.join(text)
        pickle_in = open("model.pkl", 'rb')
        model = pickle.load(pickle_in)
        pickle_in = open("vectorizer.pkl", 'rb')
        vectorizer = pickle.load(pickle_in)
        transformed_input = vectorizer.transform([text])

        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        sentiments = SentimentIntensityAnalyzer()
        pos = sentiments.polarity_scores(input_text)["pos"]
        neg = sentiments.polarity_scores(input_text)["neg"]
        neu = sentiments.polarity_scores(input_text)["neu"]



        if model.predict(transformed_input)  == 0:
            st.success(f"𝐍𝐞𝐠𝐚𝐭𝐢𝐯𝐞 𝐒𝐭𝐚𝐭𝐞𝐦𝐞𝐧𝐭 😔\n The review is {round((neg)*100,2)}% negative, {round((pos)*100,2)}% positive and {round((neu)*100,2)}% neutral")

        elif model.predict(transformed_input) == 1:
            st.success(f"𝐏𝐨𝐬𝐢𝐭𝐢𝐯𝐞 𝐒𝐭𝐚𝐭𝐞𝐦𝐞𝐧𝐭 😃\n The review is {round((pos)*100,2)}% positive,\n {round((neu)*100,2)}% neutral and \n {round((neg)*100,2)}% negative")

            st.snow()
        else:
            st.success(f"Neutral 😶\n The review is {round((neu)*100,2)}% neutral")
        

elif option == "Keywords":
    st.header("Keywords")
    if st.button("Keywords"):
        
        r=Rake(language='english')
        r.extract_keywords_from_text(input_text)
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Get the important phrases
        phrases = r.get_ranked_phrases()
        # Display the important phrases
        st.write("These are the **keywords** causing the above sentiment:")
        st.snow()
        for i, p in enumerate(phrases):
            st.write(i+1, p)


