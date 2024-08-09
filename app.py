import pandas as pd 
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

#loading dataset
data=pd.read_csv('amazon_product.csv')

#droping unwanted columns
data.drop("id",axis=1,inplace=True)

#defining stem and tokenizer
stemmer=SnowballStemmer('english')
def tokenize_stem(text):
    tokens=nltk.word_tokenize(text.lower())
    stemmed=[stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

data['stemmed_tokens']=data.apply(lambda row:tokenize_stem(row['Title']+" "+row['Description']),axis=1)

#defining tfidf and cosine similiarity check
tfidfv=TfidfVectorizer(tokenizer=tokenize_stem)
def cosine_sim(txt1,txt2):
    matrix=tfidfv.fit_transform([txt1,txt2])
    return cosine_similarity(matrix)[0,1]

#recommend product based on similarity or defining search function
def search_product(query):
    stemmed_query=tokenize_stem(query)
    #calculate cosine similarity btwn query and tokens columns
    data['similarity']=data['stemmed_tokens'].apply(lambda x:cosine_sim(stemmed_query,x))
    res=data.sort_values(by=['similarity'],ascending=False).head(10)[['Title','Description','Category']]
    return res


st.title("Search Engine and Recommendation system")
query=st.text_input("enter product name")
submit=st.button("Search")
if submit:
    res=search_product(query)
    st.write(res)