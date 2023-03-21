import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import numpy as np
from random import sample
import json
import os 
import string
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import networkx as nx
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from spacy import displacy
from spacy.matcher import Matcher 
nlp = spacy.load('en_core_web_sm')
from sklearn.metrics.pairwise import cosine_similarity
nlp = spacy.load('en_core_web_sm')
pd.set_option('display.max_colwidth', 200)

import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec


#######-----------Utility Functions----------

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
def makeittext(text):
    all_text=''
    for i in text:
        all_text+=i+' '
    return all_text
def filter_data(query,x):
    if any(elem in query for elem in x):
        return True
    else:
        return False
def pre_process_query(query):
    query=query.lower()
    query=query.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize( query)
    query=' '.join([word for word in words if word.casefold() not in stop_words])
    ps = PorterStemmer()
    words = nltk.word_tokenize(query)
    stemmed_words = [ps.stem(word) for word in words]
    query=' '.join(stemmed_words)
    return query
def filter_documents(query,df):
    query=pre_process_query(query)
    query_entity=get_subject_target_relationship(query)
    df_source=df[df['source']==query_entity[0]]
    df_target=df[df['target']==query_entity[1]]
    df_relation=df[df['relationship']==query_entity[2]]
    filtered_df=pd.concat([df_source,df_target,df_relation])
    return filtered_df
def tokenize(text):
    stemmer = gensim.parsing.PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]


def get_rankings(filtered,df,query):
    paper_id_unique=filtered['paper_id'].unique()

    filtered_df = df[df['paper_id'].isin(paper_id_unique)]
    vectorizer = TfidfVectorizer()
    text = vectorizer.fit_transform(filtered_df['text_together'])
    quer_vector=vectorizer.transform([query])
    cos_sim=cosine_similarity(quer_vector, text)
    cosim_shape=cos_sim.shape
    filtered_df['cos_sim']=cos_sim.reshape(cosim_shape[1],cosim_shape[0])
    filtered_df=filtered_df.sort_values(by='cos_sim',ascending=False)
    filtered_df=filtered_df[filtered_df['title']!=None]
    return filtered_df
def get_subject_target_relationship(text):
    doc = nlp(text)
    subject = ''
    target = ''
    relationship = ''
    for token in doc:
        if 'subj' in token.dep_:
            subject = token.text
        if 'obj' in token.dep_:
            target = token.text
        if 'prep' in token.dep_:
            relationship = token.text

    return [subject, target, relationship] 



from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()


### get all saved dataframe
df_entity_rel=pd.read_csv('./df_all.csv')
df_realationship=pd.read_csv('./df_realationship.csv')


def show_knowledge_graph(query,df_realationship):
    df_pdf_non= df_realationship[(df_realationship['relationship']!="") & (df_realationship['target']!='') & (df_realationship['source']!='')]
    query=pre_process_query(query)
    query_entity=get_subject_target_relationship(query)
    df_source=df_pdf_non[df_pdf_non['source']==query_entity[0]]
    df_target=df_pdf_non[df_pdf_non['target']==query_entity[1]]
    df_relation=df_pdf_non[df_pdf_non['relationship']==query_entity[2]]
    # plotting graphs
    
    G=nx.from_pandas_edgelist(df_pdf_non,'source','target', 
                              edge_attr=True, create_using=nx.MultiDiGraph())


    plt.figure()
    pos = nx.spring_layout(G, k = 1) # k regulates the distance between nodes
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=15, edge_cmap=plt.cm.Blues, pos = pos)
    # plt.show()
    st.pyplot(plt)









# df=get_rankings(df_pdf_non,query)
# df_filtered=filter_docs(df,'viral')


st.title('Information Retrieval App')
query = st.text_input('Enter a query:')



df_pdf_non= pd.read_csv('./df_pdf_non.csv')
df_text=pd.read_csv('./df_text.csv')

# query='Why should be wear mask in COVID'
if query!="":
    
    df=filter_documents(query,df_pdf_non)
    if df.shape[0]!=0:
        cos_sim=get_rankings(df,df_text,query)
        st.dataframe(cos_sim[['paper_id','title','cos_sim']] ) 
        
    else:
        st.write('Could not find any articles')



    # df=get_rankings(df_pdf_non,query)
    # df_filtered=filter_docs(df,'viral')
        
    # if st.button('Show Knowledge Graph'):
    #     show_knowledge_graph(query,df_realationship)
