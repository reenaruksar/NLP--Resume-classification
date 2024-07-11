# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:57:28 2024

@author: Ruksar shaik
"""
# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import click
import spacy
import docx2txt
import pdfplumber
from pickle import load
import requests
import re
import os
import sklearn
import PyPDF2
import nltk
import pickle as pk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot  as plt
stop=set(stopwords.words('english'))
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
from sklearn.feature_extraction.text import TfidfVectorizer

#----------------------------------------------------------------------------------------------------

st.title(' RESUME CLASSIFICATION BY GROUP2')
st.markdown('<style>h1{color: Yellow;}</style>', unsafe_allow_html=True)
st.subheader('Upload your Resume Files Below')

# FUNCTIONS
def extract_skills(resume_text):

    nlp_text = nlp(resume_text) 
    noun_chunks = nlp_text.noun_chunks

    tokens = [token.text for token in nlp_text if not token.is_stop] 
            
    
    data = pd.read_csv(r"E:\Resume Classification\Resume Classification\skills.csv") 
            
    
    skills = list(data.columns.values)
            
    skillset = []
            
    
    for token in tokens:                
        if token.lower() in skills:
            skillset.append(token)
            
   
    for token in noun_chunks:           
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
            
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def getText(filename):
    # Create empty string 
    fullText = ''
    
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        for para in doc:
            fullText += para
    else:  
        with pdfplumber.open(filename) as pdf_file:
            # Use PyPDF2's PdfReader to read text from PDF
            pdoc = PyPDF2.PdfReader(filename)
            number_of_pages = len(pdoc.pages)
            
            # Extract text from each page
            for page_num in range(number_of_pages):
                page = pdoc.pages[page_num]
                page_content = page.extract_text()
                fullText += page_content
             
    return fullText


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))

    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
            
    return resume


def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type=pd.DataFrame([], columns=['Uploaded File',  'Predicted Profile','Skills',])
filename = []
predicted = []
skills = []


import pickle as pk
model = pk.load(open(r'E:\Resume Classification\Resume Classification\model_DT.pkl', 'rb'))
Vectorizer = pk.load(open(r'E:\Resume Classification\Resume Classification\vector.pkl', 'rb'))

upload_file = st.file_uploader('',
                                type= ['docx','pdf'],accept_multiple_files=True)
  
for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned=preprocess(display(doc_file))
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        skills.append(extract_skills(extText))
        
if len(predicted) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())
    
    select = ['PeopleSoft','SQL Developer','React JS Developer','Workday']
    st.subheader('Select as per Requirement')
    option = st.selectbox('Fields',select)

    if option == 'PeopleSoft':
        st.table(file_type[file_type['Predicted Profile'] == 'PeopleSoft'])
    elif option == 'SQL Developer':
        st.table(file_type[file_type['Predicted Profile'] == 'SQL Developer'])
    elif option == 'React JS Developer':
        st.table(file_type[file_type['Predicted Profile'] == 'React JS Developer'])
    elif option == 'Workday':
        st.table(file_type[file_type['Predicted Profile'] == 'Workday'])
        
        
        
        
        
        
        
        
        
