import streamlit as st
import numpy as np
import joblib
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

classify=joblib.load(open("spamdetection.pkl","rb"))
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # convert to lowercase
    text = text.lower()
    # remove special characters
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text
def spam_detection(message):
    text=clean_text(message)
    return classify.predict([text])

def main():
    
    html_temp="""
    <div style="padding:10px">
    <h4 style="text-align:center;">Streamlit Spam Detection ML App </h4>
    </div>
    
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    with st.form("my_form",clear_on_submit = True):
        message=st.text_area("Enter your message",placeholder="Enter your message")
        submitted = st.form_submit_button("Predict")
    if submitted:
        result=spam_detection(message)
        if result[0]==0 and message!="":
            
            st.success("**"+message+"**"+" is not spam")
        else:
            if message!="":
                st.error( "Spam message indentified")
                st.write("**"+message+"**",indent=2)
    if st.button("About"):
        st.text("Machine Learning based Spam Detection")
        st.text("Built with Streamlit")

if __name__ == "__main__":
    main()
