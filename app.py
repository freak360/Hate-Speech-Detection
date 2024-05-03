import pickle
import streamlit as st
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Defining a function that will clean the text
def clean_text(text):
    punc = list(punctuation)
    stop = stopwords.words('english')
    bad_tokens = punc + stop
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()
    word_tokens = [t for t in tokens if t.isalpha()]
    clean_tokens = [lemma.lemmatize(t.lower()) for t in word_tokens if t not in bad_tokens]
    return ' '.join(clean_tokens)


# defining the main function and creating the ui
def main():
    st.set_page_config(page_title='Hate Speech Detector', page_icon='🤖')
    st.header("Hate Speech Detection System")
    text = st.text_area('Enter Text')
    submit = st.button('Submit')
    if submit:
        itext = clean_text(text)
        vectorized_text = vectorizer.transform([itext])
        response = model.predict(vectorized_text)[0]
        if response == 0:
            st.write('Hate Speech')
        if response == 1:
            st.write('Offensive Language')
        else:
            st.write('No hate speech or offensive language detected')

if __name__ == "__main__":
    main()

