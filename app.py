import streamlit as st
import pickle
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))



st.title('SMS Spam Classifier')
input_sms = st.text_input('Enter a message')



import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    # converting into lower case
    text = text.lower()
    text = nltk.word_tokenize(text)

    # removing non-alphanumeric characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # removing stopwords and punctuation
    cleaned_text = []
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            cleaned_text.append(i)

    root_words = []
    for i in cleaned_text:
        root_words.append(ps.stem(i))

    return " ".join(root_words)

#step 1. preprocess
transformed_sms = transform_text(input_sms)

#step2. vectorize
vector_input = tfidf.transform([transformed_sms])

#step3. predict
result = model.predict(vector_input)[0]


#step4. Display
if result ==1:
    st.header('Spam')
else:
    st.header('Not spam')
