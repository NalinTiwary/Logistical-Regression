import pandas as pd
import numpy as np
import nltk
import spacy
import os
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import gdown
!python -m spacy download en_core_web_md
import en_core_web_md

nlp = en_core_web_md.load()

gdown.download('https://drive.google.com/uc?id=1u0tnEF2Q1a7H_gUEH-ZB3ATx02w8dF4p', 'yelp_final.csv', True)
data_file  = 'yelp_final.csv'

def is_good_review(stars):
    if stars==3:
        return True
    else:
        return False

nltk.download('wordnet')
nltk.download('punkt')
language_model = spacy.load('en_core_web_sm')
yelp = pd.read_csv(data_file)
yelp.drop(labels=['business_id','user_id'],inplace=True,axis=1
nltk.download('stopwords', quiet=True)

yelp['is_good_review'] = yelp['stars'].apply(is_good_review)
X = yelp['text']
y = yelp['is_good_review']
bow_transformer = CountVectorizer(analyzer=tokenize, max_features=800).fit(X)
X = bow_transformer.transform(X)

logistic_model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
logistic_model.fit(X_train,y_train)

X1=input('Please enter a Restraunt review')
prediction = logistic_model.predict(bow_transformer.transform([X1]))
if prediction:
  print ("This was a GOOD review!")
else:
  print ("This was a BAD review!")
  
preds = logistic_model.predict(X_test)

cm = confusion_matrix(y_test, preds)
TP = cm[0][0]
TN = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]

accuracy = 1-((FP+FN)/(len(y_test)))

print('The accuracy of the model is %0.2f percent' %(accuracy*100))
