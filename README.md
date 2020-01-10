# Logistical Regression
This repository contains Codes which implement Logistical Regression and Multinomial Naive Bayes to classify restaurant reviews.

## SciKit
SciKit is a python library used commonly to create and use various Machine Learning Models(statistical) in Python.

### Installation 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install sklearn.
```bash
pip install sklearn
```
### Usage

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report  #imporrting evaluation metrics

bow_transformer = CountVectorizer(analyzer=tokenize, max_features=800).fit(X)  #Creating word vectors 
X = bow_transformer.transform(X)  # transforming all the reviews into vectors

logistic_model = LogisticRegression()  #Creating a Logistical Regression Model
nb_model = MultinomialNB() #Creating a Naive-Bayes model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101) #Splitting x and y into training a test data in a 80-20(4:1) ratio
logistic_model.fit(X_train,y_train) #training the model on the traing data

preds = logistic_model.predict(X_test) uding the testing data to make prdictions with the model for evaluation
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
