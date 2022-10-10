#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:44:10 2022

@author: Rachel Perry & Thomas Chang
"""

# =============================================================================
# Followed guide from course reading:
# https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
# =============================================================================

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from datetime import datetime
start = datetime.now()

# Set Random Seed
np.random.seed(500) # remove to get accuracy range

# Read in the dataset
wikiData = pd.read_csv('wiki_train.csv', encoding='latin-1')
twitterData = pd.read_csv('twitterData_Labeled.csv', encoding='latin-1')

wikiData = wikiData.sample(n=2000)

# Step 1: Data Pre-processing

# Step 1a: Remove blank rows if any
wikiData['comment_text'].dropna(inplace=True)

# Step 1b - Remove stop words, non-alphabet characters, and word lemmatizing
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def text_preprocessing(text):
    # Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

    # Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)


wikiData['text_final'] = wikiData['comment_text'].map(text_preprocessing)

print(wikiData['text_final'].head())

# Step - 2: Pre-Process Test set

# Step 2a: Remove blank rows if any
twitterData['text'].dropna(inplace=True)

# Step 2b - Use text_preprocessing function as on the Training set
twitterData['text_final'] = twitterData['text'].map(text_preprocessing)

print(twitterData['text_final'].head())


# Find most frequent word
from collections import Counter
p = Counter(" ".join(wikiData['text_final']).split()).most_common(10)
rslt = pd.DataFrame(p, columns=['Word', 'Frequency'])
print(rslt.head(1))


c = Counter(wikiData['text_final'])
print(c.most_common(1))

# Train and Test datasets
Train_X = wikiData['text_final']
Train_Y = wikiData['toxic']
Test_X = twitterData['text_final']
Test_Y = twitterData['ttoxic'] # annotator 1
# Test_Y = twitterData['rtoxic'] # annotator 2


# Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comparison to the corpus
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(wikiData['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Step - 4b - Balance the classes for each Train and Test set
from imblearn.over_sampling import RandomOverSampler

# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy=0.5)
# fit and apply the transform
Test_X_over, Test_Y_over = oversample.fit_resample(Test_X_Tfidf, Test_Y)
Train_X_over, Train_Y_over = oversample.fit_resample(Train_X_Tfidf, Train_Y)

# Step - 5: Now we can run different algorithms to classify out data check for accuracy

# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_over,Train_Y_over)

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_over)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y_over)*100)
print(confusion_matrix(Test_Y_over, predictions_NB))

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_over,Train_Y_over)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_over)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y_over)*100)
print(confusion_matrix(Test_Y_over, predictions_SVM))

# Classifier - Algorithm - Random Forests
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(Train_X_over, Train_Y_over)

predictions_RFC = rfc.predict(Test_X_over)
print('RF Accuracy Score -> ', accuracy_score(Test_Y_over, predictions_RFC)*100)
print(confusion_matrix(Test_Y_over, predictions_RFC))

# Append accuracy scores to list
nb_accuracy = []
a = accuracy_score(predictions_NB, Test_Y_over)*100
nb_accuracy.append(a)

svm_accuracy = []
a = accuracy_score(predictions_SVM, Test_Y_over)*100
svm_accuracy.append(a)

rfc_accuracy = []
a = accuracy_score(Test_Y_over, predictions_RFC)*100
rfc_accuracy.append(a)

# Step - 6: Compare accuracies to unbalanced classes

# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("Unbalanced Class: Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print(confusion_matrix(Test_Y, predictions_NB))

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("Unbalanced Class: SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print(confusion_matrix(Test_Y, predictions_SVM))

# Classifier - Algorithm - Random Forests
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(Train_X_Tfidf, Train_Y)

predictions_RFC = rfc.predict(Test_X_Tfidf)
print('Unbalanced Class: RF Accuracy Score -> ', accuracy_score(Test_Y, predictions_RFC)*100)
print(confusion_matrix(Test_Y, predictions_RFC))

# Append accuracy scores to list
a = accuracy_score(predictions_NB, Test_Y)*100
nb_accuracy.append(a)

a = accuracy_score(predictions_SVM, Test_Y)*100
svm_accuracy.append(a)

a = accuracy_score(Test_Y, predictions_RFC)*100
rfc_accuracy.append(a)


# Plot
X = ['Balanced Classes', 'Unbalanced Training Dataset']
X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, nb_accuracy, 0.2, label = 'NB')
plt.bar(X_axis, svm_accuracy, 0.2, label = 'SVM')
plt.bar(X_axis + 0.2, rfc_accuracy, 0.2, label = 'RFC')
plt.xticks(X_axis, X)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Sampled")
plt.legend()
plt.show()



now = datetime.now()
print('Time elapsed: ', now-start)

