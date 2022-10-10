"""
PART 2 [6 POINTS]
Download a YouTube spam collection dataset available from this link. 
This is a public set of comments collected for spam research. It has five 
datasets composed of 1,956 real messages extracted from five videos. 
These five videos are popular pop songs that were among the 10 most viewed in 
the collection period.

All five datasets have the following attributes:
Attribute
    Attribute Explained
COMMENT_ID
    Unique ID representing the comment
AUTHOR
    Author ID
DATE
    Date the comment is posted
CONTENT
    The comment
TAG	
    Attribute Explained
 
INSTRUCTIONS
For this exercise use any four of these five datasets to build a spam 
filter with the Naïve Bayes approach. 

Use that filter to check the accuracy of the remaining dataset.
Make sure to report the details of your training and the model.
"""
import pandas as pd
import statistics 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

yt_data = pd.read_csv('/Users/rachel/Desktop/IMT 574 Datasets/YouTube-Spam-Collection-v1/Youtube03-LMFAO.csv')
# X Dataframe, removes punctuation, and make lower
X = yt_data['CONTENT']
y = yt_data['CLASS']

yt_data['CLASS'].value_counts(normalize=True)
# 1    0.538813
# 0    0.461187
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Validate train and test sets accurately represent the original set
y_train.value_counts(normalize=True)
# 1    0.534286
# 0    0.465714
y_test.value_counts(normalize=True)
# 1    0.556818
# 0    0.443182

#%% # COUNT VECTORIZER
cv_report = []
cv_report = pd.DataFrame(cv_report, columns = ['i', 'accuracy'])
# loop 20 iterations to calculate an average accuracy for each data shuffle
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Convert a collection of comments to a matrix of token counts
    vect = CountVectorizer(stop_words='english')
    # fit & transform
    X_train_counts = vect.fit_transform(X_train)
    # 350x699 sparse matrix of type float64 with 1787 stored elements
    # Create Multinomial NB Classifier
    mnb_og = MultinomialNB()
    # Fit to the data
    mnb_og.fit(X_train_counts, y_train)
    # Predict and test accuracy
    # Transform the test set to a readable format
    X_test_dtm = vect.transform(X_test)
    X_test_dtm .toarray()
    y_test_pred = mnb_og.predict(X_test_dtm)
    cv_report = cv_report.append({'i':i,
                                  'accuracy':accuracy_score(y_test,
                                                            y_test_pred)},
                                  ignore_index = True)
cv_output = cv_report['accuracy']
print("COUNT VECTORIZER (ONLY)")
print("Average Accuracy:", format(statistics.mean(cv_output),".3f"))
print("Accuracy Range:", 
      format(min(cv_output),".3f"), "-",
      format(max(cv_output),".3f"))
# PRINT LAST RUN METRICS TO REPORT
print("ACCURACY SCORE:")
print(accuracy_score(y_test, y_test_pred))
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_test_pred))
print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_test_pred))
"""
COUNT VECTORIZER (ONLY)
AVERAGE SCORES OVER 1000 TRIALS ON A TRAIN/TEST 80/20 SPLIT
Average Accuracy: 0.886
Accuracy Range: 0.773 - 0.966

FINAL TRAIL REPORT:
ACCURACY SCORE: 0.875
CONFUSION MATRIX:
[32  9]
[2  45]
CLASSIFICATION REPORT:
              precision    recall  f1-score   support

           0       0.94      0.78      0.85        41
           1       0.83      0.96      0.89        47

    accuracy                           0.88        88
   macro avg       0.89      0.87      0.87        88
weighted avg       0.88      0.88      0.87        88
"""
#%% # COUNT VECTORIZER AND TF-IDF REPRESENTATION
tfidf_report = []
tfidf_report = pd.DataFrame(tfidf_report, columns = ['i', 'accuracy'])
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Convert a collection of comments to a matrix of token counts
    vect = CountVectorizer(stop_words='english')
    # fit & transform
    X_train_counts = vect.fit_transform(X_train)
    # Transform the count matrix to a normalized tf or tf-idf representation.
    # Attempted different parameter combinations to optimize accuracy
    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True, norm='l2')
    #  fit & transform
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    # 350x699 sparse matrix of type float64 with 1787 stored elements
    # Make the Naive Bayes Classifier and fit the tfidf set
    mnb = MultinomialNB().fit(X_train_tfidf, y_train)
    # Transform the test set to a readable format
    X_test_dtm = vect.transform(X_test)
    X_test_dtm .toarray()
    # Predict and test accuracy
    y_test_pred = mnb.predict(X_test_dtm)
    tfidf_report = tfidf_report.append({'i':i,
                                        'accuracy':accuracy_score(y_test,
                                                                  y_test_pred)},
                                       ignore_index = True)
tfidf_output = tfidf_report['accuracy']
print("COUNT VECTORIZER + TFIDF REPRESENTATION")
print("Average Accuracy:", format(statistics.mean(tfidf_output),".3f"))
print("Accuracy Range:", 
      format(min(tfidf_output),".3f"), "-",
      format(max(tfidf_output),".3f"))
# PRINT LAST RUN METRICS TO REPORT
print("ACCURACY SCORE:")
print(accuracy_score(y_test, y_test_pred))
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_test_pred))
print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_test_pred))
"""
COUNT VECTORIZER + TFIDF REPRESENTATION
AVERAGE SCORES OVER 1000 TRIALS ON A TRAIN/TEST 80/20 SPLIT
Average Accuracy: 0.886
Accuracy Range: 0.773 - 0.977

FINAL TRAIL REPORT:
ACCURACY SCORE: 0.897
CONFUSION MATRIX:
[29  8]
[1  50]
CLASSIFICATION REPORT:
              precision    recall  f1-score   support

           0       0.97      0.78      0.87        37
           1       0.86      0.98      0.92        51

    accuracy                           0.90        88
   macro avg       0.91      0.88      0.89        88
weighted avg       0.91      0.90      0.90        88
"""

#%% USING PIPELINE METHOD
pipe_report = []
pipe_report = pd.DataFrame(pipe_report, columns = ['i', 'accuracy'])
for i in range(1000):
    pipe = Pipeline([('vector', CountVectorizer()),
                     ('tfidf', TfidfTransformer()), 
                     ('mulNB', MultinomialNB())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipe.fit(X_train, y_train)
    pipe_report = pipe_report.append({'i':i,
                                      'accuracy':pipe.score(X_test, y_test)},
                                       ignore_index = True)
pipe_output = pipe_report['accuracy']
print("PIPELINE")
print("Average Accuracy:", format(statistics.mean(pipe_output),".3f"))
print("Accuracy Range:", 
      format(min(pipe_output),".3f"), "-",
      format(max(pipe_output),".3f"))

scores = cross_validate(pipe, X_train, y_train)
scores['test_score'].mean()
"""
PIPELINE
Average Accuracy: 0.900
Accuracy Range: 0.784 - 0.977

CV results:
{'fit_time': array([0.01127815, 0.00739312, 0.01160812, 0.00694466, 0.00675511]),
 'score_time': array([0.00170803, 0.00198197, 0.001719  , 0.00150609, 0.0020318 ]),
 'test_score': array([0.82857143, 0.92857143, 0.87142857, 0.9       , 0.92857143])}
mean: 0.891
"""


"""

"""