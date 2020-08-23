import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

reviews = pd.read_excel("reviews.xlsx")


def spliter(rev, sen, length):
    n = int(0.7*length)
    x_train = rev[:n]
    x_test = rev[n:]
    y_train = sen[:n]
    y_test = sen[n:]
    return x_train, x_test, y_train, y_test


vector = CountVectorizer()
X_train, X_test, Y_train, Y_test = spliter(reviews["review"], reviews["sentiment"], reviews.shape[0])
X_train = vector.fit_transform(X_train)
X_test = vector.transform(X_test.values.astype('U'))
scores = cross_val_score(LogisticRegression(), X_train, Y_train, cv=5)
print("Mean cross-validation accuracy", np.mean(scores))

logr = LogisticRegression()
logr.fit(X_train, Y_train)
print("training set score for Logistic regression model is", logr.score(X_train, Y_train))
print("test set score for logistic regression model is", logr.score(X_test, Y_test))
# Multinomial
nb = MultinomialNB()
nb.fit(X_train, Y_train)
print("training set score for multinomial model is", nb.score(X_train, Y_train))
print("test set score for multinomial model is", nb.score(X_test, Y_test))

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
print("training set score for random forest model is ", rf.score(X_train, Y_train))
print("test set score for random forest model is", rf.score(X_test, Y_test))
