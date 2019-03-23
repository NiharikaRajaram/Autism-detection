# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 19:58:41 2018

@author: Rajaram
"""

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import webbrowser


# path contains the filename of the dataset
path = "Toddler Autism dataset July 2018 (1).csv"
# reading the contents of the dataset
df = read_csv(path,names=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","class"])
X = np.array(df.drop("a",axis=1))
Y = np.array(df["class"])
# splitting the dataset such that 80% of the data is trained and the remaining 20% is for testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

# object creation of KNeighborsClassifier
knc = KNC()
# fit the training values
knc.fit(X_train,Y_train)
# predicting the outputs for the test values
result = knc.predict(X_test)

print("Predicted Expected")
# to count the number of wrong predictions
wrongPredict = 0
# printing the predcitions for the test values
for i,j in zip(result, Y_test):
    print(str(i) + "\t" + str(j))
    if i!=j:
        wrongPredict += 1
# printing the number of wrong predictions
print("Wrong Predicts = " + str(wrongPredict))

# printing the accuracy score
acc = accuracy_score(Y_test, result)
print("Accuracy: " + str(acc))

# printing the confusion matrix
cm = confusion_matrix(Y_test, result)
print()
print("CONFUSION MATRIX: ")
print(cm)

# printing the f1 score
from sklearn.metrics import f1_score
f1 = f1_score(Y_test, result, average='binary')
print()
print("f1 score: " + str(f1))
print()

# predicting the value for the given single input
example = np.array([0,1,1,0,1,1,0,0,0,1,33,1,1,1,0,1,1])
example = example.reshape(1,-1)
answer = knc.predict(example)

if(answer == 1):
    print("Autism detected")
    webbrowser.open("https://www.helpguide.org/articles/autism-learning-disabilities/helping-your-child-with-autism-thrive.htm")
else:
    print("Autism not detected")