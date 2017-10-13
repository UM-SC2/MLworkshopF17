import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection

emails = pd.read_csv("spambase/spambase.data", header=-1)

xmail = np.array(emails.iloc[:,:-1])
ymail = np.array(emails.iloc[:,-1])

fractest = 0.2

## Split into test and train data with model_selection.train_test_split.
## In order for the program to work, please save your test labels as ytest.
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(xmail, ymail, test_size=fractest)
# Create your linear_model.LogisticRegression() object
logreg = linear_model.LogisticRegression()
# Fit your model object
logreg.fit(xtrain, ytrain)
# Make predictions on the test set. Please save your predictions as ypred.
ypred = logreg.predict(xtest)
#
#
#
## Data output
print('Coefficients: \n', logreg.coef_)
fracWrong = np.sum(np.absolute(ytest - ypred))/ytest.shape[0]
print('Fraction correct: \n', 1-fracWrong)
