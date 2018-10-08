import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from matplotlib import pyplot as plt

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

## Data output
print('Coefficients: \n', logreg.coef_)
fracWrong = np.sum(np.absolute(ytest - ypred))/ytest.shape[0]
print('Fraction correct: \n', 1-fracWrong)

## Confusion Matrix
# A confusion matrix shows a direct comparison between your guesses and the actual
# labels. Scikit-learn can do this natively and can be visualized with matplotlib.
cnf_matrix = metrics.confusion_matrix(ytest, ypred)
print("Not normalized confusion matrix.")
print(cnf_matrix)
norm_cnf_matrix = cnf_matrix.astype('float')/cnf_matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(1)
cax = ax.imshow(norm_cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
cbar = fig.colorbar(cax, boundaries=np.linspace(0,1, 1e4), ticks=[0, 0.25, 0.5, 0.75, 1.0])
ticks = cbar.get_ticks()
cbar.set_ticks(ticks)
cbar.set_ticklabels(["{}%".format(round(t*100)) for t in ticks])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not Spam', 'Spam'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Not Spam', 'Spam'])
ax.set_xlabel("Predicted Number")
ax.set_ylabel("Actual Number")
ax.set_title("Normalized Confusion Matrix")

fig.savefig("normedCnfMatrix-spambase.png")
