import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn import linear_model
from sklearn import model_selection

mnist = fetch_mldata("MNIST original")

# There are 70000 samples of 28x28 pictures of digits in this dataset. The algorithm takes a long time to train with this data, but if you use too little data, the fit won't be as good. feel free to play with this as necessary. 
dataset_size = 1000
choices = np.random.randint(mnist.data.shape[0], size=dataset_size)

xdigits = mnist.data[choices]
ydigits = mnist.target[choices]

fractest = 0.2

## Split into test and train data with model_selection.train_test_split.
## In order for the program to work, please save your test labels as ytest.
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(xdigits, ydigits, test_size=fractest)
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
print('Non-zero Coefficients \n', np.nonzero(logreg.coef_))
fracWrong = np.sum(np.sign(np.absolute(ytest - ypred)))/ytest.shape[0]
print('Fraction correct: \n', 1-fracWrong)
