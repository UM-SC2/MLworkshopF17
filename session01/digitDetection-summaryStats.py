import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from matplotlib import pyplot as plt

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

## Data output
print('Coefficients: \n', logreg.coef_)
print('Non-zero Coefficients \n', np.nonzero(logreg.coef_))
fracWrong = np.sum(np.sign(np.absolute(ytest - ypred)))/ytest.shape[0]
print('Fraction correct: \n', 1-fracWrong)

## Pictoral Summary
# To make beter sense of your results, make a figure that shows the actual
# labels (top) compared to their predicted labels (bottom). Differences
# between the top and bottom show the mis-labeled digits
sorted_inds = np.argsort(ytest, axis=0)

fig1, ax1 = plt.subplots(2)
ax1[0].scatter(np.arange(xtest.shape[0]), ytest[sorted_inds], facecolors="none", edgecolors="black", label="Actual Label")
ax1[1].scatter(np.arange(xtest.shape[0]), ypred[sorted_inds], facecolors="none", edgecolors="red", label="Predicted Label")
ax1[0].legend(loc='best')
ax1[1].legend(loc='best')
ax1[1].set_xlabel('Picture #')
ax1[0].set_ylabel('Picture Label')
ax1[1].set_ylabel('Picture Label')

fig1.savefig("pictoralSummary-digitDetection.png")

## Incorrect Guess Histogram
# Show a figure indicating the true digit compared to how many errors it made
# and what those erros were (restatement of a confusion matrix).
fig2, ax2 = plt.subplots(1)
results = []
for n in range(10):
    n_results = []
    yt = ytest[ytest==n]
    yp = ypred[ytest==n]
    for m in range(10):
        res = np.sum(np.sign(np.absolute(yt[yp==m] - yp[yp==m])))
        n_results.append(res)
    results.append(n_results)

results = np.array(results)

for m in range(10):
    if m > 0:
        ax2.bar(np.arange(10), results[:,m], bottom=np.sum(results[:,:m], axis=1), label=m)
    else:
        ax2.bar(np.arange(10), results[:,m], label=m)
ax2.set_xlabel("Actual Number")
ax2.set_xticks(np.arange(10))
ax2.set_xticklabels(np.arange(10))
ax2.set_ylabel("Incorrect Guesses")
ax2.legend(loc='best')
ax2.set_title("Incorrect Guess Histogram")

fig2.savefig("incorrectGuessHistogram-digitDetection.png")

## Confusion Matrix
# A confusion matrix shows a direct comparison between your guesses and the actual
# labels. Scikit-learn can do this natively and can be visualized with matplotlib.
cnf_matrix = metrics.confusion_matrix(ytest, ypred)
print("Not normalized confusion matrix.")
print(cnf_matrix)
norm_cnf_matrix = cnf_matrix.astype('float')/cnf_matrix.sum(axis=1)[:, np.newaxis]
fig3, ax3 = plt.subplots(1)
cax = ax3.imshow(norm_cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
cbar = fig3.colorbar(cax)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
ax3.set_xlabel("Predicted Number")
ax3.set_ylabel("Actual Number")
ax3.set_title("Normalized Confusion Matrix")

fig3.savefig("normedCnfMatrix-digitDetection.png")
