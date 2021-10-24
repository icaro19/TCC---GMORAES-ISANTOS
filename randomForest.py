from numpy import mean
from numpy import std

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def RF_TCC(x, y):
    model = RandomForestClassifier()
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

def RF_TCC_Predict(x, y, row):
    # define the model
    model = RandomForestClassifier()
    # fit the model on the whole dataset
    model.fit(x, y)
    # make a single prediction
    yhat = model.predict(row)
    print('Predicted Class: ', yhat)

def RF_TCC_TesteKFoldDifferentValues(x, y):

    model = RandomForestClassifier()

    for i in range(2, 21):
        cv = RepeatedStratifiedKFold(n_splits=i, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        print('Accuracy with %d splits: %.3f (%.3f)' % (i, mean(n_scores), std(n_scores)))