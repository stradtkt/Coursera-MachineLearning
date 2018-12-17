import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def answer_zero():
    return len(cancer['feature_names'])

answer_zero()

def answer_one():
    columns = np.append(cancer.feature_names, 'target')
    index = pd.RangeIndex(start=0, stop=569, step=1)
    data = np.column_stack((cancer.data, cancer.target))
    df = pd.DataFrame(data=data, index=index, columns=columns)
    return df

answer_one()

def answer_two():
    cancerdf = answer_one()
    index = ['malignant', 'benign']
    malignant = np.where(cancerdf['target'] == 0.0)
    bening = np.where(cancerdf['target'] == 1.0)
    data = [np.size(malignant), np.size(bening)]
    series = pd.Series(data, index=index)
    return series 

answer_two()

def answer_three():
    cancerdf = answer_one()
    X = cancerdf.drop('target', axis=1)
    y = cancerdf.get('target')
    return X, y

answer_three()

def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test

answer_four()

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)
    return knn

answer_five()

def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn = answer_five()
    return knn.predict(means)

answer_six()


def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    test_pred = knn.predict(X_test)
    return test_pred

answer_seven()


def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score = knn.score(X_test, y_test)
    return score

answer_eight()