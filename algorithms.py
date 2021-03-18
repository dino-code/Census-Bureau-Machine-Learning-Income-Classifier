from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import preprocessing
import functions as fun
from sklearn.ensemble import AdaBoostClassifier


xtrain, ytrain, filterCol = fun.computeX(
    'census-income.data.csv')
xtest, ytest, filterCol = fun.computeX('census-income.test.csv')

fun.baggingEnsemble(xtrain, ytrain, xtest, ytest, filterCol)
