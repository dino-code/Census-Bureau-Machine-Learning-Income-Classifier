from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import statistics
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

''' Removes unknown values, encodes dicrete data, and normalizes continuous data '''


def pcc(x, y):
    """ Function computes the PCC of the given data.

        Args:
                x: feature data
                y: labels

        Retuns:
                correlation: the PCC of the passed data

        """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov_x_y = np.sum((x - mean_x) * (y - mean_y))
    stdv_x = np.sum(np.square(x - mean_x))
    stdv_y = np.sum(np.square(y - mean_y))
    std = stdv_x * stdv_y
    if std == 0:
        std = 1
    correlation = cov_x_y / np.sqrt(std)

    return correlation


def optimalFeatures(X: np.array, Y):
    """ Function optimalFeatures sorts the most optimal features from the passed dataset

        Args: 
                X: feature data
                Y: labels

        Returns: 
                indexes: sorted list of computed 'r' values (PCC) from greatest to least
        """
    r = []
    for x in X.T:
        r.append(pcc(x, Y))
    r = np.absolute(r)
    r_sorted = sorted(r, reverse=True)

    xt = [q for q in range(X.shape[1])]

    indexes = []  # store indexes of r sorted
    for i in range(X.shape[1]):
        # print(xt[np.where(r == r_sorted[i])[0][0]], ': ',  r_sorted[i])
        a = np.where(r == r_sorted[i])[0][0]
        indexes.append(a)
    return indexes

# function used within accuracyFilter to encode each column


def columnEncoder(dataSubset):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(dataSubset)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    new_xtrCol = onehot_encoder.fit_transform(integer_encoded)

    return new_xtrCol


def accuracyFilter(algo, xtr, ytr, xte, yte, filterCol, isTrain):
    """ Function accuracyFilter takes in the filtered data and applies it to the passed classifier 'algo'.
        Function uses OneHot Encoder to convert discrete data into numerical data

        Args: 
                algo: passed classifier
                xtr: training feature data
                ytr: training labels
                xte: testing feature data
                yte: testing labels
                filterCol: filter column determined by sorted PCC values

        Returns: 
                predicted: 'np.array' of predicted values
        """

    grandAcc = 0
    predicted = []
    continuousColumns = [0, 1, 2, 3, 4, 5]
    discreteColumns = [i for i in range(
        xtr.T.shape[0]) if i not in continuousColumns]

    new_xtr = np.array([])
    new_xte = np.array([])

    # Data are ordinal encoded and by column
    Xtr = xtr.T
    Xte = xte.T

    for i in range(0, Xtr.shape[0]):
        print('ITERATION *********** == ', i)
        if filterCol[i] in discreteColumns:
            # We encoded it this way because we are encoding each feature as we add it to the
            # filter selection for accuracy testing. The normal one hot encoding would encode all data at once and
            # during feature selection there would be no distinguishing between features
            # Train data hotOneEncoding
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(Xtr[filterCol[i]])
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            new_xtrCol = onehot_encoder.fit_transform(integer_encoded)

            # For test data hot one encoding
            label_encoder2 = LabelEncoder()
            integer_encoded2 = label_encoder2.fit_transform(Xte[filterCol[i]])
            onehot_encoder2 = OneHotEncoder(sparse=False)
            integer_encoded2 = integer_encoded2.reshape(
                len(integer_encoded2), 1)
            new_xteCol = onehot_encoder2.fit_transform(integer_encoded2)

            if i == 0:
                new_xtr = new_xtrCol
                new_xte = new_xteCol

            else:
                new_xtr = np.column_stack((new_xtr, new_xtrCol))
                new_xte = np.column_stack((new_xte, new_xteCol))

                if isTrain == False:
                    if new_xtr.shape[1] > new_xte.shape[1]:
                        array = np.zeros(new_xte.shape[0]).T
                        array = array.reshape(new_xte.shape[0], 1)
                        new_xte = np.concatenate((new_xte, array), axis=1)
                    if new_xte.shape[1] > new_xtr.shape[1]:
                        array = np.zeros(new_xtr.shape[0]).T
                        array = array.reshape(new_xtr.shape[0], 1)
                        new_xtr = np.concatenate((new_xtr, array), axis=1)

        # If data is Continuous
        else:
            if i == 0:
                new_xtr = Xtr[filterCol[i]]
                new_xte = Xte[filterCol[i]]
            else:
                new_xtr = np.column_stack((new_xtr, Xtr[filterCol[i]]))
                new_xte = np.column_stack((new_xte, Xte[filterCol[i]]))

        # Compute Accuracy
        if i == 0:
            algo.fit(new_xtr.reshape(-1, 1), ytr)
            algo.predict(new_xte.reshape(-1, 1))
            acc = algo.score(new_xte.reshape(-1, 1), yte)
            print('acc: ', acc)
            if acc > grandAcc:
                grandAcc = acc
                print('grandAcc: ', grandAcc)
                predicted = algo.predict(new_xte.reshape(-1, 1))

        else:
            algo.fit(new_xtr, ytr)
            algo.predict(new_xte)
            acc = algo.score(new_xte, yte)
            print('acc: ', acc)

            if acc > grandAcc:
                grandAcc = acc
                print('grandAcc: ', grandAcc)
                predicted = algo.predict(new_xte)
    print('--------------------------------------\nAccuracy: ', grandAcc)
    return np.array(predicted)


def computeX(csv):
    """ Function computeX takes in raw .csv file and converts data into usable pieces of data for the rest of the code.
        Proceeds through the data and removes redundant data and data with missing features.
        Converts discrete data into numerical form using Ordinal Endcoder

        Args:
                csv: .csv file containing raw data

        Returns: 
                xtrainFilter: cleaned and converted feature data
                ytrain: corresponding labels
                filterColumns: columns with the most optimal data points, calculated by the PCC 'r' value of the features
        """
    # Create a dataframe from csv
    trainData = pd.read_csv(csv, header=None)
    # We decided to deal with missing data by removing it because only about 6% of tuples contained missing data.
    trainData = trainData.replace(' ?', np.nan)
    # Here we drop discrete education values in favor of our continuous education values.
    trainData = trainData.drop(columns=[3])

    # Now we separate discrete values from continuous values
    continuousColumns = [0, 2, 4, 10, 11, 12]
    xtrainContinuous = np.array(trainData[continuousColumns])

    impC = KNNImputer(n_neighbors=7, weights='uniform')
    impC.fit(xtrainContinuous)
    xtrainContinuous = impC.transform(xtrainContinuous)

    discreteValues = [
        i for i in trainData.columns.values if i not in continuousColumns and i != 14]
    xtrainDiscrete = np.array(trainData[discreteValues])

    impD = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    impD.fit(xtrainDiscrete)
    xtrainDiscrete = impD.transform(xtrainDiscrete)

    # Pull a Series of class labels from trainData and convert it to a numpy array
    ytrain = trainData.iloc[:, -1]
    ytrain = ytrain.to_numpy()
    # Convert class labels to 0 is <=50K, else 1.
    if csv == 'census-income.test.csv':
        ytrain = np.where(ytrain == ' <=50K.', 0, 1)
    else:
        ytrain = np.where(ytrain == ' <=50K', 0, 1)

    # Here we normalize the continuous values using max
    xtrainContinuous = preprocessing.normalize(
        xtrainContinuous, norm='max', axis=0, copy=False, return_norm=False)

    OrdinalEnc = OrdinalEncoder()
    OrdinalEnc.fit(xtrainDiscrete)
    xtrainFilter = OrdinalEnc.transform(xtrainDiscrete)

    xtrainFilter = np.concatenate((xtrainContinuous, xtrainFilter), axis=1)
    filterColumns = optimalFeatures(xtrainFilter, ytrain)

    return xtrainFilter, ytrain, filterColumns


def downSampling(X, Y):
    """ Function downSampling takes dataset and down samples fairly (random) to be used

        Args: 
                X: feature data
                Y: labels

        Returns:
                X: downsampled feature data
                Y: corresponding labels to the downsampled data
        """
    i_class0 = np.where(Y == 0)[0]
    i_class1 = np.where(Y == 1)[0]

    n_class0 = len(i_class0)
    n_class1 = len(i_class1)

    i_class0_downsampled = np.random.choice(
        i_class0, size=n_class1, replace=False)

    X = np.concatenate(
        (X[i_class1], X[i_class0_downsampled]), axis=0)
    Y = np.concatenate(
        (Y[i_class1], Y[i_class0_downsampled]), axis=0)
    return X, Y


def upSampling(X, Y):
    """ Function upSampling takes dataset and up samples to be used

        Args: 
                X: feature data
                Y: labels

        Returns:
                X: upsampled feature data
                Y: corresponding labels to the upsampled data
    """
    i_class0 = np.where(Y == 0)[0]
    i_class1 = np.where(Y == 1)[0]

    n_class0 = len(i_class0)
    n_class1 = len(i_class1)

    i_class1_downsampled = np.random.choice(
        i_class1, size=n_class0, replace=True)

    X = np.concatenate(
        (X[i_class0], X[i_class1_downsampled]), axis=0)
    Y = np.concatenate(
        (Y[i_class0], Y[i_class1_downsampled]), axis=0)
    return X, Y


def confusionMtrx(ytest, y_pred):
    """ Function confusionMtrx creates an error matrix based on the predicted labels and the true (test) labels.

        Args: 
                ytest: true labels
                y_pred: predicted labels

        Returns:
                true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp)
        """
    confusion_matrix(ytest, y_pred)
    tn, fp, fn, tp = confusion_matrix(ytest, y_pred).ravel()
    return tn, fp, fn, tp


def calcStatistics(tn, fp, fn, tp):
    """ Function calcStatistics takes the results of the confusion Matrix to report precision.

        Args: 
                tn: true negatives
                fp: false positives
                fn: false negatives
                tp: true positives

        Returns: 
                void: prints out accuracies and precisions
        """
    accuracy = (tp+tn) / (tp+tn+fn+fp)
    print('Total Accuracy: ', accuracy)
    recall = tp / (tp+fn)
    print('Recall: ', recall)
    precision = tp / (tp+fp)
    print('Precision: ', precision)
    f_measure = (2*recall*precision) / (recall + precision)
    print('F-measure: ', (f_measure))


def ensembleAcc(ensembleArr, y_true):
    """ Function ensembleAcc takes in sampled data and test (true) labels, creates list using the mode of the data

        Args: 
                ensembleArr: sampled data features
                y_true: test labels

        Returns:
                ensemblePred: 'np.array' of most frequent data points
        """
    ensembleArr = np.array(ensembleArr).T
    ensemblePred = []
    for i in ensembleArr:
        ensemblePred.append(statistics.mode(i))

    ensemblePred = np.array(ensemblePred)

    return ensemblePred


def runClassifier(classifier, dataTransformation, xtrain, ytrain, xtest, ytest, isFilter, filterCol):
    """Function runClassifier runs the passed 'classifier' with and with out filtering

    Args:
       classifier: given classifier: clf,  rf, gn , SVM, neigh
       dataTransformation: Normal clean, up-sampled, down-sampled, SMOTE 
       xtrain: training feature data
       ytrain: training labels
       xtest: testing feature data
       ytest: testing labels
       isFilter: boolean var
       filterCol: filter column determined by sorted PCC values

    Returns: 
       Accuracies of classifiers
        """
    if isFilter == False:
        continuousColumns = [0, 1, 2, 3, 4, 5]
        discreteColumns = [i for i in range(
            xtrain.T.shape[0]) if i not in continuousColumns]

        xtrainContinuous = xtrain.T[continuousColumns]
        xtrainDiscrete = xtrain.T[discreteColumns]

        enc = OneHotEncoder()
        enc.fit(xtrainDiscrete.T)
        xtrainDiscrete = enc.transform(xtrainDiscrete.T).toarray()

        xtrain = np.concatenate((xtrainContinuous.T, xtrainDiscrete), axis=1)

        xtestContinuous = xtest.T[continuousColumns]
        xtestDiscrete = xtest.T[discreteColumns]

        enc = OneHotEncoder()
        enc.fit(xtestDiscrete.T)
        xtestDiscrete = enc.transform(xtestDiscrete.T).toarray()

        xtest = np.concatenate((xtestContinuous.T, xtestDiscrete), axis=1)

        array = np.zeros(xtest.shape[0]).T
        array = array.reshape(xtest.shape[0], 1)
        xtest = np.concatenate((xtest, array), axis=1)
        classifier.fit(xtrain, ytrain)
        print(dataTransformation + ': ', classifier.score(xtest, ytest))

        return classifier.predict(xtest)

    if isFilter == True:
        print(dataTransformation + ': ')

        return accuracyFilter(classifier, xtrain, ytrain, xtest, ytest, filterCol, False)


def baggingEnsemble(xtrain, ytrain, xtest, ytest, filterCol):
    """ Function baggingEnsemble applies cleaned data to five different popular classifiers: 
                Decision Tree
                Random Forest
                Gaussian Naive Bayes
                Support Vector Machine
                K Nearest Neighbors

        Each of these classifiers are implemented on the same dataset but using different sampling methods:
                Down sampling,
                Up sampling
                Normal (cleaned, no change)
                Synthetic Minority Oversampling TEchnique (SMOTE) sampled

        We run these calculations for each classifier using filtered and unfiltered features.
        Finally, we take a majority vote using each of the classifiers to come up with labels

        Args:
                xtrain: training data features (with discrete data converted to numerical values)
                ytrain: training data labels (converted to 'high' (1) and 'low' (0))
                xtest: testing data features (with discrete data converted to numerical values)
                ytest: testing data features (converted to 'high' (1) and 'low' (0))
                filterCol: filter column determined by sorted PCC values

        Returns: 
                Internal functions report accuracies for each sampled data for each classifier as it runs for each pass.

    """
    predictedDown, predictedUp, predictedNormal = [], [], []

    Xd, Yd = downSampling(xtrain, ytrain)
    Xu, Yu = upSampling(xtrain, ytrain)

    clf = tree.DecisionTreeClassifier()
    rf = RandomForestClassifier()
    gn = GaussianNB()
    SVM = svm.SVC()
    neigh = KNeighborsClassifier(n_neighbors=31)
    predictions = [predictedNormal, predictedDown, predictedUp]

    xtrains = [xtrain, Xd, Xu]
    ytrains = [ytrain, Yd, Yu]

    dataTransformation = ['Normal Data',
                          'Down Sampling', 'Up Sampling']

    classifiers = [clf, rf, gn, SVM, neigh]

    print('\nNo Filtering')
    print('---------------------')

    for i in range(0, 5):
        print(type(classifiers[i]), '\n')
        for j in range(0, 3):
            ## trainPred, testPred
            predictions[j].append(runClassifier(classifiers[i], dataTransformation[j], xtrains[j], ytrains[j], xtest, ytest,
                                                False, filterCol))
        print()

    print('\nEnsemble Learning: \n------------------------------------------')
    for j in range(0, 3):

        ensemblePred = ensembleAcc(predictions[j], ytest)
        print('\n' + dataTransformation[j] + ' ensemble accuracy:',
              np.equal(ensemblePred, ytest).sum() / ytest.shape[0])
        tn, fp, fn, tp = confusionMtrx(ytest, ensemblePred)
        calcStatistics(tn, fp, fn, tp)

    predictedDown, predictedUp, predictedNormal = [], [], []

    XdF, YdF = downSampling(xtrain, ytrain)
    XuF, YuF = upSampling(xtrain, ytrain)
    predictions = [predictedNormal, predictedDown, predictedUp]

    xtrains = [xtrain, XdF, XuF]
    ytrains = [ytrain, YdF, YuF]

    for i in range(0, 5):
        print(type(classifiers[i]), '\n')
        for j in range(0, 3):
            predictions[j].append(runClassifier(classifiers[i], dataTransformation[j], xtrains[j], ytrains[j], xtest, ytest,
                                                True, filterCol))
        print()

    print('\nEnsemble Learning: \n------------------------------------------')
    for j in range(0, 3):
        ensemblePred = ensembleAcc(predictions[j], ytest)
        print('\n' + dataTransformation[j] + ' ensemble accuracy:',
              np.equal(ensemblePred, ytest).sum() / ytest.shape[0])
        tn, fp, fn, tp = confusionMtrx(ytest, ensemblePred)
        calcStatistics(tn, fp, fn, tp)
