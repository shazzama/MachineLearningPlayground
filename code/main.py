import os
import pickle
import gzip
import argparse
import numpy as np

from knn import KNN
from linreg import LinearRegression
from sklearn.preprocessing import LabelBinarizer
from utils import test_and_plot

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

    if question == "2":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        print(X.shape)
        Xtest, ytest = test_set
        print(Xtest.shape)

        # Xsplit = np.split(X, 5)
        # ysplit = np.split(y, 5)
        #
        # Xtrain = Xsplit[0]
        # Xvalidate = Xsplit[1]
        #
        # ytrain = ysplit[0]
        # yvalidate = ysplit[1]
        #
        #
        # model = KNN(1)
        # model.fit(Xtrain, ytrain)
        # y_pred = model.predict(Xtrain)
        # tr_error = np.mean(y_pred != ytrain)
        #
        # y_pred = model.predict(Xvalidate)
        # va_error = np.mean(y_pred != yvalidate)
        # print("Training error: %.3f" % tr_error)
        # print("Validate error: %.3f" % va_error)




# IMPLEMENTATIOn 2
        split_amount = 5
        neighbours_to_try = 3

        best_error = 100
        best_num_neighbours = 2

        Xsplit = np.split(X, split_amount)
        ysplit = np.split(y, split_amount)


        for split_index in range(split_amount):
            Xtrain = Xsplit[split_index]
            ytrain = ysplit[split_index]
            print(Xtrain.shape)
            for num_of_neighbours in range(2,neighbours_to_try):
                print("num neighbours: %.3f" % num_of_neighbours)
                model = KNN(num_of_neighbours)
                model.fit(Xtrain, ytrain)
                y_pred = model.predict(Xtrain)
                print(y_pred)
                print(ytrain)
                train_error = np.mean(y_pred != ytrain)
                if train_error < best_error:
                    best_error = train_error
                    best_num_neighbours = num_of_neighbours
                    print("Training error: %.3f" % train_error)

        model = KNN(best_num_neighbours)
        model.fit(Xtest, ytest)
        y_pred = model.predict(Xtest)
        test_error = np.mean(y_pred != ytest)
        print("Test error: %.3f" % test_error)

    if question == "3":
        print("Linear Regression")
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        model = LinearRegression(1)
        model.fit(X, y)
        # tr_err = model.predict(X)
        # print(tr_err)
        # print("Train error: %.3f" % tr_err)
        #
        # te_err = model.predict(Xtest)
        # print("Test error: %.3f" % te_err)

        test_and_plot(model, X, y, Xtest, ytest, "LinearReg", "linreg.pdf")



    else:
        print("Unknown question: %s" % question)
