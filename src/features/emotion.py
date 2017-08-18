#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame, read_csv
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer

import context
from data.textfilter import normalize_text
from settings.paths import DATA_DIRS


class EmotionClassifier(object):

    def __init__(self, init=False):

        self.emo_df = self.make_dataset() if init \
            else read_csv(DATA_DIRS["VALENCE_AROUSAL"])

        self.vectorizer = None
        self.clf = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.df = None

    @staticmethod
    def pred_ints(model, X, percentile=95):

        err_down = []
        err_up = []
        for x in xrange(X.shape[0]):
            preds = []
            for pred in model.estimators_:
                preds.append(pred.predict(X[x])[0])

            err_down.append(
                np.percentile(preds, (100 - percentile) / 2.)
            )
            err_up.append(
                np.percentile(preds, 100 - (100 - percentile) / 2.)
            )

        return err_down, err_up

    def make_dataset(self):

        self.emo_df = read_csv(DATA_DIRS["VALENCE_AROUSAL_RAW"])
        self.emo_df.dropna(inplace=True)

        self.emo_df["text"] = self.emo_df["Anonymized Message"].apply(
            lambda x: " ".join(normalize_text(x))
        )

        self.emo_df["text"].replace("", np.nan, inplace=True)
        self.emo_df.dropna(inplace=True)

        self.emo_df["valence"] = self.emo_df.apply(
            lambda x: np.mean((x["Valence1"], x["Valence2"])), axis=1
        )

        self.emo_df["arousal"] = self.emo_df.apply(
            lambda x: np.mean((x["Arousal1"], x["Arousal2"])), axis=1
        )

        self.emo_df.to_csv(DATA_DIRS["VALENCE_AROUSAL"], index=False)

    def train_model(self, target):

        self.vectorizer = TfidfVectorizer(
            tokenizer=normalize_text,
            max_features=2000,
            ngram_range=(1, 5),
        )

        # Use `fit` to learn the vocabulary
        self.vectorizer.fit(self.emo_df["Anonymized Message"])

        # Use `tranform` to generate the vectorized matrix
        X = self.vectorizer.transform(self.emo_df["text"])
        Y = self.emo_df[target]

        # Evaluate accuracy of model on test set
        size = len(self.emo_df["text"])
        trainsize = int(0.25 * size)
        idx = range(size)
        # shuffle the data
        np.random.shuffle(idx)

        self.x_train, self.x_test = X[idx[:trainsize]], X[idx[trainsize:]]
        self.y_train, self.y_test = Y[idx[:trainsize]], Y[idx[trainsize:]]

        self.clf = ensemble.RandomForestRegressor(
            n_estimators=500, min_samples_leaf=1, n_jobs=-1
        )
        self.clf.fit(self.x_train, self.y_train)

    def get_pred_int(self):

        err_down, err_up = self.pred_ints(
            self.clf, self.x_test, percentile=90
        )

        truth = self.y_test
        correct = 0.0
        for i, val in enumerate(truth):
            if err_down[i] <= val <= err_up[i]:
                correct += 1

        print "Regressor generated with %0.2f%% percentile" \
            % (correct / len(truth) * 100)

    def predict_score(self, labels, data):

        # Use `fit` to learn the vocabulary
        self.vectorizer.fit(data)

        # Use `tranform` to generate the vectorized matrix
        X = self.vectorizer.transform(data)

        err_down, err_up = self.pred_ints(
            self.clf, X, percentile=90
        )

        def titlify(text):
            return text.title().replace("-", " ")

        titles = [titlify(i.split("(")[0]) for i in labels]
        albums = [titlify(i.split("(")[-1][:-1]) for i in labels]

        self.df = DataFrame({
            "title": titles,
            "album": albums,
            "lower": err_down,
            "upper": err_up,
        })

        self.df["mean_pred"] = self.df.apply(
            lambda x: np.mean((x["lower"], x["upper"])), axis=1
        )

        self.df["std_dev"] = self.df.apply(
            lambda x: (x["mean_pred"] - x["lower"]) / 2.0, axis=1
        )

        self.df = self.df[["title", "album", "mean_pred", "std_dev"]]
