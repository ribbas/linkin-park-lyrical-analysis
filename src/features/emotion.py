#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas import read_csv
from sklearn import ensemble, linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import context
from data.textfilter import normalize_text
from settings.paths import DATA_DIRS


class EmotionClassifier(object):

    def __init__(self, init=False):

        self.emo_df = self.make_dataset() if init \
            else read_csv(DATA_DIRS["VALENCE_AROUSAL"])

    def make_dataset(self):

        self.emo_df = read_csv(DATA_DIRS["VALENCE_AROUSAL_RAW"])
        self.emo_df.dropna(inplace=True)

        self.emo_df["text"] = self.emo_df["Anonymized Message"].apply(
            lambda x: " ".join(normalize_text(x))
        )

        self.emo_df["text"].replace('', np.nan, inplace=True)
        self.emo_df.dropna(inplace=True)

        self.emo_df["valence"] = self.emo_df.apply(
            lambda x: np.mean((x["Valence1"], x["Valence2"])), axis=1
        )

        self.emo_df["arousal"] = self.emo_df.apply(
            lambda x: np.mean((x["Arousal1"], x["Arousal2"])), axis=1
        )

        self.emo_df.to_csv(DATA_DIRS["VALENCE_AROUSAL"], index=False)

    def test(self):

        vectorizer = CountVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            stop_words='english',
            binary=True
        )

        # Use `fit` to learn the vocabulary of the titles
        vectorizer.fit(self.emo_df["text"])

        # Use `tranform` to generate the vectorized matrix
        x_new_text_features = vectorizer.transform(self.emo_df["text"])

        # Set target variable name
        target = 'Valence1'

        # Set X and y
        X = x_new_text_features
        y = self.emo_df[target]

        # Create separate training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=128)

        clf = linear_model.SGDClassifier(alpha=1e-3, n_iter=20, random_state=42)

        # Train model on training set
        clf.fit(x_train, y_train)

        # Evaluate accuracy of model on test set
        print "Accuracy 1: %0.3f" % clf.score(x_test, y_test)

        clf = ensemble.RandomForestClassifier(n_estimators=200)
        clf.fit(x_train, y_train)
        print "Accuracy 2: %0.3f" % clf.score(x_test, y_test)


if __name__ == '__main__':

    x = EmotionClassifier()
    x.test()
