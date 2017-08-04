#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import context
from data.textfilter import normalize_text


class CosineSimilarity(object):

    def __init__(self, data, labels):

        self.data = data
        self.labels = labels
        self.df = None

    def cos_sim(self):

        vectorizer = TfidfVectorizer(
            tokenizer=normalize_text,
            max_features=500,
            ngram_range=(1, 5),
        )
        tfidf = vectorizer.fit_transform(self.data)
        cos_sim = linear_kernel(tfidf, tfidf).flatten()

        num_row = len(self.labels)
        # max_row = num_row if num_row < n_rows else n_rows

        temp_df = []
        for index in range(num_row):
            song_row = cos_sim[
                index * num_row: (index + 1) * num_row
            ]
            song_row_indices = song_row.argsort()[::-1]
            row = {self.labels[i]: song_row[i] for i in song_row_indices}
            temp_df.append(row)

        self.df = DataFrame(temp_df, index=self.labels)

        labels_sorted = self.df.mean().argsort()[::-1]
        self.labels = [self.labels[i] for i in labels_sorted]
        self.df = self.df.reindex(self.labels)
        self.df = self.df[self.labels]
