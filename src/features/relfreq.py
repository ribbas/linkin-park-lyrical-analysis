#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

import context
from data.textfilter import normalize_text


class RelativeFrequency(object):

    def __init__(self, data, labels):

        self.data = data
        self.labels = labels
        self.df = None

    @staticmethod
    def unique_enough(sorted_freqs):

        def is_unique(seq1, seq2, dif):
            seq1 = set(seq1.split())
            seq2 = set(seq2.split())

            intersects = [x for x in seq1 if x in seq2]

            return seq1 != seq2 and len(intersects) < dif \
                and len(intersects) < len(seq1) + len(seq2)

        filtered_uniques = []
        for idx1, pair1 in enumerate(sorted_freqs):

            idx2 = 0
            while idx2 < len(sorted_freqs):
                unique_thresh = is_unique(
                    pair1[0], sorted_freqs[idx2][0], 1)
                if not unique_thresh:
                    sorted_freqs.pop(idx2)
                    idx2 -= 1
                idx2 += 1

            filtered_uniques.append(pair1)

        return filtered_uniques

    def rel_freq(self):

        vectorizer = CountVectorizer(
            tokenizer=normalize_text,
            max_features=2000,
            ngram_range=(2, 8),
            stop_words="english",
        )
        count = vectorizer.fit_transform(self.data)

        vocab = vectorizer.vocabulary_.items()
        freqs = (
            (word, count.getcol(idx).sum()) for word, idx in vocab
        )
        sorted_freqs = sorted(freqs, key=lambda x: -x[1])
        sorted_freqs = [
            pair for pair in sorted_freqs if len(set(pair[0].split())) > 1
        ]

        filtered_uniques = self.unique_enough(sorted_freqs)

        self.df = DataFrame(
            dict(filtered_uniques).items(), columns=["term", "freq"]
        )

        self.df.sort_values("freq", inplace=True, ascending=False)
        weight = float(self.df["freq"].sum())
        self.df = self.df.head(10)
        self.df["freq"] = self.df["freq"].apply(lambda x: x / weight)
        self.df = self.df.reset_index()
        self.df = self.df[["term", "freq"]]
