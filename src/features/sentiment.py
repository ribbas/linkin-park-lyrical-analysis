#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pandas import DataFrame
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


class CompoundSentiment(object):

    def __init__(self, data, labels):

        self.data = data
        self.labels = labels
        self.df = None

    @staticmethod
    def get_compound(text, title):

        title, album = title.split("(")
        title = title.strip()
        album = album.replace(")", "")

        compound = 0
        raw_lyrics = []
        for sentence in text:
            raw_lyrics.append(
                (sentence, sid.polarity_scores(sentence)["compound"])
            )
            compound += sid.polarity_scores(sentence)["compound"]

        sorted_lyrics = sorted(raw_lyrics, key=lambda x: x[-1])

        return {
            "title": title,
            "album": album,
            "sentences": sorted_lyrics,
            "norm_comp": compound / (len(set(text)) - 1)
        }

    def get_sentiment(self):

        self.df = DataFrame(
            (self.get_compound(lyrics, title)
                for lyrics, title in zip(self.data, self.labels))
        )
        self.df = self.df[["title", "norm_comp", "sentences", "album"]]
