#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame, Series, concat

from data.filemgmt import vectorize_docs
from features.emotion import EmotionClassifier
from features.relfreq import RelativeFrequency
from features.simsongs import CosineSimilarity
from features.sentiment import CompoundSentiment


class DataframeGenerator(object):

    def __init__(self, albums):

        self.albums = albums
        self.rel_freq = []
        self.cos_sim = []
        self.cos_sim_all = None
        self.doc_sent = None
        self.phrase_sent = None
        self.valence_arousal = None

    def init_dfs(self):

        # populate rel_freq
        for album in self.albums:

            data, labels = vectorize_docs(
                artist="linkin-park",
                albums=[album],
                keep_album=True,
            )

            rfobj = RelativeFrequency(data=data, labels=labels)
            rfobj.rel_freq()
            self.rel_freq.append(rfobj.df)

        print "rel_freq generated"

        # populate cos_sim
        for album in self.albums:

            data, labels = vectorize_docs(
                artist="linkin-park",
                albums=[album],
                keep_album=False,
                titlify=True,
            )

            csobj = CosineSimilarity(data=data, labels=labels)
            csobj.cos_sim()
            self.cos_sim.append(csobj.df)

        print "cos_sim generated"

        data, labels = vectorize_docs(
            artist="linkin-park",
            artist_only=True,
            titlify=True,
        )

        cs_allobj = CosineSimilarity(data=data, labels=labels)
        cs_allobj.cos_sim()
        order = np.argsort(-cs_allobj.df.values, axis=1)[:, 1:5 + 1]
        self.cos_sim_all = DataFrame(
            cs_allobj.df.columns[order],
            columns=["Top {}".format(i)
                     for i in range(1, 6)],
            index=cs_allobj.df.index
        )

        print "cos_sim_all generated"

        # generate doc_sent by preserving the individual sentences during text
        # normalization
        data, labels = vectorize_docs(
            artist="linkin-park",
            artist_only=True,
            keep_album=True,
            normalized=True,
            sentences=True,
        )

        saobj = CompoundSentiment(data=data, labels=labels)
        saobj.get_sentiment()
        doc_sent = saobj.df
        self.doc_sent = doc_sent[["title", "album", "norm_comp"]]

        print "doc_sent generated"

        phrase_sent = doc_sent.set_index(
            ["title", "album", "norm_comp"]
        )["sentences"].apply(Series).stack()
        phrase_sent = phrase_sent.reset_index()
        phrase_sent.drop("level_3", axis=1, inplace=True)
        phrase_sent.columns = ["title", "album", "norm_comp", "sentences"]
        phrase_sent.drop_duplicates(subset="sentences", inplace=True)
        phrase_sent[["phrase", "sent_score"]] = phrase_sent[
            "sentences"].apply(Series)
        phrase_sent.sort_values("sent_score", inplace=True)
        phrase_sent["num_words"] = phrase_sent["phrase"].apply(
            lambda x: len(x.split(" ")))
        phrase_sent = phrase_sent.reset_index()
        self.phrase_sent = phrase_sent[
            ["phrase", "sent_score", "num_words", "title", "album", "norm_comp"]
        ]

        print "phrase_sent generated"

        self.extreme_phrase_sent = self.phrase_sent.iloc[:2]
        self.extreme_phrase_sent = self.extreme_phrase_sent.append(
            self.phrase_sent.iloc[4:12])
        self.extreme_phrase_sent = self.extreme_phrase_sent.append(
            self.phrase_sent.iloc[:-12:-1][::-1][:5])
        self.extreme_phrase_sent = self.extreme_phrase_sent.append(
            self.phrase_sent.iloc[:-12:-1][::-1][6:11])

        print "extreme_phrase_sent generated"

        data, labels = vectorize_docs(
            artist="linkin-park",  # specify artist
            artist_only=True,
            keep_album=True,  # option to use the album name as a delimiter
        )

        ec_obj = EmotionClassifier()
        ec_obj.train_model("valence")
        # ec_obj.get_pred_int()
        ec_obj.predict_score(labels, data)
        valence = ec_obj.df

        print "valence generated"

        ec_obj = EmotionClassifier()
        ec_obj.train_model("arousal")
        # ec_obj.get_pred_int()
        ec_obj.predict_score(labels, data)
        arousal = ec_obj.df

        print "arousal generated"

        valence = valence.set_index(valence["title"])
        arousal = arousal.set_index(arousal["title"])

        valence = valence[["album", "mean_pred", "std_dev"]]
        arousal = arousal[["mean_pred", "std_dev"]]

        self.valence_arousal = concat([valence, arousal], axis=1)
        self.valence_arousal.columns = [
            "album",
            "valence_pred",
            "valence_std_dev",
            "arousal_pred",
            "arousal_std_dev"
        ]

        self.valence_arousal.sort_index(inplace=True)

        print "valence-arousal generated"
