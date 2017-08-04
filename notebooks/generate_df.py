#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame

from data.filemgmt import vectorize_docs
from features.relfreq import RelativeFrequency
from features.simsongs import CosineSimilarity
from features.sentiment import CompoundSentiment
from settings.artistinfo import LINKIN_PARK_ALBUMS


class DataframeGenerator(object):

    def __init__(self):

        self.rel_freq = []
        self.cos_sim = []
        self.cos_sim_all = None
        self.doc_sent = None
        self.phrase_sent = None

    def init_dfs(self):

        # populate rel_freq
        for album in LINKIN_PARK_ALBUMS:

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
        for album in LINKIN_PARK_ALBUMS:

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
        self.doc_sent = saobj.df

        print "doc_sent generated"

        # generate phrase_sent using doc_sent
        sorted_phrases = set(
            self.doc_sent.iloc[self.doc_sent["norm_comp"].argsort()].sum()
            ["sentences"])
        sorted_phrases = (sorted(sorted_phrases, key=lambda x: x[-1]))

        # HARDCODED TO GET RID OF VERY SIMILAR PHRASES
        # i.e. same phrase with an extra article
        sorted_phrases = (
            sorted_phrases[:2] + sorted_phrases[4:12] +
            sorted_phrases[:-12:-1][::-1][:5] +
            sorted_phrases[:-12:-1][::-1][6:11]
        )

        self.phrase_sent = DataFrame(sorted_phrases, columns=[
            "phrase", "sentiment"])

        print "phrase_sent generated"
