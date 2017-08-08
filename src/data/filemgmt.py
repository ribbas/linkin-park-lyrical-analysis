#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""settings

"""

import context
from structs import Album, Artist


def vectorize_docs(artist, albums=[], artist_only=False, keep_album=False,
                   normalized=False, sentences=False, titlify=False):

    lyrics = []
    albums = albums if not artist_only else []

    if albums:
        for album in albums:

            album_obj = Album(
                artist=artist,
                album=album,
                normalized=normalized,
                sentences=sentences,
            )
            lyrics.append(album_obj.lyrics(
                keep_album=keep_album,
                titlify=titlify
            ))

    else:

        artist_obj = Artist(
            artist=artist,
            normalized=normalized,
            sentences=sentences,
            keep_album=keep_album,
            titlify=titlify,
        )
        lyrics = artist_obj.lyrics()[artist]

    data = []
    labels = []
    for album in lyrics:
        for track in album["tracklist"]:
            data.append(track["lyrics"])
            labels.append(track["title"])

    return data, labels
