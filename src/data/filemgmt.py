#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""settings

"""

import json
from os import path, walk
from string import punctuation

import context
from structs import Album, Artist
from settings.paths import LOG_DIRS
from settings.artistinfo import ARTISTS, LINKIN_PARK_ALBUMS


class MP3Files(object):

    def __init__(self, local=False, spotify=False, spotify_creds=()):

        if local != (spotify and spotify_creds):
            if local:
                self.local = True
                mp3_path = walk("MP3_DIR")  # need actual MP3DIR

                _all_songs = [
                    (dirpath, filenames) for
                    (dirpath, _, filenames) in mp3_path
                ]

                _songs = []

                for dirs in _all_songs:

                    album, songs = dirs

                    _songs.extend([path.join(album, song) for song in songs])

                self.songs = sorted(_songs)

            elif (spotify and spotify_creds):
                from .spotify import SpotifyWrapper

                self.local = False
                tracks = SpotifyWrapper(
                    client_creds=spotify_creds,
                    artist=ARTISTS[0],
                    albums=LINKIN_PARK_ALBUMS
                )

                self.songs = tracks.get_tracklist()

    def make_lyrics_list(self):

        lyrics_list = []

        if self.local:
            def lyrics_url(song):

                song_name = path.basename(song).partition(
                    "-")[-1].replace(".mp3", "").replace("_", "-").lower()

                punctuations = punctuation.replace("-", "")
                for p in punctuations:
                    song_name = song_name.replace(p, "")

                album_name = "::" + path.basename(path.dirname(song))
                artist_name = path.basename(path.dirname(path.dirname(song)))

                return "-lyrics-".join((song_name, artist_name)) + album_name

            lyrics_list = [lyrics_url(song) for song in self.songs]

        else:

            for album, songs in self.songs.items():
                album = "::" + album.replace(" ", "-").lower()
                for song in songs:
                    song_name = song.lower().split(
                        '(')[0].replace(' ', '-').strip()
                    punctuations = punctuation.replace("-", "")
                    for p in punctuations:
                        song_name = song_name.replace(p, "")

                    lyrics_list.append(
                        "-lyrics-".join((song_name, ARTISTS[0])) + album
                    )

        with open(LOG_DIRS["LYRICS_LIST"], "w") as songs_list:
            songs_list.write("\n".join(lyrics_list))


def dump_lyrics(obj, file_path):

    with open(file_path, 'w') as file_name:
        json.dump(obj, file_name)
        return


def get_lyrics(file_path):

    with open(file_path) as file_name:
        return json.load(file_name)


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
