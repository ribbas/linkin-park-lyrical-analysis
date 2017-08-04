#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob

import context
from textfilter import normalize_text
from settings.paths import LYRICS_OUT_DIR


class Album(object):

    def __init__(self, artist, album, normalized=False, sentences=False):

        self.artist = artist
        self.album = album
        self.normalized = normalized
        self.sentences = sentences
        self.album_path = LYRICS_OUT_DIR.format(artist=artist, album=album)

    def lyrics(self, keep_album=False, titlify=False):
        """Return lyrics of all the tracks in the album in the JSON format:

            {
                "album": "album1_name",
                "tracklist": [
                    {
                        "title": "song1_title",
                        "album": "album1",
                        "artist": "artist1",
                        "lyrics": "lorem ipsum ...",
                    },
                    {
                        "title": "song2_title",
                        "album": "album1",
                        "artist": "artist1",
                        "lyrics": "lorem ipsum ...",
                    },
                    ...
                ]
            }

        """

        tracklist = []
        song_files = glob.iglob(self.album_path + "/*.txt")

        for song in song_files:

            lyrics = ""
            with open(song) as song_file:
                lyrics = song_file.read()
                if self.normalized:
                    lyrics = normalize_text(lyrics, sentences=self.sentences)

            song_title = song.partition(
                self.album)[-1][1:-4].replace(self.artist, "")[:-8]

            title = "{song} ({album})".format(
                song=song_title, album=self.album
            ) if keep_album else song_title

            if titlify:
                title = title.title().replace("-", " ")

            song_info = {
                "title": title,
                "lyrics": lyrics,
                "artist": self.artist,
            }

            tracklist.append(song_info)

        output = {
            "album": self.album,
            "tracklist": tracklist,
        }

        return output


class Artist(object):

    def __init__(self, artist, normalized=False, sentences=False,
                 keep_album=False, titlify=False):

        self.artist = artist
        self.normalized = normalized
        self.sentences = sentences
        self.titlify = titlify
        self.keep_album = keep_album
        self.album_path = LYRICS_OUT_DIR.format(artist=artist, album="")

    def lyrics(self):
        """Return lyrics of all the tracks by the artist in the JSON format:

            {
                "artist1": [
                    {
                        "album": "album1_name",
                        "tracklist": [
                            {
                                "title": "song1_title",
                                "album": "album1",
                                "artist": "artist1",
                                "lyrics": "lorem ipsum ...",
                            },
                            {
                                "title": "song2_title",
                                "album": "album1",
                                "artist": "artist1",
                                "lyrics": "lorem ipsum ...",
                            },
                            ...
                        ]
                    },
                    {
                        "album": "album2_name",
                        "tracklist": [
                            {
                                "title": "song1_title",
                                "album": "album1",
                                "artist": "artist1",
                                "lyrics": "lorem ipsum ...",
                            },
                            {
                                "title": "song2_title",
                                "album": "album1",
                                "artist": "artist1",
                                "lyrics": "lorem ipsum ...",
                            },
                            ...
                        ]
                    },
                    ...
                ],
                ...
            }

        """
        album_list = []
        album_dirs = glob.iglob(self.album_path + "/*")

        for album in album_dirs:
            album_obj = Album(
                artist=self.artist,
                album=album.split(self.artist)[-1][1:],
                normalized=self.normalized,
                sentences=self.sentences
            )
            album_list.append(album_obj.lyrics(
                keep_album=self.keep_album,
                titlify=self.titlify
            ))

        output = {self.artist: album_list}

        return output
