#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""settings

"""

from os import path

BASE_DIR = path.abspath(path.dirname(path.dirname(path.dirname(__file__))))
DATA_DIR = path.join(BASE_DIR, "data")
RAW_DATA_DIR = path.join(DATA_DIR, "raw")
DATA_DIRS = {
    "DATA_DIR": DATA_DIR,
    "RAW_DATA_DIR": RAW_DATA_DIR,
    "LYRICS_DIR": path.join(RAW_DATA_DIR, "lyrics"),
}

LOG_DIR = path.join(BASE_DIR, "logs")
LOG_DIRS = {
    "LOG_DIR": LOG_DIR,
    "FAIL": path.join(LOG_DIR, "failed.txt"),
    "QUEUE": path.join(LOG_DIR, "queue.txt"),
    "LYRICS_LIST": path.join(LOG_DIR, "lyrics_urls.txt"),
}

LYRICS_OUT_DIR = path.join(DATA_DIRS["LYRICS_DIR"], "{artist}", "{album}")
LYRICS_OUT = path.join(DATA_DIRS["LYRICS_DIR"], "{artist}", "{album}",
                       "{file}.txt")

DIRS = (
    DATA_DIRS["DATA_DIR"],
    DATA_DIRS["RAW_DATA_DIR"],
    DATA_DIRS["LYRICS_DIR"],
    LOG_DIRS["LOG_DIR"],
)
