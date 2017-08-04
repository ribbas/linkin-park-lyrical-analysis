#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import sys

SRC_PATH = path.join(path.dirname(path.dirname(path.abspath(__file__))), "src")
sys.path.append(SRC_PATH)

import data
import settings
import features
