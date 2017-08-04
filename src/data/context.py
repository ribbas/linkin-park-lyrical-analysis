#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import sys

SRC_PATH = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(SRC_PATH)

import features
import settings
