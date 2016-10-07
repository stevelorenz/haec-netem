#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# About  : Logger config file
"""

import logging

# create logger and set level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# do not pass events to handlers of high level loggers
logger.propagate = 0

# create StreamHandler
ST_HANDLER = logging.StreamHandler(stream=None)
ST_HANDLER.setLevel(logging.DEBUG)

# create Formatter
DATEFMT = "%Y-%m-%d %H:%M:%S"
FORMATTER = "%(levelname)s %(lineno)d: %(message)s"
SH_FORMATTER = logging.Formatter(FORMATTER)

# set logger
ST_HANDLER.setFormatter(SH_FORMATTER)
logger.addHandler(ST_HANDLER)
