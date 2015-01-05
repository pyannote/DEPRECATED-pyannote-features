#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2015 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from __future__ import unicode_literals

from ..audio.yaafe import YaafeCompound, YaafeZCR, YaafeMFCC


class SpeechActivityDetectionFeatures(YaafeCompound):
    """Features for speech activity detection"""

    def __init__(self, sample_rate=16000, block_size=512, step_size=256):

        extractors = [
            YaafeZCR(
                sample_rate=sample_rate,
                block_size=block_size,
                step_size=step_size),
            YaafeMFCC(
                e=True, coefs=11, De=True, DDe=True, D=True, DD=True,
                sample_rate=sample_rate,
                block_size=block_size,
                step_size=step_size)
        ]

        super(SpeechActivityDetectionFeatures, self).__init__(
            extractors,
            sample_rate=sample_rate,
            block_size=block_size, step_size=step_size)
