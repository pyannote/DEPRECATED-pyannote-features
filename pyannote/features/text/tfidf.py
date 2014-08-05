#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014 CNRS

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

from preprocessing import TextPreProcessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TFIDF(object):

    def __init__(self, preprocessing=None, binary=False):
        super(TFIDF, self).__init__()

        if preprocessing is None:
            preprocessing = TextPreProcessing()
        self.preprocessing = preprocessing

        _ = lambda x: x
        self._cv = CountVectorizer(tokenizer=_, analyzer=_, preprocessor=_,
                                   binary=binary)
        self._tfidf = TfidfTransformer(
            norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

    def fit(self, documents):
        counts = self._cv.fit_transform(
            [self.preprocessing(d) for d in documents])
        self._tfidf.fit(counts)

    def transform(self, documents):
        counts = self._cv.transform([self.preprocessing(d) for d in documents])
        return self._tfidf.transform(counts)


