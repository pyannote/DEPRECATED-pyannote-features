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


import nltk

POS_MAPPING = {
    nltk.corpus.reader.wordnet.ADJ: {'JJ', 'JJR', 'JJS'},
    nltk.corpus.reader.wordnet.ADJ_SAT: {},
    nltk.corpus.reader.wordnet.ADV: {'RB', 'RBR', 'RBS'},
    nltk.corpus.reader.wordnet.NOUN: {'NN', 'NNP', 'NNPS', 'NNS'},
    nltk.corpus.reader.wordnet.VERB: {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
    # None: {'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'LS', 'MD', 'PDT', 'POS',
    #        'PRP', 'PRP$', 'RP', 'SYM', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB',
    #        '.', ',', "''", ':'},
}

POS_INV_MAPPING = {
    tag: wordnet_pos_tag
    for wordnet_pos_tag, pos_tags in POS_MAPPING.iteritems()
    for tag in pos_tags
}


class TextPreProcessing(object):
    """Text pre-processing

    Parameters
    ----------
    tokenize : func or boolean, optional
        Set tokenizing function (string --> list)
        If `tokenize` is False, assumes pre-tokenized list input.
        Defaults to NLTK word-punct tokenizer.
    stopwords : iterable or boolean, optional
        Only words not in `stopwords` are kept.
        If `stopwords` is False, keep all words.
        Defaults to NLTK English stopwords.
    pos_tag : func, optional
        Set pos_tagging function (list --> pos list)
        Defaults to NLTK pos_tag.
    keep_pos : set or boolean, optional
        Only words with POS-tag in `keep_pos` are kept.
        If `keep_pos` is False, keep all words.
        Defaults to NLTK Wordnet's {ADJ, NOUN, ADV, VERB}
    lemmatize : func or boolean, optional
        Set lemmatizing function (word, pos --> lemma)
        If `lemmatize` is False, do not apply lemmatization.
        Defaults to NLTK WordNet lemmatizer.
    stem: func, or boolean, optional
        Set stemming function
        If `stem` is False, do not apply stemming.
        Defaults to NLTK Porter stemmer.

    """

    def __init__(self, tokenize=True, lemmatize=True, stem=True,
                 stopwords=True, pos_tag=True, keep_pos=True, min_length=2):

        super(TextPreProcessing, self).__init__()

        if tokenize is True:
            self.tokenize = nltk.WordPunctTokenizer().tokenize
        else:
            self.tokenize = tokenize

        if stopwords is True:
            self.stopwords = nltk.corpus.stopwords.words('english')
        else:
            self.stopwords = stopwords

        if pos_tag is True:
            self.pos_tag = nltk.pos_tag
        else:
            self.pos_tag = pos_tag

        if lemmatize is True:
            self.lemmatize = nltk.WordNetLemmatizer().lemmatize
        else:
            self.lemmatize = lemmatize

        if keep_pos is True:
            self.keep_pos = {nltk.corpus.reader.wordnet.ADJ,
                             nltk.corpus.reader.wordnet.NOUN,
                             nltk.corpus.reader.wordnet.ADV,
                             nltk.corpus.reader.wordnet.VERB}
        else:
            self.keep_pos = keep_pos

        if stem is True:
            self.stem = nltk.stem.PorterStemmer().stem
        else:
            self.stem = stem

        self.min_length = min_length

    def __call__(self, text):

        # tokenize
        if self.tokenize is False:
            tokenized = text
        else:
            tokenized = self.tokenize(text.lower())

        # pos-tag
        pos_tagged = self.pos_tag(tokenized)

        # remove stop words
        if self.stopwords is False:
            stopworded = pos_tagged
        else:
            stopworded = [(word, pos_tag) for word, pos_tag in pos_tagged
                          if word not in self.stopwords]

        # remove pos words
        if self.keep_pos is False:
            filtered = stopworded
        else:
            filtered = [(word, tag) for word, tag in stopworded
                        if POS_INV_MAPPING.get(tag, None) in self.keep_pos]

        # lemmatize
        if self.lemmatize is False:
            lemmatized = [word for word, _ in filtered]
        else:
            lemmatized = [
                self.lemmatize(word, pos=POS_INV_MAPPING.get(
                    tag, nltk.corpus.reader.wordnet.NOUN))
                for word, tag in filtered]

        # stem
        if self.stem is False:
            stemmed = lemmatized
        else:
            stemmed = [self.stem(word) for word in lemmatized]

        # filter short stems
        return [stem for stem in stemmed if len(stem) > self.min_length]
