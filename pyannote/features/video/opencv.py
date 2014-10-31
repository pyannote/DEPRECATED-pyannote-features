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

import cv
import cv2
import numpy as np
from pyannote.core.segment import SlidingWindow
from pyannote.core.feature import SlidingWindowFeature


class OpenCVFeatureExtractor(object):

    def extract(self, path, pbar=None):

        capture = cv2.VideoCapture(path)

        # frame size
        # height = int(capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        # width = int(capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))

        # video "size"
        framePerSecond = capture.get(cv.CV_CAP_PROP_FPS)
        frameCount = int(capture.get(cv.CV_CAP_PROP_FRAME_COUNT))
        # duration = frameCount / framePerSecond

        data = np.NaN * np.ones((frameCount, self.get_dimension()))

        while True:

            f = int(capture.get(cv.CV_CAP_PROP_POS_FRAMES))

            success, frame = capture.read()
            if not success:
                break

            data[f, :] = self.process_frame(frame)

            if pbar:
                pbar.update(1. * (f + 1) / frameCount)

        pbar.finish()

        duration = step = 1. / framePerSecond
        sliding_window = SlidingWindow(start=0., duration=duration, step=step)
        return SlidingWindowFeature(data, sliding_window)


COLOR_RANGES = {
    'B': [0, 255],  # blue
    'G': [0, 255],  # green
    'R': [0, 255],  # red
    'H': [0, 179],  # hue
    'S': [0, 255],  # saturation
    'V': [0, 255],  # value
}


class ColorHistogramFeatureExtractor(OpenCVFeatureExtractor):
    """

    Parameters
    ----------
    B, G, R, H, S, V : int, optional
        Number of bins in blue, green, red, hue, saturation and value channels.
        Setting it to 0 (default) means the corresponding channel is not used.

    Usage
    -----
    >>> extractor = ColorHistogramFeatureExtractor(H=32)
    >>> features = extractor.extract('/path/to/video.mkv')

    """

    def __init__(self, H=0, S=0, V=0, R=0, G=0, B=0):
        super(ColorHistogramFeatureExtractor, self).__init__()

        # number of bins for each channel
        self.bins = {'H': H, 'S': S, 'V': V, 'R': R, 'G': G, 'B': B}

        # number of bins/variation range for each selected channel
        self._bins, self._ranges = [], []
        for i, channel in enumerate('BGRHSV'):
            if self.bins[channel]:
                self._bins.append(self.bins[channel])
                self._ranges.append(COLOR_RANGES[channel])

    def get_dimension(self):
        return np.prod(self._bins)

    def process_frame(self, bgr):

        # get frame dimension
        width, height, _ = bgr.shape

        # convert to HSV (hue-saturation-value) only if needed afterwards
        if self.bins['H'] or self.bins['S'] or self.bins['V']:
            hsv = cv2.cvtColor(bgr, cv.CV_BGR2HSV)
            # reshape to (WxH, 3)
            hsv = hsv.T.reshape((3, -1)).T

        # reshape BGR only if needed afterwards
        if self.bins['B'] or self.bins['G'] or self.bins['R']:
            # reshape to (WxH, 3)
            bgr = bgr.T.reshape((3, -1)).T

        # this numpy array will have one row per channels
        if not hasattr(self, '_X'):
            # get histogram dimension
            dimension = sum([v > 0 for k, v in self.bins.iteritems()])
            self._X = np.empty((width * height, dimension))

        # d is current row index
        d = 0

        # add B, G and/or R channel if requested
        for i, channel in enumerate('BGR'):
            if self.bins[channel]:
                self._X[:, d] = bgr[:, i]
                d = d + 1

        # add H, S and/or V channel if requested
        for i, channel in enumerate('HSV'):
            if self.bins[channel]:
                self._X[:, d] = hsv[:, i]
                d = d + 1

        # compute multi-dimensional histogram
        histogram, _ = np.histogramdd(self._X,
                                      bins=self._bins,
                                      range=self._ranges)

        # return it as a 1 x nbins array
        return histogram.reshape((1, -1)) / (width * height)
