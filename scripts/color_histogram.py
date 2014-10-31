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

"""
Compute color histograms from a video

Usage:
  color_histogram [-R <R>] [-G <G>] [-B <B>] [-H <H>] [-S <S>] [-V <V>] [--numpy] [--progress] <input.mkv> <output.pkl>
  color_histogram -h | --help
  color_histogram --version

Options:
  -R <R> --red=<R>         Set number of bins in R channel [default: 0]
  -G <G> --green=<G>       Set number of bins in G channel [default: 0]
  -B <B> --blue=<B>        Set number of bins in B channel [default: 0]
  -H <H> --hue=<H>         Set number of bins in H channel [default: 0]
  -S <S> --saturation=<S>  Set number of bins in S channel [default: 0]
  -V <V> --value=<V>       Set number of bins in V channel [default: 0]
  --progress               Show progress bar.
  --numpy                  Save as numpy array.
  -h --help                Show this screen.
  --version                Show version.
"""

from pyannote.features.video.opencv import ColorHistogramFeatureExtractor
from docopt import docopt
import numpy as np
import pickle
from progressbar import ProgressBar, Percentage, Bar, ETA


FMT_PICKLE = 'pkl'
FMT_NUMPY = 'npy'


def do_it(input_video, output_file, format=FMT_PICKLE, progress=False,
          R=0, G=0, B=0, H=0, S=0, V=0):

    if progress:
        pbar = ProgressBar(
            widgets=[Percentage(), ' ', Bar(), ' ', ETA()],
            maxval=1.).start()
    else:
        pbar = None

    extractor = ColorHistogramFeatureExtractor(R=R, G=G, B=B, H=H, S=S, V=V)
    features = extractor.extract(input_video, pbar=pbar)

    with open(output_file, 'wb') as f:

        if format == FMT_PICKLE:
            pickle.dump(features, f)

        elif format == FMT_NUMPY:
            np.save(f, features.data)

if __name__ == '__main__':

    arguments = docopt(__doc__, version='Color Histogram 1.0')

    r = int(arguments['--red'])
    g = int(arguments['--green'])
    b = int(arguments['--blue'])
    h = int(arguments['--hue'])
    s = int(arguments['--saturation'])
    v = int(arguments['--value'])

    input_video = arguments['<input.mkv>']
    output_file = arguments['<output.pkl>']

    progress = arguments['--progress']

    if arguments['--numpy']:
        format = FMT_NUMPY
    else:
        format = FMT_PICKLE

    do_it(input_video, output_file, format=format, progress=progress,
          R=r, G=g, B=b, H=h, S=s, V=v)
