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
Compute MFCC coefficients from an audio file

Usage:
  mfcc [-n <coefs>] [-D] [--DD] [-e] [--De] [--DDe] [--numpy] <input.wav> <output.pkl>
  mfcc -h | --help
  mfcc --version

Options:
  -e --energy              Extract energy.
  -n <N> --coefs <N>       Append <N> MFCC coefficients [default: 11]
  --De                     Append energy first derivative.
  -D --D                   Append first derivatives.
  --DDe                    Append energy second derivative.
  --DD                     Append second derivatives.
  --numpy                  Save as numpy array.
  -h --help                Show this screen.
  --version                Show version.
"""

from pyannote.features.audio.yaafe import YaafeMFCC
from docopt import docopt
import numpy as np
import pickle


FMT_PICKLE = 'pkl'
FMT_NUMPY = 'npy'


def do_it(input_wav, output_file, format=FMT_PICKLE,
          sample_rate=16000, block_size=512, step_size=256,
          e=False, coefs=11, De=False, DDe=False, D=False, DD=False):

    extractor = YaafeMFCC(
        sample_rate=sample_rate, block_size=block_size, step_size=step_size,
        e=e, coefs=coefs, De=De, DDe=DDe, D=D, DD=DD)

    features = extractor.extract(input_wav)

    with open(output_file, 'wb') as f:

        if format == FMT_PICKLE:
            pickle.dump(features, f)

        elif format == FMT_NUMPY:
            np.save(f, features.data)

if __name__ == '__main__':

    arguments = docopt(__doc__, version='MFCC 1.0')

    e = arguments['--energy']
    coefs = int(arguments['--coefs'])
    De = arguments['--De']
    D = arguments['--D']
    DDe = arguments['--DDe']
    DD = arguments['--DD']

    input_wav = arguments['<input.wav>']
    output_file = arguments['<output.pkl>']

    if arguments['--numpy']:
        format = FMT_NUMPY
    else:
        format = FMT_PICKLE

    do_it(input_wav, output_file, format=format,
          e=e, coefs=coefs, De=De, DDe=DDe, D=D, DD=DD)
