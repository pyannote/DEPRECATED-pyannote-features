#!/usr/bin/env python
# encoding: utf-8

# Copyright 2013 Herve BREDIN (bredin@limsi.fr)

# This file is part of PyAnnote.
#
#     PyAnnote is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     PyAnnote is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with PyAnnote.  If not, see <http://www.gnu.org/licenses/>.


import scipy.io.wavfile
import yaafelib
from pyannote.core.feature import SlidingWindowFeature
from pyannote.core.segment import SlidingWindow
import numpy as np


class YaafeFrame(SlidingWindow):
    """Yaafe frames

    Parameters
    ----------
    blockSize : int, optional
        Window size (in number of samples). Default is 512.
    stepSize : int, optional
        Step size (in number of samples). Default is 256.
    sampleRate : int, optional
        Sample rate (number of samples per second). Default is 16000.

    References
    ----------
    http://yaafe.sourceforge.net/manual/quickstart.html

    """
    def __init__(self, blockSize=512, stepSize=256, sampleRate=16000):

        duration = 1. * blockSize / sampleRate
        step = 1. * stepSize / sampleRate
        start = -0.5 * duration

        super(YaafeFrame, self).__init__(
            duration=duration, step=step, start=start
        )


class YaafeFeatureExtractor(object):
    """

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    block_size : int, optional
        Defaults to 512.
    step_size : int, optional
        Defaults to 256.

    """

    def __init__(
        self, sample_rate=16000, block_size=512, step_size=256
    ):

        super(YaafeFeatureExtractor, self).__init__()

        self.sample_rate = sample_rate
        self.block_size = block_size
        self.step_size = step_size

    def get_flow_and_stack(self, **kwargs):
        raise NotImplementedError(
            'get_flow_and_stack method must be implemented'
        )

    def extract(self, wav):
        """Extract features

        Parameters
        ----------
        wav : string
            Path to wav file.

        Returns
        -------
        features : SlidingWindowFeature

        """

        # hack
        data_flow, stack = self.get_flow_and_stack()

        engine = yaafelib.Engine()
        engine.load(data_flow)

        sample_rate, raw_audio = scipy.io.wavfile.read(wav)
        assert sample_rate == self.sample_rate, "sample rate mismatch"

        audio = np.array(raw_audio, dtype=np.float64, order='C').reshape(1, -1)

        features = engine.processAudio(audio)
        data = np.hstack([features[name] for name in stack])

        sliding_window = YaafeFrame(
            blockSize=self.block_size, stepSize=self.step_size,
            sampleRate=self.sample_rate)

        return SlidingWindowFeature(data, sliding_window)


class YaafeMFCC(YaafeFeatureExtractor):

    """
        | e    |  energy
        | c1   |
        | c2   |  coefficients
        | c3   |
        | ...  |
        | Δe   |  energy first derivative
        | Δc1  |
    x = | Δc2  |  coefficients first derivatives
        | Δc3  |
        | ...  |
        | ΔΔe  |  energy second derivative
        | ΔΔc1 |
        | ΔΔc2 |  coefficients second derivatives
        | ΔΔc3 |
        | ...  |


    Parameters
    ----------

    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    block_size : int, optional
        Defaults to 512.
    step_size : int, optional
        Defaults to 256.

    e : bool, optional
        Energy. Defaults to True.
    coefs : int, optional
        Number of coefficients. Defaults to 11.
    De : bool, optional
        Keep energy first derivative. Defaults to False.
    D : bool, optional
        Add first order derivatives. Defaults to False.
    DDe : bool, optional
        Keep energy second derivative. Defaults to False.
    DD : bool, optional
        Add second order derivatives. Defaults to False.

    Notes
    -----
    Default Yaafe values:
        * fftWindow = Hanning
        * melMaxFreq = 6854.0
        * melMinFreq = 130.0
        * melNbFilters = 40

    """

    def __init__(
        self, sample_rate=16000, block_size=512, step_size=256,
        e=True, coefs=11, De=False, DDe=False, D=False, DD=False,
    ):

        super(YaafeMFCC, self).__init__(
            sample_rate=sample_rate,
            block_size=block_size,
            step_size=step_size
        )

        self.e = e
        self.coefs = coefs
        self.De = De
        self.DDe = DDe
        self.D = D
        self.DD = DD

    def get_flow_and_stack(self):

        feature_plan = yaafelib.FeaturePlan(sample_rate=self.sample_rate)
        stack = []

        # --- coefficients
        # 0 if energy is kept
        # 1 if energy is removed
        definition = (
            "mfcc: "
            "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
            "blockSize=%d stepSize=%d" % (
                0 if self.e else 1,
                self.coefs + self.e * 1,
                self.block_size, self.step_size
            )
        )
        assert feature_plan.addFeature(definition)
        stack.append('mfcc')

        # --- 1st order derivatives
        if self.D or self.De:
            definition = (
                "mfcc_d: "
                "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                "blockSize=%d stepSize=%d > Derivate DOrder=1" % (
                    0 if self.De else 1,
                    self.D * self.coefs + self.De * 1,
                    self.block_size, self.step_size
                )
            )
            assert feature_plan.addFeature(definition)
            stack.append('mfcc_d')

        # --- 2nd order derivatives
        if self.DD or self.DDe:
            definition = (
                "mfcc_dd: "
                "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                "blockSize=%d stepSize=%d > Derivate DOrder=2" % (
                    0 if self.DDe else 1,
                    self.DD * self.coefs + self.DDe * 1,
                    self.block_size, self.step_size
                )
            )
            assert feature_plan.addFeature(definition)
            stack.append('mfcc_dd')

        # --- prepare the Yaafe engine
        data_flow = feature_plan.getDataFlow()

        return data_flow, stack
