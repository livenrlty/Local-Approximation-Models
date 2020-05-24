import numpy as np
import scipy.fftpack as fftpack

from .base import BaseLAM
    
class FFT(BaseLAM):
    def __init__(self, n_harmonics):
        super(FFT, self).__init__()
        self.n_harmonics = n_harmonics
        self.name = 'fft_{}'.format(self.n_harmonics)

    def fit_row(self, row):
        params = fftpack.fft(row)
        pred = fftpack.ifft(params)
        top_harmonics_idx = np.argsort(np.abs(params))[-self.n_harmonics - 1: -1]
        return (pred, params[top_harmonics_idx])