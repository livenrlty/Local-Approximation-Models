import numpy as np
import scipy.fftpack as fftpack
import statsmodels.tsa.ar_model as ar_model
import statsmodels.tsa.arima_model as arima_model

class BaseLAM:
    def __init__(self):
        self._preds = None
        self._params = None

    
    def fit_row(self, row):
        raise NotImplementedError('Method is not implemented')
    
    def fit(self, rows):
        if len(rows.shape) != 2:
            raise ValueError('Unsupported data shape')
        
        pred_rows = [self.fit_row(row) for row in rows]
        self._preds = np.vstack([item[0] for item in pred_rows])
        self._params = np.vstack([item[1] for item in pred_rows])
            
        
class AR(BaseLAM):
    def __init__(self, maxlag):
        super(AR, self).__init__()
        self.maxlag = maxlag
    
    def fit_row(self, row):
        model = ar_model.AR(row).fit(maxlag=self.maxlag)
        head = row[:self.maxlag]
        tail = model.predict(start=self.maxlag, end=len(row) - 1)
        pred = np.concatenate((head, tail))
        params = model.params
        return (pred, params)

class ARMA(BaseLAM):
    def __init__(self, p, q):
        super(ARMA, self).__init__()
        self.p = p
        self.q = q
    
    def fit_row(self, row):
        model = arima_model.ARMA(row, (self.p, self.q)).fit()
        pred = model.predict(start=0, end=len(row) - 1)
        params = model.params
        return (pred, params)

class ARIMA(BaseLAM):
    def __init__(self, p, d, q):
        super(ARIMA, self).__init__()
        self.p = p
        self.d = d
        self.q = q

    def fit_row(self, row):
        model = arima_model.ARIMA(row, (self.p, self.d, self.q)).fit()
        pred = model.predict()
        params = model.params
        return (pred, params)
    
class FFT(BaseLAM):
    def __init__(self, n_harmonics):
        super(FFT, self).__init__()
        self.n_harmonics = n_harmonics

    def fit_row(self, row):
        params = fftpack.fft(row)
        pred = fftpack.ifft(params)
        top_harmonics_idx = np.argsort(np.abs(params))[-self.n_harmonics - 1: -1]
        return (pred, params[top_harmonics_idx])