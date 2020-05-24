import numpy as np
import statsmodels.tsa.ar_model as ar_model
import statsmodels.tsa.arima_model as arima_model

from .base import BaseLAM

class AR(BaseLAM):
    def __init__(self, maxlag):
        super(AR, self).__init__()
        self.maxlag = maxlag    
        self.name = 'ar_{}'.format(self.maxlag)
    
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
        self.name = 'arma_{}_{}'.format(self.p, self.q)
    
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
        self.name = 'arima_{}_{}_{}'.format(self.p, self.d, self.q)

    def fit_row(self, row):
        model = arima_model.ARIMA(row, (self.p, self.d, self.q)).fit()
        pred = model.predict()
        params = model.params
        return (pred, params)