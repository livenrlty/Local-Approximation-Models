import numpy as np

class BaseLAM:
    def __init__(self):
        self._preds = None
        self._params = None
        self.name = None

    
    def fit_row(self, row):
        raise NotImplementedError('Method is not implemented')
    
    def fit(self, rows):
        if len(rows.shape) != 2:
            raise ValueError('Unsupported data shape')
        
        pred_rows = [self.fit_row(row) for row in rows]
        self._preds = np.vstack([item[0] for item in pred_rows])
        self._params = np.vstack([item[1] for item in pred_rows])
