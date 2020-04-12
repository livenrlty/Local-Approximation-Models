import numpy as np

class Ensemble:
    def __init__(self, models):
        self.models = models
        self.models_indices = []
        self.params = None
        
    def fit(self, rows):
        self.models_indices = []
        preds = []
        params = []
        
        for i, model in enumerate(self.models):
            model.fit(rows)
            params.append(model._params)
            self.models_indices += str(i)*model._params.shape[1]
        
        self.params = np.hstack(params).real