import pmdarima as pm
import numpy as np

def auto_arima_pred(sample, n_preds, input_window_size=64):
    window = sample.squeeze()[-input_window_size:]
    model = pm.auto_arima(window)
    return model.predict(n_preds)