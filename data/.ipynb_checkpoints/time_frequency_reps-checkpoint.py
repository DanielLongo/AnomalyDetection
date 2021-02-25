import torch
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np
# from Siyi
from scipy.fftpack import fft, diff


def computeTimeFreqRep(signals, freq=12, overlap=0.25, window=1):
    """
    Args:
        signals: EEG segment, shape (number of channels, number of data points)
        freq: sampling frequency in Hz
        overlap: proportion of overlap with previous segment
        window: window size in seconds
    Returns:
        S: time-frequency representation, shape (number of channels, number of data points)
        where number of data points depends on freq, overlap and window.
    """
    n_ch, t_len = signals.shape

​
    start_time = 0
    end_time = min(t_len, start_time + int(freq * window))

    signal_segs = [signals[:, start_time:end_time]]
    ​
    while end_time < t_len:
        offset = int(freq * window * overlap)
        start_time = end_time - offset
        end_time = min(t_len, start_time + int(freq * window))
        curr_seg = signals[:, start_time:end_time]
        if curr_seg.shape[1] < int(freq * window):
            diff = int(freq * window) - curr_seg.shape[1]
            curr_seg = np.concatenate((curr_seg, np.zeros((n_ch, diff))), axis=1)
        signal_segs.append(curr_seg)

    signal_segs = np.concatenate(signal_segs, axis=1)

    # FFT
    S = computeFFT(signal_segs)
    ​
    return S
​

def computeFFT(signals):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
    Returns:
        S: FFT of signals, (number of channels, number of data points)
    """
    N = signals.shape[1]

​
    # fourier transform
    fourier_signal = fft(signals, axis=1)

    S = 1.0 / N * np.abs(fourier_signal)

    return S

# end of Siyi's code

def convert_to_tfr(signals, window_size, freq=250):
    """
    Args:
        signals: EEG segment, shape (number of data points)
        window: window size in datapoints
        freq: sampling frequency in Hz
    Returns:
        S: time-frequency representation, shape (number of channels, number of data points, c)
        where number of data points depends on window.
    """
    if len(signals.shape) == 1:
        signals = signals.unsqueeze(0)
    length = signals.shape[1]
    assert(window_size < length)
    num_segments = length // window_size
    segments = []
    for i in range(num_segments):
        tfr = computeTimeFreqRep(signals[:, i*window_size: (i+1)*window_size])
        segments.append(torch.from_numpy(tfr))
    segments = torch.stack(segments, dim=0)
    segments = segments.transpose(1, 0) 
    return segments