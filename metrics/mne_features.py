import mne_features
import numpy as np


def get_features_1c(x, selected_features, sfreq=250):
    x = x.reshape(1, 1, -1)
    features = mne_features.feature_extraction.extract_features(
        x,
        sfreq=sfreq,
        selected_funcs=selected_features
    )
    return features.squeeze()


def get_features_of_batch(x, selected_features, sfreq=250):
    features = []
    for example in x:
        features.append(get_features_1c(example, selected_features, sfreq=sfreq))
    features = np.asarray(features)
    features = np.sum(features, axis=0) / features.shape[1]
    return features


def compute_feature_similarity(dataset_features, sample, selected_features):
    sample_features = get_features_of_batch(sample, selected_features)
    diff = abs((dataset_features - sample_features))
    if  dataset_features != 0:
        diff /=  dataset_features
        diff = abs(diff)
    diff = np.sum(diff)
    assert(diff >= 0)
    return diff


class MNEFeatureCriterion(object):
    def __init__(self, selected_features, sfreq=250):
        self.selected_features = selected_features
        self.sfreq = sfreq

    def get_features_diff(self, x, y):
        x, y = x.cpu().numpy(), y.cpu().numpy()

        # so zero crossings works
        x, y = x - .5, y - .5
        x_features = get_features_1c(x, self.selected_features, sfreq=self.sfreq)
        y_features = get_features_1c(y, self.selected_features, sfreq=self.sfreq)
        diff = np.sum(abs((x_features - y_features)))
        if np.sum(x_features) != 0:
            diff /= np.sum(abs(x_features))
            
        assert(diff >= 0)
        return diff
