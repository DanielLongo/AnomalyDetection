import torch
from torch.utils.data import Dataset
from dynaconf import settings
import pandas as pd
from .utils import load_eeg_file, get_recordings_df
from .filtering import butter_lowpass_filter2
from progress.bar import ChargingBar
import multiprocessing as mp


class EEGDataset(Dataset):

    def __init__(self, csv_file, length, select_channels, max_num_examples=None, root_dir=None, transform=None, remove_short_recordings=True, filter_freq=False):

        if root_dir is None:
            self.root_dir = settings.DATASET_DIR
        else:
            self.root_dir = root_dir

        self.recordings_df = get_recordings_df(csv_file, max_num_examples)
        self.length = length
        self.select_channels = select_channels
        self.transform = transform
        self.filter_freq = filter_freq

        if remove_short_recordings:
            self.remove_recordings_too_short()

    def __len__(self):
        return len(self.recordings_df)

    def __getitem__(self, idx):
        filename, start_pos = self.recordings_df.iloc[idx, :2]
        signals, _  = load_eeg_file(filename)

        # only keep selected channels
        signals = signals[self.select_channels]
        # only output selected portion
        signals = signals[:, start_pos: start_pos + self.length]
        
        if self.filter_freq:
            signals = butter_lowpass_filter2(signals, cutoff=settings.LOW_PASS_FILTER_CUTOFF, fs=settings.FREQUENCY).copy()

        signals = torch.from_numpy(signals).type('torch.FloatTensor')
        if self.transform:
            signals = self.transform(signals)

        return signals

    def recording_is_sufficient_length(self, idx):
        # returns idx if insufficient otherwise None
        cur_recording = self.__getitem__(idx)
        if cur_recording.shape[1] != self.length:
            return idx
        return None
    
    def remove_recordings_too_short(self):

        print("Removing recordings of insufficient length...")
        original_len = self.__len__()

        pool = mp.Pool(mp.cpu_count())
        remove_indices = pool.map(self.recording_is_sufficient_length, list(range(original_len)))
        remove_indices = [i for i in remove_indices if i is not None] # remove all nones

        self.recordings_df = self.recordings_df.drop(remove_indices)
        print(f"Removed {len(remove_indices)} of {original_len} recordings. There are now {self.__len__()} recordings.")






                
            
        

if __name__ == "__main__":
    dataset = EEGDataset(
        csv_file = settings.TRAIN_DATASET_CSV,
        length = 1000,
        select_channels = [0,1,2],
    )

    print("Length of the dataset", dataset.__len__())
    print("Shape of dataset item", dataset[0].shape)



