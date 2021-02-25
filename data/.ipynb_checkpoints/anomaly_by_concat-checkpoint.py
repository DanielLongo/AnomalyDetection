# create anomaly set by concatenating two unique clips from different patients 

import torch
from torch.utils.data import Dataset
from .utils import load_eeg_file, get_recordings_df
from .eeg_dataset import EEGDataset
from dynaconf import settings

class AnomalyDatasetByConcat(EEGDataset):
    
    def __init__(self, first_recording_length, second_recording_length, *args, **kwargs):
        
        # not concatenating clips in the begining (only checking length in super constructor)
        self.concat_clips = False
        
        super().__init__(*args, **kwargs)
        
        self.first_recording_length = first_recording_length
        self.second_recording_length = second_recording_length
        
        assert((self.first_recording_length + self.second_recording_length) == self.length), f"Length of the two concatenated segments {self.first_recording_length} and {self.second_recording_length} should equal length param {self.length}"
        
        self.concat_clips = True
    
    def __getitem__(self, first_idx):
        
        # if not concat clips defaults to super method
        if not self.concat_clips:
            return super().__getitem__(first_idx)
        
        second_idx = torch.randint(0, (super().__len__() - 1), (1,))[0].item()
        if second_idx == first_idx:
            # don't want the same clip concatenated to itself
            return self.__getitem__(first_idx)
        
        first_tensor = super().__getitem__(first_idx)
        second_tensor = super().__getitem__(second_idx)
        
        out = torch.cat((first_tensor[:,:self.first_recording_length], second_tensor[:,-self.second_recording_length:]), dim=1)
        return out
        
        
if __name__ == "__main__":
    dataset = AnomalyDatasetByConcat(
        first_recording_length = 800,
        second_recording_length = 200,
        csv_file = settings.TRAIN_DATASET_CSV,
        length = 1000,
        select_channels = [0,1,2],
        max_num_examples = 100,
    )

    print("Length of the dataset", dataset.__len__())
    print("Shape of dataset item", dataset[0].shape)