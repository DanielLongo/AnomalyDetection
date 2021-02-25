import h5py
import pandas as pd
import torch

def load_eeg_file(filename):
    hdf = h5py.File(filename, "r")
    rec = hdf["record-0"]
    signals = rec["signals"]
    specs = {
        "sample_frequency": rec.attrs["sample_frequency"],
        "number_channels": rec.attrs["number_channels"]
    }
    return signals, specs

def get_recordings_df(csv_file, max_num_examples, print_results_info=True):
        recordings_df = pd.read_csv(csv_file)
        if print_results_info:
            print(f"Found {len(recordings_df)} recordings")

        if max_num_examples is not None:
            recordings_df = recordings_df[:max_num_examples]
            if print_results_info:
                print(f"By set limit only using {len(recordings_df)} recordings")
        
        return recordings_df
    
    
def normalize(x):
    x_range = x.max()-x.min()
    if x_range == 0:
        return torch.zeros_like(x)
    return (x-x.min())/x_range