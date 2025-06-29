# %%
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# %%
class EEGMMIDBDataset(Dataset):
  def __init__(self, pickle_path, label_map=None, normalize=False, target_class:str=None, purpose:str='aug', onehot=False):
    """
    Args:
    pickle_path (str): path to the .pkl file with EEG data
    label_map (dict): optional mapping from string label to integer class
    target_class (str): optional mapping that will return rows of data that match str
    purpose (str): 'aug' or 'eegnet' changes shape of signal output based on purpose
    onehot (bool): if y output is a one hot encoding

    return:

    """
    self.df = pd.read_pickle(pickle_path)
    self.normalize = normalize
    self.target_class = target_class
    self.purpose = purpose
    self.onehot = onehot

    # default label
    if label_map is None:
      self.label_map = {
          'left_hand': 0,
          'right_hand': 1,
          'both_hands': 2,
          'both_feet': 3,
          'rest': 4
      }
    else:
      self.label_map = label_map

    self.num_classes = len(self.label_map)


    # filter by class
    if target_class is not None:
      self.df = self.df[self.df['label'] == target_class].reset_index(drop=True)


  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
      row = self.df.iloc[idx]
      signal = row['X']  # shape: (channels, timepoints)
      label = self.label_map[row['label']]


      if self.normalize:
        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)


      # Pad or truncate to 640
      if signal.shape[-1] < 640:
          #print(idx)
          pad_width = 640 - signal.shape[-1]
          pad = np.zeros((signal.shape[0], pad_width), dtype=np.float32)
          signal = np.concatenate([signal, pad], axis=1)
      elif signal.shape[-1] > 640:
          signal = signal[..., :640]


      # adjust shape of signal
      if self.purpose == 'aug':
        # Add dummy spatial dimension → (channels, 1, timepoints)
        signal = np.expand_dims(signal, axis=1).astype(np.float32) # -> (64, 1, 640)
      elif self.purpose == 'eegnet':
        signal = np.expand_dims(signal, axis=2).astype(np.float32) # -> (64, 640, 1)

      # adjust type of y
      if self.onehot:
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        label_tensor[label] = 1.0
      else:
        label_tensor = torch.tensor(label, dtype=torch.long)



      return torch.tensor(signal), label_tensor


# %%
# import os
# # mount to google drive
# from google.colab import drive
# drive.mount('/content/drive')

# # defining path
# drive_path = "/content/drive/MyDrive/eeg_data"
# os.makedirs(drive_path, exist_ok=True)

# %%
# from torch.utils.data import DataLoader
# import pandas as pd

# train_dataset = EEGMMIDBDataset(pickle_path="/content/drive/MyDrive/eeg_data/eegmmidb_train_df.pkl",purpose="gan")
# val_dataset   = EEGMMIDBDataset("/content/drive/MyDrive/eeg_data/eegmmidb_val_df.pkl")
# test_dataset  = EEGMMIDBDataset("/content/drive/MyDrive/eeg_data/eegmmidb_test_df.pkl")

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
# from torch.utils.data import DataLoader

# purposes = ['aug', 'eegnet']

# for p in purposes:
#   train_dataset = EEGMMIDBDataset(pickle_path="/content/drive/MyDrive/eeg_data/eegmmidb_train_df.pkl",purpose=p)
#   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#   for X, y in train_loader:
#     print(f'PURPOSE: {p}')
#     print(f'EEG Shape: {X.shape}')
#     print(f'Label Shape: {y.shape}')
#     break



