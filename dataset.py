import os
import pandas as pd
import math
import random
import numpy as np
import torch

from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm


def interpolate_random_samples(cgm, cgm_time, max_len):
    """
    Linearly interpolates `max_len` random samples from irregular CGM data.

    Args
    ----
    cgm : 1-D tensor of CGM values
    cgm_time : 1-D tensor of time stamps (need not be sorted)
    max_len : int – number of points to sample

    Returns
    -------
    interpolated_cgm : 1-D tensor  (max_len,)
    interpolated_times : 1-D tensor (max_len,)
    """
    # --- 1. sort time–value pairs once ---------------------------------------
    sorted_time, order = cgm_time.sort()          # ascending
    sorted_cgm         = cgm[order]

    # --- 2. draw uniformly within [t_min, t_max] -----------------------------
    t_min, t_max = sorted_time[0], sorted_time[-1]
    interpolated_times = torch.rand(
        max_len, device=cgm.device, dtype=sorted_time.dtype
    ).mul_(t_max - t_min).add_(t_min).sort().values

    # --- 3. locate surrounding indices --------------------------------------
    indices = torch.searchsorted(sorted_time, interpolated_times)
    indices = indices.clamp(1, len(sorted_time) - 1)   # keep 1 ≤ idx ≤ n-1

    left  = indices - 1
    right = indices

    # --- 4. linear interpolation --------------------------------------------
    t_left,  t_right  = sorted_time[left],  sorted_time[right]
    v_left, v_right   = sorted_cgm[left],   sorted_cgm[right]

    weight = (interpolated_times - t_left) / (t_right - t_left)
    interpolated_cgm = v_left + weight * (v_right - v_left)

    return interpolated_cgm, interpolated_times
    

class CGMDataset(Dataset):
    def __init__(self, file, max_len=2048, split="train", pred_time=60, seed=42, augment=True, max_deviation=45, max_range=1.5):
        self.max_len = max_len
        self.split = split
        self.augment = augment
        self.pred_time = pred_time // 5
        self.data = []
        self.max_deviation = max_deviation
        self.max_range = max_range

        # Load and process data
        all_data = self.load_all_user_data(file)
        for dataset_name in all_data:
            for user_name in all_data[dataset_name]:
                user_data = all_data[dataset_name][user_name]
                if all(k in user_data for k in ['data_type=cgm', 'data_type=basal', 'data_type=bolus']):
                    self.data.append({
                        "cgm": user_data['data_type=cgm'],
                        "basal": user_data['data_type=basal'],
                        "bolus": user_data['data_type=bolus']
                    })

        # Set seed for reproducible shuffling
        random.Random(seed).shuffle(self.data)

        # Train/validation split
        split_idx = int(0.8 * len(self.data))
        if self.split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]


    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self, uid):
        """
        Returns one randomly-sampled training example with
        • max_len input points (CGM, basal, bolus) in [0, input_window] and
        • max_len target points in (input_window, input_window+pred_time].
        All time axes are in 5-minute units, strictly non-negative, and sorted.
        """
        while True:
            # ---------- 1. pick a user and a window long enough ----------
            uid        = random.randrange(len(self.data))        # allow idx = 0
            user_data  = self.data[uid]
            cgm_df     = user_data["cgm"]
            basal_df   = user_data["basal"]
            bolus_df   = user_data["bolus"]

            augment = self.augment
            if augment and random.random() < 0.5:
                augment = False

            
            input_window = self.max_len
            if augment == True:
                input_window = int(random.triangular(0.5, 1.5, 1) * self.max_len)
            if len(cgm_df) <= input_window + self.pred_time:
                continue
    
            # ---------- 2. slice raw CGM for this sample ----------
            start_idx = random.randint(0, len(cgm_df) - input_window - self.pred_time)
            raw_cgm   = cgm_df.iloc[start_idx : start_idx + input_window + self.pred_time].sort_values('datetime') 
    
            input_cgm = raw_cgm.iloc[:input_window]
            
            start_time, end_time = input_cgm['datetime'].min(), input_cgm['datetime'].max()

            
            if (end_time - start_time).total_seconds() / 300 > input_window * self.max_range:
                continue

            
            
            # ---------- 3. helper to cut & timestamp other streams ----------
            def windowize(df, value_col):
                """
                Extract rows between start_time and end_time (inclusive) and
                return (times, values) as 1-D float32 tensors of equal length.
                """
                mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
            
                # relative times in 5-minute units
                rel = ((df.loc[mask, 'datetime'] - start_time)
                       .dt.total_seconds()
                       .to_numpy(dtype='float64')) / 300.0
            
                vals = df.loc[mask, value_col].to_numpy(dtype='float32')
            
                times = torch.from_numpy(rel).float().clamp_(min=0)   # (N,)
                vals  = torch.from_numpy(vals)                        # (N,)
            
                return times, vals
    
            basal_time, basal = windowize(basal_df, 'basal_rate')
            bolus_time, bolus = windowize(bolus_df, 'bolus')

    
            # ---------- 4. CGM → tensors, interpolate to max_len ----------
            cgm_vals  = torch.as_tensor(input_cgm['cgm'].to_numpy(dtype='float32'))
            cgm_time  = torch.as_tensor(
                ((input_cgm['datetime'] - start_time).dt.total_seconds() / 300.0).to_numpy(),
                dtype=torch.float32
            ).clip_(min=0)

            if ((cgm_time[1:] - cgm_time[:-1]) > self.max_deviation/5).sum() != 0:
                continue
    
            if augment == True:
                cgm, cgm_time = interpolate_random_samples(cgm_vals, cgm_time, self.max_len)
                #cgm = cgm + torch.randn(self.max_len) * 5
                basal_time = basal_time + (torch.randn(basal_time.shape) - 0.5) * 2 * 0.05
                bolus_time = bolus_time + (torch.randn(bolus_time.shape) - 0.5) * 2 * 0.05
            else:
                cgm, cgm_time = cgm_vals, cgm_time
    
            # ---------- 5. target CGM for forecasting ----------
            target_df   = raw_cgm.iloc[self.pred_time:]
            target_vals = torch.as_tensor(target_df['cgm'].to_numpy(dtype='float32'))
            target_time = torch.as_tensor(
                ((target_df['datetime'] - start_time).dt.total_seconds() / 300.0).to_numpy(),
                dtype=torch.float32
            ).clip_(min=0)

            if augment == True:
                target_cgm, target_time = interpolate_random_samples(target_vals, target_time, self.max_len)
            else:
                #indices = torch.randint(0, len(target_vals), (self.max_len,))
                #target_cgm, target_time = target_vals[indices], target_time[indices]
                target_cgm, target_time = target_vals, target_time
                
            # ---------- 6. prediction horizons ----------
            pred_time = torch.rand(self.max_len) * (self.pred_time - 1) + 1  # in 5-min units
            pred_time = torch.minimum(pred_time, target_time - cgm_time[0] - 0.05).clamp(min=0)
    
            return {
                'cgm'        : cgm,
                'cgm_time'   : cgm_time,
                'basal'      : basal,
                'basal_time' : basal_time,
                'bolus'      : bolus,
                'bolus_time' : bolus_time,
                'raw_cgm'    : torch.as_tensor(raw_cgm['cgm'].to_numpy(dtype='float32')),
                'raw_cgm_time': torch.as_tensor(
                    ((raw_cgm['datetime'] - start_time).dt.total_seconds() / 300.0).to_numpy(),
                    dtype=torch.float32
                ).clip_(min=0),
                'target_cgm' : target_cgm,
                'target_time': target_time,
                'pred_time'  : pred_time
            }




    @staticmethod
    def load_all_user_data(base_path):
        data = defaultdict(lambda: defaultdict(dict))
        user_entries = []
    
        # Step 1: Index all users (a user appears in all 3 category folders)
        for dataset_name in os.listdir(base_path):
            dataset_path = os.path.join(base_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
    
            # Assume cgm folder contains the superset of users
            cgm_path = os.path.join(dataset_path, 'data_type=cgm')
            if not os.path.isdir(cgm_path):
                continue
    
            for user_name in os.listdir(cgm_path):
                user_dirs = {
                    'cgm': os.path.join(dataset_path, 'data_type=cgm', user_name),
                    'basal': os.path.join(dataset_path, 'data_type=basal', user_name),
                    'bolus': os.path.join(dataset_path, 'data_type=bolus', user_name)
                }
                if all(os.path.isdir(p) for p in user_dirs.values()):
                    user_entries.append((dataset_name, user_name, user_dirs))
    
        # Step 2: Load each user's full data with tqdm
        for dataset_name, user_name, user_dirs in tqdm(user_entries, desc="Users loaded"):
            for dtype, path in user_dirs.items():
                parquet_files = [f for f in os.listdir(path) if f.endswith('.parquet')]
                dfs = []
                for file in parquet_files:
                    try:
                        df = pd.read_parquet(os.path.join(path, file))
                        dfs.append(df)
                    except Exception as e:
                        print(f"Error reading {file} for user {user_name}: {e}")
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
    
                    # Drop NaNs from relevant columns based on dtype
                    if dtype == 'cgm':
                        df = df.dropna(subset=['cgm'])
                    elif dtype == 'basal':
                        df = df.dropna(subset=['basal_rate'])
                    elif dtype == 'bolus':
                        df = df.dropna(subset=['bolus'])
    
                    data[dataset_name][user_name][f"data_type={dtype}"] = df
                else:
                    data[dataset_name][user_name][f"data_type={dtype}"] = pd.DataFrame()
    
        return data


def collate_fn(batch):
    # Stack fixed-length tensors
    cgm = torch.stack([item['cgm'] for item in batch])
    cgm_time = torch.stack([item['cgm_time'] for item in batch])
    target_cgm = torch.stack([item['target_cgm'] for item in batch])
    target_time = torch.stack([item['target_time'] for item in batch])
    pred_time = torch.stack([item['pred_time'] for item in batch])

    raw_cgm = [item['raw_cgm'] for item in batch]
    raw_cgm_time = [item['raw_cgm_time'] for item in batch]

    # Collect basal and bolus before padding
    basal_list = [item['basal'] for item in batch]
    basal_time_list = [item['basal_time'] for item in batch]
    bolus_list = [item['bolus'] for item in batch]
    bolus_time_list = [item['bolus_time'] for item in batch]

    # Safely compute maximum times
    all_times = [cgm_time.max(), target_time.max()]
    for basal_time in basal_time_list:
        if len(basal_time) > 0:
            all_times.append(basal_time.max())
    for bolus_time in bolus_time_list:
        if len(bolus_time) > 0:
            all_times.append(bolus_time.max())
    
    max_time_value = max(t.item() for t in all_times) + 1000  # add large buffer

    # Pad basal and bolus sequences
    def pad_sequence(sequences, time_sequences, pad_value=0.0, pad_time_value=max_time_value):
        if len(sequences) == 0:
            return torch.zeros(0, 0), torch.full((0, 0), pad_time_value)

        lengths = [len(seq) for seq in sequences]
        max_len = ((max(lengths)//16) * 16 + 16) if lengths else 0

        padded = torch.zeros(len(sequences), max_len)
        padded_time = torch.full((len(sequences), max_len), pad_time_value)

        for i, (seq, time_seq) in enumerate(zip(sequences, time_sequences)):
            if len(seq) > 0:
                padded[i, :len(seq)] = seq
                padded_time[i, :len(seq)] = time_seq
        return padded, padded_time

    basal, basal_time = pad_sequence(basal_list, basal_time_list)
    bolus, bolus_time = pad_sequence(bolus_list, bolus_time_list)

    return {
        'cgm': cgm,
        'cgm_time': cgm_time,
        'basal': basal,
        'basal_time': basal_time,
        'bolus': bolus,
        'bolus_time': bolus_time,
        'raw_cgm': raw_cgm,
        'raw_cgm_time': raw_cgm_time,
        'target_cgm': target_cgm,
        'target_time': target_time,
        'pred_time': pred_time
    }
