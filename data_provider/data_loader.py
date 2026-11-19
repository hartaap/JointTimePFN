import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data_provider.mask_sampling import sample_mask
import warnings
import pyarrow.dataset as ds
warnings.filterwarnings('ignore')

_arrow_cache = {}  # module-level cache so train/val/test share one loaded DataFrame


def make_variable_channel_collate(min_channel, max_channel):
    # Collate that draws one channel count per batch, with a fresh k-channel subset and mask
    def collate(batch):
        fulls = [b[2] for b in batch]
        marks = [b[3] for b in batch]
        T, Nfull = fulls[0].shape
        rng = np.random.default_rng()
        upper = max_channel if 0 < max_channel < Nfull else Nfull
        k = int(rng.integers(min_channel, upper + 1)) if 0 < min_channel < upper else Nfull
        iv, mk, fl = [], [], []
        for full in fulls:
            keep = rng.choice(Nfull, size=k, replace=False)
            f = full[:, keep]
            m = torch.from_numpy(sample_mask(T, k, rng)[0])
            iv.append(f * (1.0 - m)); mk.append(m); fl.append(f)
        return torch.stack(iv), torch.stack(mk), torch.stack(fl), torch.stack(marks)
    return collate


class Dataset_GaussianPCoregionalization(Dataset):
    def __init__(self, root_path=None, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='h', n_variables = 10, seq_len = 192, pred_len = 96, stride = -1, noise = 0.1, synthetic_length = 1024, series_range=None,
                 return_mask=False, max_channel=-1, min_channel=-1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.n_variables = n_variables
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.noise = noise
        self.synthetic_length = synthetic_length
        self.series_range = series_range
        if stride == -1:
            self.stride = 1
        else:
            self.stride = stride
        self.return_mask = return_mask
        self.max_channel = max_channel   # >0 -> randomly subsample this many channels per sample (low-N training)
        self.min_channel = min_channel   # >0 -> channel count drawn per batch in make_variable_channel_collate
        self._read_data()

    def _get_rng(self):
        # Per-worker RNG: each DataLoader worker process gets its own
        # numpy Generator so masks differ across workers and across epochs.
        pid = os.getpid()
        if not hasattr(self, '_rng') or getattr(self, '_rng_pid', None) != pid:
            wi = torch.utils.data.get_worker_info()
            wid = wi.id if wi is not None else 0
            self._rng = np.random.default_rng((pid * 1009 + wid) & 0xffffffff)
            self._rng_pid = pid
        return self._rng


    def _read_data(self):
        if self.data_path is not None:
            all_path = os.path.join(self.root_path, self.data_path)
            if all_path not in _arrow_cache:
                _arrow_cache[all_path] = ds.dataset(all_path, format="arrow").to_table().to_pandas()
            full_df = _arrow_cache[all_path]
            if self.series_range is not None:
                self.dataset = full_df.iloc[self.series_range[0]:self.series_range[1]].reset_index(drop=True)
            else:
                self.dataset = full_df
        else:
            raise ValueError("Data path is not provided")


    def __getitem__(self, index):
        index_var = index % self.dataset.shape[0]
        index_interior = index // self.dataset.shape[0]
        
        s_begin = index_interior*self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len


        # get data
        data_x = self.dataset["target"][index_var]
        # convert to numpy
        data_x = data_x.reshape(-1,self.synthetic_length)

        # NaN -> column mean + small noise (the arrow view is read-only, so copy only if needed)
        inds = np.where(np.isnan(data_x))
        if len(inds[0]) > 0:
            col_mean = np.nanmean(data_x, axis=0)
            data_x = np.array(data_x, copy=True)
            data_x[inds] = np.take(col_mean, inds[1]) + 0.1*np.random.normal(0, 1, len(inds[1]))


        # transpose
        data_x = data_x.T
        
        seq_x = data_x[s_begin:s_end]
        seq_y = data_x[r_begin:r_end]

        #create a multiplicative noise with a gaussian distribution
        noise_x = np.random.normal(1, self.noise, seq_x.shape)
        seq_x = seq_x*noise_x

        noise_y = np.random.normal(1, self.noise, seq_y.shape)
        seq_y = seq_y*noise_y

        seq_x_mark = torch.zeros((seq_x.shape[0], 4))
        seq_y_mark = torch.zeros((seq_y.shape[0], 4))

        if isinstance(seq_x, np.ndarray):
            seq_x = torch.from_numpy(seq_x)

        if isinstance(seq_y, np.ndarray):
            seq_y = torch.from_numpy(seq_y)

        if self.return_mask:

            full = torch.cat([seq_x, seq_y[-self.pred_len:]], dim=0).float()
            T, N = full.shape
            rng = self._get_rng()
            if self.min_channel > 0:
                # variable N: a [B, T, N] batch cannot stack ragged N, so the channel subsample and mask
                # are applied per batch in make_variable_channel_collate. the placeholders are replaced there
                full_mark = torch.zeros((T, 4))
                placeholder = torch.zeros((T, N), dtype=torch.float32)
                return placeholder, placeholder, full, full_mark
            if 0 < self.max_channel < N:                 # fixed low-N: per-window subsample (batch N stays constant)
                keep = rng.choice(N, size=self.max_channel, replace=False)
                full = full[:, keep]; T, N = full.shape
            mask_np, pattern = sample_mask(T, N, rng)
            mask = torch.from_numpy(mask_np)
            input_vals = full * (1.0 - mask)
            full_mark = torch.zeros((T, 4))
            return input_vals, mask, full, full_mark

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.dataset.shape[0]*((self.synthetic_length - self.seq_len - self.pred_len)//self.stride + 1)