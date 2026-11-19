import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D
from utils.magnitude_max_pooling import magnitude_max_pooling_1d


    


class DataEmbedding_FeaturePatching(nn.Module):
    def __init__(self, seq_len,  patch_size,  embed_dim = 512, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_FeaturePatching, self).__init__()
        self.seq_len = seq_len 
        self.patch_size = patch_size
        self.n_of_patches = (seq_len - patch_size)//(patch_size//2) + 1
        self.inner_dim = patch_size * 10
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv1d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv1d(1, 3, kernel_size=9)
        self.conv3 = nn.Conv1d(1, 3, kernel_size=15)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.fc1 = nn.Linear(self.inner_dim, embed_dim*4)
        self.fc2 = nn.Linear(embed_dim*4, embed_dim)
        self.pe  = PositionalEncoding2D(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.sigm = nn.GELU()

    def forward(self, x, x_mark):
        B, L, N = x.shape
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]

        if x_mark is not None:
            N += x_mark.shape[2]
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)

        x = x.reshape(-1, 1, L)
        x_1 = F.pad(x, (4, 0), mode = 'replicate')
        x_1 = self.conv1(x_1)
        x_2 = F.pad(x, (8, 0), mode = 'replicate')
        x_2 = self.conv2(x_2)
        x_3 = F.pad(x, (14, 0), mode = 'replicate')
        x_3 = self.conv3(x_3)
        x_1 = F.pad(x_1, (2, 0), mode = 'constant', value = 0)
        x_2 = F.pad(x_2, (4, 0), mode = 'constant', value = 0)
        x_3 = F.pad(x_3, (6, 0), mode = 'constant', value = 0)

        x_1 = magnitude_max_pooling_1d(x_1, 3, 1)
        x_2 = magnitude_max_pooling_1d(x_2, 5, 1)
        x_3 = magnitude_max_pooling_1d(x_3, 7, 1)

        

        x_1 = x_1.reshape(B, N, 3, L)
        x_2 = x_2.reshape(B, N, 3, L)
        x_3 = x_3.reshape(B, N, 3, L)
        x = x.reshape(B, N, 1, L)
        

        x_1 = x_1.unfold(3, self.patch_size, self.patch_size//2)
        x_2 = x_2.unfold(3, self.patch_size, self.patch_size//2)
        x_3 = x_3.unfold(3, self.patch_size, self.patch_size//2)
        x = x.unfold(3, self.patch_size, self.patch_size//2)


        x_1 = x_1.permute(0, 1, 3, 2, 4)
        x_2 = x_2.permute(0, 1, 3, 2, 4)
        x_3 = x_3.permute(0, 1, 3, 2, 4)
        x = x.permute(0, 1, 3, 2, 4)


        x = torch.cat([x, x_1, x_2, x_3], dim = 3)


        x = x.reshape(B, N, self.n_of_patches, -1)
        x = self.gelu1(self.fc1(x))
        x = self.fc2(x)
        x = self.pe(x) + x #apply 2D positional encodings

        x = x.reshape(B, -1, self.embed_dim)

        return self.dropout(x)


#For ablation

class DataEmbedding_FeaturePatchingMasked(nn.Module):
    """DataEmbedding_FeaturePatching with (value, mask) per channel and two-sided padding."""

    def __init__(self, seq_len, patch_size, embed_dim=512, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_FeaturePatchingMasked, self).__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_of_patches = (seq_len - patch_size) // (patch_size // 2) + 1
        self.inner_dim = patch_size * 11
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv1d(2, 3, kernel_size=5)
        self.conv2 = nn.Conv1d(2, 3, kernel_size=9)
        self.conv3 = nn.Conv1d(2, 3, kernel_size=15)
        self.gelu1 = nn.GELU()
        self.fc1 = nn.Linear(self.inner_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.pe = PositionalEncoding2D(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, m, x_mark):
        # x: [B, L, N] values (masked cells already zeroed);  m: [B, L, N] mask (1 = predict)
        B, L, N = x.shape
        x = x.permute(0, 2, 1)                       # [B, N, L]
        m = m.permute(0, 2, 1)
        if x_mark is not None:
            K = x_mark.shape[2]
            N += K
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)
            m = torch.cat([m, torch.zeros(B, K, L, device=m.device, dtype=m.dtype)], 1)  # marks always observed

        xm = torch.stack([x, m], dim=2).reshape(-1, 2, L)      # [B*N, 2, L]

        # (k-1) split evenly: 5->(2,2), 9->(4,4), 15->(7,7); pooling 3/5/7 -> (1,1)/(2,2)/(3,3).
        x_1 = self.conv1(F.pad(xm, (2, 2), mode='replicate'))
        x_2 = self.conv2(F.pad(xm, (4, 4), mode='replicate'))
        x_3 = self.conv3(F.pad(xm, (7, 7), mode='replicate'))
        x_1 = F.pad(x_1, (1, 1), mode='constant', value=0)
        x_2 = F.pad(x_2, (2, 2), mode='constant', value=0)
        x_3 = F.pad(x_3, (3, 3), mode='constant', value=0)
        x_1 = magnitude_max_pooling_1d(x_1, 3, 1)
        x_2 = magnitude_max_pooling_1d(x_2, 5, 1)
        x_3 = magnitude_max_pooling_1d(x_3, 7, 1)

        x_1 = x_1.reshape(B, N, 3, L)
        x_2 = x_2.reshape(B, N, 3, L)
        x_3 = x_3.reshape(B, N, 3, L)
        xm = xm.reshape(B, N, 2, L)

        p, s = self.patch_size, self.patch_size // 2
        x_1 = x_1.unfold(3, p, s).permute(0, 1, 3, 2, 4)
        x_2 = x_2.unfold(3, p, s).permute(0, 1, 3, 2, 4)
        x_3 = x_3.unfold(3, p, s).permute(0, 1, 3, 2, 4)
        xm = xm.unfold(3, p, s).permute(0, 1, 3, 2, 4)

        out = torch.cat([xm, x_1, x_2, x_3], dim=3).reshape(B, N, self.n_of_patches, -1)
        out = self.fc2(self.gelu1(self.fc1(out)))
        out = self.pe(out) + out
        return self.dropout(out.reshape(B, -1, self.embed_dim))
