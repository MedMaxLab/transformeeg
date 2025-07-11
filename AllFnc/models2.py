# ==========================================================================
# This file contains a backup copy of the transformEEG architecture
# modified to gather materials for the supplementary materials
# In particular, transformEEG was modified to include the possibility of
# adding more attention heads, a positional encoding, or a class token
# See Supplementary materials sections A.5 and A.7
# ==========================================================================

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def reset_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


class Conv1DEncoder(nn.Module):

    def __init__(
        self,
        Chans,
        D1 = 2,
        D2 = 2,
        kernLength1 = 5,
        kernLength2 = 5,
        pool  = 4,
        stridePool = 2,
        dropRate = 0.2,
        ELUAlpha = .1,
        batchMomentum = 0.25,
        seed = None
    ):
        self._reset_seed(seed)
        super(Conv1DEncoder, self).__init__()

        self.D1 = D1
        F1 = Chans*D1
        self.blck1 = nn.Sequential(
            nn.Conv1d(Chans, F1, kernLength1, padding = 'same', groups=Chans),
            nn.BatchNorm1d(F1, momentum = batchMomentum),
            nn.ELU(ELUAlpha),
        )
        
        #self.pool1 = nn.MaxPool1d(pool, stridePool)
        self.pool1 = nn.AvgPool1d(pool, stridePool)
        self.drop1 = nn.Dropout1d(dropRate)

        self.blck2 = nn.Sequential(
            nn.Conv1d(F1, F1, kernLength2, padding = 'same', groups = F1),
            nn.BatchNorm1d(F1, momentum = batchMomentum),
            nn.ELU(ELUAlpha),
        )

        self.D2 = D2
        F2 = Chans*D1*D2
        self.blck3 = nn.Sequential(
            nn.Conv1d(F1, F2, kernLength2, padding = 'same', groups = F1),
            nn.BatchNorm1d(F2, momentum = batchMomentum),
            nn.ELU(ELUAlpha),
        )

        self.pool2 = nn.AvgPool1d(pool, stridePool)#, padding=1)
        self.drop2 = nn.Dropout1d(dropRate)

        self.blck4 = nn.Sequential(
            nn.Conv1d(F2, F2, kernLength2, padding = 'same', groups = F2),
            #nn.Conv1d(F2, F2, 1, padding = 'same'),
            nn.BatchNorm1d(F2, momentum = batchMomentum),
            nn.ELU(ELUAlpha),
        )

    def _reset_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    @torch.no_grad()
    def _pad_input_for_depth(self, x: torch.tensor, D: int=2) -> torch.tensor:
        """
        pad input 3D tensor on the chanel dimension to make it compatible
        with the output of a depthwise canv1d with depth bigger than 1
        """
        out = torch.cat(tuple(x for i in range(D)), -2)
        for i in range(D):
            out[:, i::2, :]= x
        return out

    def forward(self, x):
        x1 = self.blck1(x)
        #x1 = self._pad_input_for_depth(x, self.D1) + x1
        x1 = self.pool1(x1)
        x1 = self.drop1(x1)
        x2 = self.blck2(x1)
        x2 = x1 + x2 
        x3 = self.blck3(x2)
        #x3 = self._pad_input_for_depth(x2, self.D2) + x3
        x3 = self.pool2(x3)
        x3 = self.drop2(x3)
        x4 = self.blck4(x3)
        x4 = x3 + x4
        return x4

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_len: int = 256,
        n: int = 10000
    ):
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term1 = torch.pow(n, torch.arange(0, math.ceil(d_model/2))/d_model)
        if d_model%2 == 0:
            div_term2 = div_term1
        else:
            div_term2 = div_term1[:-1]

        print(div_term1.shape, div_term2.shape)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position/div_term1)
        pe[0, :, 1::2] = torch.cos(position/div_term2)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[0,:x.size(1)]


class TransformEEG(nn.Module):

    def __init__(
        self,
        nb_classes: int,
        Chan: int,
        D1: int,
        D2: int,
        Nhead: int,
        Nlayer: int, 
        seed: int=None
    ):

        self._reset_seed(seed)
        super(TransformEEG, self).__init__()
        self.Chan = Chan
        Features = Chan*D1*D2
        self.Features = Features
        
        self.token_gen = Conv1DEncoder(
            Chan,
            D1            = D1,      # 2
            D2            = D2,      # 2
            kernLength1   = 5,
            kernLength2   = 5,
            pool          = 4,       # 4
            stridePool    = 2,       # 2
            dropRate      = 0.2,
            ELUAlpha      = 0.1,
            batchMomentum = 0.25,
        )
        #self.pos_enc = PositionalEncoding(Features, 512, 1000)
        #self.cls_token = torch.nn.Parameter(
        #    torch.randn(1,1,Features)/math.sqrt(Features),
        #    requires_grad=True
        #)
        self.has_cls_token = False
        
        self._reset_seed(seed)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                Features,
                nhead                = Nhead,
                dim_feedforward      = Features,
                dropout              = 0.2,
                activation           = torch.nn.functional.hardswish,
                batch_first          = True
            ),
            num_layers           = Nlayer,
            enable_nested_tensor = False
        )
        self.pool_lay = torch.nn.AdaptiveAvgPool1d((1))

        self.shuffle_features = False
        if Nhead>1 and Features>Chan:
            reshaper = []
            N = Features//Chan
            for k in range(N):
                for i in range(Chan):
                    reshaper.append(i*N+k)
            reshaper = torch.tensor(reshaper, dtype=torch.long)
            self.reshaper = torch.nn.Parameter(reshaper, requires_grad=False)
            self.shuffle_features = True
        
        self._reset_seed(seed)
        self.linear_lay = nn.Sequential(
            nn.Linear(Features, Features//2 if Features//2>64 else 64),
            nn.LeakyReLU(),
            nn.Linear(Features//2 if Features//2>64 else 64, 1 if nb_classes <= 2 else nb_classes)
        )
        
    def _reset_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    def forward(self, x, y=None):
        x = self.token_gen(x)
        if self.has_cls_token:
            x = torch.permute(x, [0,2,1])
            x = torch.cat((self.cls_token.expand(x.shape[0],-1,-1), x), 1)
            #x = self.pos_enc(x)
            x = self.transformer(x)
            x = torch.permute(x, [0,2,1])
            x = x[:,:,0]
        else:
            x = torch.permute(x, [0,2,1])
            if self.shuffle_features:
                x = x[:,:,self.reshaper]
            #x = self.pos_enc(x)
            x = self.transformer(x)
            x = torch.permute(x, [0,2,1])
            #x = x[:,:,-1]
            x = self.pool_lay(x)
            x = x.squeeze(-1)
        x = self.linear_lay(x)
        return x

class PSDNetFinal(nn.Module):

    def __init__(self, nb_classes, Chan, Features, temporal=2000, spectral=180, seed=None):

        self._reset_seed(seed)
        super(PSDNetFinal, self).__init__()

        self.Chan = Chan
        self.Features = Features
        self.temporal = temporal
        self.spectral = spectral

        self.token_gen_eeg = Conv1DEncoder(
            Chan,
            D1            = 2,
            D2            = 2,
            kernLength1   = 5,
            kernLength2   = 5,
            pool          = 4,       # 4
            stridePool    = 2,       # 2
            dropRate      = 0.2,
            ELUAlpha      = 0.1,
            batchMomentum = 0.25,
        )

        #self.token_gen_psd = PSDBlockBig(self.spectral, embedding=64, seed=seed)
        self.token_gen_psd = Conv1DEncoder(
            Chan,
            D1            = 2,
            D2            = 2,
            kernLength1   = 5,
            kernLength2   = 5,
            pool          = 4,       # 4
            stridePool    = 2,       # 2
            dropRate      = 0.2,
            ELUAlpha      = 0.1,
            batchMomentum = 0.25,
        )

        
        self._reset_seed(seed)
        # self.transformer_eeg = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         Features,
        #         nhead                = 1,
        #         dim_feedforward      = Features,
        #         dropout              = 0.2,
        #         activation           = torch.nn.functional.hardswish,
        #         batch_first          = True
        #     ),
        #     num_layers           = 2,
        #     enable_nested_tensor = False
        # )
        self.transformer_eeg = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                Features,
                nhead                = 1,
                dim_feedforward      = Features,
                dropout              = 0.2,
                activation           = "gelu",
                batch_first          = True
            ),
            num_layers           = 2,
        )
        self.pool_lay = torch.nn.AdaptiveAvgPool1d((1))
        
        self._reset_seed(seed)
        self.linear_lay = nn.Sequential(
            #nn.Linear(2*Features, Features//2 if Features//2>64 else 64),
            nn.Linear(Features, Features//2 if Features//2>64 else 64),
            nn.LeakyReLU(),
            nn.Linear(Features//2 if Features//2>64 else 64, 1 if nb_classes <= 2 else nb_classes)
        )


    def _reset_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)


    def forward(self, x):

        # psd branch
        xf = x[..., self.temporal:]
        xf = self.token_gen_psd(xf)
        xf = torch.permute(xf, [0,2,1])

        # temp branch
        xt = x[..., :self.temporal]
        xt = self.token_gen_eeg(xt)
        xt = torch.permute(xt, [0,2,1])
        #xt = torch.cat((xt, xf), 1) # Mid fusion concat sequence
        #xt = self.transformer_eeg(xt)
        xt = self.transformer_eeg(xt, xf) # Mid fusion with decoder layer
        xt = torch.permute(xt, [0,2,1])
        xt = self.pool_lay(xt)
        xt = xt.squeeze(-1)

        # final head
        #x = torch.cat((xt, xf), -1) # Late fusion concat sequence
        #x = self.linear_lay(xt)
        x = self.linear_lay(xt)
        return x