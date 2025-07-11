import copy
from itertools import chain, combinations
import math
import numpy as np
import random
from scipy.signal import firwin, freqz
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

__all__ = [
    "PSDNetFinal",
    "Conformer",
    "TransformEEG",
    "XEEGNet",
    "EEGResNet18"
]


# -----------------------------
#       Subject head model
# -----------------------------
class XEEGNetEncoder(nn.Module):

    def __init__(
        self,
        Chans,
        F1=40,
        K1=25,
        F2=40,
        Pool=75,
        p=0.2,
        log_activation_base="e",
        norm_type='batchnorm',
        random_temporal_filter = True,
        spatial_depthwise = False,
        spatial_only_positive = False,
        global_pooling = False,
        Fs=-1,
        freeze_temporal=0,
        bias = [True, True, True],
        seed = None
    ):

        super(XEEGNetEncoder, self).__init__()

        # Set seed before initializing layers
        self.custom_seed = seed
        self._reset_seed(seed)
                
        self.Fs = Fs
        self.chans = Chans
        self.freeze_temporal = freeze_temporal
        self.spatial_only_positive = spatial_only_positive
        self.bias_1conv = bias[0]
        self.bias_2conv = bias[1]
        self.bias_dense = bias[2]
        self.do_global_pooling = global_pooling
        if self.Fs <=0 and not(random_temporal_filter):
            raise ValueError(
                "to properly initialize non random temporal fir filters, "
                "Fs (sampling rate) must be given"
            )
        
        if random_temporal_filter:
            self.conv1 = nn.Conv2d(1, F1, (1, K1), stride=(1, 1), bias=self.bias_1conv)
        else:
            self.conv1 = nn.Conv2d(1, F1, (1, K1), stride=(1,1), bias=self.bias_1conv)
            self._initialize_custom_temporal_filter(self.custom_seed)

        if spatial_depthwise:
            self.conv2 = nn.Conv2d(F1, F2, (Chans, 1), stride=(1, 1), groups = F1, bias = self.bias_2conv)
        else:
            self.conv2 = nn.Conv2d(F1, F2, (Chans, 1), stride=(1, 1), bias = self.bias_2conv)
            
        if "batch" in norm_type.casefold(): 
            self.batch1 = nn.BatchNorm2d(F2,affine=True)
        elif "instance" in norm_type.casefold():
            self.batch1 = nn.InstanceNorm2d(F2)
        else:
            raise ValueError(
                "normalization layer type can be 'batchnorm' or 'instancenorm'"
            )
        
        if log_activation_base in ["e", torch.e]: 
            self.log_activation = lambda x: torch.log(torch.clamp(x, 1e-7, 1e4))
        elif log_activation_base in ["10", 10]:
            self.log_activation = lambda x: torch.log10(torch.clamp(x, 1e-7, 1e4))
        elif log_activation_base in ["db", "dB"]:
            self.log_activation = lambda x: 10*torch.log10(torch.clamp(x, 1e-7, 1e4))
        elif log_activation_base == "logrelu":
            self.log_activation = lambda x: torch.log(torch.nn.functional.relu(x)+1)
        elif log_activation_base in ["linear"]:
            self.log_activation = lambda x: x
        else:
            raise ValueError(
                "allowed activation base are 'e' for torch.log, "
                "'10' for torch.log10, and 'dB' for 10*torch.log10"
            )
        
        if not self.do_global_pooling:
            self.pool2 = nn.AvgPool2d((1, Pool), stride=(1, max(1, Pool//5)))
        else:
            self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        self.drop1 = nn.Dropout(p)
        self.flatten = nn.Flatten()

    def _reset_seed(self, seed):
        if self.custom_seed is not None:
            torch.manual_seed(self.custom_seed)
            np.random.seed(self.custom_seed)
            random.seed(self.custom_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.custom_seed)
                torch.cuda.manual_seed_all(self.custom_seed)

    def forward(self, x):
        if self.freeze_temporal:
            self.freeze_temporal -= 1
            self.conv1.requires_grad_(False)
        else:
            self.conv1.requires_grad_(True)        
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        if self.spatial_only_positive:
            x = self.conv2._conv_forward(x, self._get_spatial_softmax(), self.conv2.bias)
        else:
            x = self.conv2(x)

        x = self.batch1(x)
        x = torch.square(x)
        
        if self.do_global_pooling:
            x = self.global_pooling(x)
        else:
            x = self.pool2(x)

        x = self.log_activation(x)
        x = self.drop1(x)
        x = self.flatten(x)
        
        return x

    @torch.no_grad()
    def _get_spatial_softmax(self):
        return torch.softmax(self.conv2.weight, -2)

    @torch.no_grad()
    def _get_spatial_zero(self):
        return self.conv2.weight-torch.sum(self.conv2.weight,-2, keepdim=True)
    
    @torch.no_grad()
    def _initialize_custom_temporal_filter(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        if self.conv1.weight.shape[-1] >= 75:
            bands = (
                ( 0.5,  4.0), #delta
                ( 4.0,  8.0), #theta
                ( 8.0, 12.0), #alpha
                (12.0, 16.0), #beta1 
                (16.0, 20.0), #beta2
                (20.0, 28.0), #beta3
                (28.0, 45.0)  #gamma
            )
        else:
            bands = (
                ( 0.5,  8.0),
                ( 8.0, 16.0),
                (16.0, 28.0),
                (28.0, 45.0)
            )
        F, KernLength = self.conv1.weight.shape[0], self.conv1.weight.shape[-1]
        comb = self._powerset(bands)
        #if F <= len(comb):
        for i in range(np.min([F,len(comb)])):
            filt_coeff = firwin(
                KernLength,
                self._merge_tuples(comb[i]),
                pass_zero=False,
                fs=self.Fs
            )
            self.conv1.weight.data[i,0,0] = torch.from_numpy(filt_coeff)

    @torch.no_grad()
    def _powerset(self, s):
        return tuple(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))

    @torch.no_grad()
    def _merge_tuples(self, tuples):
        merged = [num for tup in tuples for num in tup]
        merged = sorted(merged)
        if len(merged)>2:
            new_merged = [merged[0]]
            for i in range(1, len(merged)-2, 2):
                if merged[i] != merged[i+1]:
                    new_merged.append(merged[i])
                    new_merged.append(merged[i+1])
            new_merged.append(merged[-1])
            return sorted(new_merged)  
        return merged

    @torch.no_grad()
    def _combinatorial_op(self, N, k):
        return int((math.factorial(N))/(math.factorial(k)*math.factorial(N-k)))
        
        
class XEEGNet(nn.Module):
    def __init__(
        self,
        nb_classes,
        Chans,
        Samples,
        F1=40,
        K1=25,
        F2=40,
        Pool=75,
        p=0.2,
        log_activation_base="e",
        norm_type='batchnorm',
        random_temporal_filter = True,
        spatial_depthwise = False,
        spatial_only_positive = False,
        global_pooling = False, 
        bias = [True, True, True],
        Fs=-1,
        freeze_temporal=0,
        dense_hidden=None,
        return_logits=True,
        seed = None
    ):

        super(XEEGNet, self).__init__()

        # Set seed before initializing layers
        self.custom_seed = seed
        self._reset_seed(seed)

        self.nb_classes = nb_classes
        self.return_logits = return_logits
        self.encoder = XEEGNetEncoder(
            Chans, F1, K1, F2, Pool, p, log_activation_base,
            norm_type,
            random_temporal_filter,
            spatial_depthwise,
            spatial_only_positive,
            global_pooling,
            Fs, freeze_temporal, bias, seed
        )
        if global_pooling:
            self.emb_size = F2
        else:
            self.emb_size = F2 * ((Samples - K1 + 1 - Pool) // max(1,int(Pool//5)) + 1)

        self._reset_seed(seed)
            
        if dense_hidden is None or dense_hidden<=0:
            self.Dense = nn.Linear(self.emb_size, 1 if nb_classes <= 2 else nb_classes, bias=bias[2])
        else:
            self.Dense = nn.Sequential(
                nn.Linear(self.emb_size, dense_hidden, bias=True),
                nn.ELU(alpha=1.0),
                nn.Linear(dense_hidden, 1 if nb_classes <= 2 else nb_classes,bias=bias[2])
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.Dense(x)

        if not (self.return_logits):
            if self.nb_classes <= 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x

    @torch.no_grad()
    def plot_temporal_response(self, filter):
        self.encoder.plot_temporal_response(filter)

    def _reset_seed(self, seed):
        if self.custom_seed is not None:
            torch.manual_seed(self.custom_seed)
            np.random.seed(self.custom_seed)
            random.seed(self.custom_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.custom_seed)
                torch.cuda.manual_seed_all(self.custom_seed)

# ------------------------------
#           ResNet
# ------------------------------
class ResBlock(nn.Module):
    """
    :meta private:
    """
    def __init__(
        self,
        inplanes: int,
        outplanes: int,
        kernel_size: tuple=(1,7),
        stride: tuple=(1,1),
        kernel_reduction: int = 2,
        mix: bool=False,
        force_conv_skip: bool=False,
        seed: int = None,
    ):

        self._reset_seed(seed)
        super(ResBlock, self).__init__()
        self.stride = stride

        if (inplanes != outplanes) or force_conv_skip:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, outplanes, kernel_size=(1, 1), stride=stride,
                    padding="valid",
                    bias=False,
                ),
                nn.BatchNorm2d(outplanes),
            )
        else:
            self.downsample = lambda x: x

        self.conv1 = nn.Conv2d(
            inplanes, outplanes, kernel_size=kernel_size, stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

        if mix:
            new_kernel_size = (kernel_size[1], kernel_size[0])
            self.conv2 = nn.Conv2d(
                outplanes, outplanes,
                kernel_size=new_kernel_size, stride=(1, 1),
                padding="same", bias=False,
            )
        else:
            if kernel_size[0]>1:
                kernel_size = (kernel_size[0]-kernel_reduction, 1)
                self.conv2 = nn.Conv2d(
                    outplanes, outplanes,
                    kernel_size=kernel_size, stride=(1, 1),
                    padding="same", bias=False,
                )
            else:
                kernel_size = (1, kernel_size[1]-kernel_reduction)
                self.conv2 = nn.Conv2d(
                    outplanes, outplanes,
                    kernel_size=kernel_size, stride=(1, 1),
                    padding=(0, kernel_size[1] // 2), bias=False,
                )
        self.bn2 = nn.BatchNorm2d(outplanes)

    def _reset_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    def forward(self, x):
        """
        :meta private:
        """
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class EEGResNet18(nn.Module):
    """
    Pytorch implementation of the Resnet 18 Encoder for EEG data.

    This version uses both temporal and spatial convolutional layers
    (so conv2d with horizontal or vertical kernel).
    See ResNet for the reference paper which inspired this implementation.

    The expected **input** is a **3D tensor** with size
    (Batch x Channels x Samples).

    Parameters
    ----------
    Chans: int
        The number of EEG channels.
    inplane: int, optional
        The number of output filters of the first convolutional layer.
        At embedding size before the final classification head is inplane*8.

        Default = 64
    kernLength: int, optional
        The length of the temporal convolutional layer.

        Default = 25
    kernDiff: int, optional
        The reduction of the 

        Default = 2


    Example
    -------
    >>> import selfeeg.models
    >>> import torch
    >>> x = torch.randn(4,8,512)
    >>> mdl = models.EEGResnet18Encoder(8)
    >>> out = mdl(x)
    >>> print(out.shape) # shoud return torch.Size([4, 296])
    >>> print(torch.isnan(out).sum()) # shoud return 0

    """

    def __init__(
        self,
        nb_classes: int,
        inplane: int = 32,
        kernLength: int = 9,
        kernDiff: int = 2,
        block: nn.Module = ResBlock,
        seed: int = None,
    ):

        assert inplane > 0, "inplane must be a positive integer"
        assert kernLength > 0, "kernLength must be a positive integer"
        assert kernDiff >= 0, "kernDiff must be a non negative integer"

        self._reset_seed(seed)
        super(EEGResNet18, self).__init__()
        self.inplane = inplane
        self.kernLength = kernLength
        self.kernDiff = kernDiff
        self.block = ResBlock

        curr_plane = self.inplane
        curr_kern = self.kernLength
        self.preblock = nn.Sequential(
            nn.Conv2d(1, curr_plane, (1, curr_kern), padding="same"),
            nn.BatchNorm2d(curr_plane),
            nn.ReLU()
        )

        #  RESIDUAL BLOCKS
        self._reset_seed(seed)
        self.stage1 = nn.Sequential(*[
            self.block(
                curr_plane, curr_plane, kernel_size=(1,curr_kern),
                kernel_reduction = self.kernDiff, stride=(1,2), force_conv_skip=True
            ),
            self.block(
                curr_plane, curr_plane, kernel_size=(1,curr_kern),
                kernel_reduction = self.kernDiff
            ),
        ])

        self._reset_seed(seed)
        curr_kern -=2
        self.stage2 = nn.Sequential(*[
            self.block(
                curr_plane, curr_plane*2, kernel_size=(1,curr_kern), 
                stride = (1,2), kernel_reduction = self.kernDiff,
            ),
            self.block(
                curr_plane*2, curr_plane*2, kernel_size=(1,curr_kern),
                kernel_reduction = self.kernDiff,
            ),
        ])
        curr_plane *= 2

        self._reset_seed(seed)
        curr_kern -=2
        self.stage3 = nn.Sequential(*[
            self.block(
                curr_plane, curr_plane*2, kernel_size=(1,curr_kern),
                stride=(1,2), mix=True, kernel_reduction=0,
            ),
            self.block(
                curr_plane*2, curr_plane*2, kernel_size=(1,curr_kern), 
                mix=True, kernel_reduction=0,
            ),
        ])
        curr_plane *= 2

        self._reset_seed(seed)
        self.stage4 = nn.Sequential(*[
            self.block(
                curr_plane, curr_plane*2, kernel_size=(curr_kern,1),
                kernel_reduction = self.kernDiff
            ),
            self.block(
                curr_plane*2, curr_plane*2, kernel_size=(curr_kern,1),
                kernel_reduction = self.kernDiff
            ),
        ])

        curr_plane *= 2
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self._reset_seed(seed)
        self.head = nn.Linear(curr_plane, 1 if nb_classes <= 2 else nb_classes)

    def _reset_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    @torch.autocast(device_type="cuda")
    def forward(self, x):
        """
        :meta private:
        """
        x = torch.unsqueeze(x, 1)
        x = self.preblock(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pooling(x) 
        x = x.squeeze()
        x = self.head(x)
        return x


# =============================
#         EEGConformer
# =============================
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, Chan=32):
        # self.patch_size = patch_size
        super().__init__()

        # this is a surrogate of shallowNet, square and log are missing
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (Chan, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
    
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)    
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        #memorize as class attributes
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        #After combining the results of the different "heads" to reshape to original dimensions
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # ""residual connection"""
        res = x
        x = self.fn(x, **kwargs)
        x += res 
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
     
class GELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #erf is a function of gaussian error
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0))) 

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
                 
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p)
                )
            ),
            
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                )
            )
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        # Call TransformerEncoderBlock "#depth" times, depth = number of calls
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, nb_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 256),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256,32),   
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1 if nb_classes <= 2 else nb_classes)
        )

    def forward(self, x):
        x = self.clshead(x)  # Use clshead
        out = self.fc(x)
        return out   #Remove x,

class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=2, chan=32, seed=None, **kwargs):
        self._reset_seed(seed)
        super().__init__(
            PatchEmbedding(emb_size, chan),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

    def _reset_seed(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)



# ==========================
#       TransformEEG
# ==========================
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
        self.pool1 = nn.AvgPool1d(pool, stridePool)#, padding=1)
        #self.pool1 = nn.Conv1d(F1, F1, pool, stride=stridePool, padding = 'valid', groups=F1)
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
            #nn.Conv1d(F2, F2, 1, padding = 'same'),
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

        #F3 = F2
        #self.skipblck5 = nn.Sequential(
        #    nn.Conv1d(F2, F3, kernLength2, padding = 'same', groups = F2),
        #    nn.BatchNorm1d(F3, momentum = batchMomentum),
        #    nn.ELU(ELUAlpha),
        #)
        #
        #self.pool3 = nn.AvgPool1d(pool, stridePool)
        #self.drop3 = nn.Dropout1d(dropRate)
        #
        #self.blck6 = nn.Sequential(
        #    nn.Conv1d(F3, F3, kernLength2, padding = 'same', groups = F3),
        #    nn.BatchNorm1d(F3, momentum = batchMomentum),
        #    nn.ELU(ELUAlpha),
        #)

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

    def __init__(self, nb_classes, Chan, Features, seed=None):

        self._reset_seed(seed)
        super(TransformEEG, self).__init__()
        self.Chan = Chan
        self.Features = Features
        
        self.token_gen = Conv1DEncoder(
            Chan,
            D1            = 2,      # 2
            D2            = 2,      # 2
            kernLength1   = 5,
            kernLength2   = 5,
            pool          = 4,       # 4
            stridePool    = 2,       # 2
            dropRate      = 0.2,
            ELUAlpha      = 0.1,
            batchMomentum = 0.25,
        )
        #self.pos_enc = PositionalEncoding(Features, 0, 125)
        
        self._reset_seed(seed)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                Features,
                nhead                = 1,
                dim_feedforward      = Features,
                dropout              = 0.2,
                activation           = torch.nn.functional.hardswish, #torch.nn.ELU(0.1),#
                batch_first          = True
            ),
            num_layers           = 2,
            enable_nested_tensor = False
        )
        self.pool_lay = torch.nn.AdaptiveAvgPool1d((1))
        
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

    def forward(self, x):
        x = self.token_gen(x)
        #x = x.squeeze(-1)
        x = torch.permute(x, [0,2,1])
        #x = self.pos_enc(x)
        x = self.transformer(x)
        x = torch.permute(x, [0,2,1])
        x = self.pool_lay(x)
        x = x.squeeze(-1)
        x = self.linear_lay(x)
        return x



# =================================
#       TransformEEG with PSD
# =================================
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
        self.transformer_eeg = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                Features,
                nhead                = 1,
                dim_feedforward      = Features,
                dropout              = 0.2,
                activation           = torch.nn.functional.hardswish,
                batch_first          = True
            ),
            num_layers           = 2,
            enable_nested_tensor = False
        )
        self.pool_lay = torch.nn.AdaptiveAvgPool1d((1))

        self._reset_seed(seed)
        self.transformer_psd = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                Features,
                nhead                = 1,
                dim_feedforward      = Features,
                dropout              = 0.2,
                activation           = torch.nn.functional.hardswish,
                batch_first          = True
            ),
            num_layers           = 2,
            enable_nested_tensor = False
        )
        self.pool_lay_f = torch.nn.AdaptiveAvgPool1d((1))
        
        self._reset_seed(seed)
        self.linear_lay = nn.Sequential(
            nn.Linear(2*Features, Features//2 if Features//2>64 else 64),
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
        xf = self.transformer_psd(xf)
        xf = torch.permute(xf, [0,2,1])
        xf = self.pool_lay_f(xf)
        xf = xf.squeeze(-1)

        # temp branch
        xt = x[..., :self.temporal]
        xt = self.token_gen_eeg(xt)
        xt = torch.permute(xt, [0,2,1])
        xt = self.transformer_eeg(xt)
        xt = torch.permute(xt, [0,2,1])
        xt = self.pool_lay(xt)
        xt = xt.squeeze(-1)

        # final head
        x = torch.cat((xt, xf), -1)
        x = self.linear_lay(x)
        return x