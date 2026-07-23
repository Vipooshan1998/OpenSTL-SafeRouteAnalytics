import torch
from torch import nn

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)


class SimVP_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        cords_depth = kwargs.pop('cords_depth', 3)
        cords_timesteps = kwargs.pop('cords_timesteps', 1)
        if model_type == 'cordsnet':
            self.hid = MidCORDSNet(
                T * hid_S, hid_T, N_T,
                depth=cords_depth,
                timesteps=cords_timesteps,
            )
        else:
            # self.hid = ContinuousDynamicsNet(hid_S, hid_T, N_T)
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        # Print the size (shape) of Y
        #print(f"Shape of Y: {Y.shape}")
        # Optionally, print the data type of Y
        #print(f"Data type of Y: {Y.dtype}")
        return Y


# class MidIncepNet(nn.Module):
#     """The hidden Translator of IncepNet for SimVPv1"""

#     def __init__(self, channel_in, channel_hid, N2, incep_ker=[3,5,7,11], groups=8, **kwargs):
#         super(MidIncepNet, self).__init__()
#         assert N2 >= 2 and len(incep_ker) > 1
#         self.N2 = N2
#         enc_layers = [gInception_ST(
#             channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
#         for i in range(1,N2-1):
#             enc_layers.append(
#                 gInception_ST(channel_hid, channel_hid//2, channel_hid,
#                               incep_ker=incep_ker, groups=groups))
#         enc_layers.append(
#                 gInception_ST(channel_hid, channel_hid//2, channel_hid,
#                               incep_ker=incep_ker, groups=groups))
#         dec_layers = [
#                 gInception_ST(channel_hid, channel_hid//2, channel_hid,
#                               incep_ker=incep_ker, groups=groups)]
#         for i in range(1,N2-1):
#             dec_layers.append(
#                 gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
#                               incep_ker=incep_ker, groups=groups))
#         dec_layers.append(
#                 gInception_ST(2*channel_hid, channel_hid//2, channel_in,
#                               incep_ker=incep_ker, groups=groups))

#         self.enc = nn.Sequential(*enc_layers)
#         self.dec = nn.Sequential(*dec_layers)

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         x = x.reshape(B, T*C, H, W)

#         # encoder
#         skips = []
#         z = x
#         for i in range(self.N2):
#             z = self.enc[i](z)
#             if i < self.N2-1:
#                 skips.append(z)
#         # decoder
#         z = self.dec[0](z)
#         for i in range(1,self.N2):
#             z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

#         y = z.reshape(B, T, C, H, W)
#         return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y


class CORDSBlock(nn.Module):
    """A lightweight residual block used by the compact CORDS-style translator."""

    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = max(channels * expansion, 16)
        self.norm = nn.GroupNorm(1, channels)
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.dw(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return residual + x


class MidCORDSNet(nn.Module):
    """A compact CORDS-inspired translator for SimVP.

    This version keeps the same interface as the original translator but
    replaces the expensive recurrent-style stack with a smaller residual
    refinement path that is faster and easier to optimize.
    """

    def __init__(self, channel_in, channel_hid, N2, depth=3, timesteps=1, expansion=2):
        super().__init__()
        assert depth >= 1

        self.channel_in = channel_in
        self.depth = depth
        self.timesteps = max(1, timesteps)

        self.proj_in = nn.Conv2d(channel_in, channel_hid, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList([
            CORDSBlock(channel_hid, expansion=expansion) for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(channel_hid, channel_in, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = self.proj_in(x)

        for _ in range(self.timesteps):
            residual = z
            for block in self.blocks:
                z = block(z)
            z = residual + torch.sigmoid(self.alpha) * (z - residual)

        z = self.proj_out(z)
        return z.reshape(B, T, C, H, W)


class ContinuousDynamicsBlock(nn.Module):
    """
    Continuous spatiotemporal latent dynamics block.

    h_{k+1} = (1-a)h_k + aF(h_k)

    where
        a = sigmoid(alpha)

    """

    def __init__(self,
                 channels,
                 expansion=4,
                 drop=0.0):
        super().__init__()

        hidden = channels * expansion

        self.norm = nn.BatchNorm3d(channels)

        # Temporal and spatial mixing
        self.temporal_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False
        )
        self.spatial_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 5, 5),
            padding=(0, 2, 2),
            groups=channels,
            bias=False
        )

        # Channel expansion
        self.pwconv1 = nn.Conv3d(
            channels,
            hidden,
            kernel_size=1,
            bias=False
        )

        self.act = nn.GELU()

        # Channel projection
        self.pwconv2 = nn.Conv3d(
            hidden,
            channels,
            kernel_size=1,
            bias=False
        )

        self.dropout = nn.Dropout3d(drop)

        # Learnable continuous-time coefficient
        self.alpha = nn.Parameter(torch.zeros(1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d)):
            nn.init.kaiming_normal_(
                m.weight,
                mode='fan_out',
                nonlinearity='relu'
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x = self.temporal_dw(x)
        x = self.spatial_dw(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pwconv2(x)

        alpha = torch.sigmoid(self.alpha)
        out = residual + alpha * (x - residual)

        return out


class ContinuousDynamicsNet(nn.Module):
    """
    Replacement for MidMetaNet.

    Input:
        (B,T,C,H,W)

    Output:
        (B,T,C,H,W)
    """

    def __init__(self,
                 channel_in,
                 channel_hid,
                 N2,
                 **kwargs):

        super().__init__()

        self.N2 = N2

        layers = []

        # First projection
        layers.append(
            nn.Conv3d(
                channel_in,
                channel_hid,
                kernel_size=1,
                bias=False
            )
        )

        # Continuous dynamics
        for _ in range(N2):
            layers.append(
                ContinuousDynamicsBlock(
                    channel_hid
                )
            )

        # Projection back
        layers.append(
            nn.Conv3d(
                channel_hid,
                channel_in,
                kernel_size=1,
                bias=False
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W)
        z = self.net(x)
        y = z.permute(0, 2, 1, 3, 4)
        return y
