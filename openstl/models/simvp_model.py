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
        # defer Encoder/Decoder creation to avoid NameError during import-time instantiation
        self.enc = None
        self.dec = None
        self._enc_cfg = dict(C_in=C, C_hid=hid_S, N_S=N_S, spatio_kernel=spatio_kernel_enc, act_inplace=act_inplace)
        self._dec_cfg = dict(C_hid=hid_S, C_out=C, N_S=N_S, spatio_kernel=spatio_kernel_dec, act_inplace=act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
        # model_type = 'cordsnet'
        if model_type == 'cordsnet':
            self.hid = MidCORDSNet(T*hid_S, hid_T, N_T)
        else:
            # self.hid = ContinuousDynamicsNet(hid_S, hid_T, N_T)
            self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                input_resolution=(H, W), model_type=model_type,
                mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape

        # lazy-init encoder/decoder if classes were not available at module import
        # if self.enc is None or self.dec is None:
            # cfg = self._enc_cfg
            # self.enc = Encoder(cfg['C_in'], cfg['C_hid'], cfg['N_S'], cfg['spatio_kernel'], act_inplace=cfg['act_inplace'])
            # dcfg = self._dec_cfg
            # self.dec = Decoder(dcfg['C_hid'], dcfg['C_out'], dcfg['N_S'], dcfg['spatio_kernel'], act_inplace=dcfg['act_inplace'])
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


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


class ChannelSE(nn.Module):
    """Squeeze-and-Excitation for flattened (T*C) channel maps."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class TemporalMixer(nn.Module):
    """Lightweight temporal mixer using depthwise 3D convs to improve time/channel coupling.

    Operates on (B, T, C, H, W) and returns same shape.
    """

    def __init__(self, channels, kernel_size=(3,3,3), expansion=2, drop=0.0):
        super().__init__()
        hidden = channels * expansion
        self.dw = nn.Conv3d(channels, channels, kernel_size=kernel_size,
                            padding=(kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2),
                            groups=channels, bias=False)
        self.pw1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout3d(drop)

    def forward(self, x):
        # x: (B, T, C, H, W) -> permute to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        residual = x
        x = self.dw(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw2(x)
        out = residual + x
        out = out.permute(0, 2, 1, 3, 4)
        return out



class MidCORDSNet(nn.Module):
    """A CORDSNet-inspired translator for SimVP.

    This translator operates on flattened spatiotemporal latent features
    in the same style as other SimVP hidden translators.
    """

    def __init__(self, channel_in, channel_hid, N2, depth=8, timesteps=2):
        super().__init__()
        assert depth >= 4 and depth % 2 == 0

        self.channel_in = channel_in
        self.depth = depth
        self.blockdepth = int(depth / 2 - 1)
        self.timesteps = timesteps

        self.relu = nn.ReLU(inplace=False)
        self.inp_conv = nn.Conv2d(channel_in, channel_in, kernel_size=7, stride=1, padding=3, bias=False)
        self.inp_avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False)
        self.inp_skip = nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=1, padding=1, bias=False)

        self.area_conv = nn.ModuleList([
            nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=1, padding=1, bias=True)
            for _ in range(depth)
        ])
        self.area_area = nn.ModuleList([
            nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(depth)
        ])
        self.skip_area = nn.ModuleList([
            nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0, bias=False)
            for _ in range(self.blockdepth)
        ])

        self.out_conv = nn.Conv2d(channel_in, channel_in, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

        # Optional helpers to improve latent representation
        # created lazily in forward when T and C are known
        self.temporal_mixer = None
        self.time_embedding = None
        self.channel_se = ChannelSE(channel_in)
        # residual projection and blending (keeps original _rnn intact)
        self.res_conv = nn.Conv2d(channel_in, channel_in, kernel_size=1, bias=False)
        self.res_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, T, C, H, W = x.shape

        # lazy create temporal_mixer and time embeddings when T and C are known
        if self.temporal_mixer is None:
            tm = TemporalMixer(C)
            # register module so it's moved with .to(device)
            self.add_module('temporal_mixer', tm)
            self.temporal_mixer = tm

        if self.time_embedding is None:
            te = nn.Parameter(torch.zeros(T, C, 1, 1))
            self.register_parameter('time_embedding', te)
            self.time_embedding = te

        # apply lightweight temporal mixing to improve time-channel coupling
        x_mixed = self.temporal_mixer(x)

        # run CORDSNet on flattened channels
        flat = x_mixed.reshape(B, T*C, H, W)
        out = self._run_cordsnet(flat)
        # blend with a lightweight residual projection from input
        out = out + torch.sigmoid(self.res_alpha) * self.res_conv(flat)
        y = out.reshape(B, T, C, H, W)

        # add learnable time embeddings (broadcasted)
        y = y + self.time_embedding.unsqueeze(0)

        # apply channel squeeze-excite on flattened channels for better channel weighting
        y_flat = y.reshape(B, T*C, H, W)
        y_se = self.channel_se(y_flat)

        return y_se.reshape(B, T, C, H, W)

    def _run_cordsnet(self, img):
        batch_size = img.size(0)
        device = img.device
        dtype = img.dtype
        rs = [torch.zeros(batch_size, self.channel_in, img.size(2), img.size(3), device=device, dtype=dtype)
              for _ in range(self.depth)]

        with torch.no_grad():
            for _ in range(self.timesteps):
                for j in range(self.depth - 1, -1, -1):
                    rs[j] = self._rnn(j, rs, img * 0)

        for _ in range(self.timesteps):
            for j in range(self.depth - 1, -1, -1):
                rs[j] = self._rnn(j, rs, img)

        out = self.relu(rs[-1] + rs[-2])
        out = self.out_conv(out)
        return out

    def _rnn(self, area, r, img):
        inp = self.inp_avgpool(self.inp_conv(img))
        if area == 0:
            areainput = inp
        elif area == 1:
            areainput = self.relu(r[0]) + self.inp_skip(inp)
        elif area == 2:
            areainput = self.relu(r[1]) + self.relu(r[0])
        elif area == 3:
            areainput = self.relu(r[2]) + self.skip_area[0](self.relu(r[1]))
        elif area == 4:
            areainput = self.relu(r[3]) + self.relu(r[2])
        elif area == 5:
            areainput = self.relu(r[4]) + self.skip_area[1](self.relu(r[3]))
        elif area == 6:
            areainput = self.relu(r[5]) + self.relu(r[4])
        elif area == 7:
            areainput = self.relu(r[6]) + self.skip_area[2](self.relu(r[5]))
        else:
            raise ValueError(f"Unsupported CORDSNet area: {area}")

        alpha = torch.sigmoid(self.alpha)
        r[area] = (1 - alpha) * r[area] + alpha * self.relu(
            self.area_conv[area](r[area]) + self.area_area[area](areainput)
        )
        return r[area]


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