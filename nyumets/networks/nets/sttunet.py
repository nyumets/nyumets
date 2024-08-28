from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

from monai.networks.nets.basic_unet import TwoConv, Down, UpCat

from nyumets.networks.convlstm2d import ConvLSTM2DCell
from nyumets.networks.convlstm3d import ConvLSTM3DCell


class STTUNet(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        num_layers: int = 1,  # for rnn
        kernel_size: tuple = (3, 3, 3),  # for convlstm cell
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        """
        TODO: update documentation
        
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)

        self._check_kernel_size_consistency(kernel_size)

        if num_layers != 1:
            raise NotImplementedError

        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.spatial_dims = spatial_dims

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        if spatial_dims == 3:
            self.convlstm = ConvLSTM3DCell(input_dim=fea[4],
                                           hidden_dim=fea[4],
                                           kernel_size=self.kernel_size,
                                           bias=bias)
        elif spatial_dims == 2:
            self.convlstm = ConvLSTM2DCell(input_dim=fea[4],
                                           hidden_dim=fea[4],
                                           kernel_size=self.kernel_size,
                                           bias=bias)

        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        if self.spatial_dims < 2 and self.spatial_dims > 3:
            raise NotImplementedError("STTUnet only accepts spatial dims of 2 or 3.")

        x = x.unsqueeze(0)  # add batch dimension (previous batch dimension is num_timepoints)

        if self.spatial_dims == 3:
            b, _, _, h, w, d = x.size()
            hidden_state = self._init_hidden(batch_size=b, image_size=(h//16,w//16,d//16))
        elif self.spatial_dims == 2:
            b, _, _, h, w = x.size()
            hidden_state = self._init_hidden(batch_size=b, image_size=(h//16,w//16))   

        seq_len = x.size(1)

        outputs = []

        # initialize hidden and cell states
        h, c = hidden_state[0]

        for t in range(seq_len):

            if self.spatial_dims == 3:
                x_t = x[:,t,:,:,:,:]
            if self.spatial_dims == 2:
                x_t = x[:,t,:,:,:]

            x0 = self.conv_0(x_t)
            x1 = self.down_1(x0)
            x2 = self.down_2(x1)
            x3 = self.down_3(x2)
            x4 = self.down_4(x3)

            h, c = self.convlstm(input_tensor=x4, cur_state=[h, c])

            u4 = self.upcat_4(h, x3)
            u3 = self.upcat_3(u4, x2)
            u2 = self.upcat_2(u3, x1)
            u1 = self.upcat_1(u2, x0)

            logits = self.final_conv(u1)
            outputs.append(logits)

        stacked_outputs = torch.stack(outputs, dim=0)
        stacked_outputs = stacked_outputs.squeeze(1)  # remove added batch dim
        return stacked_outputs
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.convlstm.init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')