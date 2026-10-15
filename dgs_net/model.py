import os
import sys
from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from timm.models.layers import DropPath
import torch_scatter
from collections import OrderedDict

try:
    import flash_attn
except ImportError:
    flash_attn = None


from .serialization import encode
from .loss import matrix_to_quaternion, compute_rotation_matrix_from_ortho6d


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


# class PDNorm(PointModule):
#     def __init__(
#         self,
#         num_features,
#         norm_layer,
#         context_channels=256,
#         conditions=("ScanNet", "S3DIS", "Structured3D"),
#         decouple=True,
#         adaptive=False,
#     ):
#         super().__init__()
#         self.conditions = conditions
#         self.decouple = decouple
#         self.adaptive = adaptive
#         if self.decouple:
#             self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
#         else:
#             self.norm = norm_layer
#         if self.adaptive:
#             self.modulation = nn.Sequential(
#                 nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
#             )

#     def forward(self, point):
#         assert {"feat", "condition"}.issubset(point.keys())
#         if isinstance(point.condition, str):
#             condition = point.condition
#         else:
#             condition = point.condition[0]
#         if self.decouple:
#             assert condition in self.conditions
#             norm = self.norm[self.conditions.index(condition)]
#         else:
#             norm = self.norm
#         point.feat = norm(point.feat)
#         if self.adaptive:
#             assert "context" in point.keys()
#             shift, scale = self.modulation(point.context).chunk(2, dim=1)
#             point.feat = point.feat * (1.0 + scale) + shift
#         return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            ))
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


class Grasp_Points_enc(PointModule):
    def __init__(
        self,

    ):
        super().__init__()
        act_layer = nn.GELU
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        # TODO: check remove spconv
        self.stem = PointSequential(
            conv1=spconv.SubMConv3d(
                64,
                128,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="gp_conv1",
            ),
            bn1 = bn_layer(128),
            act1 = act_layer(),
            conv2=spconv.SubMConv3d(
                128,
                64,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="gp_conv2",
            ),
            bn2 = bn_layer(64),
            act2 = act_layer(),
            conv3=spconv.SubMConv3d(
                64,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="gp_conv3",
            ),
            bn3 = bn_layer(32),
            act3 = act_layer(),
            conv4=spconv.SubMConv3d(
                32,
                8,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="gp_conv4",
            )
        )

    def forward(self, point: Point):
        point = self.stem(point)
        return point
    
class Grasp_Generator_dec(PointModule):
    def __init__(
        self,
        batch_size
    ):
        super().__init__()
        self.batch_size = batch_size
        act_layer = nn.GELU
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # TODO: check remove spconv
        
        self.stem = PointSequential(
            conv1=spconv.SubMConv3d(
                64,
                128,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="grasp_conv1",
            ),
            bn = bn_layer(128),
            act = act_layer(),
            conv2=spconv.SubMConv3d(
                128,
                256,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="grasp_conv2",
            ),
        )
        self.grasp_joint = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 40, kernel_size=1, bias=False)
        )
        self.grasp_offset = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, kernel_size=1, bias=False)
        )

    def forward(self, point: Point):
        point = self.stem(point)
        features=point.feat.view(self.batch_size, -1 , 256).permute(0,2,1)
        pre_joint = self.grasp_joint(features)
        pre_offset = self.grasp_offset(features)
        return pre_joint.permute(0,2,1), pre_offset.permute(0,2,1)

class PointTransformerV3(PointModule):
    def __init__(
        self,
        
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        batch_size=1,
        is_training=False,
        # pdnorm_bn=False,
        # pdnorm_ln=False,
        # pdnorm_decouple=True,
        # pdnorm_adaptive=False,
        # pdnorm_affine=True,
        # pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.batch_size = batch_size
        self.is_training = is_training

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # # norm layers
        # if pdnorm_bn:
        #     bn_layer = partial(
        #         PDNorm,
        #         norm_layer=partial(
        #             nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
        #         ),
        #         conditions=pdnorm_conditions,
        #         decouple=pdnorm_decouple,
        #         adaptive=pdnorm_adaptive,
        #     )
        # else:
        #     bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # if pdnorm_ln:
        #     ln_layer = partial(
        #         PDNorm,
        #         norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
        #         conditions=pdnorm_conditions,
        #         decouple=pdnorm_decouple,
        #         adaptive=pdnorm_adaptive,
        #     )
        # else:
        #     ln_layer = nn.LayerNorm
        
        
        act_layer = nn.GELU
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        
        self.embedding = Embedding(
            in_channels=in_channels,            #3
            embed_channels=enc_channels[0],     #32
            norm_layer=bn_layer,
            act_layer=act_layer,
        )
        
        self.embedding_stage2 = Embedding(
            in_channels = 10,        
            embed_channels = 64,     #32
            norm_layer = bn_layer,
            act_layer = act_layer,
        )

        # encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],      #32
                        num_heads=enc_num_head[s],     #2
                        patch_size=enc_patch_size[s],   #1024
                        mlp_ratio=mlp_ratio,        #4
                        qkv_bias=qkv_bias,        #True
                        qk_scale=qk_scale,        #None
                        attn_drop=attn_drop,        #0.0
                        proj_drop=proj_drop,        #0.0
                        drop_path=enc_drop_path_[i],        #0.0
                        norm_layer=ln_layer,    #nn.LayerNorm
                        act_layer=act_layer,    #nn.GELU
                        pre_norm=pre_norm,      #True
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,      #False
                        enable_flash=enable_flash,    #True
                        upcast_attention=upcast_attention,      #False
                        upcast_softmax=upcast_softmax,     #False
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
        self.dec = PointSequential()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])]
            dec_drop_path_.reverse()
            dec = PointSequential()
            dec.add(
                SerializedUnpooling(
                    in_channels=dec_channels[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),
                name="up",
            )
            for i in range(dec_depths[s]):
                dec.add(
                    Block(
                        channels=dec_channels[s],
                        num_heads=dec_num_head[s],
                        patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.dec.add(module=dec, name=f"dec{s}")

        self.grasp_points_enc = Grasp_Points_enc()
        
        # enc_stage2
        enc_stage2 = PointSequential()
        self.enc_stage2 = PointSequential()
        self.enc_stage2_depth = 2
        enc_stage2_drop_path = [x.item() for x in torch.linspace(0, 0.2, self.enc_stage2_depth)]
        for i in range(self.enc_stage2_depth):
            enc_stage2.add(
                Block(
                    channels=64,
                    num_heads=4,
                    patch_size=1024,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=enc_stage2_drop_path[i],
                    norm_layer=ln_layer,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    order_index=i % len(self.order),
                    cpe_indice_key=f"stage{s}",
                    enable_rpe=enable_rpe,
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                ),
                name=f"block{i}",
            )
        self.enc_stage2.add(module=enc_stage2, name=f"enc_stage2")
        
        self.grasp_generator = Grasp_Generator_dec(batch_size = self.batch_size)

    def forward(self, data_dict, gt=None):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        point = self.grasp_points_enc(point)
        
        x=point.feat.view(self.batch_size, -1 , 8)
        pred_graspable = x[:,:,:2]
        pred_pose = x[:,:,2:]
        
        # stage2
        if self.is_training:
            with torch.no_grad():
                point_ = is_training_prosess(data_dict ,gt ,self.batch_size)
            
        else:
            with torch.no_grad():
                feat1 = data_dict['feat'].view(self.batch_size, -1 , 3)
                grasp_points_label = torch.argmax(pred_graspable, dim=2)
                feat2 = grasp_points_label.unsqueeze(-1).repeat(1, 1, 3).float()
                feat3 = compute_rotation_matrix_from_ortho6d(pred_pose.view(-1 , 6))
                feat3 = matrix_to_quaternion(feat3).view(self.batch_size, -1 , 4)
                feature_ = torch.cat([feat1, feat2, feat3], dim=-1)
                new_data_dict = {
                    "coord": data_dict["coord"],
                    "grid_size": data_dict["grid_size"],
                    "batch": data_dict["batch"],
                    "feat": feature_.view(-1 , 10)
                }
                point_ = Point(new_data_dict)
        point_.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point_.sparsify()
        point_ = self.embedding_stage2(point_)
        point_ = self.enc_stage2(point_)
        
        pre_joint, pre_offset = self.grasp_generator(point_)
        pred_joint_init = pre_joint[:,:,:20]
        pred_joint_final = pre_joint[:,:,20:]
        pred_point_offset = pre_offset
        
        output = {
            'pred_graspable':       pred_graspable,
            'pred_pose':            pred_pose,
            'pred_point_offset':    pred_point_offset,
            'pred_joint_init':      pred_joint_init,
            'pred_joint_final':     pred_joint_final,
        }
        return output


def is_training_prosess(data_dict, gt, batch_size):
    feat1 = data_dict['feat']
    feat1 = feat1.view(batch_size, -1 , 3)
    device = feat1.device
    # Stage-II conditioning maps background (-1) to zero, but the loss still
    # needs the original -1/0/1 labels. Do not mutate the target tensor.
    grasp_points_label = gt['grasp_points_mask'].to(device).clone()
    grasp_points_label[grasp_points_label == -1] = 0
    feat2 = grasp_points_label.unsqueeze(-1).repeat(1, 1, 3).float()
    feat3 = gt['grasp_pose'].to(device).contiguous().float()
    feature = torch.cat([feat1, feat2, feat3], dim=-1) 
    new_data_dict = {
        "coord": data_dict["coord"],
        "grid_size": data_dict["grid_size"],
        "batch": data_dict["batch"],
        "feat": feature.view(-1 , 10)
    }
    return Point(new_data_dict)


def dataloader_preprosess(batch_points_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = torch.tensor(0.01, device=device)
    if isinstance(batch_points_data, dict):
        B, num_points, _ = batch_points_data['points'].shape
        points = batch_points_data['norm_points'].to(device=device, dtype=torch.float32)
        data_dict = {
            "coord": points.reshape(-1, 3),
            "grid_size": grid_size,
            "batch": torch.arange(B, device=device).repeat_interleave(num_points),
            "feat": points.reshape(-1, 3),
        }
    elif isinstance(batch_points_data, torch.Tensor):
        B, num_points, _ = batch_points_data.shape
        points = batch_points_data.to(device=device, dtype=torch.float32).reshape(-1, 3)
        data_dict = {
            "coord": points,
            "grid_size": grid_size,
            "batch": torch.arange(B, device=device).repeat_interleave(num_points),
            "feat": points,
        }
    else:
        raise TypeError("Unsupported type for batch_points_data. It must be either a dictionary or a tensor.")
    return data_dict


def dataloader_preprosess_half(batch_points_data, use_fp16: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = torch.tensor(0.01, device=device)
    if isinstance(batch_points_data, dict):
        for k in batch_points_data.keys():
            batch_points_data[k] = batch_points_data[k].to(device).float()
        B, num_points, _ = batch_points_data['points'].shape
        coord = batch_points_data['norm_points'].view(-1, 3)
        feat = batch_points_data['norm_points'].view(-1, 3)
    elif isinstance(batch_points_data, torch.Tensor):
        B, num_points, _ = batch_points_data.shape
        tensor = batch_points_data.to(device).float()
        coord = tensor.view(-1, 3)
        feat = tensor.view(-1, 3)
    else:
        raise TypeError("Unsupported type for batch_points_data. It must be either a dictionary or a tensor.")
    
    if use_fp16:
        coord = coord.half()
        feat = feat.half()
        grid_size_ = grid_size.half()
    else:
        grid_size_ = grid_size

    data_dict = {
        "coord": coord,
        "grid_size": grid_size_,
        "batch": torch.arange(B, device=device).repeat_interleave(num_points),
        "feat": feat,
    }
    return data_dict



