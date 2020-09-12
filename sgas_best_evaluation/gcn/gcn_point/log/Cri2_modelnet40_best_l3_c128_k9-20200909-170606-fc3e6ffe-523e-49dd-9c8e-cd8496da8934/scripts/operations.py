import __init__
import torch.nn as nn
# from gcn.gcn_lib.dense import BasicConv, Conv1, Conv2, Conv3, Conv4, Conv5
from gcn.gcn_lib.dense import BasicConv, Conv1


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity(),
    'conv_1x1': lambda C, stride, affine: BasicConv([C, C], 'relu', 'batch', bias=False),
    # 原子操作�?1
    'Conv1': lambda C, stride, affine: Conv1(C, C, 'relu', 'batch', bias=False), 
    # # 原子操作�?2
    # 'Conv2': lambda C, stride, affine: Conv2(C, C, 'relu', 'batch', bias=False),
    # # 原子操作�?3
    # 'Conv3': lambda C, stride, affine: Conv3(C, C, 'relu', 'batch', bias=False),
    # # 原子操作�?4
    # 'Conv4': lambda C, stride, affine: Conv4(C, C, 'relu', 'batch', bias=False),
    # # 原子操作�?5
    # 'Conv5': lambda C, stride, affine: Conv5(C, C, 'relu', 'batch', bias=False),
}


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, edge_index=None, att=None):
        return x



class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x, edge_index=None, att=None):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


# OPS = {
#     'none': lambda C, stride, affine: Zero(stride),
#     'skip_connect': lambda C, stride, affine: Identity(),
#     'conv_1x1': lambda C, stride, affine: BasicConv([C, C], 'relu', 'batch', bias=False),
#     'edge_conv': lambda C, stride, affine: GraphConv2d(C, C, 'edge', 'relu', 'batch', bias=False),
#     'mr_conv': lambda C, stride, affine: GraphConv2d(C, C, 'mr', 'relu', 'batch', bias=False),
#     'gat': lambda C, stride, affine: GraphConv2d(C, C, 'gat', 'relu', 'batch', bias=False),
#     'semi_gcn': lambda C, stride, affine: GraphConv2d(C, C, 'gcn', 'relu', 'batch', bias=False),
#     'gin': lambda C, stride, affine: GraphConv2d(C, C, 'gin', 'relu', 'batch', bias=False),
#     'sage': lambda C, stride, affine: GraphConv2d(C, C, 'sage', 'relu', 'batch', bias=False),
#     'res_sage': lambda C, stride, affine: GraphConv2d(C, C, 'rsage', 'relu', 'batch', bias=False)
# }


# class Identity(nn.Module):

#     def __init__(self):
#         super(Identity, self).__init__()

#     def forward(self, x, edge_index=None):
#         return x


# class Zero(nn.Module):

#     def __init__(self, stride):
#         super(Zero, self).__init__()
#         self.stride = stride

#     def forward(self, x, edge_index=None):
#         if self.stride == 1:
#             return x.mul(0.)
#         return x[:, :, ::self.stride, ::self.stride].mul(0.)



