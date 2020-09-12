import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select
from .torch_edge import DilatedKnn2d, add_self_loops, remove_self_loops
import torch.nn.functional as F
from torch_geometric.nn import inits


##############################
#    Atomic operators
##############################
class Conv1(nn.Module):
    """
    input:  x.shape=[B, C, N, 1]
            edge_index.shape=[B, C, N, k]
    output: 
    
    There are five definitions of atomic operators：
    1: x_i = batched_index_select(x, edge_index[1])
    2: x_j = batched_index_select(x, edge_index[0])
    3: xij_sub = (x_i - x_j)
    4: xce_sub = (x_i - m_i)
    5: xij_eud = ||x_i - x_j||
    
    batched_index_select: Copy "x_i" k times, or find the k neighbor points "x_j" of all the points, return shape=[B, C, N, k]
    op: The possible combinations of the above operations, the fixed combination operation is 3 times, so the shape is [B, C*3, N, k]. 
        Currently, 5 operations are used, so there are altogether 10 possible combinations. Turning list "op" into "op_tensor".
    alpha: Alpha first multiplies the Concat[x_i, x_j] and performs a series of dimensional transformations to ensure alignment.
           If the alpha is multiplied by OP_tensors, the score's shape[0] = 10. If the score's shape[0] is taken, the top1 operation 
           and dimensionality reduction operation are performed to obtain the final op's index. Finally, the op[index] is convolved with the 
           self.out to obtain out, out' shape is [B, C, N, 1], then return out_value. 
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=False, negative_slope=0.2):
        super(Conv1, self).__init__()
        self.nn = BasicConv([2 * in_channels, out_channels], act, norm, bias)
        self.out = BasicConv([in_channels, out_channels], act, norm, bias)
        self.att = nn.Parameter(torch.Tensor(10, 1, 1, 1, 1))
        self.negative_slope = negative_slope

        self.reset_parameters()

    def reset_parameters(self):
        inits.glorot(self.att)

    def forward(self, x, edge_index):
        edge_index = add_self_loops(edge_index)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        xij_sub = x_i - x_j
        m_i = torch.mean(x_i, dim=-1, keepdim=True)
        xce_sub = x_i - m_i 
        vij = x_i - x_j
        xij_eud = torch.norm(vij, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1, 1)

        ATOM_OPS = [x_i, x_j, xij_sub, xce_sub, xij_eud]
        op = []
        op_nums = len(ATOM_OPS)
        # concat operation: 
        for i in range(op_nums):
            for j in range(i, op_nums):
                if i == j:
                    pass
                else:
                    concat_fir = self.nn(torch.cat([ATOM_OPS[i], ATOM_OPS[j]], dim=1))
                    for k in range(j, op_nums):
                        if i == j or i == k or j == k:
                            pass
                        else:
                            concat_sec = self.nn(torch.cat([concat_fir, ATOM_OPS[k]], dim=1))
                            op.append(concat_sec)
        op_tensor= torch.mean(torch.tensor([item.cpu().detach().numpy() for item in op]).cuda(), dim=1).unsqueeze(1)

        alpha = (self.nn(torch.cat([x_i, x_j], dim=1)).unsqueeze(0) * self.att).sum(dim=1, keepdim=True)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=-1)

        score = alpha * op_tensor
        # score.shape=[10, 1, 3, 1024, 9], score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True).shape=[10, 1, 1, 1, 1]
        _, index = torch.topk(score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True), 1, dim=0, largest=True, sorted=True)
        index = index.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
        out = self.out(op[index])
        out_value, _ = torch.max(out, dim=-1, keepdim=True)
        
        return out_value

class Conv2(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=False, negative_slope=0.2):
        super(Conv2, self).__init__()
        self.nn = BasicConv([2 * in_channels, out_channels], act, norm, bias)
        self.out = BasicConv([in_channels, out_channels], act, norm, bias)
        self.att = nn.Parameter(torch.Tensor(10, 1, 1, 1, 1))
        self.negative_slope = negative_slope

        self.reset_parameters()

    def reset_parameters(self):
        inits.glorot(self.att)

    def forward(self, x, edge_index):
        edge_index = add_self_loops(edge_index)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        xij_sub = x_i - x_j
        m_i = torch.mean(x_i, dim=-1, keepdim=True)
        xce_sub = x_i - m_i 
        vij = x_i - x_j
        xij_eud = torch.norm(vij, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1, 1)

        ATOM_OPS = [x_i, x_j, xij_sub, xce_sub, xij_eud]
        op = []
        op_nums = len(ATOM_OPS)
        # concat operation: 
        for i in range(op_nums):
            for j in range(i, op_nums):
                if i == j:
                    pass
                else:
                    concat_fir = self.nn(torch.cat([ATOM_OPS[i], ATOM_OPS[j]], dim=1))
                    for k in range(j, op_nums):
                        if i == j or i == k or j == k:
                            pass
                        else:
                            concat_sec = self.nn(torch.cat([concat_fir, ATOM_OPS[k]], dim=1))
                            op.append(concat_sec)
        op_tensor= torch.mean(torch.tensor([item.cpu().detach().numpy() for item in op]).cuda(), dim=1).unsqueeze(1)

        alpha = (self.nn(torch.cat([x_i, x_j], dim=1)).unsqueeze(0) * self.att).sum(dim=1, keepdim=True)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=-1)
        score = alpha * op_tensor
        # score.shape=[10, 1, 3, 1024, 9], score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True).shape=[10, 1, 1, 1, 1]
        _, index = torch.topk(score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True), 1, dim=0, largest=True, sorted=True)
        index = index.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
        out = self.out(op[index])
        out_value, _ = torch.max(out, dim=-1, keepdim=True)
        # import pdb;pdb.set_trace()
        return out_value

class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=False, negative_slope=0.2):
        super(Conv3, self).__init__()
        self.nn = BasicConv([2 * in_channels, out_channels], act, norm, bias)
        self.out = BasicConv([in_channels, out_channels], act, norm, bias)
        self.att = nn.Parameter(torch.Tensor(10, 1, 1, 1, 1))
        self.negative_slope = negative_slope

        self.reset_parameters()

    def reset_parameters(self):
        inits.glorot(self.att)

    def forward(self, x, edge_index):
        edge_index = add_self_loops(edge_index)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        xij_sub = x_i - x_j
        m_i = torch.mean(x_i, dim=-1, keepdim=True)
        xce_sub = x_i - m_i 
        vij = x_i - x_j
        xij_eud = torch.norm(vij, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1, 1)

        ATOM_OPS = [x_i, x_j, xij_sub, xce_sub, xij_eud]
        op = []
        op_nums = len(ATOM_OPS)
        # concat operation: 
        for i in range(op_nums):
            for j in range(i, op_nums):
                if i == j:
                    pass
                else:
                    concat_fir = self.nn(torch.cat([ATOM_OPS[i], ATOM_OPS[j]], dim=1))
                    for k in range(j, op_nums):
                        if i == j or i == k or j == k:
                            pass
                        else:
                            concat_sec = self.nn(torch.cat([concat_fir, ATOM_OPS[k]], dim=1))
                            op.append(concat_sec)
        op_tensor= torch.mean(torch.tensor([item.cpu().detach().numpy() for item in op]).cuda(), dim=1).unsqueeze(1)

        alpha = (self.nn(torch.cat([x_i, x_j], dim=1)).unsqueeze(0) * self.att).sum(dim=1, keepdim=True)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=-1)
        score = alpha * op_tensor
        # score.shape=[10, 1, 3, 1024, 9], score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True).shape=[10, 1, 1, 1, 1]
        _, index = torch.topk(score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True), 1, dim=0, largest=True, sorted=True)
        index = index.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
        out = self.out(op[index])
        out_value, _ = torch.max(out, dim=-1, keepdim=True)
        # import pdb;pdb.set_trace()
        return out_value

class Conv4(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=False, negative_slope=0.2):
        super(Conv4, self).__init__()
        self.nn = BasicConv([2 * in_channels, out_channels], act, norm, bias)
        self.out = BasicConv([in_channels, out_channels], act, norm, bias)
        self.att = nn.Parameter(torch.Tensor(10, 1, 1, 1, 1))
        self.negative_slope = negative_slope

        self.reset_parameters()

    def reset_parameters(self):
        inits.glorot(self.att)

    def forward(self, x, edge_index):
        edge_index = add_self_loops(edge_index)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        xij_sub = x_i - x_j
        m_i = torch.mean(x_i, dim=-1, keepdim=True)
        xce_sub = x_i - m_i 
        vij = x_i - x_j
        xij_eud = torch.norm(vij, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1, 1)

        ATOM_OPS = [x_i, x_j, xij_sub, xce_sub, xij_eud]
        op = []
        op_nums = len(ATOM_OPS)
        # concat operation: 
        for i in range(op_nums):
            for j in range(i, op_nums):
                if i == j:
                    pass
                else:
                    concat_fir = self.nn(torch.cat([ATOM_OPS[i], ATOM_OPS[j]], dim=1))
                    for k in range(j, op_nums):
                        if i == j or i == k or j == k:
                            pass
                        else:
                            concat_sec = self.nn(torch.cat([concat_fir, ATOM_OPS[k]], dim=1))
                            op.append(concat_sec)
        op_tensor= torch.mean(torch.tensor([item.cpu().detach().numpy() for item in op]).cuda(), dim=1).unsqueeze(1)

        alpha = (self.nn(torch.cat([x_i, x_j], dim=1)).unsqueeze(0) * self.att).sum(dim=1, keepdim=True)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=-1)
        score = alpha * op_tensor
        # score.shape=[10, 1, 3, 1024, 9], score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True).shape=[10, 1, 1, 1, 1]
        _, index = torch.topk(score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True), 1, dim=0, largest=True, sorted=True)
        index = index.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
        out = self.out(op[index])
        out_value, _ = torch.max(out, dim=-1, keepdim=True)
        # import pdb;pdb.set_trace()
        return out_value

class Conv5(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=False, negative_slope=0.2):
        super(Conv5, self).__init__()
        self.nn = BasicConv([2 * in_channels, out_channels], act, norm, bias)
        self.out = BasicConv([in_channels, out_channels], act, norm, bias)
        self.att = nn.Parameter(torch.Tensor(10, 1, 1, 1, 1))
        self.negative_slope = negative_slope

        self.reset_parameters()

    def reset_parameters(self):
        inits.glorot(self.att)

    def forward(self, x, edge_index):
        edge_index = add_self_loops(edge_index)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        xij_sub = x_i - x_j
        m_i = torch.mean(x_i, dim=-1, keepdim=True)
        xce_sub = x_i - m_i 
        vij = x_i - x_j
        xij_eud = torch.norm(vij, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1, 1)

        ATOM_OPS = [x_i, x_j, xij_sub, xce_sub, xij_eud]
        op = []
        op_nums = len(ATOM_OPS)
        # concat operation: 
        for i in range(op_nums):
            for j in range(i, op_nums):
                if i == j:
                    pass
                else:
                    concat_fir = self.nn(torch.cat([ATOM_OPS[i], ATOM_OPS[j]], dim=1))
                    for k in range(j, op_nums):
                        if i == j or i == k or j == k:
                            pass
                        else:
                            concat_sec = self.nn(torch.cat([concat_fir, ATOM_OPS[k]], dim=1))
                            op.append(concat_sec)
        op_tensor= torch.mean(torch.tensor([item.cpu().detach().numpy() for item in op]).cuda(), dim=1).unsqueeze(1)

        alpha = (self.nn(torch.cat([x_i, x_j], dim=1)).unsqueeze(0) * self.att).sum(dim=1, keepdim=True)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=-1)
        score = alpha * op_tensor
        # score.shape=[10, 1, 3, 1024, 9], score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True).shape=[10, 1, 1, 1, 1]
        _, index = torch.topk(score.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True), 1, dim=0, largest=True, sorted=True)
        index = index.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
        out = self.out(op[index])
        out_value, _ = torch.max(out, dim=-1, keepdim=True)
        # import pdb;pdb.set_trace()
        return out_value

class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        edge_index = remove_self_loops(edge_index)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        # import pdb;pdb.set_trace()
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):

        edge_index = remove_self_loops(edge_index)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        # import pdb;pdb.set_trace()
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GATConv2d(nn.Module):
    r"""Revised one head graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True,
                 negative_slope=0.2, dropout=0):
        super(GATConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, 2 * out_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        inits.glorot(self.att)
        inits.zeros(self.bias)

    def forward(self, x, edge_index):
        x = self.nn(x)
        edge_index = add_self_loops(edge_index)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])

        # x_i BxCxNxk
        alpha = (torch.cat([x_i, x_j], dim=1) * self.att).sum(dim=1, keepdim=True)  # -1 xk
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=-1)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        x_j = x_j * alpha

        aggr_out = x_j.sum(dim=-1, keepdim=True)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class SemiGCNConv2d(nn.Module):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True):
        super(SemiGCNConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        inits.zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        # import pdb;pdb.set_trace()
        x = self.nn(x)
        edge_index = add_self_loops(edge_index)
        x_j = batched_index_select(x, edge_index[0])

        deg = edge_index.shape[-1]
        norm = 1 / deg
        x_j = x_j * norm
        aggr_out = x_j.sum(dim=-1, keepdim=True)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GINConv2d(nn.Module):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,
                 act='relu', norm=None, bias=True,
                 eps=0, train_eps=False):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        edge_index = remove_self_loops(edge_index)
        # import pdb;pdb.set_trace()
        x_j = batched_index_select(x, edge_index[0])
        out = self.nn((1 + self.eps) * x + torch.sum(x_j, dim=-1, keepdim=True))
        return out


class RSAGEConv2d(nn.Module):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be :math:`\ell_2`-normalized. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act='relu', norm=True, bias=True,
                 relative=False,
                 normlization=True):
        super(RSAGEConv2d, self).__init__()
        self.relative = relative
        self.nn = BasicConv([out_channels + in_channels, out_channels], act, norm=None, bias=False)
        self.pre_nn = BasicConv([in_channels, out_channels], act, norm=None, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.normlization = normlization

    def forward(self, x, edge_index):
        """"""
        x_j = batched_index_select(x, edge_index[0])
        if self.relative:
            x_i = batched_index_select(x, edge_index[1])
            x_j = self.pre_nn(x_j-x_i)
        else:
            x_j = self.pre_nn(x_j)
        aggr_out, _ = torch.max(x_j, -1, keepdim=True)

        out = self.nn(torch.cat((x, aggr_out), dim=1))
        if self.bias is not None:
            out = out + self.bias
        if self.normlization:
            out = F.normalize(out, dim=1)
        return out


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gat':
            self.gconv = GATConv2d(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gcn':
            self.gconv = SemiGCNConv2d(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'sage':
            self.gconv = RSAGEConv2d(in_channels, out_channels, act, norm, bias, relative=False)
        elif conv.lower() == 'rsage':
            self.gconv = RSAGEConv2d(in_channels, out_channels, act, norm, bias, relative=True)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1,
                 conv='edge', act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0):
        super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnn2d(kernel_size, dilation,
                                              self_loop=True, stochastic=stochastic, epsilon=epsilon)

    def forward(self, x):
        edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)


class ResDynBlock2d(nn.Module):
    """
    Residual Dynamic graph convolution block
        :input: (x0, x1, x2, ... , xi), batch
        :output:(x0, x1, x2, ... , xi ,xi+1) , batch
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, res_scale=1):
        super(ResDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon)
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x) + x * self.res_scale


class DenseDynBlock2d(nn.Module):
    """
    Dense Dynamic graph convolution block
    """

    def __init__(self, in_channels, out_channels=64, kernel_size=9, dilation=1, conv='edge',
                 act='relu', norm=None, bias=True, stochastic=False, epsilon=0.0):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon)

    def forward(self, x):
        dense = self.body(x)
        return torch.cat((x, dense), 1)

