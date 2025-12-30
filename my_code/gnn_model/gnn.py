import torch
from torch import Tensor
from torch.nn import GRUCell,  Parameter,Linear, ModuleList, ReLU, Sequential,GRU
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool,GINEConv,BatchNorm, PNAConv,Set2Set,NNConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
from typing import Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GAT(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels1,hidden_channels2, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels1, heads=4)
        self.lin1 = torch.nn.Linear(in_channels, 4 * hidden_channels1)
        self.conv2 = GATConv(4 * hidden_channels1, hidden_channels1, heads=4)
        self.lin2 = torch.nn.Linear(4 * hidden_channels1, 4 * hidden_channels1)
        self.conv3 = GATConv(4 * hidden_channels1, 64, heads=6,
                             concat=False)
        self.lin3 = torch.nn.Linear(4 * hidden_channels1, 64)
        self.set2set = Set2Set(64, processing_steps=3)

        self.lin4 = torch.nn.Linear(2*64+2, hidden_channels2)
        self.lin5 = torch.nn.Linear(hidden_channels2, out_channels)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index,data.edge_attr) + self.lin1(data.x))
        x = F.elu(self.conv2(x, data.edge_index,data.edge_attr) + self.lin2(x))
        x = self.conv3(x, data.edge_index,data.edge_attr) + self.lin3(x)

        out = self.set2set(x, data.batch)
        tem = torch.tensor(data.t).to(device)
        pre = torch.tensor(data.p).to(device)
        out = torch.cat([out, tem], 1).to(torch.float32)
        out = torch.cat([out, pre], 1).to(torch.float32)
        out = F.relu(self.lin4(out))
        out = self.lin5(out)
        return out.view(-1)

# #CO2 capacity
# class GIN(torch.nn.Module):
#     def __init__(self,in_channels, hidden_channels1,hidden_channels2, out_channels):
#         super().__init__()
#         self.conv1 = GINEConv(torch.nn.Linear(in_channels, hidden_channels1),edge_dim = 6)
#         self.lin1 = torch.nn.Linear(in_channels,  hidden_channels1)
#         self.conv2 = GINEConv(torch.nn.Linear(hidden_channels1, hidden_channels1),edge_dim = 6)
#         self.lin2 = torch.nn.Linear( hidden_channels1,  hidden_channels1)
#         self.conv3 = GINEConv(torch.nn.Linear(hidden_channels1, 64),edge_dim = 6)
#         self.lin3 = torch.nn.Linear(hidden_channels1, 64)
#         self.set2set = Set2Set(64, processing_steps=3)
#
#         self.lin4 = torch.nn.Linear(2 * 64 + 2, hidden_channels2)
#         self.lin5 = torch.nn.Linear(hidden_channels2, 32)
#         self.lin6 = torch.nn.Linear(32, 1)
#
#     def forward(self, data):
#         x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr) + self.lin1(data.x))
#         x = F.elu(self.conv2(x, data.edge_index, data.edge_attr) + self.lin2(x))
#         x = self.conv3(x, data.edge_index, data.edge_attr) + self.lin3(x)
#
#         out = self.set2set(x, data.batch)
#         tem = torch.tensor(data.t).to(device)
#         pre = torch.tensor(data.p).to(device)
#         out = torch.cat([out, tem], 1).to(torch.float32)
#         out = torch.cat([out, pre], 1).to(torch.float32)
#         out = F.relu(self.lin4(out))
#         out = F.relu(self.lin5(out))
#         out = self.lin6(out)
#         return out.view(-1)

class GIN(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels1,hidden_channels2, out_channels):
        super().__init__()
        self.conv1 = GINEConv(torch.nn.Linear(in_channels, hidden_channels1),edge_dim = 6)
        self.lin1 = torch.nn.Linear(in_channels,  hidden_channels1)
        self.conv2 = GINEConv(torch.nn.Linear(hidden_channels1, hidden_channels1),edge_dim = 6)
        self.lin2 = torch.nn.Linear( hidden_channels1,  hidden_channels1)
        self.conv3 = GINEConv(torch.nn.Linear(hidden_channels1, 64),edge_dim = 6)
        self.lin3 = torch.nn.Linear(hidden_channels1, 64)
        self.set2set = Set2Set(64, processing_steps=3)

        self.lin4 = torch.nn.Linear(2 * 64 + 2, hidden_channels2)
        self.lin5 = torch.nn.Linear(hidden_channels2, 1)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr) + self.lin1(data.x))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr) + self.lin2(x))
        x = self.conv3(x, data.edge_index, data.edge_attr) + self.lin3(x)

        out = self.set2set(x, data.batch)
        tem = torch.tensor(data.t).to(device)
        pre = torch.tensor(data.p).to(device)
        out = torch.cat([out, tem], 1).to(torch.float32)
        out = torch.cat([out, pre], 1).to(torch.float32)
        out = F.relu(self.lin4(out))
        out = self.lin5(out)

        return out.view(-1)

class PNA(torch.nn.Module):
    def __init__(self,in_channels: int, hidden_channels: int,
            out_channels: int, deg):
        super().__init__()
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.lin1 = Linear(in_channels, hidden_channels)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.set2set = Set2Set(hidden_channels, processing_steps=3)
        for _ in range(3):
            conv = PNAConv(in_channels=hidden_channels, out_channels=hidden_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=6, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
        # self.mlp = Sequential(Linear(hidden_channels * 2 + 2, 64), ReLU(), Linear(64, out_channels))
        self.mlp = Sequential(Linear(hidden_channels*2+2, 64), ReLU(), Linear(64, 32), ReLU(),
                              Linear(32, out_channels))

    def forward(self,data):
        x = F.relu((self.lin1(data.x)))
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, data.edge_index, data.edge_attr)))

        out = self.set2set(x, data.batch)
        tem = torch.tensor(data.t).to(device)
        pre = torch.tensor(data.p).to(device)
        out = torch.cat([out, tem], 1).to(torch.float32)
        out = torch.cat([out, pre], 1).to(torch.float32)

        out = self.mlp(out)
        return out.view(-1)


class MPNN(torch.nn.Module):
    def __init__(self,in_channels: int, hidden_channels: int,
                 out_channels: int):
        super().__init__()
        self.lin0 = torch.nn.Linear(in_channels, hidden_channels)

        nn = Sequential(Linear(6, 64), ReLU(), Linear(64, hidden_channels * hidden_channels))
        self.conv = NNConv(hidden_channels, hidden_channels, nn, aggr='add')
        self.gru = GRU(hidden_channels, hidden_channels)

        self.set2set = Set2Set(hidden_channels, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * hidden_channels +2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 64)
        self.lin3 = torch.nn.Linear(64, 1)


    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(4):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        tem = torch.tensor(data.t).to(device)
        pre = torch.tensor(data.p).to(device)
        out = torch.cat([out, tem], 1).to(torch.float32)
        out = torch.cat([out, pre], 1).to(torch.float32)
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out))
        out = self.lin3(out)
        return out.view(-1)

class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)

class AttentiveFP(torch.nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, edge_dim: int, num_layers: int,
                 num_timesteps: int, dropout: float = 0.0):
        super().__init__()

        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)
        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels+2, 64)
        self.lin3 = Linear(64,32)
        self.lin4 = Linear(32, out_channels)


        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
    def forward(self, data):
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(data.x))

        h = F.elu_(self.gate_conv(x, data.edge_index, data.edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, data.edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(data.batch.size(0), device=data.batch.device)
        edge_index = torch.stack([row, data.batch], dim=0)

        out = global_add_pool(x, data.batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        tem = torch.tensor(data.t).to(device)
        pre = torch.tensor(data.p).to(device)
        out = torch.cat([out, tem], 1).to(torch.float32)
        out = torch.cat([out, pre], 1).to(torch.float32)
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = self.lin4(out)
        return out.view(-1)