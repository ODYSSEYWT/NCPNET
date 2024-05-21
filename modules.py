import torch
import dgl
from memorys import *
from layers import *


class GeneralModel(torch.nn.Module):

    def __init__(self,
                 dim_node,
                 dim_edge,
                 sample_param,
                 memory_param,
                 gnn_param,
                 train_param,
                 combined=False,
                 device=None):
        super(GeneralModel, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param

        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(
                    memory_param, 2 * memory_param['dim_out'] + dim_edge,
                    memory_param['dim_out'], memory_param['dim_time'],
                    dim_node, device).to(device)
            elif memory_param['memory_update'] == 'rnn':
                self.memory_updater = RNNMemeoryUpdater(
                    memory_param, 2 * memory_param['dim_out'] + dim_edge,
                    memory_param['dim_out'], memory_param['dim_time'],
                    dim_node, device).to(device)
            elif memory_param['memory_update'] == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(
                    memory_param, 2 * memory_param['dim_out'] + dim_edge,
                    memory_param['dim_out'], memory_param['dim_time'],
                    train_param, device).to(device)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']

        self.layers = torch.nn.ModuleDict()

        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(
                    self.dim_node_input,
                    dim_edge,
                    gnn_param['dim_time'],
                    gnn_param['att_head'],
                    train_param['dropout'],
                    train_param['att_dropout'],
                    gnn_param['dim_out'],
                    combined=combined,
                    device=device).to(device)

            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' +
                                str(h)] = TransfomerAttentionLayer(
                                    gnn_param['dim_out'],
                                    dim_edge,
                                    gnn_param['dim_time'],
                                    gnn_param['att_head'],
                                    train_param['dropout'],
                                    train_param['att_dropout'],
                                    gnn_param['dim_out'],
                                    combined=False,
                                    device=device).to(device)
        elif gnn_param['arch'] == 'identity':
            self.gnn_param['layer'] = 1
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = IdentityNormLayer(
                    self.dim_node_input).to(device)
                if 'time_transform' in gnn_param and gnn_param[
                        'time_transform'] == 'JODIE':
                    self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(
                        gnn_param['dim_out']).to(device)
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'],
                                         gnn_param['dim_out'])
            self.combiner = self.combiner.to(device)

        self.device = device
        self.layers = self.layers.to(device)
        self.edge_predictor = self.edge_predictor.to(device)

    def forward(self, mfgs, neg_samples=1):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param[
                        'time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) +
                                      't'](rst, mfgs[l][h].srcdata['mem_ts'],
                                           mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return self.edge_predictor(out.to(self.device),
                                   neg_samples=neg_samples)

    def get_emb(self, mfgs):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param[
                        'time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) +
                                      't'](rst, mfgs[l][h].srcdata['mem_ts'],
                                           mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out


class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


class NexModel(torch.nn.Module):

    def __init__(self, k, alpha, sigma=0.01):
        super(NexModel, self).__init__()
        self.k = k
        self.alpha = alpha
        self.sigma = sigma
        rho = 0.99
        initals = rho**(np.arange(k, 0, -1))
        self.weights = torch.nn.Parameter(
            torch.tensor(initals, requires_grad=True, dtype=torch.float))
        self.weights_normalized = torch.zeros_like(self.weights)

    def forward(self, cal_smx, cal_labels):
        cal_smx_sorted = cal_smx
        if len(cal_labels.shape) == 1:
            cal_labels_sorted = cal_labels
        else:
            if cal_labels.shape[1] >= 2:
                cal_labels_sorted = torch.argmax(cal_labels, dim=1)
            else:
                cal_labels_sorted = cal_labels

        R = cal_smx[torch.arange(cal_smx_sorted.shape[0]), cal_labels_sorted]
        R_sort, ord_R = torch.sort(R)

        weights_sig = torch.nn.Sigmoid()(self.weights)
        self.weights_normalized = weights_sig / (torch.sum(weights_sig) + 1)

        # differentiable
        weights_cumsum = torch.cumsum(self.weights_normalized[ord_R], dim=0)
        weigths_cumsum_resi = weights_cumsum - (1 - self.alpha)
        soft_weight = torch.nn.Softmax(dim=0)(-weigths_cumsum_resi**2 /
                                              self.sigma)
        qhat = torch.sum(R_sort * soft_weight)

        return qhat, qhat


class SimpleMLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.FC_hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.FC_hidden2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = torch.nn.Linear(hidden_dim, output_dim)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        h = self.ReLU(self.FC_hidden(x))
        h = self.ReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat
