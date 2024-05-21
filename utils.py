from sklearn.metrics import roc_auc_score
import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
from modules import *

def load_feat(d, rand_de=0, rand_dn=0):
    node_feats = None
    if os.path.exists('DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('DATA/{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    if rand_de > 0:
        if d == 'LASTFM':
            edge_feats = torch.randn(1293103, rand_de)
        elif d == 'MOOC':
            edge_feats = torch.randn(411749, rand_de)
        elif d == 'DBLP':
            edge_feats = torch.randn(146739, rand_de)
        elif d == 'taobao':
            edge_feats = torch.randn(77436, rand_de)
        elif d == 'money':
            edge_feats = torch.randn(5078345, rand_de)
    if rand_dn > 0:
        if d == 'LASTFM':
            node_feats = torch.randn(1980, rand_dn)
        elif d == 'MOOC':
            node_feats = torch.randn(7144, rand_dn)
        elif d == 'DBLP':
            node_feats = torch.randn(2390, rand_dn)
        elif d == 'taobao':
            node_feats = torch.randn(82567, rand_dn)
        elif d == 'money':
            node_feats = torch.randn(515080, rand_dn)
    return node_feats, edge_feats

def load_graph(d):
    df = pd.read_csv('DATA/{}/edges.csv'.format(d))
    g = np.load('DATA/{}/ext_full.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def to_dgl_blocks(ret, hist, reverse=False, cuda=True, device=None):
    mfgs = list()
    for r in ret:
        if not reverse:
            b = dgl.create_block((r.col(), r.row()), num_src_nodes=r.dim_in(), num_dst_nodes=r.dim_out())
            b.srcdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
            b.srcdata['ts'] = torch.from_numpy(r.ts())
        else:
            b = dgl.create_block((r.row(), r.col()), num_src_nodes=r.dim_out(), num_dst_nodes=r.dim_in())
            b.dstdata['ID'] = torch.from_numpy(r.nodes())
            b.edata['dt'] = torch.from_numpy(r.dts())[b.num_src_nodes():]
            b.dstdata['ts'] = torch.from_numpy(r.ts())
        b.edata['ID'] = torch.from_numpy(r.eid())
        if device is not None:
            mfgs.append(b.to(device))
        else:
            mfgs.append(b)
        # if cuda:
        #     mfgs.append(b.to('cuda:0'))
        # else:
        #     mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def node_to_dgl_blocks(root_nodes, ts, cuda=True, device=None):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if device is not None:
        mfgs.insert(0, [b.to(device)])
    else:
        mfgs.insert(0, [b])
    # if cuda:
    #     mfgs.insert(0, [b.to('cuda:0')])
    # else:
    #     mfgs.insert(0, [b])
    return mfgs

def mfgs_to_cuda(mfgs):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to('cuda:0')
    return mfgs


def prepare_input(mfgs,
                  node_feats,
                  edge_feats,
                  combine_first=False,
                  pinned=False,
                  nfeat_buffs=None,
                  efeat_buffs=None,
                  nids=None,
                  eids=None,
                  device=None):
    if combine_first:
        for i in range(len(mfgs[0])):
            if mfgs[0][i].num_src_nodes() > mfgs[0][i].num_dst_nodes():
                num_dst = mfgs[0][i].num_dst_nodes()
                ts = mfgs[0][i].srcdata['ts'][num_dst:]
                nid = mfgs[0][i].srcdata['ID'][num_dst:].float()
                nts = torch.stack([ts, nid], dim=1)
                unts, idx = torch.unique(nts, dim=0, return_inverse=True)
                uts = unts[:, 0]
                unid = unts[:, 1]
                # import pdb; pdb.set_trace()
                b = dgl.create_block((idx + num_dst, mfgs[0][i].edges()[1]),
                                     num_src_nodes=unts.shape[0] + num_dst,
                                     num_dst_nodes=num_dst,
                                     device=device)
                b.srcdata['ts'] = torch.cat(
                    [mfgs[0][i].srcdata['ts'][:num_dst], uts], dim=0)
                b.srcdata['ID'] = torch.cat(
                    [mfgs[0][i].srcdata['ID'][:num_dst], unid], dim=0)
                b.edata['dt'] = mfgs[0][i].edata['dt']
                b.edata['ID'] = mfgs[0][i].edata['ID']
                mfgs[0][i] = b
    t_idx = 0
    t_cuda = 0
    i = 0
    if node_feats is not None:
        for b in mfgs[0]:
            b = b.to(device)
            if pinned:
                if nids is not None:
                    idx = nids[i]
                else:
                    idx = b.srcdata['ID'].cpu().long()
                torch.index_select(node_feats,
                                   0,
                                   idx,
                                   out=nfeat_buffs[i][:idx.shape[0]])
                b.srcdata['h'] = nfeat_buffs[i][:idx.shape[0]].to(device)
                i += 1
            else:
                srch = node_feats[b.srcdata['ID'].cpu().long()].float()
                b.srcdata['h'] = srch.to(device)
    i = 0
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                b = b.to(device)
                if b.num_src_nodes() > b.num_dst_nodes():
                    if pinned:
                        if eids is not None:
                            idx = eids[i]
                        else:
                            idx = b.edata['ID'].cpu().long()
                        torch.index_select(edge_feats,
                                           0,
                                           idx,
                                           out=efeat_buffs[i][:idx.shape[0]])
                        b.edata['f'] = efeat_buffs[i][:idx.shape[0]].to(device)
                        i += 1
                    else:
                        srch = edge_feats[b.edata['ID'].cpu().long()].float()
                        b.edata['f'] = srch.to(device)
    return mfgs

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if 'ID' in mfgs[0][0].edata:
        if edge_feats is not None:
            for mfg in mfgs:
                for b in mfg:
                    eids.append(b.edata['ID'].long())
    else:
        eids = None
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs


class NodeEmbMinibatch():

    def __init__(self, emb, role, label, batch_size, ts=None, device=None):
        self.role = role
        self.label = label
        self.batch_size = batch_size
        self.train_emb = emb[role == 0]
        self.val_emb = emb[role == 1]
        self.calib_emb = emb[role == 2]
        self.test_emb = emb[role == 3]

        self.train_label = label[role == 0]
        self.val_label = label[role == 1]
        self.calib_label = label[role == 2]
        self.test_label = label[role == 3]

        if ts is not None:
            self.train_ts = ts[role == 0]
            self.val_ts = ts[role == 1]
            self.calib_ts = ts[role == 2]
            self.test_ts = ts[role == 3]

        self.mode = 0
        self.s_idx = 0

    def shuffle(self):
        perm = torch.randperm(self.train_emb.shape[0])
        self.train_emb = self.train_emb[perm]
        self.train_label = self.train_label[perm]
        self.train_ts = self.train_ts[perm]

        perm = torch.randperm(self.calib_emb.shape[0])
        self.calib_emb = self.calib_emb[perm]
        self.calib_label = self.calib_label[perm]
        self.calib_ts = self.calib_ts[perm]

    def set_mode(self, mode):
        if mode == 'train':
            self.mode = 0
        elif mode == 'val':
            self.mode = 1
        elif mode == 'calib':
            self.mode = 2
        elif mode == 'test':
            self.mode = 3

        self.s_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == 0:
            emb = self.train_emb
            label = self.train_label
            ts = self.train_ts
        elif self.mode == 1:
            emb = self.val_emb
            label = self.val_label
            ts = self.val_ts
        elif self.mode == 2:
            emb = self.calib_emb
            label = self.calib_label
            ts = self.calib_ts
        else:
            emb = self.test_emb
            label = self.test_label
            ts = self.test_ts

        if self.s_idx >= emb.shape[0]:
            raise StopIteration
        else:
            end = min(self.s_idx + self.batch_size, emb.shape[0])
            curr_emb = emb[self.s_idx:end]
            curr_label = label[self.s_idx:end]
            curr_ts = ts[self.s_idx:end]
            self.s_idx += self.batch_size
            return curr_emb, curr_label, range(self.s_idx, end), curr_ts


class NodeEmbMinibatchWithCalib():
    def __init__(self, emb, role, label, batch_size, ts=None, device=None, calib_batch_size=100):
        self.role = role
        self.label = label
        self.batch_size = batch_size
        self.train_emb = emb[role == 0]
        self.val_emb = emb[role == 1]
        self.calib_train_emb = emb[role == 2]
        self.calib_valid_emb = emb[role == 3]
        self.test_emb = emb[role == 4]

        self.train_label = label[role == 0]
        self.val_label = label[role == 1]
        self.calib_train_label = label[role == 2]
        self.calib_valid_label = label[role == 3]
        self.test_label = label[role == 4]

        if ts is not None:
            self.train_ts = ts[role == 0]
            self.val_ts = ts[role == 1]
            self.calib_train_ts = ts[role == 2]
            self.calib_valid_ts = ts[role == 3]
            self.test_ts = ts[role == 4]

        self.mode = 0
        self.s_idx = 0
        self.calib_batch_size = calib_batch_size

    def shuffle(self):
        # torch.manual_seed(0)
        perm = torch.randperm(self.train_emb.shape[0])
        self.train_emb = self.train_emb[perm]
        self.train_label = self.train_label[perm]
        self.train_ts = self.train_ts[perm]

        torch.manual_seed(0)
        perm = torch.randperm(self.calib_train_emb.shape[0])
        self.calib_train_emb = self.calib_train_emb[perm]
        self.calib_train_label = self.calib_train_label[perm]
        self.calib_train_ts = self.calib_train_ts[perm]

    def set_mode(self, mode):
        if mode == 'train':
            self.mode = 0
            self.batch_size = self.batch_size
        elif mode == 'val':
            self.mode = 1
            self.batch_size = self.batch_size
        elif mode == 'calib_train':
            self.mode = 2
            self.batch_size = self.calib_batch_size
        elif mode == 'calib_valid':
            self.mode = 3
            self.batch_size = self.calib_batch_size
        elif mode == 'test':
            self.mode = 4
            self.batch_size = self.batch_size

        self.s_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == 0:
            emb = self.train_emb
            label = self.train_label
            ts = self.train_ts
        elif self.mode == 1:
            emb = self.val_emb
            label = self.val_label
            ts = self.val_ts
        elif self.mode == 2:
            emb = self.calib_train_emb
            label = self.calib_train_label
            ts = self.calib_train_ts
        elif self.mode == 3:
            emb = self.calib_valid_emb
            label = self.calib_valid_label
            ts = self.calib_valid_ts
        else:
            emb = self.test_emb
            label = self.test_label
            ts = self.test_ts

        if self.s_idx >= emb.shape[0]:
            raise StopIteration
        else:
            if self.mode not in [0, 2]:
                end = min(self.s_idx + self.batch_size, emb.shape[0])
                curr_emb = emb[self.s_idx:end]
                curr_label = label[self.s_idx:end]
                curr_ts = ts[self.s_idx:end]
                self.s_idx += self.batch_size
            else:
                batch_size = self.batch_size
                # batch_size = self.batch_size if self.mode != 2 else self.calib_batch_size
                end = min(self.s_idx + batch_size, emb.shape[0])

                if (end == self.s_idx + batch_size):
                    curr_emb = emb[self.s_idx:end]
                    curr_label = label[self.s_idx:end]
                    curr_ts = ts[self.s_idx:end]
                else:
                    if self.batch_size > emb.shape[0]:
                        curr_emb = emb[self.s_idx:end]
                        curr_label = label[self.s_idx:end]
                        curr_ts = ts[self.s_idx:end]
                    else:
                        gap = self.s_idx + batch_size - emb.shape[0]
                        curr_emb = torch.cat((emb[self.s_idx:end], emb[:gap]), dim=0)
                        curr_label = torch.cat((label[self.s_idx:end], label[:gap]), dim=0)
                        curr_ts = torch.cat((ts[self.s_idx:end], ts[:gap]), dim=0)

                self.s_idx += batch_size

        return curr_emb, curr_label, range(self.s_idx, end), curr_ts


def get_eff_loss(scores, qhat, tau=0.1):
    c = (scores - qhat) / tau
    c_sign = torch.sign(c)
    sample_weight = torch.sum(c_sign, dim=1)
    eff = torch.mean(torch.sum(c, dim=1) * sample_weight)

    return eff


def get_cov_loss(scores, labels, qhat, tau=0.1):
    c = (scores - qhat) / tau
    if len(labels.shape) == 1:
        c_label = c[torch.arange(c.shape[0]), labels]
    else:
        if labels.shape[1] >= 2:
            labels_dense = torch.argmax(labels, dim=1)
            c_label = c[torch.arange(c.shape[0]), labels_dense]
        else:
            c_label = c[torch.arange(c.shape[0]), labels]

    cov = torch.mean(torch.relu(c_label))

    return cov

def get_crossentropy_loss(scores, labels):
    ce_loss = torch.nn.CrossEntropyLoss()
    loss = torch.mean(ce_loss(scores, labels))

    return loss

def nonconformity_score(scores):
    ncs = torch.abs(torch.tensor(1.0, dtype=float) - scores)

    return ncs


def evaluate(cal_smx, cal_labels, qhat):
    prediction_sets = cal_smx <= qhat
    if len(cal_labels.shape) == 1:
        cov = prediction_sets[np.arange(prediction_sets.shape[0]), cal_labels].mean()
        eff = np.mean(np.sum(prediction_sets, axis=1))
    else:
        if cal_labels.shape[1] >= 2:
            labels_dense = np.argmax(cal_labels, axis=1)
            cov = prediction_sets[np.arange(prediction_sets.shape[0]), labels_dense].mean()
            size_each = np.sum(prediction_sets, axis=1)
            size_adjust = [x if x >= 1 else cal_smx.shape[1] for x in size_each]
            eff = np.mean(size_adjust)
        else:
            cov = prediction_sets[np.arange(prediction_sets.shape[0]), cal_labels].mean()
            size_each = np.sum(prediction_sets, axis=1)
            size_adjust = [x if x >= 1 else 2 for x in size_each]
            eff = np.mean(size_adjust)

    return cov, eff, prediction_sets


def eval_acc(y_true, y_pred):
    if (len(y_true.shape) == 1) or (y_true.shape[1] == 1):
        acc = torch.sum(y_true.reshape(-1, 1) == torch.argmax(y_pred, 1, keepdim=True)) / y_true.shape[0]
    else:
        acc = torch.sum(torch.argmax(y_true, 1, keepdim=True) == torch.argmax(y_pred, 1, keepdim=True)) / y_true.shape[0]

    return acc
