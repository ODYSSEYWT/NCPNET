import argparse
import copy
import os
import hashlib

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, default='', help='path to config file')
parser.add_argument('--batch_size', type=int, default=4000)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model', type=str, default='', help='name of stored model to load')
parser.add_argument('--posneg', default=False, action='store_true', help='for positive negative detection, whether to sample negative nodes')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')

parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--exp', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.05)

parser.add_argument('--calib_epochs', type=int, default=1000)
parser.add_argument('--lr_calib', type=float, default=0.005)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--cov_weight', type=float, default=1.0)
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--calib_batch_size', type=int, default=100)
parser.add_argument('--epsilon', type=float, default=0.0)
args=parser.parse_args()

if args.data in ['WIKI', 'REDDIT']:
    args.posneg = True

import torch
import time
import random
import dgl
import numpy as np
import pandas as pd
from modules import *
from sampler import *
from utils import *
from tqdm import tqdm
import wandb


wandb.init(
    project=f"UQTG-{args.data}",
    name="{}_{}".format(
        args.config.split("/")[-1].split(".")[0], int(time.time())
    ),
    config={
        "lr": args.lr_calib,
        "epochs": args.calib_epochs,
    },
)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

torch.autograd.set_detect_anomaly(True)

ldf = pd.read_csv('./DATA/{}/labels.csv'.format(args.data))

role = ldf['ext_roll'].values
labels = ldf['label'].values.astype(np.int64)

emb_file_name = hashlib.md5(str(torch.load(args.model, map_location=torch.device('cpu'))).encode('utf-8')).hexdigest() + '.pt'
if not os.path.isdir('embs'):
    os.mkdir('embs')
if not os.path.isfile('embs/' + emb_file_name):
    print('Generating temporal embeddings..')

    node_feats, edge_feats = load_feat(args.data, args.rand_edge_features, args.rand_node_features)
    g, df = load_graph(args.data)
    sample_param, memory_param, gnn_param, train_param = parse_config(
        args.config)
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True
    model = GeneralModel(gnn_dim_node,
                         gnn_dim_edge,
                         sample_param,
                         memory_param,
                         gnn_param,
                         train_param,
                         combined=combine_first,
                         device=device).to(device)
    mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge, _device=device) if memory_param['type'] != 'none' else None
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
        if node_feats is not None:
            node_feats = node_feats.to(device)
        if edge_feats is not None:
            edge_feats = edge_feats.to(device)
        if mailbox is not None:
            mailbox.move_to_gpu()

    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(
            g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
            sample_param['num_thread'], 1, sample_param['layer'],
            sample_param['neighbor'], sample_param['strategy'] == 'recent',
            sample_param['prop_time'], sample_param['history'],
            float(sample_param['duration']))
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)

    params_dict = torch.load(args.model)
    model.load_state_dict(params_dict)

    processed_edge_id = 0

    def forward_model_to(time):
        global processed_edge_id
        if processed_edge_id >= len(df):
            return
        while df.time[processed_edge_id] < time:
            rows = df[processed_edge_id:min(
                processed_edge_id + train_param['batch_size'], len(df))]
            if processed_edge_id < train_edge_end:
                model.train()
            else:
                model.eval()
            root_nodes = np.concatenate([
                rows.src.values, rows.dst.values,
                neg_link_sampler.sample(len(rows))
            ]).astype(np.int32)
            ts = np.concatenate(
                [rows.time.values, rows.time.values,
                 rows.time.values]).astype(np.float32)
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end],
                                   ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'], device=device)
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts, device=device)
            mfgs = prepare_input(mfgs,
                                 node_feats,
                                 edge_feats,
                                 combine_first=combine_first,
                                 device=device)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])
            with torch.no_grad():
                pred_pos, pred_neg = model(mfgs)
                if mailbox is not None:
                    eid = rows['Unnamed: 0'].values
                    mem_edge_feats = edge_feats[
                        eid] if edge_feats is not None else None
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        block = to_dgl_blocks(ret,
                                              sample_param['history'],
                                              reverse=True, device=device)[0][0]
                    mailbox.update_mailbox(
                        model.memory_updater.last_updated_nid,
                        model.memory_updater.last_updated_memory, root_nodes,
                        ts, mem_edge_feats, block)
                    mailbox.update_memory(
                        model.memory_updater.last_updated_nid,
                        model.memory_updater.last_updated_memory, root_nodes,
                        model.memory_updater.last_updated_ts)
            processed_edge_id += train_param['batch_size']
            if processed_edge_id >= len(df):
                return

    def get_node_emb(root_nodes, ts):
        forward_model_to(ts[-1])
        if sampler is not None:
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
        if gnn_param['arch'] != 'identity':
            mfgs = to_dgl_blocks(ret, sample_param['history'], device=device)
        else:
            mfgs = node_to_dgl_blocks(root_nodes, ts, device=device)
        mfgs = prepare_input(mfgs,
                             node_feats,
                             edge_feats,
                             combine_first=combine_first,
                             device=device)
        if mailbox is not None:
            mailbox.prep_input_mails(mfgs[0])
        with torch.no_grad():
            ret = model.get_emb(mfgs)
        return ret.detach().cpu()

    emb = list()
    for _, rows in tqdm(ldf.groupby(ldf.index // args.batch_size)):
        emb.append(
            get_node_emb(rows.node.values.astype(np.int32),
                         rows.time.values.astype(np.float32)))
    emb = torch.cat(emb, dim=0)
    ldf['emb'] = emb.tolist()

    torch.save(emb, 'embs/' + emb_file_name)
    print('Saved to embs/' + emb_file_name)
else:
    print('Loading temporal embeddings from embs/' + emb_file_name)
    emb = torch.load('embs/' + emb_file_name)
    ldf['emb'] = emb.tolist()

model = NodeClassificationModel(emb.shape[1], args.dim, labels.max() + 1).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.posneg:
    labels = torch.from_numpy(labels).type(torch.int32).to(device)

role = torch.from_numpy(role).type(torch.int32).to(device)
emb = emb.to(device)
emb_all = copy.deepcopy(emb)

split = [0.5, 0.6, 0.7, 0.8]
ldf['ts_role'] = np.zeros(ldf.shape[0])

perm = torch.randperm(ldf.shape[0])
ldf.loc[perm[int(split[0] * ldf.shape[0]):int(split[1] * ldf.shape[0])], 'ts_role'] = 1
ldf.loc[perm[int(split[1] * ldf.shape[0]):int(split[2] * ldf.shape[0])], 'ts_role'] = 2
ldf.loc[perm[int(split[2] * ldf.shape[0]):int(split[3] * ldf.shape[0])], 'ts_role'] = 3
ldf.loc[perm[int(split[3] * ldf.shape[0]):], 'ts_role'] = 4

print("Train: {}/{}, Valid: {}/{}, Calib train: {}/{}, Calib valid: {}/{}, Test: {}/{}".format(
    ldf[ldf['ts_role'] == 0].shape[0], ldf[(ldf['ts_role'] == 0) & (ldf['label'] == 1)].shape[0],
    ldf[ldf['ts_role'] == 1].shape[0], ldf[(ldf['ts_role'] == 1) & (ldf['label'] == 1)].shape[0],
    ldf[ldf['ts_role'] == 2].shape[0], ldf[(ldf['ts_role'] == 2) & (ldf['label'] == 1)].shape[0],
    ldf[ldf['ts_role'] == 3].shape[0], ldf[(ldf['ts_role'] == 3) & (ldf['label'] == 1)].shape[0],
    ldf[ldf['ts_role'] == 4].shape[0], ldf[(ldf['ts_role'] == 4) & (ldf['label'] == 1)].shape[0]))

ts_role = torch.from_numpy(ldf['ts_role'].values).to(device)
pos_label = ldf['label'] == 1
ts = torch.from_numpy(ldf['time'].values).to(device)

if args.posneg:
    minibatch = NodeEmbMinibatchWithCalib(emb_all[pos_label], ts_role[pos_label], labels[pos_label], args.batch_size, ts[pos_label], device, args.calib_batch_size)

if args.posneg:
    role = role[labels == 1]
    emb_neg = emb[labels == 0].to(device)
    emb = emb[labels == 1]
    labels = torch.ones(emb.shape[0], dtype=torch.int64).to(device)
    labels_neg = torch.zeros(emb_neg.shape[0], dtype=torch.int64).to(device)

    emb_neg_train = emb_all[np.logical_and(ts_role.cpu().numpy() == 0, (ldf['label'].values == 0))].to(device)
    emb_neg_valid = emb_all[np.logical_and(ts_role.cpu().numpy() == 1, (ldf['label'].values == 0))].to(device)
    emb_neg_calib_train = emb_all[np.logical_and(ts_role.cpu().numpy() == 2, (ldf['label'].values == 0))].to(device)
    emb_neg_calib_valid = emb_all[np.logical_and(ts_role.cpu().numpy() == 3, (ldf['label'].values == 0))].to(device)
    emb_neg_test = emb_all[np.logical_and(ts_role.cpu().numpy() == 4, (ldf['label'].values == 0))].to(device)

    ts_neg_calib_train = ts[np.logical_and(ts_role.cpu().numpy() == 2, (ldf['label'].values == 0))]
    ts_neg_test = ts[np.logical_and(ts_role.cpu().numpy() == 4, (ldf['label'].values == 0))]


def final_loss(eff_loss, cov_loss, weight=1.0):
    return eff_loss + weight * cov_loss

def nonconformity_score(scores):
    ncs = torch.abs(torch.tensor(1.0, dtype=float) - scores)

    return ncs

for exp_i in range(args.exp):
    seed = 0

    if not os.path.isdir('models'):
        os.mkdir('models')
    save_path = 'models/node_' + args.model.split('/')[-1]
    best_e = 0
    best_acc = 0
    for e in range(args.epoch):
        # train model
        minibatch.set_mode('train')
        minibatch.shuffle()
        model.train()
        for emb, label, _, _ in minibatch:
            emb = emb.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            if args.posneg:
                neg_node_train_sampler = NegLinkSampler(emb_neg_train.shape[0], seed)
                neg_idx = neg_node_train_sampler.sample(emb.shape[0])
                emb = torch.cat([emb, emb_neg_train[neg_idx]], dim=0)
                label = torch.cat([label, labels_neg[neg_idx]], dim=0)
                label = torch.nn.functional.one_hot(label)
            pred = model(emb)

            loss = loss_fn(pred, label.float())
            loss.backward()
            optimizer.step()

        # valid model
        minibatch.set_mode('val')
        model.eval()
        accs = list()
        with torch.no_grad():
            for emb, label, _, _ in minibatch:
                emb = emb.to(device)
                label = label.to(device)
                if args.posneg:
                    neg_node_valid_sampler = NegLinkSampler(emb_neg_valid.shape[0], seed)
                    neg_idx = neg_node_valid_sampler.sample(emb_neg_valid.shape[0])
                    emb = torch.cat([emb, emb_neg_valid[neg_idx]], dim=0)
                    label = torch.cat([label, labels_neg[neg_idx]], dim=0)
                    label = torch.nn.functional.one_hot(label)
                pred = model(emb)
                prob = pred.softmax(dim=1)

                acc = eval_acc(label, prob)
                accs.append(acc)
            acc = float(torch.tensor(accs).mean())

        if acc > best_acc:
            best_e = e
            best_acc = acc
            torch.save(model.state_dict(), save_path)
    print("best acc: ", best_acc)
    model.load_state_dict(torch.load(save_path))

    best_efficiency = np.inf
    best_cov = 0.0
    best_qhat = 0.0
    best_calib_epoch = 0
    best_acc = 0.0
    best_loss = np.inf
    
    calib_neg_ratio = 1
    if args.posneg:
        q_model = NexModel(min(minibatch.calib_train_label.shape[0], args.calib_batch_size) * (calib_neg_ratio + 1), args.alpha, args.sigma).to(device)

    optimizer_c = torch.optim.Adam([{'params': q_model.parameters(), 'lr': args.lr_calib}], lr=args.lr_calib)
    calib_save_path = 'models/node_calib_{}'.format(args.model.split('/')[-1])

    for calib_i in range(args.calib_epochs):
        minibatch.set_mode('calib_train')
        minibatch.shuffle()

        model.eval()
        q_model.train()
        optimizer_c.zero_grad()

        for emb, label, _, times in minibatch:
            emb = emb.to(device)
            label = label.to(device)
            times = times.to(device)

            if args.posneg:
                neg_node_calib_sampler = NegLinkSampler(emb_neg_calib_train.shape[0], seed)
                neg_idx = neg_node_calib_sampler.sample(emb.shape[0] * calib_neg_ratio)
                emb = torch.cat([emb, emb_neg_calib_train[neg_idx]], dim=0)
                label = torch.cat([label, labels_neg[neg_idx]], dim=0)
                label = torch.nn.functional.one_hot(label)
                times = torch.cat([times, ts_neg_calib_train[neg_idx]], dim=0)

            with torch.no_grad():
                pred = model(emb)
                prob = pred.softmax(dim=1)

            nonconf_scores = nonconformity_score(prob, label)
            qhat_train, tv = q_model(nonconf_scores, label)

            pred_train_loss = loss_fn(pred, label.float())
            eff_train_loss = get_eff_loss(nonconf_scores, qhat_train, args.tau)
            cov_train_loss = get_cov_loss(nonconf_scores, label, qhat_train, args.tau)

            calib_loss = final_loss(eff_train_loss, cov_train_loss, args.cov_weight)

            calib_loss.backward()
            optimizer_c.step()

            wandb.log(
                {
                    "Train/loss": calib_loss,
                    "Train/Pred_loss": pred_train_loss,
                    "Train/Eff_loss": eff_train_loss,
                    "Train/Cov_loss": cov_train_loss,
                    "Train/Q": qhat_train,
                    "Train/weights": torch.sum(q_model.weights_normalized).item(),
                    "Train/weights_distribution": wandb.Histogram(q_model.weights_normalized.detach().cpu().numpy())
                },
                commit=False
            )

        minibatch.set_mode('calib_valid')
        model.eval()
        q_model.eval()

        accs = list()
        covs = list()
        effs = list()
        calib_test_losses = list()
        pred_losses = list()
        cov_losses = list()
        eff_losses = list()

        with torch.no_grad():
            for emb, label, _, _ in minibatch:
                emb = emb.to(device)
                label = label.to(device)

                if args.posneg:
                    neg_node_calib_sampler = NegLinkSampler(emb_neg_calib_valid.shape[0], seed)
                    neg_idx = neg_node_calib_sampler.sample(emb_neg_calib_valid.shape[0])
                    emb = torch.cat([emb, emb_neg_calib_valid[neg_idx]], dim=0)
                    label = torch.cat([label, labels_neg[neg_idx]], dim=0)
                    label = torch.nn.functional.one_hot(label)

                pred = model(emb)
                prob = pred.softmax(dim=1)

                # calibration model
                pred_loss = loss_fn(pred, label.float())

                nonconf_score = nonconformity_score(prob, label)
                eff_loss = get_eff_loss(nonconf_scores, qhat_train, args.tau)
                cov_loss = get_cov_loss(nonconf_score, label, qhat_train, args.tau)

                calib_test_loss = final_loss(eff_loss, cov_loss, args.cov_weight)

                acc = eval_acc(label, prob)
                cov, eff, single_hit, prediction_sets = evaluate(nonconf_score.detach().cpu().numpy(), label.detach().cpu().numpy(), qhat=qhat_train.detach().cpu().numpy())

                accs.append(acc)
                covs.append(cov)
                effs.append(eff)
                calib_test_losses.append(calib_test_loss)
                pred_losses.append(pred_loss)
                cov_losses.append(cov_loss)
                eff_losses.append(eff_loss)

        acc = float(torch.tensor(accs).mean())
        cov = float(torch.tensor(covs).mean())
        eff = float(torch.tensor(effs).mean())
        calib_test_loss = float(torch.tensor(calib_test_losses).mean())
        pred_loss = float(torch.tensor(pred_losses).mean())
        cov_loss = float(torch.tensor(cov_losses).mean())
        eff_loss = float(torch.tensor(eff_losses).mean())

        wandb.log(
                {
                    "Calib/loss": calib_test_loss,
                    "Calib/Pred_loss": pred_loss,
                    "Calib/Eff_loss": eff_loss,
                    "Calib/Cov_loss": cov_loss,
                    "Calib/ACC": acc,
                    "Calib/COV": cov,
                    "Calib/EFF": eff,
                }
            )

        if (eff <= best_efficiency) & (cov >= 1 - args.alpha + args.epsilon):
            best_acc = acc
            best_cov = cov
            best_efficiency = eff
            best_qhat = qhat_train.tolist()
            best_calib_epoch = calib_i
            best_loss = calib_test_loss

    print("best_epoch: {}, best_qhat: {:.4f}, best_acc: {:.4f}, best_cov: {:.4f}, best_efficiency: {:.4f}".format(best_calib_epoch, best_qhat, best_acc, best_cov, best_efficiency))

    acc_list = list()
    cov_list = list()
    eff_list = list()

    minibatch.set_mode('test')
    model.eval()
    q_model.eval()

    label_test_list = list()
    preds_test_list = list()
    prediction_sets_list = list()
    times_test_list = list()

    with torch.no_grad():
        for emb, label, _, ts in minibatch:
            emb = emb.to(device)
            label = label.to(device)
            if args.posneg:
                neg_node_test_sampler = NegLinkSampler(emb_neg_test.shape[0], seed)
                neg_idx = neg_node_test_sampler.sample(emb_neg_test.shape[0])
                emb = torch.cat([emb, emb_neg_test[neg_idx]], dim=0)
                label = torch.cat([label, labels_neg[neg_idx]], dim=0)
                label = torch.nn.functional.one_hot(label)
                times = torch.cat([times, ts_neg_test[neg_idx]], dim=0)
            pred = model(emb)
            prob = pred.softmax(dim=1)

            # calibration model
            acc = eval_acc(label, prob)

            label_test_list.extend(label.tolist())
            preds_test_list.extend(prob.tolist())
            times_test_list.append(times.tolist())
            
            nonconf_score = nonconformity_score(prob, label)
            cov, eff, single_hit, prediction_sets = evaluate(nonconf_score.detach().cpu().numpy(), label.detach().cpu().numpy(), qhat=best_qhat)
            prediction_sets_list.extend(prediction_sets.astype(int))

            acc_list.append(acc)
            cov_list.append(cov)
            eff_list.append(eff)
    acc = float(torch.tensor(acc_list).mean())
    cov = float(torch.tensor(cov_list).mean())
    eff = float(torch.tensor(eff_list).mean())

    print("Test acc: {:.4f}, cov: {:.4f}, eff: {:.4f}\n".format(acc, cov, eff))
