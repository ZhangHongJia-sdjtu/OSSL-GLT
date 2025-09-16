import copy
import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import LayerNorm, LSTM, Linear, Dropout, ReLU
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch, dropout_edge
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_curve, auc
import random


def time_sequence_graph(num_nodes):
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


num_heads_setting = 4
bonus_d_model = 8 * num_heads_setting

batch_size = 512

pseudo_label_initial_threshold = 0.9
pseudo_label_min_threshold = 0.85

pseudo_label_loss_weight = 0.4
consistency_loss_weight = 0.4

dropout_rate_transformer = 0.4
dropout_rate_gcn = 0.3
noise_std = 0.15

learning_rate = 0.005
weight_decay = 1e-4

clip_grad_norm_value = 1.0

predict_threshold = 0.7

num_rounds = 1


file_path = '#your data#'
sheet = pd.read_excel(file_path)
data_rows = sheet.values

def semi_split(dataset, labeled_ratio=0.10):
    labels = [data.y.item() for data in dataset]
    labeled_idx, unlabeled_idx = train_test_split(range(len(dataset)), train_size=labeled_ratio, stratify=labels, random_state=42)
    labeled_set = [dataset[i] for i in labeled_idx]
    unlabeled_set = [copy.deepcopy(dataset[i]) for i in unlabeled_idx]
    for data in unlabeled_set:
        data.y = torch.tensor(-1, dtype=torch.long)
    return labeled_set + unlabeled_set, labeled_idx

graph_list = []
for row in data_rows:
    features = row[:-1].reshape(#图数据格式#).T
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(int(row[-1]) , dtype=torch.long)
    edge_index = time_sequence_graph(x.shape[0])
    graph_list.append(Data(x=x, edge_index=edge_index, y=y))


new_input_dataset, remaining_dataset = train_test_split(graph_list, test_size=0.9, stratify=[g.y.item() for g in graph_list], random_state=42)


remaining_labels = [g.y.item() for g in remaining_dataset]
train_dataset_raw, test_dataset = train_test_split(remaining_dataset, test_size=0.2, stratify=remaining_labels, random_state=42)


train_dataset, labeled_idx = semi_split(train_dataset_raw)


def split_new_data_for_online_update(dataset):
    dataset_copy = copy.deepcopy(dataset)
    n = len(dataset_copy)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)

    half = n // 2
    labeled_indices = indices[:half]
    unlabeled_indices = indices[half:]

    for i in unlabeled_indices:
        dataset_copy[i].y = torch.tensor(-1, dtype=torch.long)

    return dataset_copy

new_dataset_for_online = split_new_data_for_online_update(new_input_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
new_loader = DataLoader(new_dataset_for_online, batch_size=batch_size, shuffle=True)

def train(epoch):
    model.train()
    total_loss = 0
    temperature = 2.0 * math.exp(-0.03 * epoch)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits1, feat1 = model(data)
        logits2, feat2 = model(data)

        valid_mask = data.y != -1
        loss_supervised = weighted_ce_loss(logits1[valid_mask], data.y[valid_mask]) if valid_mask.sum() > 0 else 0
        loss_pseudo, loss_consist = 0, 0

        if (~valid_mask).sum() > 0:
            unlabeled_logits = logits1[~valid_mask]
            probs = F.softmax(unlabeled_logits / temperature, dim=1)
            confidence, pseudo_labels = probs.max(dim=1)

            threshold = max(pseudo_label_initial_threshold - 0.1 * (epoch / 50), pseudo_label_min_threshold)
            mask = confidence > threshold
            if mask.sum() > 0:
                loss_pseudo = F.cross_entropy(unlabeled_logits[mask], pseudo_labels[mask])
            loss_consist = consistency_loss(feat1, feat2)

        loss = loss_supervised + (pseudo_label_loss_weight * loss_pseudo + consistency_loss_weight * loss_consist)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)



def test(loader):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits, _ = model(data)
            _, pred = logits.max(dim=1)

            mask = data.y != -1
            if mask.sum() == 0:
                continue

            pred = pred[mask]
            labels = data.y[mask]

            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, np.array([], dtype=int), np.array([], dtype=int)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc = correct / total
    return acc, all_labels, all_preds


class HybridPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.learnable_pe = torch.nn.Parameter(torch.zeros(1, max_len, d_model))
        torch.nn.init.xavier_uniform_(self.learnable_pe)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        if d_model % 2 == 1:

            div_term = div_term[:-1]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('sinusoidal_pe', pe.unsqueeze(0))
        self.ln = LayerNorm(d_model)

    def forward(self, x):
        pos_emb = self.learnable_pe[:, :x.size(1), :] + self.sinusoidal_pe[:, :x.size(1), :]
        return self.ln(x + pos_emb)


class CustomMultiHeadTransformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.4):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = torch.nn.Sequential(
            LayerNorm(d_model),
            Linear(d_model, 128),
            ReLU(),
            Dropout(dropout),
            Linear(128, d_model)
        )
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, key_padding_mask=~mask if mask is not None else None)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.ffn(x))
        return x


class SemiGCNTransformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, noise_std=noise_std):
        super().__init__()
        self.noise_std = noise_std

        # LSTM：输入为每个节点的特征维度，输出为隐藏单元维度，双向
        self.lstm = LSTM(input_size="节点特征维度",
                         hidden_size="LSTM隐藏单元维度",
                         batch_first=True,
                         bidirectional=True)

        # GCN 第一层：输入为节点特征维度，输出为第一层GCN特征维度
        self.conv1 = GCNConv("节点特征维度", "第一层GCN输出特征维度")
        # GCN 第二层：输入为第一层输出维度，输出为第二层GCN特征维度
        self.conv2 = GCNConv("第一层GCN输出特征维度", "第二层GCN输出特征维度")
        self.ln1 = LayerNorm("第二层GCN输出特征维度")
        self.ln2 = LayerNorm("第二层GCN输出特征维度")
        # 拼接GCN输出和LSTM输出后的特征维度映射到Transformer输入维度
        self.cat_linear = Linear("第二层GCN输出特征维度 + LSTM输出维度", "Transformer输入特征维度(d_model)")
        # 位置编码
        self.pe = HybridPositionalEncoding(d_model="Transformer输入特征维度(d_model)")
        # Transformer编码器
        self.transformer = CustomMultiHeadTransformer(d_model="Transformer输入特征维度(d_model)",
                                                      num_heads="多头注意力头数",
                                                      dropout="Transformer dropout率")
        # 分类头
        self.fc = torch.nn.Sequential(
            LayerNorm("Transformer输出特征维度拼接后的总维度"),
            Linear("分类头输入维度", "隐藏层维度"),
            ReLU(),
            Dropout("分类头dropout率"),
            Linear("隐藏层维度", "分类类别数")
        )
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        num_graphs = batch.max().item() + 1
        # LSTM输入 reshape
        x_lstm_in = x.view(num_graphs, -1, "节点特征维度")
        lstm_out, _ = self.lstm(x_lstm_in)
        lstm_out = lstm_out.contiguous().view(-1, "LSTM输出维度")

        edge_index_dropped, _ = dropout_edge(edge_index, p="GCN边dropout概率", training=self.training)
        # GCN前向
        x_gcn = F.leaky_relu(self.ln1(self.conv1(x, edge_index_dropped)))
        x_gcn = F.dropout(x_gcn, p="GCN特征dropout概率", training=self.training)
        x_gcn = F.leaky_relu(self.ln2(self.conv2(x_gcn, edge_index_dropped)))
        # 拼接 LSTM 和 GCN 特征
        x_cat = torch.cat([x_gcn, lstm_out], dim=1)
        x_cat = self.cat_linear(x_cat)

        # Transformer输入
        x_dense, mask = to_dense_batch(x_cat, batch)
        x_pe = self.pe(x_dense)
        x_trans = self.transformer(x_pe, mask)
        x_trans = x_trans.view(-1, "Transformer输出特征维度")

        # 全局池化
        x_pooled_transformer = global_mean_pool(x_trans, batch)
        x_pooled_cat = global_mean_pool(x_cat, batch)
        x_combined = torch.cat([x_pooled_transformer, x_pooled_cat], dim=1)

        # 分类输出
        logits = self.fc(x_combined)
        return logits, x_trans


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_weight = torch.tensor([3.0, 1.5])


def weighted_ce_loss(logits, labels):
    return F.cross_entropy(logits, labels, weight=class_weight.to(device))


def consistency_loss(f1, f2):
    return F.mse_loss(f1, f2)


def online_update(model, fine_tune_loader, test_loader, optimizer, epochs=50, patience=30, update_confidence_threshold=0.75):
    print('\n=== Starting Online Update ===')

    model.eval()
    best_acc = -1
    best_labels = np.array([], dtype=int)
    best_preds = np.array([], dtype=int)
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        batches_processed = 0


        for data in fine_tune_loader:
            data = data.to(device)

            logits, _ = model(data)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)


            pseudo_labels = data.y.clone().to(device)

            for i in range(data.num_graphs):
                if pseudo_labels[i] == -1 and conf[i].item() > update_confidence_threshold:
                    pseudo_labels[i] = pred[i]

            valid_mask = (pseudo_labels >= 0) & (pseudo_labels < logits.shape[1])
            if valid_mask.sum() == 0:
                continue

            loss = F.cross_entropy(logits[valid_mask], pseudo_labels[valid_mask], weight=class_weight.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            batches_processed += 1

        acc, labels, preds = test(test_loader)
        print(f'[Online Update] Epoch {epoch:02d}, Test Acc: {acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            best_labels = labels.copy() if isinstance(labels, np.ndarray) else np.array(labels)
            best_preds = preds.copy() if isinstance(preds, np.ndarray) else np.array(preds)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print('Early stopping triggered in Online Update.')
            break

    return best_acc, best_labels, best_preds



best_train_metrics_per_round = []
round_epoch_metrics = []
best_confusion_matrices = []

for round_idx in range(1, num_rounds + 1):
    print(f'\n========== Round {round_idx} / {num_rounds} ==========')
    model = SemiGCNTransformer(d_model=bonus_d_model, num_heads=num_heads_setting).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    best_test_acc = -1
    best_metrics = {
        'Train Accuracy': 0,
        'Precision Class 0': 0,
        'Recall Class 0': 0,
        'F1 Class 0': 0,
        'Precision Class 1': 0,
        'Recall Class 1': 0,
        'F1 Class 1': 0,
        'Test Accuracy': 0,
        'Epoch': 0
    }

    no_improve_epochs = 0

    for epoch in range(1, 101):
        loss = train(epoch)
        scheduler.step()

        train_acc, _, _ = test(train_loader)
        test_acc, test_labels, test_preds = test(test_loader)


        try:
            if len(test_labels) > 0:
                tl_np = test_labels
                tp_np = test_preds
                test_precision_list = precision_score(tl_np, tp_np, average=None, labels=[0, 1], zero_division=0)
                test_recall_list = recall_score(tl_np, tp_np, average=None, labels=[0, 1], zero_division=0)
                test_f1_list = f1_score(tl_np, tp_np, average=None, labels=[0, 1], zero_division=0)
                precision_0, precision_1 = test_precision_list[0], test_precision_list[1]
                recall_0, recall_1 = test_recall_list[0], test_recall_list[1]
                f1_0, f1_1 = test_f1_list[0], test_f1_list[1]
            else:
                precision_0 = precision_1 = recall_0 = recall_1 = f1_0 = f1_1 = 0.0
        except Exception:
            precision_0 = precision_1 = recall_0 = recall_1 = f1_0 = f1_1 = 0.0

        print(f'[Round {round_idx}] Epoch {epoch:03d} | Loss: {loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Test Acc: {test_acc:.4f} | '
              f'P0: {precision_0:.4f} R0: {recall_0:.4f} F10: {f1_0:.4f} | '
              f'P1: {precision_1:.4f} R1: {recall_1:.4f} F11: {f1_1:.4f}')

        round_epoch_metrics.append({
            'Round': round_idx,
            'Epoch': epoch,
            'Loss': loss,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision Class 0': precision_0,
            'Recall Class 0': recall_0,
            'F1 Class 0': f1_0,
            'Precision Class 1': precision_1,
            'Recall Class 1': recall_1,
            'F1 Class 1': f1_1,
        })

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_metrics['Train Accuracy'] = train_acc
            best_metrics['Precision Class 0'] = precision_0
            best_metrics['Recall Class 0'] = recall_0
            best_metrics['F1 Class 0'] = f1_0
            best_metrics['Precision Class 1'] = precision_1
            best_metrics['Recall Class 1'] = recall_1
            best_metrics['F1 Class 1'] = f1_1
            best_metrics['Test Accuracy'] = test_acc
            best_metrics['Epoch'] = epoch
            no_improve_epochs = 0


            if len(test_labels) == 0:
                cm = np.array([[0, 0], [0, 0]])
            else:
                try:
                    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
                except Exception:
                    cm = np.array([[0, 0], [0, 0]])

            best_confusion_matrices.append({
                'Round': round_idx,
                'Epoch': epoch,
                'Confusion Matrix': cm
            })

        else:
            no_improve_epochs += 1


    best_metrics['Round'] = round_idx
    best_train_metrics_per_round.append(best_metrics)


    print(f"\n========== Online Update Round {round_idx} ==========")
    fine_tune_acc, fine_tune_labels, fine_tune_preds = online_update(model, new_loader, test_loader, optimizer, epochs=50, patience=30)
    print(f"Online Updated Model Test Accuracy: {fine_tune_acc:.4f}")
