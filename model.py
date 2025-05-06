import json
import math
import os
from statistics import mean, stdev

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import collections
import utils
import pdb
from utils import prediction2label
from scipy.stats import kendalltau


class ordinal_loss(nn.Module):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    def __init__(self, weight_class=False):
        super(ordinal_loss, self).__init__()
        self.weights = weight_class

    def forward(self, predictions, targets):
        # Fill in ordinalCoefficientVariationLoss target function, i.e. 0 -> [1,0,0,...]
        modified_target = torch.zeros_like(predictions)
        for i, target in enumerate(targets):
            modified_target[i, 0:target + 1] = 1

        # if torch tensor is empty, return 0
        if predictions.shape[0] == 0:
            return 0
        # loss
        if self.weights is not None:
            # pdb.set_trace()
            return torch.sum((self.weights * F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))
        else:
            return torch.sum((F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ContextAttention(nn.Module):
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)
        self.head_size = int(size / num_head)
        self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def get_attention(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        # attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
        similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)
        return similarity

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        if self.head_size != 1:
            attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
            similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
            similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)
            similarity[x.sum(-1) == 0] = -1e4  # mask out zero padded_ones
            softmax_weight = torch.softmax(similarity, dim=1)

            x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
            weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1, 1, 1, x_split.shape[-1])
            attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
        else:
            softmax_weight = torch.softmax(attention, dim=1)
            attention = softmax_weight * x

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip Connection
        out = self.relu(out)
        return out


def get_conv_layer(rep_name):
    if "pianoroll" in rep_name:
        in_channels = 2
        kernel_width = (3, 4, 4)  # 88
    elif "mel" in rep_name:
        in_channels = 1
        kernel_width = (3, 4, 4)  # 64
    elif "cqt" in rep_name:
        in_channels = 1
        kernel_width = (3, 4, 4)  # 88
    else:
        raise ValueError("Representation not implemented")

    if "5" in rep_name:
        kernel_height = (3, 4, 4)
    elif "10" in rep_name:
        kernel_height = (4, 5, 5)
    elif "20" in rep_name:
        kernel_height = (4, 6, 6)
    else:
        raise ValueError("Representation not implemented")

    convs = nn.Sequential(
        ResidualBlock(in_channels, 64, 3, 1, 1),
        nn.MaxPool2d((kernel_height[0], kernel_width[0])),  # Adjusted pooling to handle increased length
        nn.Dropout(0.1),
        ResidualBlock(64, 128, 3, 1, 1),
        nn.MaxPool2d((kernel_height[1], kernel_width[1])),  # Adjusted pooling
        nn.Dropout(0.1),
        ResidualBlock(128, 256, 3, 1, 1),
        nn.MaxPool2d((kernel_height[2], kernel_width[2])),  # Adjusted pooling
        nn.Dropout(0.1)
    )
    return convs


class multimodal_cnns(nn.Module):

    def __init__(self, modality_dropout, only_cqt=False, only_pr=False):
        super().__init__()

        self.midi_branch = get_conv_layer("pianoroll5")
        self.audio_branch = get_conv_layer("cqt5")
        self.modality_dropout = modality_dropout
        self.only_cqt = only_cqt
        self.only_pr = only_pr

    def forward(self, x):
        x_midi, x_audio = x
        # print("input", x_midi.shape, x_audio.shape)
        x_midi = self.midi_branch(x_midi)
        x_audio = self.audio_branch(x_audio)
        # do a modality dropout
        if self.only_cqt:
            x_midi = torch.zeros_like(x_midi, device=x_midi.device)
        elif self.only_pr:
            x_audio = torch.zeros_like(x_audio, device=x_audio.device)
        # print(x_midi.shape, x_audio.shape)
        x_midi_trimmed = x_midi[:, :, :x_audio.size(2), :]

        cnns_out = torch.cat((x_midi_trimmed, x_audio), 1)

        return cnns_out


class AudioModel(nn.Module):
    def __init__(self, num_classes, rep, modality_dropout, only_cqt=False, only_pr=False):
        super(AudioModel, self).__init__()

        # All Convolutional Layers in a Sequential Block
        if "pianoroll" in rep:
            conv = get_conv_layer(rep)
        elif "cqt" in rep:
            conv = get_conv_layer(rep)
        elif "mel" in rep:
            conv = get_conv_layer(rep)
        elif "multi" in rep:
            conv = multimodal_cnns(modality_dropout, only_cqt, only_pr)
        self.conv_layers = conv

        # Calculate the size of GRU input feature
        self.gru_input_size = 512 if "multi" in rep else 256

        # GRU Layer
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=128, num_layers=2,
                          batch_first=True, bidirectional=True)

        self.context_attention = ContextAttention(size=256, num_head=4)
        self.non_linearity = nn.ReLU()

        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x1, kk):
        # Applying Convolutional Block
        # print(x1.shape)
        # pdb.set_trace()
        x = self.conv_layers(x1)
        # Reshape for GRU input
        x = x.flatten(-2).transpose(1, 2)  # Reshaping to [batch, seq_len, features]
        # print(x.shape)
        # GRU part
        # print(x.shape)
        x, _ = self.gru(x)
        # Attention
        x = self.context_attention(x)
        # classiffier
        x = self.non_linearity(x)
        x = self.fc(x)
        return x


def get_mse_macro(y_true, y_pred):
    mse_each_class = []
    for true_class in set(y_true):
        tt, pp = zip(*[[tt, pp] for tt, pp in zip(y_true, y_pred) if tt == true_class])
        mse_each_class.append(mean_squared_error(y_true=tt, y_pred=pp))
    return mean(mse_each_class)


def get_cqt(rep, k):
    inp_data = utils.load_binary(f"../videos_download/{rep}/{k}.bin")
    inp_data = torch.tensor(inp_data, dtype=torch.float32).cuda()
    inp_data = inp_data.unsqueeze(0).unsqueeze(0).transpose(2, 3)
    return inp_data


def get_pianoroll(rep, k):
    inp_pr = utils.load_binary(f"../videos_download/{rep}/{k}.bin")
    inp_on = utils.load_binary(f"../videos_download/{rep}/{k}_onset.bin")
    inp_pr = torch.from_numpy(inp_pr).float().cuda()
    inp_on = torch.from_numpy(inp_on).float().cuda()
    inp_data = torch.stack([inp_pr, inp_on], dim=1)
    inp_data = inp_data.unsqueeze(0).permute(0, 1, 2, 3)
    return inp_data

def compute_model_basic(model_name, rep, modality_dropout, only_cqt=False, only_pr=False):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    data = utils.load_json("../videos_download/split_audio.json")
    mse, acc = [], []
    predictions = []
    if only_cqt:
        cache_name = model_name + "_cqt"
    elif only_pr:
        cache_name = model_name + "_pr"
    else:
        cache_name = model_name
    if not os.path.exists(f"cache/{cache_name}.json"):
        for split in range(5):
            #load_model
            model = AudioModel(11, rep, modality_dropout, only_cqt, only_pr)
            checkpoint = torch.load(f"models/{model_name}/checkpoint_{split}.pth",  map_location='cuda:0')
            # print(checkpoint["epoch"])
            # print(checkpoint.keys())
            # pdb.set_trace()
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.cuda()
            pred_labels, true_labels = [], []
            predictions_split = {}
            model.eval()
            with torch.inference_mode():
                for k, ps in data[str(split)]["test"].items():
                    # computar el modelo
                    if "cqt" in rep:
                        inp_data = get_cqt(rep, k)
                    elif "pianoroll" in rep:
                        inp_data = get_pianoroll(rep, k)
                    elif rep == "multimodal5":
                        x1 = get_pianoroll("pianoroll5", k)
                        x2 = get_cqt("cqt5", k)[:, :, :x1.shape[2]]
                        inp_data = [x1, x2]
                    log_prob = model(inp_data, None)
                    pred = prediction2label(log_prob).cpu().tolist()[0]
                    print(k, ps, pred)
                    predictions_split[k] = {
                        "true": ps,
                        "pred": pred
                    }
                    # pdb.set_trace()
                    true_labels.append(ps)
                    pred_labels.append(pred)
            # pdb.set_trace()
            predictions.append(predictions_split)
            mse.append(get_mse_macro(true_labels, pred_labels))
            acc.append(balanced_accuracy_score(true_labels, pred_labels))
        # with one decimal
        print(f"mse: {mean(mse):.1f}({stdev(mse):.1f})", end=" ")
        print(f"acc: {mean(acc)*100:.1f}({stdev(acc)*100:.1f})")
        utils.save_json({
            "mse": mse,
            "acc": acc,
            "predictions": predictions
        }, f"cache/{cache_name}.json")
    else:
        data = utils.load_json(f"cache/{cache_name}.json")
        tau_c, mse, acc = [], [], []
        for i in range(5):
            pred, true = [], []
            for k, dd in data["predictions"][i].items():
                pred.append(dd["pred"])
                true.append(dd["true"])
            tau_c.append(kendalltau(x=true, y=pred).statistic)
            mse.append(get_mse_macro(true, pred))
            acc.append(balanced_accuracy_score(true, pred))
        print(model_name, end="// ")
        print(f"& {mean(mse):.2f}({stdev(mse):.2f})", end=" ")
        print(f"& {mean(acc) * 100:.1f}({stdev(acc) * 100:.2f})", end=" ")
        print(f"& {mean(tau_c):.3f}({stdev(tau_c):.3f})")


def compute_ensemble(truncate=False):
    round_func = lambda x: math.ceil(x) if truncate else math.floor(x)
    data_pr = utils.load_json(f"cache/audio_midi_cqt5_ps_v5.json")
    data_cqt = utils.load_json(f"cache/audio_midi_pianoroll_ps_5_v4.json")
    tau_c, mse, acc = [], [], []
    for i in range(5):
        pred, true = [], []
        for k, dd in data_pr["predictions"][i].items():
            cqt_pred = data_cqt["predictions"][i][k]
            pred.append(round_func((dd["pred"] + cqt_pred["pred"])/2))
            true.append(dd["true"])
        tau_c.append(kendalltau(x=true, y=pred).statistic)
        mse.append(get_mse_macro(true, pred))
        acc.append(balanced_accuracy_score(true, pred))
    print("ensemble", end="// ")
    print(f"& {mean(mse):.2f}({stdev(mse):.2f})", end=" ")
    print(f"& {mean(acc) * 100:.1f}({stdev(acc) * 100:.2f})", end=" ")
    print(f"& {mean(tau_c):.3f}({stdev(tau_c):.3f})")


def load_json(name_file):
    with open(name_file, 'r') as fp:
        data = json.load(fp)
    return data




