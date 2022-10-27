import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader

import dgl
import dgl.function as fn

from utils.synthetic import synthetic_dataset, get_model_parameters

def collate(samples):
    # The input `samples` is a list of pairs
    g_list, states, anchor_pos, x_prior  = map(list, zip(*samples))

    batched_g_list = [dgl.batch([g_list[i][step] for i in range(len(g_list))]) for step in range(len(g_list[0]))]

    return dgl.batch(batched_g_list), torch.stack(states), torch.stack(anchor_pos), torch.stack(x_prior)

def prepare_data(num_steps, num_agents, P0, sigma_driving = 0.05, sigma_meas = 0.3, anchor_range = None, size = 50, batch_size = 10, partition = 'test', track_style = 1, seeds = None):

    if anchor_range is None:
        anchor_range = np.array([[0, 200], [0, 200]])

    xmin = anchor_range[0, 0]
    xmax = anchor_range[0, 1]
    ymin = anchor_range[1, 0]
    ymax = anchor_range[1, 1]
    anchor_range = np.array([[xmin, xmax], [xmin, xmax]])
    if seeds is None:
        seeds = {'test': 1001, 'train': 1009, 'val': 51}

    dataset = synthetic_dataset(partition = partition,
                                track_style = track_style,
                                num_steps = num_steps,
                                train_size = size,
                                val_size = size,
                                test_size = size,
                                anchor_range = anchor_range,
                                num_agents = num_agents,
                                P0 = P0,
                                sigma_driving = sigma_driving,
                                sigma_meas = sigma_meas,
                                start_after = 0,
                                seeds = seeds)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, collate_fn = collate, num_workers = 4)

    return dataset, data_loader
