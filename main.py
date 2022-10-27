# ------------------------------------------------------------------------
# Cooperative Localization
# Copyright (c) 2022 MIngchao Liang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------

import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

import dgl
import dgl.function as fn

import matplotlib.pyplot as plt

from utils.synthetic import synthetic_dataset, get_model_parameters
from utils.prepare_data import prepare_data
from bp_test import bp_test

def collate(samples):
    # The input `samples` is a list of pairs
    g_list, states, anchor_pos, x_prior  = map(list, zip(*samples))

    batched_g_list = [dgl.batch([g_list[i][step] for i in range(len(g_list))]) for step in range(len(g_list[0]))]

    return dgl.batch(batched_g_list), torch.stack(states), torch.stack(anchor_pos), torch.stack(x_prior)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps',
                        help='The number of time steps in each sample',
                        type = int,
                        default=50)
    parser.add_argument('--num_agents',
                        help='The number of agents in each sample',
                        type = int,
                        default = 100)
    parser.add_argument('--pos_prior_cov',
                        help='The prior position covariance of each agent',
                        type = float,
                        default = 10)
    parser.add_argument('--vel_prior_cov',
                        help='The prior velocity covariance of each agent',
                        type = float,
                        default = 0.01)
    parser.add_argument('--driving_noise_std',
                        help='The std of driving noise',
                        type = float,
                        default = 0.05)
    parser.add_argument('--meas_noise_std',
                        help='The std of measurement noise',
                        type = float,
                        default = 1)
    parser.add_argument('--num_samples',
                        help='The number of samples to generate',
                        type = int,
                        default = 1)
    parser.add_argument('--batch_size',
                        help='The size of batch',
                        type = int,
                        default = 1)
    parser.add_argument('--use_cuda',
                        help = 'Flag of using cuda',
                        action = 'store_true')
    parser.add_argument('--result_path',
                        help = 'The path to save the localization results',
                        default = None)
    args = parser.parse_args()

    if args.use_cuda:
        device = torch.device('cuda')
        extras = {"num_workers": 4, "pin_memory": True}
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        extras = {"num_workers": 4, "pin_memory": False}
        print("Using CPU")

    track_style = 3
    is_drag = True if track_style == 3 else False
    # is_drag = False

    num_steps = args.num_steps
    num_agents = args.num_agents
    P0 = torch.diag(torch.tensor([args.pos_prior_cov] * 2 + [args.vel_prior_cov] * 2))
    sigma_driving = args.driving_noise_std
    sigma_meas_inference = args.meas_noise_std
    sigma_meas_data = args.meas_noise_std
    size = args.num_samples
    batch_size = args.batch_size


    xmin = 10
    xmax = 91
    ymin = 10
    ymax = 91

    anchor_range = np.array([[xmin, xmax], [xmin, xmax]])

    seeds = {'test': 1001, 'train': 1009, 'val': 51}

    dataset, data_loader = prepare_data(num_steps = num_steps,
                                        num_agents = num_agents,
                                        P0 = P0,
                                        sigma_driving = sigma_driving,
                                        sigma_meas = sigma_meas_data,
                                        anchor_range = anchor_range,
                                        size = size,
                                        batch_size = batch_size,
                                        partition = 'test',
                                        track_style = track_style,
                                        seeds = seeds)

    random_seed = 100

    num_mc = 1
    pred_bp_list = []
    true_bp_list = []
    rmse_bp_list = []
    cov_tr_bp_list = []
    cov_bp_list = []
    particle_single_bp_list = []

    rmse_bp = torch.zeros(num_mc, size, num_agents, num_steps)
    cov_tr_bp = torch.zeros(num_mc, size, num_agents, num_steps)

    print('--------------------------------')
    print('Running bp_test')
    print('--------------------------------')
    num_particles = 50000
    for i in range(num_mc):
        pred, true, rmse, cov_tr, cov, particle_single = bp_test(data_loader = data_loader,
                                           random_seed = random_seed + i,
                                           num_steps = num_steps,
                                           num_agents = num_agents,
                                           num_particles = num_particles,
                                           P0 = P0,
                                           sigma_driving = sigma_driving,
                                           sigma_meas = sigma_meas_inference,
                                           is_drag = is_drag,
                                                                 plot = False,
                                                                 device = device)
        pred_bp_list.append(pred)
        true_bp_list.append(true)
        rmse_bp_list.append(rmse)
        cov_tr_bp_list.append(cov_tr)
        cov_bp_list.append(cov)
        particle_single_bp_list.append(particle_single)

    pred_bp = torch.stack(pred_bp_list)
    true_bp = torch.stack(true_bp_list)
    rmse_bp = torch.stack(rmse_bp_list)
    cov_tr_bp = torch.stack(cov_tr_bp_list)
    cov_bp = torch.stack(cov_bp_list)
    particle_single_bp = torch.stack(particle_single_bp_list)

    if args.result_path is not None:
        torch.save({'pred': pred_bp,
                    'true': true_bp,
                    'rmse': rmse_bp,
                    'cov_tr': cov_tr_bp,
                    'cov': cov_bp,
                    'particle_single': particle_single_bp},
                   args.result_path)

    print('rmse of bp is {:.4f}'.format(torch.mean(rmse_bp)))
    print('mse of bp is {:.4f}'.format(torch.mean(rmse_bp ** 2)))
    print('Mean cov_tr of bp is {:.4f}'.format(torch.mean(cov_tr_bp)))

if __name__ == '__main__':
    main()
