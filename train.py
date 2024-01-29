# ------------------------------------------------------------------------
# Cooperative Localization
# Copyright (c) 2022 MIngchao Liang. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader

import dgl
import dgl.function as fn

import argparse

from tqdm import tqdm

from utils.synthetic import synthetic_dataset, get_model_parameters
from utils.cooperative_localization import cooperative_localization_solver

def collate(samples):
    # The input `samples` is a list of pairs
    g_list, states, anchor_pos, x_prior  = map(list, zip(*samples))

    batched_g_list = [dgl.batch([g_list[i][step] for i in range(len(g_list))]) for step in range(len(g_list[0]))]

    return dgl.batch(batched_g_list), torch.stack(states), torch.stack(anchor_pos), torch.stack(x_prior)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps',
                        help='The number of time steps in each sample',
                        type = int,
                        default=50)
    parser.add_argument('--num_agents',
                        help='The number of agents in each sample',
                        type = int,
                        default = 25)
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
    parser.add_argument('--train_size',
                        help='The number of samples in training_set',
                        type = int,
                        default = 100)
    parser.add_argument('--val_size',
                        help='The number of samples in training_set',
                        type = int,
                        default = 10)
    parser.add_argument('--num_epochs',
                        help='The number of epochs',
                        type = int,
                        default = 10)
    parser.add_argument('--lr',
                        help='Learning rate',
                        type = int,
                        default = 1e-4)
    parser.add_argument('--batch_size',
                        help='The size of batch',
                        type = int,
                        default = 2)
    parser.add_argument('--use_cuda',
                        help = 'Flag of using cuda',
                        action = 'store_true')
    parser.add_argument('--output_dir',
                        help = 'The path to save the localization results',
                        default = './results')
    args = parser.parse_args()

    random_seed = 100
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.set_printoptions(sci_mode = False)

    if not os.path.exists('./Figs'):
        os.mkdir('./Figs')
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./Figs/train'):
        os.mkdir('./Figs/train')
    if not os.path.exists('./Figs/val'):
        os.mkdir('./Figs/val')

    if args.use_cuda:
        device = torch.device("cuda")
        extras = {"num_workers": 3, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        device = torch.device("cpu")
        extras = {"num_workers": 3, "pin_memory": False}
        print("CUDA NOT supported")

    num_steps = args.num_steps
    num_agents = args.num_agents
    P0 = torch.diag(torch.tensor([args.pos_prior_cov] * 2 + [args.vel_prior_cov] * 2))
    sigma_driving = args.driving_noise_std
    sigma_meas_inference = args.meas_noise_std
    sigma_meas_data = args.meas_noise_std
    sigma_meas = sigma_meas_data
    train_size = args.train_size
    val_size = args.val_size
    test_size = 1
    num_epochs = args.num_epochs
    lr = args.lr
     = args.
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    track_style = 3
    is_drag = True if track_style == 3 else False
    num_particles = 50000
    is_random_noise = False

    xmin = 10
    xmax = 51
    ymin = 10
    ymax = 51
    anchor_range = np.array([[xmin, xmax], [xmin, xmax]])
    P0 = 10 * torch.eye(4)
    P0[2, 2] = P0[3, 3] = 0.01
    seeds = {'test': 1001, 'train': 1009, 'val': 51}

    F, Q, _, R, W = get_model_parameters(sigma_driving = sigma_driving,
                                         sigma_meas = sigma_meas)
    F, Q, R, W = torch.from_numpy(F.astype(np.float32)), torch.from_numpy(Q.astype(np.float32)), torch.from_numpy(R.astype(np.float32)), torch.from_numpy(W.astype(np.float32))

    dataset_train = synthetic_dataset(partition = 'train',
                                      track_style = track_style,
                                      num_steps = num_steps,
                                      train_size = train_size,
                                      val_size = 1000,
                                      test_size = 1000,
                                      anchor_range = anchor_range,
                                      num_agents = num_agents,
                                      P0 = P0,
                                      sigma_driving = sigma_driving,
                                      sigma_meas = sigma_meas,
                                      is_random_noise = is_random_noise,
                                      start_after = 0,
                                      seeds = seeds)

    train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, collate_fn = collate, num_workers = 4, pin_memory = extras['pin_memory'])

    dataset_val = synthetic_dataset(partition = 'val',
                                    track_style = track_style,
                                    num_steps = num_steps,
                                    train_size = 10,
                                    val_size = val_size,
                                    test_size = test_size,
                                    anchor_range = anchor_range,
                                    num_agents = num_agents,
                                    P0 = P0,
                                    sigma_driving = sigma_driving,
                                    sigma_meas = sigma_meas,
                                    start_after = 0,
                                    seeds = seeds)
    val_loader = DataLoader(dataset_val, batch_size = batch_size, shuffle = False, collate_fn = collate, num_workers = 4, pin_memory = extras['pin_memory'])

    dim_node = 20
    dim_edge = 32
    num_steps_grad = num_steps // num_steps
    beta = 0
    gamma = 4
    cl_solver = cooperative_localization_solver(F = F, Q = Q, W = W,
                                                sigma_meas = sigma_meas,
                                                num_particles = num_particles,
                                                hybrid = True,
                                                dim_node = dim_node,
                                                dim_edge = dim_edge)
    cl_solver = cl_solver.to(device)
    optimizer = torch.optim.Adam(cl_solver.parameters(), lr = lr)

    # --------------------------------
    # Train
    # --------------------------------
    for epoch_ind in tqdm(range(num_epochs)):
        rmse_list = []
        cl_solver.train()
        for minibatch_count, (bbg, states, anchor_pos, x_prior) in enumerate(train_loader):

            batch_size, _, dim_state, _ = states.shape
            g_list = dgl.unbatch(bbg)
            bg_list = [dgl.batch(g_list[i: i + batch_size]) for i in range(0, len(g_list), batch_size)]
            states = states.to(device)

            particles = cl_solver.init_particles(x_prior.reshape(-1, dim_state), P0)
            particles = particles.to(device)


            for start_step in range(0, num_steps, num_steps_grad):
                estimated_states_list = []
                estimated_cov_list = []

                for step, bg in enumerate(bg_list[start_step: start_step + num_steps_grad]):
                    bg = bg.to(device)
                    estimated_mean, particles, estimated_cov = cl_solver.perform_estimation(bg, cl_solver.particle_predict(particles, is_drag = is_drag), num_iter = 1)
                    estimated_states_list.append(estimated_mean)
                    estimated_cov_list.append(estimated_cov)
                estimated_states = torch.stack(estimated_states_list).permute(1, 2, 0).reshape(batch_size, num_agents, dim_state, num_steps_grad)
                estimated_cov = torch.stack(estimated_cov_list)
                estimated_cov_tr = torch.sum(torch.diagonal(estimated_cov, dim1 = 2, dim2 = 3).reshape(num_steps_grad, batch_size, num_agents, dim_state), dim = (2, 3))

                mse_v = (estimated_states[:, :, :, :] - states[:, :, :, start_step : start_step + num_steps_grad]) ** 2
                mse_v_2d = mse_v[:, :, :2, :].permute(3, 0, 1, 2)
                Sigma_2d = estimated_cov.reshape(num_steps_grad, batch_size, num_agents, dim_state, dim_state)[:, :, :, :2, :2]
                m_dist = torch.matmul(torch.matmul(mse_v_2d.unsqueeze(-2), torch.inverse(Sigma_2d)), mse_v_2d.unsqueeze(-1))
                alpha = m_dist / 9.21
                regularizer = torch.mean(func.threshold(alpha - 1, threshold = 0, value = 0) * torch.sum(torch.diagonal(Sigma_2d, dim1 = -2, dim2 = -1), dim = -1))


                mse = torch.sum(mse_v[:, :, :2, :], dim = 2)
                Sigma_diag = torch.diagonal(estimated_cov[:, :, :, :], dim1 = 2, dim2 = 3).reshape(num_steps_grad, batch_size, num_agents, 4).permute(1, 2, 3, 0)
                loss = torch.mean(mse) + beta * torch.mean(torch.sum(func.threshold(mse_v[:, :, :2, :] - gamma * Sigma_diag[:, :, :2, :], threshold = 0, value = 0), dim = 2))
                optimizer.zero_grad()
                rmse_list.append(torch.mean(mse ** 0.5).item())
                loss.backward()
                optimizer.step()


        torch.save(cl_solver, 'models/model_epoch{:02d}_tmp.pt'.format(epoch_ind))
        print('Training loss: {:.4f}'.format(np.mean(rmse_list)))

    # --------------------------------
    # plot train data
    # --------------------------------
    for ind, (_, states, anchor_pos, x_prior) in enumerate(dataset_train.data):
        for agent in range(num_agents):
            plt.plot(states[agent, 0, :], states[agent, 1, :], color = 'C0')
            plt.scatter(states[agent, 0, -1], states[agent, 1, -1], color = 'C0')
        plt.scatter(anchor_pos[:, 0], anchor_pos[:, 1], color = 'k')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        # plt.legend(loc = 'best')
        plt.tight_layout(pad = 0)
        plt.savefig('Figs/train/{}.eps'.format(ind), format = 'eps')
        plt.close()

    torch.save(cl_solver, 'models/model_final_tmp.pt')


    # --------------------------------
    # Validate
    # --------------------------------
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    pred_list = []
    true_list = []
    cov_tr_list = []
    rmse_list = []

    with torch.no_grad():
        cl_solver.eval()
        for minibatch_count, (bbg, states, anchor_pos, x_prior) in enumerate(val_loader):
            batch_size, _, dim_state, _ = states.shape
            g_list = dgl.unbatch(bbg)
            bg_list = [dgl.batch(g_list[i: i + batch_size]) for i in range(0, len(g_list), batch_size)]
            states = states.to(device)

            particles = cl_solver.init_particles(x_prior.reshape(-1, dim_state), P0)
            particles = particles.to(device)

            estimated_states_list = []
            estimated_cov_list = []
            for step, bg in enumerate(tqdm(bg_list)):
                bg = bg.to(device)
                estimated_mean, particles, estimated_cov = cl_solver.perform_estimation(bg, cl_solver.particle_predict(particles, is_drag = is_drag), num_iter = 1)
                estimated_states_list.append(estimated_mean)
                estimated_cov_list.append(estimated_cov)

            estimated_states = torch.stack(estimated_states_list).detach().cpu().permute(1, 2, 0).reshape(batch_size, num_agents, dim_state, num_steps)
            estimated_cov = torch.stack(estimated_cov_list).detach().cpu()
            estimated_cov_tr = torch.sum(torch.diagonal(estimated_cov, dim1 = 2, dim2 = 3).reshape(num_steps, batch_size, num_agents, dim_state)[:, :, :, :2], dim = 3)

            rmse = torch.sum((estimated_states - states.cpu())[:, :, :2, :] ** 2, dim = 2) ** 0.5

            pred_list.append(estimated_states)
            true_list.append(states.cpu())
            cov_tr_list.append(estimated_cov_tr.permute(1, 2, 0))
            rmse_list.append(rmse)

    pred = torch.cat(pred_list, dim = 0)
    true = torch.cat(true_list, dim = 0)
    cov_tr = torch.cat(cov_tr_list, dim = 0)
    rmse = torch.cat(rmse_list, dim = 0)

    print('rmse of hybrid_val is {:.4f}'.format(torch.mean(rmse).item()))
    print('mse of hybrid_val is {:.4f}'.format(torch.mean(rmse ** 2).item()))
    print('Mean cov_tr of hybrid is {:.4f}'.format(torch.mean(cov_tr).item()))

    torch.save({'pred': pred,
                'true': true,
                'cov_tr': cov_tr,
                'rmse': rmse},
               os.path.join(output_dir, 'result_hybrid_val.pt'))

    for i in range(val_size):
        plt.figure()
        for agent in range(num_agents):
            plt.plot(true[i, agent, 0, :], true[i, agent, 1, :], color = 'C0')
            plt.scatter(true[i, agent, 0, -1], true[i, agent, 1, -1], color = 'C0')
            plt.plot(pred[i, agent, 0, :], pred[i, agent, 1, :], color = 'C1')
            plt.scatter(pred[i, agent, 0, -1], pred[i, agent, 1, -1], color = 'C1')
        plt.scatter(anchor_pos[0, :, 0], anchor_pos[0, :, 1], color = 'k')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.savefig('Figs/val/{}_hybrid.eps'.format(i), format = 'eps')
        plt.close()

