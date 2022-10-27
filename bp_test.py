import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader

import dgl
import dgl.function as fn

from tqdm import tqdm

from utils.synthetic import synthetic_dataset, get_model_parameters
from utils.cooperative_localization import cooperative_localization_solver
from utils.prepare_data import prepare_data

def bp_test(data_loader,
            random_seed,
            num_steps,
            num_agents,
            num_particles,
            P0,
            sigma_driving = 0.05,
            sigma_meas = 0.3,
            is_save = False,
            is_drag = False,
            plot = True,
            device = None):
    print('seed is {}'.format(random_seed))
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.set_printoptions(sci_mode = False)

    if device is None:
        device = torch.device('cpu')

    # sigma_driving = 0.05
    # sigma_meas = 0.3

    F, Q, _, R, W = get_model_parameters(sigma_driving = sigma_driving,
                                         sigma_meas = sigma_meas)
    F, Q, R, W = torch.from_numpy(F.astype(np.float32)), torch.from_numpy(Q.astype(np.float32)), torch.from_numpy(R.astype(np.float32)), torch.from_numpy(W.astype(np.float32))

    cl_solver = cooperative_localization_solver(F, Q, W,
                                                sigma_meas = sigma_meas,
                                                num_particles = num_particles,
                                                hybrid = False)
    cl_solver = cl_solver.to(device)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    pred_list = []
    true_list = []
    particle_single_list = []
    cov_list = []
    cov_tr_list = []
    rmse_list = []
    with torch.no_grad():
        cl_solver.eval()
        for minibatch_count, (bbg, states, anchor_pos, x_prior) in enumerate(data_loader):
            batch_size, _, dim_state, _ = states.shape
            bbg = bbg.to(device)
            g_list = dgl.unbatch(bbg)
            bg_list = [dgl.batch(g_list[i: i + batch_size]) for i in range(0, len(g_list), batch_size)]
            states = states.to(device)

            particles = cl_solver.init_particles(x_prior.reshape(-1, dim_state), P0)
            particles = particles.to(device)

            estimated_states_list = []
            estimated_cov_list = []
            particle_single_temp_list = []
            for step, bg in enumerate(tqdm(bg_list)):
                bg = bg.to(device)
                estimated_mean, particles, estimated_cov = cl_solver.perform_estimation(bg, cl_solver.particle_predict(particles, is_drag = is_drag), num_iter = 1)
                estimated_states_list.append(estimated_mean)
                estimated_cov_list.append(estimated_cov)
                particle_single_temp_list.append(particles[:, :, 0].clone())


            estimated_states = torch.stack(estimated_states_list).cpu().permute(1, 2, 0).reshape(batch_size, num_agents, dim_state, num_steps)
            estimated_cov = torch.stack(estimated_cov_list).cpu()
            particle_single = torch.stack(particle_single_temp_list).cpu().permute(1, 2, 0).reshape(batch_size, num_agents, dim_state, num_steps)


            estimated_cov_tr = torch.sum(torch.diagonal(estimated_cov, dim1 = 2, dim2 = 3).reshape(num_steps, batch_size, num_agents, dim_state)[:, :, :, :2], dim = 3).permute(1, 2, 0)

            rmse = torch.sum((estimated_states[:, :, :2, :] - states.cpu()[:, :, :2, :]) ** 2, dim = 2) ** 0.5

            pred_list.append(estimated_states)
            true_list.append(states.cpu())
            cov_list.append(estimated_cov.reshape(num_steps, batch_size, num_agents, dim_state, dim_state).permute(1, 2, 3, 4, 0))
            cov_tr_list.append(estimated_cov_tr)
            rmse_list.append(rmse)
            particle_single_list.append(particle_single)

    pred = torch.cat(pred_list, dim = 0)
    true = torch.cat(true_list, dim = 0)
    cov = torch.cat(cov_list, dim = 0)
    cov_tr = torch.cat(cov_tr_list, dim = 0)
    rmse = torch.cat(rmse_list, dim = 0)
    particle_single = torch.cat(particle_single_list, dim = 0)

    # print('Mean cov_cond of bp_test is {:.4f}'.format(torch.mean(cov_cond)))

    if is_save:
        torch.save({'pred': pred,
                    'true': true,
                    'anchor_pos': anchor_pos,
                    'cov_tr': cov_tr,
                    'rmse': rmse},
                   './Results_temp/result_bp.pt')

    if plot:
        for i in range(true.shape[0]):
            plt.figure()
            for agent in range(num_agents):
                plt.plot(true[i, agent, 0, :], true[i, agent, 1, :], color = 'C0')
                plt.scatter(true[i, agent, 0, -1], true[i, agent, 1, -1], color = 'C0')
                plt.plot(pred[i, agent, 0, :], pred[i, agent, 1, :], color = 'C1')
                plt.scatter(pred[i, agent, 0, -1], pred[i, agent, 1, -1], color = 'C1')
            plt.scatter(anchor_pos[0, :, 0], anchor_pos[0, :, 1], color = 'k')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            # plt.legend(loc = 'best')
            plt.tight_layout(pad = 0)
            plt.savefig('Figs/test/{}_bp.eps'.format(i), format = 'eps')
            plt.close()
        # plt.show()

    return pred, true, rmse, cov_tr, cov, particle_single

if __name__ == '__main__':

    num_steps = 100
    num_agents = 3
    size = 2
    num_particles = 50000
    P0 = 10 * torch.eye(4)
    random_seed = 100
    dataset, data_loader = prepare_data(num_steps = num_steps,
                               num_agents = num_agents,
                               P0 = P0,
                               size = size,
                               partition = 'test')
    bp_test(data_loader = data_loader,
            random_seed = random_seed,
            num_steps = num_steps,
            num_agents = num_agents,
            num_particles = num_particles,
            P0 = P0, plot = True)
