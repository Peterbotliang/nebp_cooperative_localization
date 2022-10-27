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

from tqdm import tqdm

from synthetic import synthetic_dataset, get_model_parameters


class cooperative_localization_solver(nn.Module):

    def __init__(self, F, Q, W, sigma_meas,
                 num_particles = 1500,
                 dim_node = 32,
                 dim_edge = None,
                 hybrid = False):

        super(cooperative_localization_solver, self).__init__()

        self.F, self.Q, self.W = F, Q, W
        self.sigma_meas = sigma_meas

        if isinstance(F, np.ndarray):
            self.F = torch.from_numpy(F.astype(np.float32))
        if isinstance(Q, np.ndarray):
            self.Q = torch.from_numpy(Q.astype(np.float32))
        if isinstance(W, np.ndarray):
            self.W = torch.from_numpy(W.astype(np.float32))

        self.num_particles = num_particles
        self.hybrid = hybrid

        self.dim_node = dim_node
        if dim_edge is None:
            self.dim_edge = num_particles
        else:
            self.dim_edge = dim_edge


    def _apply(self, fn):
        super(cooperative_localization_solver, self)._apply(fn)
        self.F = fn(self.F)
        self.Q = fn(self.Q)
        self.W = fn(self.W)
        return self

    def init_particles(self, x_prior, P0):
        rv = torch.distributions.multivariate_normal.MultivariateNormal(loc = torch.zeros_like(x_prior), covariance_matrix = P0)
        particles = (x_prior.unsqueeze(2) + rv.sample(sample_shape = (self.num_particles,)).permute(1, 2, 0))

        return particles


    def calculate_weight_anchor(self, nodes):
        anchor_pos_agent = nodes.data['anchor_pos']
        meas = nodes.data['meas']
        mask = nodes.data['mask']
        particles = nodes.data['particles']

        predicted_meas = torch.sum((particles[:, : 2, :].unsqueeze(3) - anchor_pos_agent.transpose(1, 2).unsqueeze(2).contiguous()) ** 2, dim = 1) ** 0.5
        weights_log = torch.sum((-(predicted_meas - meas.unsqueeze(1)) ** 2 / (2 * self.sigma_meas ** 2)) * mask.unsqueeze(1), dim = 2)
        weights_log = weights_log - torch.max(weights_log, dim = 1)[0].unsqueeze(1)

        return {'weights_anchor_log': weights_log}

    def calculate_weight_agent(self, edges):
        particles_dst = edges.dst['particles']
        particles_src = edges.src['particles']
        meas = edges.data['meas']
        belief_src_log = edges.src['belief_log']

        predicted_meas = torch.sum((particles_dst[:, : 2, :] - particles_src[:, : 2, :]) ** 2, dim = 1) ** 0.5
        weights = torch.exp(-(predicted_meas - meas) ** 2 / (2 * self.sigma_meas ** 2)) * torch.exp(belief_src_log)

        weights_log = torch.log(weights + torch.finfo(weights.dtype).eps)
        weights_log = weights_log - torch.max(weights_log, dim = 1)[0].unsqueeze(1)

        return {'weights_agent_log': weights_log}


    def calculate_belief(self, nodes):
        return {'belief_log': nodes.data['weights_anchor_log'] + torch.sum(nodes.mailbox['weights_agent_log'], dim = 1)}


    def perform_estimation(self, g, particles, num_iter = 1):
        num_agents, dim_state, _ = particles.shape
        with g.local_scope():
            g.ndata['particles'] = particles
            # g.ndata['particles'] = self.particle_predict(particles)
            g.ndata['belief_log'] = torch.zeros(num_agents, self.num_particles, device = particles.device)

            g.apply_nodes(self.calculate_weight_anchor)

            for _ in range(num_iter):
                g.update_all(self.calculate_weight_agent, self.calculate_belief)
            if g.num_edges() == 0:
                g.ndata['belief_log'] = g.ndata['weights_anchor_log']

            belief_log = g.ndata['belief_log']
            belief = torch.exp(belief_log - torch.max(belief_log, dim = 1)[0].unsqueeze(1))
            belief = belief / torch.sum(belief, dim = 1).unsqueeze(1)

            # estimate mean and covariance
            estimated_mean = torch.sum(g.ndata['particles'] * belief.unsqueeze(1), dim = 2)
            particles_zero_mean = g.ndata['particles'] - estimated_mean.unsqueeze(2)
            estimated_cov = torch.bmm(particles_zero_mean * belief.unsqueeze(1), particles_zero_mean.transpose(1, 2))

            ind = torch.multinomial(belief, self.num_particles, replacement = True)
            particles_resampled = torch.gather(g.ndata['particles'], dim = 2, index = ind.unsqueeze(1).expand(-1, dim_state, -1))

            return estimated_mean, particles_resampled, estimated_cov


    def particle_predict(self, particles, is_drag = False):
        num_agents, dim_state, num_particles = particles.shape
        
        rv = torch.distributions.multivariate_normal.MultivariateNormal(loc = torch.zeros(dim_state, device = particles.device), covariance_matrix = self.Q + torch.eye(dim_state, device = self.Q.device) * 1e-6)
        if is_drag:
            particles_v = particles[:, 2:, :]
            particles_proposal = torch.matmul(particles.transpose(1, 2), self.F.transpose(0, 1)).transpose(1, 2) + torch.matmul(self.W, -0.06 * ((particles_v ** 2) * torch.sign(particles_v))) + rv.rsample(sample_shape=(num_particles, num_agents)).permute(1, 2, 0)
        else:
            particles_proposal = torch.matmul(particles.transpose(1, 2), self.F.transpose(0, 1)).transpose(1, 2) + rv.rsample(sample_shape=(num_particles, num_agents)).permute(1, 2, 0)

        return particles_proposal.to(particles.device)
