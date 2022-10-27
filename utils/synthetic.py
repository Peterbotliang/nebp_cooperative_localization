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
import networkx as nx

class synthetic_dataset(Dataset):

    def __init__(self,
                 partition = 'train',
                 track_style = 1,
                 num_steps = 100,
                 train_size = 1000,
                 val_size = 1000,
                 test_size = 1000,
                 anchor_range = None,
                 num_agents = 3,
                 drag_coeff = 0.06,
                 P0 = np.eye(4) * 0.3,
                 sigma_driving = 0.05,
                 sigma_meas = 0.3,
                 is_random_noise = False,
                 start_after = 0,
                 seeds = None):

        assert partition in ['train', 'val', 'test']
        assert isinstance(P0, np.ndarray) or isinstance(P0, torch.Tensor)

        self.partition = partition
        self.track_style = track_style
        self.num_steps = num_steps

        if seeds is None:
            self.seeds = {'test': 0, 'train': 50, 'val': 51}
        else:
            self.seeds = seeds
        self.size = {'test': test_size, 'train': train_size, 'val': val_size}

        if anchor_range is None:
            self.anchor_range = np.array([[0, 200], [0, 200]])
        else:
            self.anchor_range = anchor_range

        self.drag_coeff = drag_coeff
        self.num_agents = num_agents
        self.num_max_partner_anchor = 4
        self.P0 = P0
        self.sigma_driving = sigma_driving
        self.sigma_meas = sigma_meas
        self.is_random_noise = is_random_noise

        np.random.seed(self.seeds[self.partition])
        if self.partition == 'train':
            self.data = [self._generate_sample(self.num_steps, start_after = start_after) for _ in range(self.size[self.partition])]
        elif self.partition == 'val':
            self.data = [self._generate_sample(self.num_steps, start_after = start_after) for _ in range(self.size[self.partition])]
        elif self.partition == 'test':
            self.data = [self._generate_sample(self.num_steps, start_after = start_after) for _ in range(self.size[self.partition])]

        self.data = [i for i in self.data if i]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        g_list, states, anchor_pos, x_prior = self.data[ind]
        return self.data[ind]

    def _generate_sample(self, num_steps, start_after = 0):
        # np.random.seed(seed)

        xmin = self.anchor_range[0, 0]
        xmax = self.anchor_range[0, 1]
        ymin = self.anchor_range[1, 0]
        ymax = self.anchor_range[1, 1]

        F, Q, _, R, W = get_model_parameters(sigma_driving = self.sigma_driving,
                                             sigma_meas = self.sigma_meas)
        if self.track_style == 1 or self.track_style == 3:
            anchor_pos = get_anchor_pos(self.anchor_range)
            # anchor_pos = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin]])
            # anchor_pos = np.array([[xmin, ymax], [xmax, ymin]])
            # anchor_pos = np.array([[xmin, ymin]])
        elif self.track_style == 2:
            anchor_pos = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin]])
            shift = np.random.rand()
            is_x = np.random.rand() > 0.5

        state_agents_list = []
        x0_list = []
        for agent in range(self.num_agents):
            if self.track_style == 1:
                x0 = np.hstack([np.random.rand(2) * 40 + np.array([(xmax - xmin) / 2, (ymax - ymin) / 2]),
                                np.random.multivariate_normal(np.zeros(2), self.P0[2 :, 2 :])])

            elif self.track_style == 2:
                x0_px = np.random.rand() * 140 + shift * 60
                x0_py = np.random.randn() * 0.1
                x0_vx = np.random.randn() * 0
                x0_vy = 1 + np.random.randn() * 0
                # if is_x:
                #     x0 = np.array([x0_px, x0_py, x0_vx, x0_vy])
                # else:
                #     x0 = np.array([x0_py, x0_px, x0_vy, x0_vx])

                if agent < self.num_agents // 2:
                    x0 = np.array([x0_px, x0_py, x0_vx, x0_vy])
                else:
                    x0 = np.array([x0_py, x0_px, x0_vy, x0_vx])

            elif self.track_style == 3:
                x0 = np.hstack([np.random.rand(2) * (xmax - xmin - 0) + xmin + 0,
                                np.random.multivariate_normal(np.zeros(2), self.P0[2 :, 2 :])])
                # x0_px = np.random.rand(1) * (xmax - xmin - 0) + xmin + 0
                # x0_py = np.random.rand(1) * (ymax - ymin - 0) + ymin + 0
                # x0_vx = (0.5 if x0_px < (xmax + xmin) / 2 else -0.5) + np.random.randn() * self.P0[2, 2] ** 0.5
                # x0_vy = (0.5 if x0_py < (ymax + ymin) / 2 else -0.5) + np.random.randn() * self.P0[3, 3] ** 0.5
                # x0 = np.hstack([x0_px, x0_py, x0_vx, x0_vy])


            x0_list.append(x0.copy())
            if self.track_style == 3:
                state = get_tracks(F, Q, W, x0 = x0, is_drag = True, drag_coeff = self.drag_coeff, num_steps = self.num_steps)
            else:
                state = get_tracks(F, Q, W, x0 = x0, num_steps = self.num_steps)
            state_agents_list.append(state)

        states = np.array(state_agents_list)
        if self.track_style == 1:
            measurements = get_measurements(states, anchor_pos, R,
                                            anchor_connectivity = 20,
                                            agent_connectivity = np.inf)
        elif self.track_style == 2:
            measurements = get_measurements(states, anchor_pos, R,
                                            anchor_connectivity = 140,
                                            agent_connectivity = 140)
        elif self.track_style == 3:
            if self.partition == 'train':
                measurements = get_measurements(states, anchor_pos, R,
                                                anchor_connectivity = 20,
                                                agent_connectivity = 20,
                                                is_random_noise = self.is_random_noise)
            else:
                measurements = get_measurements(states, anchor_pos, R,
                                                anchor_connectivity = 20,
                                                agent_connectivity = 20)

        g_list = []
        for step in range(self.num_steps):
            g = dgl.graph((range(0), range(0)))
            g.add_nodes(self.num_agents, {'anchor_pos': torch.zeros(self.num_agents, self.num_max_partner_anchor, 2),
                                     'meas': torch.zeros(self.num_agents, self.num_max_partner_anchor),
                                     'mask': torch.ones(self.num_agents, self.num_max_partner_anchor)})
            for agent in range(self.num_agents):
                meas_anchor = measurements['anchor'][step][agent].astype(np.float32)
                meas_anchor_ind = measurements['anchor_ind'][step][agent]
                meas_agent = measurements['agent'][step][agent].astype(np.float32)
                meas_agent_ind = measurements['agent_ind'][step][agent]

                num_partner_anchor = meas_anchor_ind.shape[0]

                g.ndata['anchor_pos'][agent, : num_partner_anchor, :] = torch.from_numpy(anchor_pos[meas_anchor_ind, :].astype(np.float32))
                g.ndata['meas'][agent, : num_partner_anchor] = torch.from_numpy(meas_anchor)
                g.ndata['mask'][agent, num_partner_anchor :] = 0

                num_partner_agent = meas_agent_ind.shape[0]
                g.add_edges(agent, meas_agent_ind, {'meas': torch.from_numpy(meas_agent).unsqueeze(1)})

                # if num_partner_anchor + num_partner_agent == 0:
                #     return None

            # if g.num_edges() > 2000:
            #     return None
            g_list.append(g)

        x_prior = np.array(x0_list) + np.random.multivariate_normal(np.zeros(4), self.P0, size = self.num_agents)
        # x_prior[:, 2 :] = 0

        # g_list.reverse()
        # x_prior = states[:, :, -1] + np.random.multivariate_normal(np.zeros(4), self.P0, size = self.num_agents)
        # x_prior[:, 2 :] = 0
        # states = np.flip(states, axis = 2)

        return g_list, torch.from_numpy(states.astype(np.float32)), torch.from_numpy(anchor_pos.astype(np.float32)), torch.from_numpy(x_prior.astype(np.float32))


def get_model_parameters(delta_t = 1.,
                         sigma_driving = 0.05,
                         sigma_meas = 0.3):

    F = np.eye(4)
    F[0, 2] = F[1, 3] = delta_t

    W = np.zeros((4, 2))
    W[0, 0] = W[1, 1] = 0.5 * delta_t ** 2
    W[2, 0] = W[3, 1] = delta_t

    Q = W @ W.T * sigma_driving ** 2

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    R = np.eye(1) * sigma_meas ** 2

    return F, Q, H, R, W


def get_tracks(F, Q, W, x0 = None, num_steps = 100, is_drag = False, drag_coeff = 0.06, start_after = 0):

    dim_state = Q.shape[0]

    if x0 is None:
        x0 = np.zeros(dim_state)

    states_list = [x0.copy()]
    for _ in range(num_steps + start_after):
        x_last = states_list[-1]
        if is_drag:
            v_last = x_last[2:]
            x = F @ x_last + W @ (-drag_coeff * (v_last ** 3 / np.abs(v_last)))+ np.random.multivariate_normal(np.zeros(dim_state), Q)
        else:
            x = F @ x_last + np.random.multivariate_normal(np.zeros(dim_state), Q)

        states_list.append(x)

    states = np.array(states_list[1 : ])[start_after :, :]

    return states.T

def get_measurements(states, anchor_pos, R,
                     anchor_connectivity = 20,
                     agent_connectivity = 20,
                     is_random_noise = False):

    num_agents, dim_state, num_steps = states.shape
    _, num_anchors = anchor_pos.shape
    dim_meas = R.shape[0]

    meas_anchor = []
    meas_anchor_ind = []
    meas_agent = []
    meas_agent_ind = []

    for step in range(num_steps):
        meas_anchor_step = []
        meas_anchor_ind_step = []
        meas_agent_step = []
        meas_agent_ind_step = []

        for agent in range(num_agents):
            # anchor distance
            true_distances = np.sum((states[agent, 0 : 2, step][np.newaxis, :] - anchor_pos) ** 2, axis = 1) ** 0.5

            meas_ind = np.argwhere(true_distances <= anchor_connectivity).squeeze(axis = 1)
            if is_random_noise:
                meas = true_distances[meas_ind] + np.random.multivariate_normal(mean = np.zeros(dim_meas), cov = (R ** 0.5 * np.random.rand()) ** 2, size = meas_ind.shape).squeeze()
            else:
                meas = true_distances[meas_ind] + np.random.multivariate_normal(mean = np.zeros(dim_meas), cov = R, size = meas_ind.shape).squeeze()

            meas_anchor_ind_step.append(np.array(meas_ind))
            meas_anchor_step.append(np.array(meas))


            # agent distance
            true_distances = np.sum((states[agent, 0 : 2, step][np.newaxis, :] - states[:, 0 : 2, step]) ** 2, axis = 1) ** 0.5

            mask = np.ones(num_agents, dtype = np.bool)
            mask[agent] = False
            meas_ind = np.argwhere(np.logical_and(true_distances <= agent_connectivity, mask)).squeeze(axis = 1)
            meas = true_distances[meas_ind] + np.random.multivariate_normal(mean = np.zeros(dim_meas), cov = R, size = meas_ind.shape).squeeze()

            meas_agent_ind_step.append(np.array(meas_ind))
            meas_agent_step.append(np.array(meas))

        meas_anchor.append(meas_anchor_step)
        meas_anchor_ind.append(meas_anchor_ind_step)
        meas_agent.append(meas_agent_step)
        meas_agent_ind.append(meas_agent_ind_step)

    measurements = {}
    measurements['anchor'] = meas_anchor
    measurements['anchor_ind'] = meas_anchor_ind
    measurements['agent'] = meas_agent
    measurements['agent_ind'] = meas_agent_ind

    return measurements

def get_anchor_pos(anchor_range, distance = 20):

    assert isinstance(anchor_range, np.ndarray)
    assert anchor_range.shape == (2, 2)

    xmin = anchor_range[0, 0]
    xmax = anchor_range[0, 1]
    ymin = anchor_range[1, 0]
    ymax = anchor_range[1, 1]

    anchor_pos_list = []
    for y in np.arange(ymin, ymax, 2 * distance):
        for x in np.arange(xmin, xmax, 2 * distance):
            anchor_pos_list.append(np.array([x, y]))

    for y in np.arange(ymin + distance, ymax, 2 * distance):
        for x in np.arange(xmin + distance, xmax, 2 * distance):
            anchor_pos_list.append(np.array([x, y]))

    # size (dim_pos, num_anchors)
    return np.array(anchor_pos_list)

def collate(samples):
    # The input `samples` is a list of pairs
    g_list, states, anchor_pos, x_prior  = map(list, zip(*samples))

    batched_g_list = [dgl.batch([g_list[i][step] for i in range(len(g_list))]) for step in range(len(g_list[0]))]

    return dgl.batch(batched_g_list), torch.stack(states), torch.stack(anchor_pos), torch.stack(x_prior)

if __name__ == '__main__':
    num_max_partner_anchor = 10
    num_agents = 25
    num_steps = 50
    xmin = 10
    xmax = 51
    ymin = 10
    ymax = 51
    anchor_range = np.array([[xmin, xmax], [xmin, xmax]])
    anchor_pos = get_anchor_pos(anchor_range)
    P0 = 0.3 * np.eye(4)
    seeds = {'test': 1001, 'train': 1009, 'val': 51}

    dataset = synthetic_dataset(partition = 'train',
                                track_style = 3,
                                num_steps = num_steps,
                                train_size = 10,
                                val_size = 1000,
                                test_size = 2,
                                anchor_range = anchor_range,
                                num_agents = num_agents,
                                P0 = P0,
                                start_after = 0,
                                seeds = seeds)
    data_loader = DataLoader(dataset, batch_size = 1, shuffle = False,
                             collate_fn = collate)

    # del dataset.data[5]
    # for ind, (_, _, _, _) in enumerate(data_loader):
    #     print(ind)

    g_list, states, anchor_pos, x_prior = dataset.data[0]
    g = g_list[0]
    print(x_prior.shape)

    plt.figure()
    nx.draw(g.to_networkx(), with_labels=True)

    fig, ax = plt.subplots()
    for i in range(num_agents):
        ax.plot(states[i, 0, :], states[i, 1, :], color = 'C0')
        ax.scatter(states[i, 0, -1], states[i, 1, -1], color = 'C0')
    ax.scatter(anchor_pos[:, 0], anchor_pos[:, 1], color = 'k')
    ax.scatter(x_prior[:, 0], x_prior[:, 1], color = 'r')
    for i in range(anchor_pos.shape[0]):
        circle = plt.Circle((anchor_pos[i, 0], anchor_pos[i, 1]), 20, fill = False)
        ax.add_patch(circle)
    ax.set_xlim([-10, 70])
    ax.set_ylim([-10, 70])

    g_list, states, anchor_pos, x_prior = dataset.data[1]
    # for g in g_list:
    #     print(g.edata['meas'].shape)
    fig, ax = plt.subplots()
    for i in range(num_agents):
        ax.plot(states[i, 0, :], states[i, 1, :], color = 'C0')
        ax.scatter(states[i, 0, -1], states[i, 1, -1], color = 'C0')
    plt.scatter(anchor_pos[:, 0], anchor_pos[:, 1], color = 'k')
    for i in range(anchor_pos.shape[0]):
        circle = plt.Circle((anchor_pos[i, 0], anchor_pos[i, 1]), 20, fill = False)
        ax.add_patch(circle)
    ax.set_xlim([-10, 70])
    ax.set_ylim([-10, 70])
    plt.show()
