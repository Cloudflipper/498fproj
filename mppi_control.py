import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def get_cartpole_mppi_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running MPPI on the cartpole environment
    The required parameters are:
     * lambda: float parameter between 0. and 1. used to weight samples.
     * Q: torch tensor fo shape (state_size, state_size) representing the state quadratic cost.
     * noise_sigma: torch tensor fo size (action_size, action_size) representing the covariance matrix  of the random action perturbations.
    """
    action_size = 1
    state_size = 4
    hyperparams = {
        'action_size': action_size,
        'state_size': state_size,
        'lambda': 0.2,
        'Q': torch.diag(torch.tensor([1.0, 0.1, 1.0,0.1])).float(),
        'noise_sigma': torch.eye(action_size) * 0.5,
    }
    # --- Your code here


    # ---
    return hyperparams


def get_panda_mppi_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running MPPI on the panda environment
    The required parameters are:
     * lambda: float parameter between 0. and 1. used to weight samples.
     * Q: torch tensor fo shape (state_size, state_size) representing the state quadratic cost.
     * noise_sigma: torch tensor fo size (action_size, action_size) representing the covariance matrix  of the random action perturbations.
    """
    action_size = 7
    state_size = 14
    hyperparams = {
        'action_size': action_size,
        'state_size': state_size,
        'lambda': 0.01,
        'Q': torch.diag(torch.tensor([5., 5., 5., 5., 5., 5., 5.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])).float(),
        'noise_sigma': torch.eye(action_size) * 30.0,
    }
    # --- Your code here


    # ---
    return hyperparams


class MPPIController(object):

    def __init__(self, env, num_samples, horizon, hyperparams):
        """

        :param env: Simulation environment. Must have an action_space and a state_space.
        :param num_samples: <int> Number of perturbed trajectories to sample
        :param horizon: <int> Number of control steps into the future
        :param hyperparams: <dic> containing the MPPI hyperparameters
        """
        self.env = env
        self.T = horizon
        self.K = num_samples
        self.lambda_ = hyperparams['lambda']
        self.action_size = env.action_space.shape[-1]
        self.state_size = env.state_space.shape[-1]
        self.goal_state = torch.zeros(self.state_size)  # This is just a container for later use
        self.Q = hyperparams['Q'] # Quadratic Cost Matrix (state_size, state_size)
        self.noise_mu = torch.zeros(self.action_size)
        self.noise_sigma = hyperparams['noise_sigma']  # Noise Covariance matrix shape (action_size, action_size)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.u_init = torch.zeros(self.action_size)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

    def reset(self):
        """
        Resets the nominal action sequence
        :return:
        """
        self.U = torch.zeros((self.T, self.action_size))# nominal action sequence (T, action_size)

    def command(self, state):
        """
        Run a MPPI step and return the optimal action.
        :param state: torch tensor of shape (state_size,)
        :return:
        """
        action = None
        perturbations = self.noise_dist.sample((self.K, self.T))    # shape (K, T, action_size)
        perturbed_actions = self.U + perturbations      # shape (K, T, action_size)
        trajectory = self._rollout_dynamics(state, actions=perturbed_actions)
        trajectory_cost = self._compute_trajectory_cost(trajectory, perturbations)
        self._nominal_trajectory_update(trajectory_cost, perturbations)
        # select optimal action
        action = self.U[0]
        # final update nominal trajectory
        self.U = torch.roll(self.U, -1, dims=0) # move u_t to u_{t-1}
        self.U[-1] = self.u_init # Initialize new end action
        return action

    def _rollout_dynamics(self, state_0, actions):
        """
        Roll out the environment dynamics from state_0 and taking the control actions given by actions
        :param state_0: torch tensor of shape (state_size,)
        :param actions: torch tensor of shape (K, T, action_size)
        :return:
         * trajectory: torch tensor of shape (K, T, state_size) containing the states along the trajectories given by
                       starting at state_0 and taking actions.
                       This tensor contains K trajectories of T length.
         TIP 1: You may need to call the self._dynamics method.
         TIP 2: At most you need only 1 for loop.
        """
        state = state_0.unsqueeze(0).expand(self.K, -1) # transform it to (K, state_size)
        
        # --- Your code here
        trajectory = torch.zeros((self.K, self.T, self.state_size), dtype=torch.float32)
        

        for t in range(self.T):
            
            next_state = self._dynamics(state, actions[:, t, :])
            trajectory[:, t, :] = next_state
            state = next_state

        return trajectory

        # ---

    def _compute_trajectory_cost(self, trajectory, perturbations):
        """
        Compute the costs for the K different trajectories
        :param trajectory: torch tensor of shape (K, T, state_size)
        :param perturbations: torch tensor of shape (K, T, action_size)
        :return:
         - total_trajectory_cost: torch tensor of shape (K,) containing the total trajectory costs for the K trajectories
        Observations:
        * The trajectory cost be the sum of the state costs and action costs along the trajectories
        * State cost should be quadratic as (state_i-goal_state)^T Q (state_i-goal_state)
        * Action costs should be given by (non_perturbed_action_i)^T noise_sigma^{-1} (perturbation_i)

        TIP 1: the nominal actions (without perturbation) are stored in self.U
        TIP 2: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references.
        """
        total_trajectory_cost = None
        # --- Your code here
        K, T, state_size = trajectory.shape
        x_goal = self.goal_state.view(1, 1, -1) 
        state_diff = trajectory - x_goal  # Shape: (K, T, state_size)
        state_cost = torch.einsum('kti,ij,ktj->kt', state_diff, self.Q, state_diff).sum(dim=1)
        
        # Compute action cost: sum over lambda * u_t^T Sigma^{-1} epsilon_t for t=0 to T-1
        Sigma_inv = torch.inverse(self.noise_sigma)
        u_nominal = self.U.unsqueeze(0)  # Shape: (1, T, action_size)
        action_cost_per_t = torch.einsum('kti,ij,ktj->kt', u_nominal, Sigma_inv, perturbations)
        action_cost = self.lambda_ * action_cost_per_t.sum(dim=1)
        
        total_trajectory_cost = state_cost + action_cost
        return total_trajectory_cost

    def _nominal_trajectory_update(self, trajectory_costs, perturbations):
        """
        Update the nominal action sequence (self.U) given the trajectory costs and perturbations
        :param trajectory_costs: torch tensor of shape (K,)
        :param perturbations: torch tensor of shape (K, T, action_size)
        :return: No return, you just need to update self.U

        TIP: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references about the action update.
        """
        # --- Your code here
        beta = torch.min(trajectory_costs)
        gamma = torch.exp(-1 / self.lambda_ * (trajectory_costs - beta))
        yita = torch.sum(gamma)
        omega = gamma / yita
        self.U += torch.sum(omega.view(-1, 1, 1) * perturbations, dim=0) 
        # ---

    def _dynamics(self, state, action):
        """
        Query the environment dynamics to obtain the next_state in a batched format.
        :param state: torch tensor of size (...., state_size)
        :param action: torch tensor of size (..., action_size)
        :return: next_state: torch tensor of size (..., state_size)
        """
        next_state = self.env.batched_dynamics(state.cpu().detach().numpy(), action.cpu().detach().numpy())
        next_state = torch.tensor(next_state, dtype=state.dtype)
        return next_state


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from tqdm import tqdm
from cartpole_env import * 

def create_env():
    env = CartpoleEnv()
    env.reset(np.array([0, np.pi, 0, 0]) + np.random.rand(4,)) 
    return env

def main():
    # 初始化环境
    env = create_env()
    
    goal_state = np.zeros(4)
    controller = MPPIController(env, num_samples=500, horizon=30, hyperparams=get_cartpole_mppi_hyperparams())
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)

    frames = []
    num_steps = 150
    
    # 进度条
    pbar = tqdm(range(num_steps))
    
    for i in pbar:
        state = env.get_state()
        state = torch.tensor(state, dtype=torch.float32)
        control = controller.command(state)
        s = env.step(control)
        
        error_i = np.linalg.norm(s - goal_state[:7])
        pbar.set_description(f'Goal Error: {error_i:.4f}')
        
        # 渲染图像
        img = env.render()

        frames.append(img)

        # 如果误差小于0.2则提前结束
        if error_i < 0.2:
            break
    
    print("creating animated gif, please wait about 10 seconds")
    
    # 创建APNG动图
    write_apng("cartpole_mppi.gif", frames, delay=10)
    print("GIF created successfully!")

    # 显示GIF（在本地环境中可能需要用合适的工具显示 GIF）
    plt.imshow(plt.imread("cartpole_mppi.gif"))
    plt.axis('off')  # 不显示坐标轴
    plt.show()

if __name__ == "__main__":
    main()
