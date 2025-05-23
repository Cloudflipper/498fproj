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
    state_size = 6
    hyperparams = {
        'action_size': action_size,
        'state_size': state_size,
        'lambda': 0.001,
        'Q': torch.diag(torch.tensor([1.0, 5.0, 5.0, 0.1, 0.1, 0.1])).float(),
        'noise_sigma': torch.eye(action_size) * 70,
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

from PIL import Image as PILImage
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from cartpole_env import * 
import os


def main():
    env = CartpoleEnv()
    # env.reset(np.array([0, 0.1, 0, 0.1, 0, 0]))
    env.reset(np.array([0, np.pi, 0, 0, 0, 0]))


    # env1 = MyCartpoleEnv()
    # env1.reset(np.array([0, 0.1, -0.05, 0, 0, 0]))

    goal_state = np.zeros(6)
    controller = MPPIController(env, num_samples=200, horizon=15, hyperparams=get_cartpole_mppi_hyperparams())
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)

    frames = []
    num_steps = 350

    # Q = np.diag([2.0, 10.0, 10.0, 0.1, 0.1, 0.1])
    Q = torch.diag(torch.tensor([1, 5, 5, 0.1, 0.1, 0.1]))

    R = np.array([0.1])

    pole1_heights = []
    pole2_heights = []
    
    pbar = tqdm(range(num_steps))
    
    for i in pbar:
        state = env.get_state()
        state = torch.tensor(state, dtype=torch.float32)
        control = controller.command(state)
        s = env.step(control)

        # state1 = env1.get_state()
        # state1 = torch.tensor(state1, dtype=torch.float32)
        # control = controller.command(state1)
        # s1 = env1.step(control)
        # s1[5] = s1[5] + s1[4]
        # s1[2] = s1[2] + s1[1]
        # diff = s - s1
        # print(diff)
        # error_i = np.linalg.norm(s - goal_state[:7])state_cost = float(state_diff.T @ Q @ state_diff)  # 转换为Python float

        state_diff = s[:6] - goal_state
        import ipdb
        # ipdb.set_trace()
        state_cost = state_diff.T @ Q.detach().cpu().numpy() @ state_diff.T
        input = control.detach().cpu().numpy()
        # control_cost = (R * input**2).item()  
        control_cost = 1* (R * input**2).item()  

        # angle_penalty = -50 * (np.cos(s[1]) + np.cos(s[2]) - 2)
        angle_penalty = -5 * (10*np.cos(s[1]) + 5*np.cos(s[1]+s[2]) - 15)

        error_i = state_cost + control_cost + angle_penalty

        pbar.set_description(f'Goal Error: {error_i:.4f}')
        
        img = env.render()

        frames.append(PILImage.fromarray(img))

        l1, l2 = 1.0, 1.0
        theta1, theta2 = s[1], s[2]

        y_pole1 = l1 * np.cos(theta1)
        y_pole2 = y_pole1 + l2 * np.cos(theta1 + theta2)

        pole1_heights.append(y_pole1)
        pole2_heights.append(y_pole2)

        if error_i < 0.005:
            break
    
    save_dir = './saved_figures'
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(pole1_heights, label='Pole1 End Height')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Goal Height')
    plt.axhline(y=np.mean(pole1_heights), color='b', linestyle='--', label='Average Height')
    plt.xlabel('Time Step')
    plt.ylabel('Height (m)')
    plt.title('Pole 1 End Height Over Time (MPPI)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pole1_height_over_time.png'))
    plt.show()

    plt.figure()
    plt.plot(pole2_heights, label='Pole2 End Height')
    plt.axhline(y=2.0, color='r', linestyle='--', label='Goal Height')
    plt.axhline(y=np.mean(pole2_heights), color='b', linestyle='--', label='Average Height')
    plt.xlabel('Time Step')
    plt.ylabel('Height (m)')
    plt.title('Pole 2 End Height Over Time (MPPI)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pole2_height_over_time.png'))
    plt.show()
    
    print("creating animated gif, please wait about 10 seconds")

    frames[0].save("cartpole_mppi.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

    print("GIF created successfully!")

    plt.imshow(plt.imread("cartpole_mppi.gif"))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
