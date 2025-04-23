import torch
import torch.linalg as linalg
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from tqdm import tqdm
from cartpole_env import *

def get_cartpole_sqp_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running SQP on the cartpole environment
    """
    action_size = 1
    state_size = 6

    hyperparams = {
        'action_size': action_size,
        'state_size': state_size,
        'lambda': 0.1,
        'Q': torch.diag(torch.tensor([1.0, 0.1, 0.1, 1.0, 0.1, 0.1])).float(),
        'R': torch.eye(action_size) * 0.1,
        'noise_sigma': torch.eye(action_size) * 0.5,
        'max_iter': 5,
        'step_size': 0.1,
        'fd_eps': 1e-4,
    }
    return hyperparams

class SQPController(object):
    def __init__(self, env, num_samples, horizon, hyperparams):
        self.env = env
        self.T = horizon
        self.lambda_ = hyperparams['lambda']
        self.action_size = env.action_space.shape[-1]
        self.state_size = env.state_space.shape[-1]
        self.goal_state = torch.zeros(self.state_size)
        self.Q = hyperparams['Q']
        self.R = hyperparams['R']
        self.u_init = torch.zeros(self.action_size)
        self.U = torch.zeros((self.T, self.action_size))
        self.noise_mu = torch.zeros(self.action_size)
        self.noise_sigma = hyperparams['noise_sigma']
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        self.max_iter = hyperparams['max_iter']
        self.step_size = hyperparams['step_size']
        self.fd_eps = hyperparams['fd_eps']

    def reset(self):
        self.U = torch.zeros((self.T, self.action_size))

    def command(self, state):
        # 对当前状态做多次 SQP 迭代
        for _ in range(self.max_iter):
            self._sqp_update(state)

        # 取最优动作
        action = self.U[0].clone()
        # 平移动作序列
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        return action

    def _rollout_dynamics(self, state_0, actions):
        """
        给定初始 state_0 和动作序列 actions (T, action_size)，
        返回滚动得到的轨迹 (T, state_size)
        """
        state = state_0.unsqueeze(0)  # (1, state_size)
        traj = torch.zeros((self.T, self.state_size), dtype=torch.float32)
        for t in range(self.T):
            next_s = self._dynamics(state, actions[t].unsqueeze(0))
            traj[t] = next_s.squeeze(0)
            state = next_s
        return traj

    def _compute_nominal_cost(self, state, U):
        """
        给定 state 和 动作序列 U，计算整个 horizon 的总代价
        """
        traj = self._rollout_dynamics(state, U)  # (T, state_size)
        # 状态二次代价
        diff = traj - self.goal_state
        state_cost = torch.einsum('ti,ij,tj->t', diff, self.Q, diff)
        # 动作二次代价
        action_cost = torch.einsum('ti,ij,tj->t', U, self.R, U)
        return (state_cost + self.lambda_ * action_cost).sum()

    def _sqp_update(self, state):
        """
        用有限差分计算梯度，然后做一次梯度下降
        """
        eps = self.fd_eps
        J0 = self._compute_nominal_cost(state, self.U)

        grad = torch.zeros_like(self.U)
        # 对每个 t, 每个动作维度 j 做有限差分
        for t in range(self.T):
            for j in range(self.action_size):
                U_plus = self.U.clone()
                U_minus = self.U.clone()
                U_plus[t, j] += eps
                U_minus[t, j] -= eps
                J_plus = self._compute_nominal_cost(state, U_plus)
                J_minus = self._compute_nominal_cost(state, U_minus)
                grad[t, j] = (J_plus - J_minus) / (2 * eps)

        # 梯度下降更新
        self.U -= self.step_size * grad

    def _dynamics(self, state, action):
        """
        调用 env.batched_dynamics (numpy) 再转回 torch
        """
        nxt = self.env.batched_dynamics(state.cpu().numpy(), action.cpu().numpy())
        return torch.tensor(nxt, dtype=state.dtype)

def create_env():
    env = CartpoleEnv()
    env.reset(np.array([0, np.pi, 0, 0, 0, 0]) + np.random.rand(6,))
    return env

def main():
    env = create_env()
    goal_state = np.zeros(4)
    controller = SQPController(env, num_samples=500, horizon=30,
                               hyperparams=get_cartpole_sqp_hyperparams())
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)

    frames = []
    num_steps = 150
    pbar = tqdm(range(num_steps))

    for i in pbar:
        s_t = torch.tensor(env.get_state(), dtype=torch.float32)
        u_t = controller.command(s_t)
        s_next = env.step(u_t.numpy())

        error = np.linalg.norm(s_next - goal_state[:7])
        pbar.set_description(f'Goal Error: {error:.4f}')

        img = env.render()
        frames.append(PILImage.fromarray(img))

        if error < 0.2:
            break

    print("Creating animated gif, please wait...")
    frames[0].save("cartpole_sqp.gif", save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    print("GIF created successfully!")

    plt.imshow(plt.imread("cartpole_sqp.gif"))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
import torch
import torch.linalg as linalg
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from tqdm import tqdm
from cartpole_env import *

def get_cartpole_sqp_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running SQP on the cartpole environment
    """
    action_size = 1
    state_size = 4

    hyperparams = {
        'action_size': action_size,
        'state_size': state_size,
        'lambda': 0.1,
        'Q': torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0])).float(),
        'R': torch.eye(action_size) * 0.1,
        'noise_sigma': torch.eye(action_size) * 0.5,
        'max_iter': 8,
        'step_size': 0.5,
        'fd_eps': 1e-4,
    }

    
    return hyperparams

class SQPController(object):
    def __init__(self, env, num_samples, horizon, hyperparams):
        self.env = env
        self.T = horizon
        self.lambda_ = hyperparams['lambda']
        self.action_size = env.action_space.shape[-1]
        self.state_size = env.state_space.shape[-1]
        self.goal_state = torch.zeros(self.state_size)
        self.Q = hyperparams['Q']
        self.R = hyperparams['R']
        self.u_init = torch.zeros(self.action_size)
        self.U = torch.zeros((self.T, self.action_size))
        self.noise_mu = torch.zeros(self.action_size)
        self.noise_sigma = hyperparams['noise_sigma']
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        self.max_iter = hyperparams['max_iter']
        self.step_size = hyperparams['step_size']
        self.fd_eps = hyperparams['fd_eps']

    def reset(self):
        self.U = torch.zeros((self.T, self.action_size))

    def command(self, state):
        # 对当前状态做多次 SQP 迭代
        for _ in range(self.max_iter):
            self._sqp_update(state)

        # 取最优动作
        action = self.U[0].clone()
        # 平移动作序列
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        return action

    def _rollout_dynamics(self, state_0, actions):
        """
        给定初始 state_0 和动作序列 actions (T, action_size)，
        返回滚动得到的轨迹 (T, state_size)
        """
        state = state_0.unsqueeze(0)  # (1, state_size)
        traj = torch.zeros((self.T, self.state_size), dtype=torch.float32)
        for t in range(self.T):
            next_s = self._dynamics(state, actions[t].unsqueeze(0))
            traj[t] = next_s.squeeze(0)
            state = next_s
        return traj

    def _compute_nominal_cost(self, state, U):
        """
        给定 state 和 动作序列 U，计算整个 horizon 的总代价
        """
        traj = self._rollout_dynamics(state, U)  # (T, state_size)
        # 状态二次代价
        diff = traj - self.goal_state
        state_cost = torch.einsum('ti,ij,tj->t', diff, self.Q, diff)
        # 动作二次代价
        action_cost = torch.einsum('ti,ij,tj->t', U, self.R, U)
        return (state_cost + self.lambda_ * action_cost).sum()

    def _sqp_update(self, state):
        """
        用有限差分计算梯度，然后做一次梯度下降
        """
        eps = self.fd_eps
        J0 = self._compute_nominal_cost(state, self.U)

        grad = torch.zeros_like(self.U)
        # 对每个 t, 每个动作维度 j 做有限差分
        for t in range(self.T):
            for j in range(self.action_size):
                U_plus = self.U.clone()
                U_minus = self.U.clone()
                U_plus[t, j] += eps
                U_minus[t, j] -= eps
                J_plus = self._compute_nominal_cost(state, U_plus)
                J_minus = self._compute_nominal_cost(state, U_minus)
                grad[t, j] = (J_plus - J_minus) / (2 * eps)

        # 梯度下降更新
        self.U -= self.step_size * grad

    def _dynamics(self, state, action):
        """
        调用 env.batched_dynamics (numpy) 再转回 torch
        """
        nxt = self.env.batched_dynamics(state.cpu().numpy(), action.cpu().numpy())
        return torch.tensor(nxt, dtype=state.dtype)

def create_env():
    env = CartpoleEnv()
    env.reset(np.array([0, np.pi, 0, 0]) + np.random.rand(4,))
    return env

def main():
    env = create_env()
    goal_state = np.zeros(4)
    controller = SQPController(env, num_samples=500, horizon=30,
                               hyperparams=get_cartpole_sqp_hyperparams())
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)

    frames = []
    num_steps = 150
    pbar = tqdm(range(num_steps))

    for i in pbar:
        s_t = torch.tensor(env.get_state(), dtype=torch.float32)
        u_t = controller.command(s_t)
        s_next = env.step(u_t.numpy())

        error = np.linalg.norm(s_next - goal_state[:7])
        pbar.set_description(f'Goal Error: {error:.4f}')

        img = env.render()
        frames.append(PILImage.fromarray(img))

        if error < 0.2:
            break

    print("Creating animated gif, please wait...")
    frames[0].save("cartpole_sqp.gif", save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    print("GIF created successfully!")

    plt.imshow(plt.imread("cartpole_sqp.gif"))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
