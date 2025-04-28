

import numpy as np
import torch
from scipy.optimize import minimize
from cartpole_env import CartpoleEnv
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os

def get_cartpole_sqp_hyperparams():
    hyperparams = {
        'Q': np.diag([1.0, 6.0, 8.0, 5, 10, 10]),
        'R': np.array([[0.1]]),
        'horizon': 50,
        'max_iter': 5,
    }
    return hyperparams


class SQPController:
    def __init__(self, env, hyperparams):
        self.env = env
        self.T = hyperparams['horizon']
        self.Q = hyperparams['Q']
        self.R = hyperparams['R']
        self.max_iter = hyperparams['max_iter']
        self.action_size = env.action_space.shape[0]
        self.state_size = env.state_space.shape[0]
        self.goal_state = np.zeros(self.state_size)
        self.U = np.zeros((self.T, self.action_size))

    def dynamics(self, state, action):
        return self.env.batched_dynamics(state.reshape(1, -1), action.reshape(1, -1)).reshape(-1)

    def rollout(self, state, U):
        traj = np.zeros((self.T, self.state_size))
        for t in range(self.T):
            state = self.dynamics(state, U[t])
            traj[t] = state
        return traj

    def cost(self, U_flat, state):
        U = U_flat.reshape(self.T, self.action_size)
        traj = self.rollout(state, U)
        state_diff = traj - self.goal_state
        cost_state = np.sum(state_diff @ self.Q @ state_diff.T)
        cost_action = np.sum(U @ self.R @ U.T)
        return cost_state + cost_action

    def command(self, state):
        U_flat = self.U.flatten()

        constraints = [] 
        bounds = [(-100, 100)] * len(U_flat)  # action bounds

        result = minimize(self.cost, U_flat, args=(state,), method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': self.max_iter, 'ftol': 1e-4})

        self.U = result.x.reshape(self.T, self.action_size)
        action = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = 0
        return action


def main():
    env = CartpoleEnv()
    # env.reset(np.array([0, np.pi, 0, 0, 0, 0]))
    env.reset(np.array([0, 0.05, -0.02, 0.03, 0, 0]))


    controller = SQPController(env, get_cartpole_sqp_hyperparams())

    frames = []
    state_diffs = []
    num_steps = 50
    goal_state = np.zeros(6)
    Q = np.diag([2.0, 10.0, 10.0, 2, 5, 5])
    R = np.array([0.1])

    pole1_heights = []
    pole2_heights = []

    pbar = tqdm(range(num_steps))
    
    # 设置VideoWriter来保存为MP4格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码器
    out = cv2.VideoWriter('cartpole_sqp.mp4', fourcc, 30.0, (env.render().shape[1], env.render().shape[0]))

    for i in pbar:
        s_t = env.get_state()
        u_t = controller.command(s_t)
        s_next = env.step(u_t)

        s = s_next
        state_diff = s[:6] - goal_state
        state_diffs.append(state_diff[:3])
        state_cost = state_diff.T @ Q @ state_diff
        input = u_t
        control_cost = 1* (R * input**2).item()
        angle_penalty = - 10 * (10*np.cos(s[1]) + 5*np.cos(s[1]+s[2]) - 15)
        error = state_cost + control_cost + angle_penalty

        # error = np.linalg.norm(s_next - goal_state)
        pbar.set_description(f'Goal Error: {error:.4f}')

        img = env.render()

        # 将每一帧添加到视频中
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV 使用BGR格式
        out.write(img_bgr)

        l1, l2 = 1.0, 1.0
        theta1, theta2 = s[1], s[2]

        y_pole1 = l1 * np.cos(theta1)
        y_pole2 = y_pole1 + l2 * np.cos(theta1 + theta2)

        pole1_heights.append(y_pole1)
        pole2_heights.append(y_pole2)

        if error < 0.02:
            break
    
        

    
    save_dir = './saved_figures'
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(pole1_heights, label='Pole1 End Height')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Goal Height')
    plt.axhline(y=np.mean(pole1_heights), color='b', linestyle='--', label='Average Height')
    plt.xlabel('Time Step')
    plt.ylabel('Height (m)')
    plt.title('Pole 1 End Height Over Time (SQP)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'sqp_high_pole1_height_over_time.png'))
    plt.show()

    plt.figure()
    plt.plot(pole2_heights, label='Pole2 End Height')
    plt.axhline(y=2.0, color='r', linestyle='--', label='Goal Height')
    plt.axhline(y=np.mean(pole2_heights), color='b', linestyle='--', label='Average Height')
    plt.xlabel('Time Step')
    plt.ylabel('Height (m)')
    plt.title('Pole 2 End Height Over Time (SQP)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'sqp_high_pole2_height_over_time.png'))
    plt.show()

    out.release()

    print("MP4 created successfully!")

    video = cv2.VideoCapture('cartpole_sqp.mp4')
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow('Cartpole SQP', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
            break
    video.release()
    cv2.destroyAllWindows()

    # state_diffs = np.array(state_diffs)
    # plt.plot(state_diffs[:, 0], label='State Diff - Dimension 1')
    # plt.plot(state_diffs[:, 1], label='State Diff - Dimension 2')
    # plt.plot(state_diffs[:, 2], label='State Diff - Dimension 3')
    # plt.xlabel('Time step')
    # plt.ylabel('State Difference')
    # plt.legend()
    # plt.title('State Differences (First 3 Dimensions) Over Time')
    # plt.show()

    save_dir = './saved_figures'
    os.makedirs(save_dir, exist_ok=True)





if __name__ == "__main__":
    main()
