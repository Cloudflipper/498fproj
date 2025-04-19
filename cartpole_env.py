import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym


class CartpoleEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        self.cartpole = None
        super().__init__(*args, **kwargs)

    def step(self, control):
        """
            Steps the simulation one timestep, applying the given force
        Args:
            control: np.array of shape (1,) representing the force to apply

        Returns:
            next_state: np.array of shape (4,) representing next cartpole state

        """
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=control[0])
        p.stepSimulation()
        return self.get_state()

    def reset(self, state=None):
        """
            Resets the environment
        Args:
            state: np.array of shape (4,) representing cartpole state to reset to.
                   If None then state is randomly sampled
        """
        if state is not None:
            self.state = state
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(6,))
        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        self.cartpole = p.loadURDF('cartpole2.urdf')
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 2, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.setJointMotorControl2(self.cartpole, 2, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        self.set_state(self.state)
        self._setup_camera()

    def get_state(self):
        """
            Gets the cartpole internal state

        Returns:
            state: np.array of shape (4,) representing cartpole state [x, theta, x_dot, theta_dot]

        """

        x, x_dot = p.getJointState(self.cartpole, 0)[0:2]
        theta, theta_dot = p.getJointState(self.cartpole, 1)[0:2]
        return np.array([x, theta, x_dot, theta_dot])

    def set_state(self, state):
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
        p.resetJointState(self.cartpole, 0, targetValue=x, targetVelocity=x_dot)
        p.resetJointState(self.cartpole, 1, targetValue=theta1, targetVelocity=theta1_dot)
        p.resetJointState(self.cartpole, 2, targetValue=theta2, targetVelocity=theta2_dot)

    def _get_action_space(self):
        action_space = gym.spaces.Box(low=-30, high=30)  # linear force # TODO: Verify that they are correct
        return action_space

    def _get_state_space(self):
        x_lims = [-5, 5]  # TODO: Verify that they are the correct limits
        theta_lims = [-np.pi, np.pi]
        x_dot_lims = [-10, 10]
        theta_dot_lims = [-5 * np.pi, 5 * np.pi]
        state_space = gym.spaces.Box(
            low=np.array([x_lims[0], theta_lims[0],theta_lims[0], x_dot_lims[0], theta_dot_lims[0], theta_dot_lims[0]], dtype=np.float32),
            high=np.array([x_lims[1], theta_lims[1],theta_lims[1], x_dot_lims[1], theta_dot_lims[1],theta_dot_lims[1]],
                          dtype=np.float32))  # linear force # TODO: Verify that they are correct
        return state_space

    def _setup_camera(self):
        self.render_h = 480
        self.render_w = 640
        base_pos = [0, 0, 0]
        cam_dist = 6
        cam_pitch = 0.3
        cam_yaw = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=self.render_w / self.render_h,
                                                        nearVal=0.1,
                                                        farVal=100.0)

    def linearize_numerical(self, state, control, eps=1e-3):
        """
            Linearizes cartpole dynamics around linearization point (state, control). Uses numerical differentiation
        Args:
            state: np.array of shape (6,) representing cartpole state
            control: np.array of shape (1,) representing the force to apply
            eps: Small change for computing numerical derivatives
        Returns:
            A: np.array of shape (6, 6) representing Jacobian df/dx for dynamics f
            B: np.array of shape (6, 1) representing Jacobian df/du for dynamics f
        """

        A = np.zeros((6, 6))
        B = np.zeros((6, 1))
        # --- Your code here
        state_tensor = torch.from_numpy(state).float().reshape(1, 6)
        control_tensor = torch.from_numpy(control).float().reshape(1, 1)

        for i in range(6):
            state_plus = state.copy()
            state_plus[i] += eps
            self.set_state(state_plus)
            f_plus = self.step(control)
            
            state_minus = state.copy()
            state_minus[i] -= eps
            self.set_state(state_minus)
            f_minus = self.step(control)
            
            A[:, i] = (f_plus - f_minus) /  (2*eps)

        control_plus = control.copy() + eps
        self.set_state(state)
        f_plus = self.step(control_plus)
        
        control_minus = control.copy() - eps
        self.set_state(state)
        f_minus = self.step(control_minus)
        
        B[:, 0] = (f_plus - f_minus) /  eps/2
        # ---
        return A, B


# def dynamics_analytic(state, action):
#     """
#         Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
#         Should support batching
#     Args:
#         state: torch.tensor of shape (B, 4) representing the cartpole state
#         control: torch.tensor of shape (B, 1) representing the force to apply

#     Returns:
#         next_state: torch.tensor of shape (B, 4) representing the next cartpole state

#     """
    
#     dt = 0.05
#     g = 9.81
#     mc = 1
#     mp = 0.1
#     l = 0.5

#     # --- Your code here
#     print(state)
#     x, theta, x_dot, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
#     force = action.squeeze(-1)  # Ensure action is (B,) if it was (B, 1)

#     # Calculate theta_double_dot
#     cos_theta = torch.cos(theta)
#     sin_theta = torch.sin(theta)
#     denominator_theta = l * (4/3 - (mp * cos_theta**2) / (mc + mp))
#     theta_dot_dot = (g * sin_theta - cos_theta * (force + mp * l * theta_dot**2 * sin_theta) / (mc + mp)) / denominator_theta

#     # Calculate x_double_dot
#     x_dot_dot_numerator = force + mp * l * (theta_dot**2 * sin_theta - theta_dot_dot * cos_theta)
#     x_dot_dot = x_dot_dot_numerator / (mc + mp)

#     # Update next state components
#     x_dot_next = x_dot + dt * x_dot_dot
#     theta_dot_next = theta_dot + dt * theta_dot_dot

#     x_next = x + dt * x_dot_next
#     theta_next = theta + dt * theta_dot_next

#     # Reshape each component to (B, 1) and concatenate
#     next_state = torch.stack([
#         x_next.unsqueeze(-1),
#         theta_next.unsqueeze(-1),
#         x_dot_next.unsqueeze(-1),
#         theta_dot_next.unsqueeze(-1)
#     ], dim=-1).squeeze(1)  # Combine to (B, 4)

#     # ---

#     return next_state


# def linearize_pytorch(state, control):
#     """
#         Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
#     Args:
#         state: torch.tensor of shape (4,) representing cartpole state
#         control: torch.tensor of shape (1,) representing the force to apply

#     Returns:
#         A: torch.tensor of shape (4, 4) representing Jacobian df/dx for dynamics f
#         B: torch.tensor of shape (4, 1) representing Jacobian df/du for dynamics f

#     """

#     # --- Your code here
#     state = state.requires_grad_(True)
#     control = control.requires_grad_(True)

#     next_state = dynamics_analytic(state.reshape(1,4), control.reshape(1,1)).squeeze(0)
#     A = torch.zeros((4, 4), dtype=state.dtype)
#     for i in range(4):
#         grad_outputs = torch.zeros_like(next_state)
#         grad_outputs[i] = 1.0
#         grads = torch.autograd.grad(next_state, state, grad_outputs=grad_outputs, retain_graph=True)[0]
#         A[i, :] = grads

#     B = torch.zeros((4, 1), dtype=control.dtype)
#     for i in range(4):
#         grad_outputs = torch.zeros_like(next_state)
#         grad_outputs[i] = 1.0
#         grads = torch.autograd.grad(next_state, control, grad_outputs=grad_outputs, retain_graph=True)[0]
#         B[i, :] = grads

#     # ---
#     return A, B
