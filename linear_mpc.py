import control
import numpy as np
import scipy.linalg
import cvxpy as cp


class LinearMPC:

    def __init__(self, A, B, Q, R, horizon):
        self.dx = A.shape[0]
        self.du = B.shape[1]
        assert A.shape == (self.dx, self.dx)
        assert B.shape == (self.dx, self.du)
        self.H = horizon
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def compute_SM(self):
        """
        Computes the S and M matrices as defined in the ipython notebook

        All the variables you need should be class member variables already

        Returns:
            S: np.array of shape (horizon * dx, horizon * du) S matrix
            M: np.array of shape (horizon * dx, dx) M matrix

        """
        

        # --- Your code here
        S = np.zeros((self.H * self.dx, self.H * self.du))
        M = np.zeros((self.H * self.dx, self.dx))
        for i in range(self.H):
            j = 0
            while i + j < self.H:
                exponent = j
                mat = np.linalg.matrix_power(self.A, exponent) @ self.B
                row = i + j
                S[row * self.dx : (row + 1) * self.dx, i * self.du : (i + 1) * self.du] = mat
                j += 1
            M[i * self.dx : (i + 1) * self.dx, :] = np.linalg.matrix_power(self.A, i + 1)
        
        # ---
        return S, M

    def compute_Qbar_and_Rbar(self):
        Q_repeat = [self.Q] * self.H
        R_repeat = [self.R] * self.H
        return scipy.linalg.block_diag(*Q_repeat), scipy.linalg.block_diag(*R_repeat)

    def compute_finite_horizon_lqr_gain(self):
        """
            Compute the controller gain G0 for the finite-horizon LQR

        Returns:
            G0: np.array of shape (du, dx)

        """
        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()

        G = -np.linalg.inv(S.T @ Qbar @ S + Rbar) @ S.T @ Qbar @ M

        # --- Your code here
        G0 = G[:self.du, :]
        # ---

        return G0

    def compute_lqr_gain(self):
        """
            Compute controller gain G for infinite-horizon LQR
        Returns:
            Ginf: np.array of shape (du, dx)

        """
        Ginf = None
        theta_T_theta, _, _ = control.dare(self.A, self.B, self.Q, self.R)

        # --- Your code here
        Ginf = -np.linalg.inv(self.R + self.B.T @ theta_T_theta @ self.B) @ self.B.T @ theta_T_theta @ self.A

        # ---
        return Ginf

    def lqr_box_constraints_qp_shooting(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing with shooting

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls

        """

        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()
        U = cp.Variable((self.H, self.B.shape[1]))
        # --- Your code here
        U_flat = cp.reshape(U, (self.H * self.B.shape[1], ),order='C')
        cost = cp.quad_form(S @ U_flat + M @ x0, Qbar) + cp.quad_form(U_flat, Rbar)

        constraints = [ U>=u_min , U <= u_max]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        # ---

        return U.value

    def lqr_box_constraints_qp_collocation(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing
            with collocation

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls
            X: np.array of shape (horizon, dx) containing sequence of optimal states

        """

        dx = self.A.shape[0]  
        du = self.B.shape[1]  
        horizon = self.H

        U = cp.Variable((horizon, du))
        X = cp.Variable((self.H + 1, dx))

    
        cost = 0
        for t in range(self.H):
            cost += cp.quad_form(X[t+1], self.Q) + cp.quad_form(U[t], self.R)

        constraints = [X[0] == x0]  
        for t in range(self.H):
            constraints += [X[t+1] == self.A @ X[t] + self.B @ U[t]]  
            constraints += [u_min <= U[t], U[t] <= u_max]  

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        return U.value, X.value[1:]

        # --- Your code here


        # ---


# def test_compute_SM():
#     A = np.array([[1.0, 1.0], [2.0, 1.0]])
#     B = np.array([[2.0], [1.0]])
#     Q = np.eye(2)
#     R = np.eye(1)
#     horizon = 3
#     mpc = LinearMPC(A, B, Q, R, horizon)
#     S, M = mpc.compute_SM()

#     print("S matrix:")
#     print(S)
#     print("M matrix:")
#     print(M)
#     for i in range(horizon):
#         An = np.linalg.matrix_power(A, i)
#         AnB = An @ B
#         print(f"A^{i}:")
#         print(An)
#         print(f"A^{i}B:")
#         print(AnB)

# test_compute_SM()