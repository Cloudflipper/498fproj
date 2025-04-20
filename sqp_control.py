import numpy as np
from scipy.linalg import block_diag, solve
from scipy.optimize import linprog

class SQPController:
    def __init__(self, dynamics, cost, eq_constraints, ineq_constraints, 
                 nx, nu, horizon=10, max_iter=100, tol=1e-4):
        """
        Sequential Quadratic Programming Controller
        
        :param dynamics: 离散状态方程 x_{k+1} = f(x_k, u_k)
        :param cost: 成本函数 J(x, u) -> scalar
        :param eq_constraints: 等式约束列表 [h_1(x,u)=0, ..., h_m(x,u)=0]
        :param ineq_constraints: 不等式约束列表 [g_1(x,u)<=0, ..., g_p(x,u)<=0]
        :param nx: 状态维度
        :param nu: 控制维度
        :param horizon: 预测时域
        :param max_iter: 最大迭代次数
        :param tol: 收敛容忍度
        """
        self.f = dynamics
        self.J = cost
        self.h = eq_constraints
        self.g = ineq_constraints
        self.nx = nx
        self.nu = nu
        self.T = horizon
        self.max_iter = max_iter
        self.tol = tol

        # 初始化变量
        self.x_seq = np.zeros((self.T+1, self.nx))  # 状态序列
        self.u_seq = np.zeros((self.T, self.nu))    # 控制序列
        self.lam = np.zeros(len(self.h))           # 等式约束乘子
        self.mu = np.zeros(len(self.g))            # 不等式约束乘子

    def command(self, x0):
        """主控制循环"""
        self.x_seq[0] = x0
        for k in range(self.max_iter):
            # 1. 计算当前轨迹成本
            J_prev = self._compute_total_cost()
            
            # 2. 构建QP子问题
            G, c, A_eq, b_eq, A_ineq, b_ineq = self._build_qp_subproblem()
            
            # 3. 求解QP问题
            delta_u = self._solve_qp(G, c, A_eq, b_eq, A_ineq, b_ineq)
            
            # 4. 线搜索确定步长
            alpha = self._line_search(delta_u)
            
            # 5. 更新控制序列
            self.u_seq += alpha * delta_u.reshape((self.T, self.nu))
            
            # 6. 前向模拟更新状态序列
            self._forward_simulate()
            
            # 7. 更新拉格朗日乘子
            self._update_multipliers()
            
            # 检查收敛
            if np.linalg.norm(delta_u) < self.tol:
                break
        
        return self.u_seq[0].copy()

    def _build_qp_subproblem(self):
        """构建二次规划子问题"""
        # 初始化QP矩阵
        n = self.T * self.nu
        G = np.zeros((n, n))
        c = np.zeros(n)
        A_eq = []
        b_eq = []
        A_ineq = []
        b_ineq = []

        # 遍历时域构建各矩阵
        for t in range(self.T):
            x = self.x_seq[t]
            u = self.u_seq[t]
            
            # 计算目标函数的二次项和一次项
            J_x, J_u = self._compute_gradients(x, u)
            J_xx, J_uu, J_ux = self._compute_hessians(x, u)
            
            # 构建Hessian矩阵块
            G_block = J_uu + self.mu @ self._ineq_hessian(x,u)
            G[t*self.nu:(t+1)*self.nu, t*self.nu:(t+1)*self.nu] = G_block
            
            # 构建梯度项
            c_block = J_u + self.lam @ self._eq_jacobian(x,u)
            c[t*self.nu:(t+1)*self.nu] = c_block

            # 构建等式约束 (动力学约束)
            A_eq_block = self._dynamics_jacobian(x, u)
            A_eq.append(A_eq_block)
            b_eq.append(self.f(x,u) - self.x_seq[t+1])

            # 构建不等式约束
            active_ineq = self._get_active_ineq_constraints(x,u)
            for j in active_ineq:
                A_ineq.append(self._ineq_jacobian(j, x,u))
                b_ineq.append(-self.g[j](x,u))

        # 组装完整约束矩阵
        A_eq = np.vstack(A_eq) if A_eq else np.zeros((0,n))
        b_eq = np.hstack(b_eq) if b_eq else np.zeros(0)
        A_ineq = np.vstack(A_ineq) if A_ineq else np.zeros((0,n))
        b_ineq = np.hstack(b_ineq) if b_ineq else np.zeros(0)

        return G, c, A_eq, -b_eq, A_ineq, -b_ineq

    def _solve_qp(self, G, c, A_eq, b_eq, A_ineq, b_ineq):
        """求解二次规划问题"""
        # 转换为标准QP形式: min 0.5 x^T G x + c^T x
        #                 s.t. A_eq x = b_eq
        #                      A_ineq x <= b_ineq
        result = linprog(c, A_ub=A_ineq, b_ub=b_ineq, 
                         A_eq=A_eq, b_eq=b_eq, 
                         bounds=(None, None))
        if not result.success:
            raise RuntimeError("QP求解失败: "+result.message)
        return result.x

    def _compute_gradients(self, x, u):
        """数值计算梯度"""
        eps = 1e-6
        J0 = self.J(x, u)
        
        # 计算状态梯度
        J_x = np.zeros_like(x)
        for i in range(len(x)):
            dx = np.zeros_like(x)
            dx[i] = eps
            J_x[i] = (self.J(x+dx, u) - J0) / eps
            
        # 计算控制梯度
        J_u = np.zeros_like(u)
        for i in range(len(u)):
            du = np.zeros_like(u)
            du[i] = eps
            J_u[i] = (self.J(x, u+du) - J0) / eps
            
        return J_x, J_u

    def _compute_hessians(self, x, u):
        """数值计算Hessian矩阵"""
        eps = 1e-6
        J_x, J_u = self._compute_gradients(x, u)
        
        # 计算状态Hessian
        J_xx = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            dx = np.zeros_like(x)
            dx[i] = eps
            J_x_plus, _ = self._compute_gradients(x+dx, u)
            J_xx[i,:] = (J_x_plus - J_x) / eps
            
        # 计算控制Hessian
        J_uu = np.zeros((len(u), len(u)))
        for i in range(len(u)):
            du = np.zeros_like(u)
            du[i] = eps
            _, J_u_plus = self._compute_gradients(x, u+du)
            J_uu[i,:] = (J_u_plus - J_u) / eps
            
        # 计算交叉项
        J_ux = np.zeros((len(u), len(x)))
        for i in range(len(u)):
            du = np.zeros_like(u)
            du[i] = eps
            J_x_plus, _ = self._compute_gradients(x, u+du)
            J_ux[i,:] = (J_x_plus - J_x) / eps
            
        return J_xx, J_uu, J_ux

    def _forward_simulate(self):
        """前向模拟生成状态轨迹"""
        for t in range(self.T):
            self.x_seq[t+1] = self.f(self.x_seq[t], self.u_seq[t])

    def _update_multipliers(self):
        """更新拉格朗日乘子"""
        # 等式约束乘子更新
        self.lam += self.alpha * (self.A_eq @ self.delta_u - self.b_eq)
        
        # 不等式约束乘子更新 (仅激活约束)
        active = self._get_active_ineq_constraints()
        self.mu[active] = np.maximum(0, self.mu[active] + self.alpha*(self.A_ineq @ self.delta_u - self.b_ineq))

    def _line_search(self, delta_u):
        """回溯线搜索确定步长"""
        alpha = 1.0
        while alpha > 1e-4:
            new_u = self.u_seq + alpha * delta_u.reshape(self.u_seq.shape)
            new_J = self._compute_total_cost(new_u)
            if new_J < self.J_prev:
                return alpha
            alpha *= 0.5
        return alpha

    def _compute_total_cost(self, u_seq=None):
        """计算总成本"""
        if u_seq is None:
            u_seq = self.u_seq
        total = 0.0
        x = self.x_seq[0].copy()
        for t in range(self.T):
            total += self.J(x, u_seq[t])
            x = self.f(x, u_seq[t])
        return total

    # 以下为约束相关工具函数
    def _dynamics_jacobian(self, x, u):
        """计算动力学方程的雅可比矩阵"""
        eps = 1e-6
        f0 = self.f(x, u)
        J = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            du = np.zeros_like(u)
            du[i] = eps
            J[:,i] = (self.f(x, u+du) - f0) / eps
        return J

    def _eq_jacobian(self, x, u):
        """等式约束雅可比矩阵"""
        return np.array([h(x,u) for h in self.h])

    def _ineq_jacobian(self, idx, x, u):
        """不等式约束雅可比矩阵"""
        g = self.g[idx]
        eps = 1e-6
        J = np.zeros(self.nu)
        for i in range(self.nu):
            du = np.zeros_like(u)
            du[i] = eps
            J[i] = (g(x, u+du) - g(x,u)) / eps
        return J

    def _get_active_ineq_constraints(self, x, u):
        """获取激活的不等式约束索引"""
        active = []
        for j in range(len(self.g)):
            if abs(self.g[j](x,u)) < 1e-4:  # 激活阈值
                active.append(j)
        return active

# 使用示例：倒立摆控制
if __name__ == "__main__":
    # 定义系统参数
    m = 0.1  # 摆杆质量
    l = 0.5  # 摆杆长度
    g = 9.81
    
    # 离散动力学模型
    def pendulum_dynamics(x, u):
        theta, omega = x
        dt = 0.05
        new_omega = omega + (3*g/(2*l)*np.sin(theta) + 3/(m*l**2)*u) * dt
        new_theta = theta + new_omega * dt
        return np.array([new_theta, new_omega])
    
    # 成本函数
    def quadratic_cost(x, u):
        return 0.5*x[0]**2 + 0.1*x[1]**2 + 0.01*u**2
    
    # 初始化控制器
    controller = SQPController(
        dynamics=pendulum_dynamics,
        cost=quadratic_cost,
        eq_constraints=[],  # 无等式约束
        ineq_constraints=[lambda x,u: u-1.0,  # 控制输入约束
                         lambda x,u: -u-1.0],
        nx=2, nu=1, horizon=10
    )
    
    # 控制测试
    x0 = np.array([np.pi/4, 0.0])  # 初始状态
    optimal_u = controller.command(x0)
    print(f"Optimal control: {optimal_u[0]:.4f}")