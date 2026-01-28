import numpy as np
import torch


class BatchNewmarkExplicitStepper:
    def __init__(self, model, n_samples, K):
        # N = Phi.shape[0]
        self.model_ = model
        self.un_ = torch.zeros((n_samples, K))
        self.udotn_ = torch.zeros((n_samples, K))
        self.uddotn_ = torch.zeros((n_samples, K))
        self.unp1_ = torch.zeros((n_samples, K))
        self.unm1_ = torch.zeros((n_samples, K))
        self.udotnp1_ = torch.zeros((n_samples, K))
        self.uddotnp1_ = torch.zeros((n_samples, K))
        self.unm1_ = torch.zeros((n_samples, K))
        self.unp1_ = torch.zeros((n_samples, K))
        self.I_ = torch.eye(K)
        self.M_ = torch.ones((n_samples, K, K))*torch.eye(K)[None]
        self.u_history_ = self.un_[:, :, None]

    def update_states_newmark(self):
        self.u_history_ = torch.cat(
            (self.u_history_, self.unp1_.unsqueeze(2)), dim=2)
        self.un_[:] = self.unp1_[:]
        self.udotn_[:] = self.udotnp1_[:]
        self.uddotn_[:] = self.uddotnp1_[:]

    def advance_newmark(self, dt):
        torch.autograd.set_detect_anomaly(True)
        gamma = 0.5
        beta = 0.25*(gamma + 0.5)**2
        self.unp1_[:] = self.un_[:] + dt*self.udotn_ + dt**2/2.*self.uddotn_[:]
        inputs = {'x': self.unp1_*1.0}
        Kx, K = self.model_.forward(inputs, return_stiffness=True)
        self.uddotnp1_[:] = 1.0*Kx[:]
        self.udotnp1_[:] = self.udotn_[:] + \
            (1. - gamma)*dt*self.uddotn_[:] + gamma*dt*self.uddotnp1_[:]

    def advance_n_steps(self, un, udotn, uddotn, dt, n_steps):
        self.un_[:] = torch.tensor(un[:])
        self.udotn_[:] = torch.tensor(udotn[:])
        self.uddotn_[:] = torch.tensor(uddotn[:])
        for i in range(0, n_steps):
            # bcs = bc_hook(i)
            self.advance_newmark(dt)
            self.update_states_newmark()
        return self.u_history_

    def get_unp1(self):
        return self.unp1_


class BatchNewmarkStepper:
    def __init__(self, model, n_samples, K):
        # N = Phi.shape[0]
        self.model_ = model
        self.un_ = torch.zeros((n_samples, K))
        self.udotn_ = torch.zeros((n_samples, K))
        self.uddotn_ = torch.zeros((n_samples, K))
        self.unp1_ = torch.zeros((n_samples, K))
        self.unm1_ = torch.zeros((n_samples, K))
        self.udotnp1_ = torch.zeros((n_samples, K))
        self.uddotnp1_ = torch.zeros((n_samples, K))
        self.unm1_ = torch.zeros((n_samples, K))
        self.unp1_ = torch.zeros((n_samples, K))
        self.I_ = torch.eye(K)
        self.M_ = torch.ones((n_samples, K, K))*torch.eye(K)[None]
        self.u_history_ = self.un_[:, :, None]

    def update_states_newmark(self):
        self.u_history_ = torch.cat(
            (self.u_history_, self.unp1_.unsqueeze(2)), dim=2)
        self.un_[:] = self.unp1_[:]
        self.udotn_[:] = self.udotnp1_[:]
        self.uddotn_[:] = self.uddotnp1_[:]

    def advance_newmark(self, dt):
        gamma = 0.5
        beta = 0.25*(gamma + 0.5)**2

        def jacobian(x):
            inputs = {'x': x}
            Kx, K = self.model_.forward(inputs, return_stiffness=True)
            Kx = -Kx
            K = -K
            J = self.M_/(dt**2 * beta) + K
            return J

        counter = 0

        def my_residual(x):
            inputs = {'x': x}
            Kx, K = self.model_.forward(inputs, return_stiffness=True)
            Kx = -Kx
            K = -K
            # LHS = (self.M_/(dt**2 * beta) + K) @ x
            LHS = torch.einsum('nij,nj->ni', (self.M_/(dt**2 * beta) + K), x)
            RHS = 1./(dt**2*beta)*(self.un_) + 1./(beta*dt) * \
                (self.udotn_) + 1./(2.*beta)*(1. - 2.*beta)*(self.uddotn_)
            residual = LHS - RHS
            return residual

        def my_newton(x):
            r = my_residual(x)
            r0_norm = torch.linalg.norm(r)
            iteration = 0
            max_its = 50
            while torch.linalg.norm(r)/r0_norm > 1e-10 and iteration < max_its:
                J = jacobian(x)
                dx = torch.linalg.solve(J, -r)
                x = x + dx
                r = my_residual(x)
                iteration += 1
            return x[:]

        inputs = self.un_*1.
        solution = my_newton(inputs)

        self.unp1_[:] = solution[:]
        self.uddotnp1_[:] = 1./(dt**2*beta)*(self.unp1_ - self.un_) - \
            1./(beta*dt)*self.udotn_ - 1./(2*beta)*(1 - 2.*beta)*self.uddotn_
        self.udotnp1_[:] = self.udotn_ + \
            (1. - gamma)*dt*self.uddotn_ + gamma*dt*self.uddotnp1_

    def advance_n_steps(self, un, udotn, uddotn, dt, n_steps):
        self.un_[:] = torch.tensor(un[:])
        self.udotn_[:] = torch.tensor(udotn[:])
        self.uddotn_[:] = torch.tensor(uddotn[:])
        for i in range(0, n_steps):
            # bcs = bc_hook(i)
            self.advance_newmark(dt)
            self.update_states_newmark()
        return self.u_history_

    def get_unp1(self):
        return self.unp1_


class NewmarkStepper:
    def __init__(self, model, K):
        # N = Phi.shape[0]
        self.model_ = model
        self.un_ = torch.zeros(K)
        self.udotn_ = torch.zeros(K)
        self.uddotn_ = torch.zeros(K)
        self.unp1_ = torch.zeros(K)
        self.unm1_ = torch.zeros(K)
        self.udotnp1_ = torch.zeros(K)
        self.uddotnp1_ = torch.zeros(K)
        self.unm1_ = torch.zeros(K)
        self.unp1_ = torch.zeros(K)
        self.I_ = torch.eye(K)
        self.M_ = torch.eye(K)
        self.u_history_ = self.un_[:, None]

    def update_states_newmark(self):
        self.u_history_ = torch.cat(
            (self.u_history_, self.unp1_.unsqueeze(1)), dim=1)
        self.un_[:] = self.unp1_[:]
        self.udotn_[:] = self.udotnp1_[:]
        self.uddotn_[:] = self.uddotnp1_[:]

    def advance_newmark(self, dt):
        gamma = 0.5
        beta = 0.25*(gamma + 0.5)**2

        def jacobian(x):
            inputs = {'x': x[None]}
            Kx, K = self.model_.forward(inputs, return_stiffness=True)
            Kx = -Kx[0]
            K = -K[0]
            J = self.M_/(dt**2 * beta) + K
            return J

        counter = 0

        def my_residual(x):
            inputs = {'x': x[None]}
            Kx, K = self.model_.forward(inputs, return_stiffness=True)
            Kx = -Kx[0]
            K = -K[0]
            LHS = (self.M_/(dt**2 * beta) + K) @ x
            RHS = 1./(dt**2*beta)*(self.M_ @ self.un_) + 1./(beta*dt)*(self.M_ @
                                                                       self.udotn_) + 1./(2.*beta)*(1. - 2.*beta)*(self.M_ @  self.uddotn_)
            residual = LHS - RHS
            return residual

        def my_newton(x):
            r = my_residual(x)
            r0_norm = torch.linalg.norm(r)
            iteration = 0
            max_its = 50
            while torch.linalg.norm(r)/r0_norm > 1e-10 and iteration < max_its:
                J = jacobian(x)
                dx = torch.linalg.solve(J, -r)
                x = x + dx
                r = my_residual(x)
                iteration += 1
            return x[:]

        inputs = self.un_*1.
        solution = my_newton(inputs)

        self.unp1_[:] = solution[:]
        self.uddotnp1_[:] = 1./(dt**2*beta)*(self.unp1_ - self.un_) - \
            1./(beta*dt)*self.udotn_ - 1./(2*beta)*(1 - 2.*beta)*self.uddotn_
        self.udotnp1_[:] = self.udotn_ + \
            (1. - gamma)*dt*self.uddotn_ + gamma*dt*self.uddotnp1_

    def advance_n_steps(self, un, udotn, uddotn, dt, n_steps):
        self.un_[:] = torch.tensor(un[:])
        self.udotn_[:] = torch.tensor(udotn[:])
        self.uddotn_[:] = torch.tensor(uddotn[:])
        for i in range(0, n_steps):
            # bcs = bc_hook(i)
            self.advance_newmark(dt)
            self.update_states_newmark()
        return self.u_history_

    def get_unp1(self):
        return self.unp1_
