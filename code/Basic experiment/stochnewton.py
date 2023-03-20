import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output


class DeterministicNewtonLogReg:
    def __init__(self, A, b, l2=0, opt_value=None):

        self.A = A
        self.b = b
        self.l2 = l2
        self.n_obj, self.dim = self.A.shape

        self.opt_value = opt_value

        self.exp_border = 20  # Обрезаем большие числа до 20, чтобы не подавать их в экспоненту

        self.x = np.zeros(self.dim)
        self.grad = np.zeros(self.dim)
        self.hessian = np.zeros((self.dim, self.dim))
        self.initialize_derivatives()

        self.loss_history = []
        self.x_norms = []
        self.grad_norms = []
        self.hess_norms = []
        self.iterations = 0  # Количество сделаннных итераций (к текущему моменту)

    def run(self, n_iterations=20, plot_loss=True, plot_norms=False, **kwargs):
        for it_num in range(n_iterations):
            self.step()
            self.update_loss()
            if plot_loss:
                self.plot_loss(**kwargs)
            elif plot_norms:
                self.plot_norms(**kwargs)
            self.iterations += 1
        return self

    def step(self):
        self.x -= np.linalg.solve(self.hessian, self.grad)
        self.update_grad()
        self.update_hessians()
        return None

    def logistic(self, x, A, b, l2):
        logits = A @ x
        logits = np.clip(logits, -self.exp_border, self.exp_border)
        loss = np.mean(np.log(1 + np.exp(-b * logits)))
        loss += (l2 / 2) * np.linalg.norm(x)**2
        return loss

    def calculate_loss(self):
        return self.logistic(self.x, self.A, self.b, self.l2)

    def update_loss(self):
        self.loss_history.append(self.calculate_loss())
        self.x_norms.append(np.linalg.norm(self.x))
        self.grad_norms.append(np.linalg.norm(self.grad))
        self.hess_norms.append(np.linalg.norm(self.hessian))
        return None

    def initialize_derivatives(self):
        self.update_grad()
        self.update_hessians()
        return self

    def update_grad(self):
        logits = self.A @ self.x
        logits = np.clip(logits, -self.exp_border, self.exp_border)
        self.grad = np.mean(self.A * (-self.b / (1 + np.exp(self.b * logits)))[:, None], axis=0)
        self.grad += self.l2 * self.x
        return self.grad

    def update_hessians(self, idxs=None):
        logits = self.A @ self.x
        logits = np.clip(logits, -self.exp_border, self.exp_border)
        self.hessian = np.mean(self.A[:, :, None] @ self.A[:, None, :] *
                               (np.exp(self.b * logits) /
                                (1 + np.exp(self.b * logits)) /
                                (1 + np.exp(self.b * logits))).reshape(-1, 1, 1),
                               axis=0)
        self.hessian += self.l2 * np.eye(self.dim)
        return self.hessian

    def plot_loss(self, yscale='log'):
        clear_output(wait=True)
        if self.opt_value is not None:
            plt.plot(np.array(self.loss_history) - self.opt_value, label='f(x) - f*')
            plt.title('Loss history')
            plt.xlabel('Number of iterations')
            plt.ylabel('f(x) - f*')
        else:
            plt.plot(self.loss_history, label='loss')
            plt.title('Loss history')
            plt.xlabel('Number of iterations')
            plt.ylabel('Loss')
        plt.yscale(yscale)
        plt.legend()
        plt.show()

    def plot_norms(self, yscale='log'):
        clear_output(wait=True)
        plt.plot(self.x_norms, label='x norm')
        plt.plot(self.grad_norms, label='grad norm')
        plt.plot(self.hess_norms, label='hess norm')
        plt.title('Norms history')
        plt.xlabel('Number of iterations')
        plt.ylabel('Norms')
        plt.yscale(yscale)
        plt.legend()
        plt.show()


class StochasticNewtonLogReg:
    def __init__(self, A, b, l2=0, opt_value=None):

        self.A = A
        self.b = b
        self.l2 = l2
        self.n_obj, self.dim = self.A.shape

        self.opt_value = opt_value

        self.exp_border = 20  # Обрезаем большие числа до 20, чтобы не подавать их в экспоненту

        self.x = np.zeros(self.dim)
        self.W = np.zeros((self.n_obj, self.dim))
        self.grads = np.zeros((self.n_obj, self.dim))
        self.hessians = np.zeros((self.n_obj, self.dim, self.dim))
        self.initialize_derivatives()

        self.batch_size = self.n_obj

        self.loss_history = []
        self.x_norms = []
        self.grads_norms = []
        self.hess_norms = []
        self.iterations = 0  # Количество сделаннных итераций (к текущему моменту)

    def run(self, n_iterations=500, batch_size=10, plot_loss=True, plot_norms=False, **kwargs):
        self.batch_size = batch_size
        for it_num in range(n_iterations):
            self.step()
            self.update_loss()
            if plot_loss:
                self.plot_loss(**kwargs)
            elif plot_norms:
                self.plot_norms(**kwargs)
            self.iterations += 1
        return self

    def step(self):
        self.x = np.linalg.solve(np.mean(self.hessians, axis=0),
                                 np.mean((self.hessians @ self.W[:, :, None]).squeeze() -
                                         self.grads, axis=0))
        idxs_to_update = self.make_batch_idxs(self.batch_size)
        self.W[idxs_to_update] = [self.x.copy() for idx in idxs_to_update]
        self.update_grads(idxs_to_update)
        self.update_hessians(idxs_to_update)
        return None

    def logistic(self, x, A, b, l2):
        logits = A @ x
        logits = np.clip(logits, -self.exp_border, self.exp_border)
        loss = np.mean(np.log(1 + np.exp(-b * logits)))
        loss += (l2 / 2) * np.linalg.norm(x) ** 2
        return loss

    def calculate_loss(self):
        return self.logistic(self.x, self.A, self.b, self.l2)

    def update_loss(self):
        self.loss_history.append(self.calculate_loss())
        self.x_norms.append(np.linalg.norm(self.x))
        self.grads_norms.append(np.linalg.norm(np.mean(self.grads, axis=0)))
        self.hess_norms.append(np.linalg.norm(np.mean(self.hessians, axis=0)))
        return None

    def make_batch_idxs(self, batch_size):
        idxs = np.random.choice(self.n_obj, size=batch_size, replace=False)
        return idxs

    def initialize_derivatives(self):
        self.update_grads()
        self.update_hessians()
        return self

    def update_grads(self, idxs=None):
        if idxs is None:
            idxs = range(self.n_obj)
        logits = (self.A[idxs, None, :] @ self.W[idxs, :, None]).squeeze()
        logits = np.clip(logits, -self.exp_border, self.exp_border)
        self.grads[idxs] = self.A[idxs] * (-self.b[idxs] / (1 + np.exp(self.b[idxs] * logits)))[:, None]
        self.grads[idxs] += self.l2 * self.W[idxs]
        return self.grads

    def update_hessians(self, idxs=None):
        if idxs is None:
            idxs = range(self.n_obj)
        logits = (self.A[idxs, None, :] @ self.W[idxs, :, None]).squeeze()
        logits = np.clip(logits, -self.exp_border, self.exp_border)
        self.hessians[idxs] = self.A[idxs, :, None] @ self.A[idxs, None, :] * \
                              (np.exp(self.b[idxs] * logits) /
                               (1 + np.exp(self.b[idxs] * logits)) /
                               (1 + np.exp(self.b[idxs] * logits))).reshape(-1, 1, 1)
        self.hessians[idxs] += np.array([self.l2 * np.eye(self.dim) for idx in idxs])
        return self.hessians

    def plot_loss(self, yscale='log'):
        if self.iterations % (self.n_obj // self.batch_size) == 0:
            clear_output(wait=True)
            if self.opt_value is not None:
                plt.plot(np.array(self.loss_history) - self.opt_value, label='f(x) - f*(x)')
                plt.title('Loss history')
                plt.xlabel('Number of iterations')
                plt.ylabel('f(x) - f*')
            else:
                plt.plot(self.loss_history, label='loss')
                plt.title('Loss history')
                plt.xlabel('Number of iterations')
                plt.ylabel('Loss')
            plt.yscale(yscale)
            plt.legend()
            plt.show()

    def plot_norms(self, yscale='log'):
        if self.iterations % (self.n_obj // self.batch_size) == 0:
            clear_output(wait=True)
            plt.plot(self.x_norms, label='x norm')
            plt.plot(self.grads_norms, label='grads norm')
            plt.plot(self.hess_norms, label='hess norm')
            plt.title('Norms history')
            plt.xlabel('Number of iterations')
            plt.ylabel('Norms')
            plt.yscale(yscale)
            plt.legend()
            plt.show()


# ------ Это пока не используется ------


# class DeterministicGDLogReg:
#     def __init__(self, A, b, l2=0):
#
#         self.A = A
#         self.b = b
#         self.l2 = l2
#         self.n_obj, self.dim = self.A.shape
#
#         self.exp_border = 20  # Обрезаем больших числа до 20, чтобы не подавать их в экспоненту
#
#         self.x = np.zeros(self.dim)
#         self.grad = np.zeros(self.dim)
#         self.initialize_derivatives()
#
#         self.lr = 1e-3
#
#         self.loss_history = []
#         self.x_norms = []
#         self.grad_norms = []
#         self.iterations = 0
#
#     def run(self, n_iterations=100, lr=1e-3, plot_loss=True, plot_norms=False, **kwargs):
#         self.lr = lr
#         for it_num in range(n_iterations):
#             self.step()
#             self.update_loss()
#             if plot_loss:
#                 self.plot_loss(**kwargs)
#             elif plot_norms:
#                 self.plot_norms(**kwargs)
#             self.iterations += 1
#         return self
#
#     def step(self):
#         self.x -= self.lr * self.grad
#         self.update_grad()
#         return None
#
#     def logistic(self, x, A, b, l2):
#         logits = A @ x
#         logits = np.clip(logits, -self.exp_border, self.exp_border)
#         loss = np.mean(np.log(1 + np.exp(-b * logits)))
#         loss += (l2 / 2) * np.linalg.norm(x) ** 2
#         return loss
#
#     def calculate_loss(self):
#         return self.logistic(self.x, self.A, self.b, self.l2)
#
#     def update_loss(self):
#         self.loss_history.append(self.calculate_loss())
#         self.x_norms.append(np.linalg.norm(self.x))
#         self.grad_norms.append(np.linalg.norm(self.grad))
#         return None
#
#     def initialize_derivatives(self):
#         self.update_grad()
#         return self
#
#     def update_grad(self):
#         logits = self.A @ self.x
#         logits = np.clip(logits, -self.exp_border, self.exp_border)
#         self.grad = np.mean(self.A * (-self.b / (1 + np.exp(self.b * logits)))[:, None], axis=0)
#         self.grad += self.l2 * self.x
#         return self.grad
#
#     def plot_loss(self, yscale='linear'):
#         clear_output(wait=True)
#         plt.plot(self.loss_history, label='loss')
#         plt.title('Loss history')
#         plt.xlabel('Number of iterations')
#         plt.ylabel('Loss')
#         plt.yscale(yscale)
#         plt.legend()
#         plt.show()
#
#     def plot_norms(self, yscale='log'):
#         clear_output(wait=True)
#         plt.plot(self.x_norms, label='x norm')
#         plt.plot(self.grad_norms, label='grad norm')
#         plt.title('Norms history')
#         plt.xlabel('Number of iterations')
#         plt.ylabel('Norms')
#         plt.yscale(yscale)
#         plt.legend()
#         plt.show()
