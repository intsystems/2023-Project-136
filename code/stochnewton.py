import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output


exp_border = 20  # Для численной стабильности логитов - обрезаем больших числа до 20, чтобы не подавать их в экспоненту


def logistic(x, A, b):
    logits = A @ x
    logits = np.clip(logits, -exp_border, exp_border)
    loss = np.mean(np.log(1 + np.exp(-b * logits)))
    return loss


class DeterministicNewtonLogReg:
    def __init__(self, A, b, l2=0, initialization=None, opt_value=None):

        self.A = A
        self.b = b
        self.l2 = l2
        self.n_obj, self.dim = self.A.shape

        self.opt_value = opt_value

        if initialization is not None:
            self.x = initialization
        else:
            self.x = np.zeros(self.dim)
        self.grad = np.zeros(self.dim)
        self.hessian = np.zeros((self.dim, self.dim))
        self.initialize_derivatives()

        self.loss_history = [self.calculate_loss(), ]
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

    def calculate_loss(self):
        loss = logistic(self.x, self.A, self.b)
        loss += (self.l2 / 2) * np.linalg.norm(self.x)**2
        return loss

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
        logits = np.clip(logits, -exp_border, exp_border)
        self.grad = np.mean(self.A * (-self.b / (1 + np.exp(self.b * logits)))[:, None], axis=0)
        self.grad += self.l2 * self.x
        return self.grad

    def update_hessians(self):
        logits = self.A @ self.x
        logits = np.clip(logits, -exp_border, exp_border)
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
    def __init__(self, A, b, l2=0, initialization=None, opt_value=None, opt_x=None):

        self.A = A
        self.b = b
        self.l2 = l2
        self.n_obj, self.dim = self.A.shape
        self.batch_size = self.n_obj

        self.opt_value = opt_value
        self.opt_x = opt_x

        if initialization is not None:
            self.x = initialization
            self.W = np.repeat(initialization[None, :], self.n_obj, axis=0)
        else:
            self.x = np.zeros(self.dim)
            self.W = np.zeros((self.n_obj, self.dim))
        self.grads = np.zeros((self.n_obj, self.dim))
        self.hessians = np.zeros((self.n_obj, self.dim, self.dim))
        self.initialize_derivatives()

        self.grads_lipschitzness = 1 / 4 * np.linalg.norm(self.A, axis=1) ** 2 + self.l2
        self.hessians_lipschitzness = 1/10 * np.linalg.norm(self.A, axis=1)**3

        self.loss_history = [self.calculate_loss(), ]
        self.dist_history = [self.calculate_dist(), ]
        self.x_norms = []
        self.grads_norms = []
        self.hess_norms = []
        self.iterations = 0  # Количество сделаннных итераций (к текущему моменту)

    def run(self, n_iterations=1000, batch_size=1, plot_loss=True, plot_norms=False, **kwargs):
        self.batch_size = batch_size
        for it_num in range(n_iterations):
            self.step(**kwargs)
            self.update_loss()
            if self.opt_x is not None:
                if self.calculate_dist() < 1e-15:
                    self.W = np.repeat(self.opt_x[None, :], self.n_obj, axis=0)
            if plot_loss:
                self.plot_loss(**kwargs)
            elif plot_norms:
                self.plot_norms(**kwargs)
            self.iterations += 1
        return self

    def step(self, **kwargs):
        self.x = np.linalg.solve(np.mean(self.hessians, axis=0),
                                 np.mean((self.hessians @ self.W[:, :, None]).squeeze() -
                                         self.grads, axis=0))
        idxs_to_update = self.make_batch_idxs(self.batch_size, **kwargs)
        self.W[idxs_to_update] = np.repeat(self.x[None, :], self.batch_size, axis=0)
        self.update_grads(idxs_to_update)
        self.update_hessians(idxs_to_update)
        return None

    def calculate_loss(self):
        loss = logistic(self.x, self.A, self.b)
        loss += (self.l2 / 2) * np.linalg.norm(self.x)**2
        return loss

    def calculate_dist(self):
        if self.opt_x is not None:
            dist = np.mean(np.linalg.norm(self.W - self.opt_x, axis=1)**2)
            if dist < 1e-15:
                dist = 0
            return dist
        return None

    def update_loss(self):
        self.loss_history.append(self.calculate_loss())
        self.dist_history.append(self.calculate_dist())
        self.x_norms.append(np.linalg.norm(self.x))
        self.grads_norms.append(np.linalg.norm(np.mean(self.grads, axis=0)))
        self.hess_norms.append(np.linalg.norm(np.mean(self.hessians, axis=0)))
        return None

    def make_batch_idxs(self, batch_size=1, strategy='nice', **kwargs):
        idxs = None
        if strategy == 'nice':
            idxs = np.random.choice(self.n_obj, size=batch_size, replace=False)
        elif strategy == 'imp-grad':
            p = self.grads_lipschitzness / np.sum(self.grads_lipschitzness)
            idxs = np.random.choice(self.n_obj, size=batch_size, p=p)
        elif strategy == 'imp-hess':
            p = self.hessians_lipschitzness / np.sum(self.hessians_lipschitzness)
            idxs = np.random.choice(self.n_obj, size=batch_size, p=p)
        return idxs

    def initialize_derivatives(self):
        self.update_grads()
        self.update_hessians()
        return self

    def update_grads(self, idxs=None):
        if idxs is None:
            idxs = range(self.n_obj)
        logits = (self.A[idxs, None, :] @ self.W[idxs, :, None]).squeeze()
        logits = np.clip(logits, -exp_border, exp_border)
        self.grads[idxs] = self.A[idxs] * (-self.b[idxs] / (1 + np.exp(self.b[idxs] * logits)))[:, None]
        self.grads[idxs] += self.l2 * self.W[idxs]
        return self.grads

    def update_hessians(self, idxs=None):
        if idxs is None:
            idxs = range(self.n_obj)
        logits = (self.A[idxs, None, :] @ self.W[idxs, :, None]).squeeze()
        logits = np.clip(logits, -exp_border, exp_border)
        self.hessians[idxs] = self.A[idxs, :, None] @ self.A[idxs, None, :] * \
                              (np.exp(self.b[idxs] * logits) /
                               (1 + np.exp(self.b[idxs] * logits)) /
                               (1 + np.exp(self.b[idxs] * logits))).reshape(-1, 1, 1)
        self.hessians[idxs] += np.repeat(self.l2 * np.eye(self.dim)[None, :, :], self.batch_size, axis=0)
        return self.hessians

    def plot_loss(self, yscale='log', plot_dist=True, **kwargs):
        if self.iterations % (self.n_obj // self.batch_size) == 0:
            clear_output(wait=True)
            if self.opt_x is not None and plot_dist:
                plt.plot(self.dist_history, label='W')
                plt.title('Dist history')
                plt.xlabel('Number of iterations')
                plt.ylabel('W')
            elif self.opt_value is not None:
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

    def plot_norms(self, yscale='log', **kwargs):
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


# Не используется
class DeterministicGDLogReg:
    def __init__(self, A, b, l2=0, initialization=None):

        self.A = A
        self.b = b
        self.l2 = l2
        self.n_obj, self.dim = self.A.shape

        if initialization is not None:
            self.x = initialization
        else:
            self.x = np.zeros(self.dim)

        self.grad = np.zeros(self.dim)
        self.initialize_derivatives()

        self.lr = 1e-2

        self.loss_history = [self.calculate_loss(), ]
        self.x_norms = []  # Для отладки
        self.grad_norms = []  # Для отладки
        self.iterations = 0

    def run(self, n_iterations=10000, lr=None, plot_loss=True, plot_norms=False, **kwargs):
        if lr is not None:
            self.lr = lr
        for it_num in range(n_iterations):
            self.step()
            self.update_loss()
            if plot_loss:
                self.plot_loss(**kwargs)
            elif plot_norms:  # Для отладки
                self.plot_norms(**kwargs)
            self.iterations += 1
        return self

    def step(self):
        self.x -= self.lr * self.grad
        self.update_grad()
        return None

    def calculate_loss(self):
        loss = logistic(self.x, self.A, self.b)
        loss += (self.l2 / 2) * np.linalg.norm(self.x)**2
        return loss

    def update_loss(self):
        self.loss_history.append(self.calculate_loss())
        self.x_norms.append(np.linalg.norm(self.x))
        self.grad_norms.append(np.linalg.norm(self.grad))
        return None

    def initialize_derivatives(self):
        self.update_grad()
        return self

    def update_grad(self):
        logits = self.A @ self.x
        logits = np.clip(logits, -exp_border, exp_border)
        self.grad = np.mean(self.A * (-self.b / (1 + np.exp(self.b * logits)))[:, None], axis=0)
        self.grad += self.l2 * self.x
        return self.grad

    def plot_loss(self, yscale='linear'):
        if self.iterations % 100 == 0:
            clear_output(wait=True)
            plt.plot(self.loss_history, label='loss')
            plt.title('Loss history')
            plt.xlabel('Number of iterations')
            plt.ylabel('Loss')
            plt.yscale(yscale)
            plt.legend()
            plt.show()

    def plot_norms(self, yscale='log'):  # Для отладки
        if self.iterations % 100 == 0:
            clear_output(wait=True)
            plt.plot(self.x_norms, label='x norm')
            plt.plot(self.grad_norms, label='grad norm')
            plt.title('Norms history')
            plt.xlabel('Number of iterations')
            plt.ylabel('Norms')
            plt.yscale(yscale)
            plt.legend()
            plt.show()
