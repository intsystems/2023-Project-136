import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

from IPython.display import clear_output


exp_border = 20  # Для численной стабильности логитов - обрезаем больших числа до 20, чтобы не подавать их в экспоненту
# For numerical stability


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
        self.iterations = 0  # Количество сделаннных итераций (к текущему моменту)
        # Current number of iterations done

    def run(self, n_iterations=20, plot_loss=True, plot_norms=False, **kwargs):
        for it_num in range(n_iterations):
            self.step()
            self.update_loss()
            if plot_loss:
                self.plot_loss(**kwargs)
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
        self.iterations = 0  # Количество сделаннных итераций (к текущему моменту)
        # Current number of iterations done

    def run(self, n_iterations=1000, batch_size=1, plot_loss=True, **kwargs):
        self.batch_size = batch_size
        for it_num in range(n_iterations):
            self.step(**kwargs)
            self.update_loss()
            if self.opt_x is not None:
                if self.calculate_dist() < 1e-15:
                    self.W = np.repeat(self.opt_x[None, :], self.n_obj, axis=0)
            if plot_loss:
                self.plot_loss(**kwargs)
            self.iterations += 1
        return self

    def step(self, **kwargs):
        self.x = np.linalg.solve(np.mean(self.hessians, axis=0),
                                 np.mean((self.hessians @ self.W[:, :, None]).squeeze() -
                                         self.grads, axis=0))
        idxs_to_update = self.make_batch_idxs(self.batch_size, **kwargs)
        actual_batch_size = idxs_to_update.shape[0]
        self.W[idxs_to_update] = np.repeat(self.x[None, :], actual_batch_size, axis=0)
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
        return None

    def make_batch_idxs(self, batch_size=1, strategy='nice', tau_bin_p=1, **kwargs):
        idxs = None
        if strategy == 'nice':
            idxs = np.random.choice(self.n_obj, size=batch_size, replace=False)
        elif strategy == 'tau-ind':
            idxs = np.random.choice(self.n_obj, size=batch_size, replace=True)
        elif strategy == 'tau-bin':
            actual_batch_size = sps.binom.rvs(batch_size, tau_bin_p)
            idxs = np.random.choice(self.n_obj, size=actual_batch_size, replace=False)
        elif strategy == 'imp-grad':
            p = self.grads_lipschitzness / np.sum(self.grads_lipschitzness)
            idxs = np.random.choice(self.n_obj, size=batch_size, p=p)
        elif strategy == 'imp-hess':
            p = self.hessians_lipschitzness / np.sum(self.hessians_lipschitzness)
            idxs = np.random.choice(self.n_obj, size=batch_size, p=p)
        elif strategy == 'inv-imp-grad':
            p = 1/self.grads_lipschitzness / np.sum(1/self.grads_lipschitzness)
            idxs = np.random.choice(self.n_obj, size=batch_size, p=p)
        elif strategy == 'inv-imp-hess':
            p = 1/self.hessians_lipschitzness / np.sum(1/self.hessians_lipschitzness)
            idxs = np.random.choice(self.n_obj, size=batch_size, p=p)
        elif strategy == 'consec':
            if self.iterations == 0:
                self.order = np.random.permutation(self.n_obj)
            start = (self.iterations * batch_size) % self.n_obj
            finish = np.min((start + batch_size, self.n_obj))
            idxs = self.order[start:finish]
        return idxs

    def initialize_derivatives(self):
        self.update_grads()
        self.update_hessians()
        return self

    def update_grads(self, idxs=None):
        if idxs is None:
            idxs = np.arange(self.n_obj)
        logits = (self.A[idxs, None, :] @ self.W[idxs, :, None]).squeeze()
        logits = np.clip(logits, -exp_border, exp_border)
        self.grads[idxs] = self.A[idxs] * (-self.b[idxs] / (1 + np.exp(self.b[idxs] * logits)))[:, None]
        self.grads[idxs] += self.l2 * self.W[idxs]
        return self.grads

    def update_hessians(self, idxs=None):
        if idxs is None:
            idxs = np.arange(self.n_obj)
        logits = (self.A[idxs, None, :] @ self.W[idxs, :, None]).squeeze()
        logits = np.clip(logits, -exp_border, exp_border)
        self.hessians[idxs] = self.A[idxs, :, None] @ self.A[idxs, None, :] * \
                              (np.exp(self.b[idxs] * logits) /
                               (1 + np.exp(self.b[idxs] * logits)) /
                               (1 + np.exp(self.b[idxs] * logits))).reshape(-1, 1, 1)
        actual_batch_size = idxs.shape[0]
        self.hessians[idxs] += np.repeat(self.l2 * np.eye(self.dim)[None, :, :], actual_batch_size, axis=0)
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