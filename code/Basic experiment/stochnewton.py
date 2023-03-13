import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression
from sklearn.metrics import log_loss as sklearn_log_loss

from IPython.display import clear_output
            
            
class StochasticNewtonLogReg:
    def __init__(self, A, b, deterministic=False, l2=None):

        self.A = np.hstack((A, np.ones((A.shape[0], 1))))
        self.b = b
        self.n_obj, self.dim = self.A.shape
        self.deterministic = deterministic

        self.eps = 1e-8
        self.exp_border = 20
        
        params_initialization_scale = np.sqrt((self.b**2).mean()) / np.sqrt((self.A**2).mean())
        self.x = np.random.rand(self.dim) * params_initialization_scale
        if self.deterministic: # Для работы в детерминированном режиме
            self.W = np.array([self.x.copy() for idx in range(self.n_obj)])
        else:
            self.W = np.random.rand(self.n_obj, self.dim) * params_initialization_scale
        
        self.l2 = 0 if l2 is None else l2
        
        self.grads = np.zeros((self.n_obj, self.dim))
        self.hessians = np.zeros((self.n_obj, self.dim, self.dim))
        self.initialize_derivatives()

        sklearn_model = sklearn_LogisticRegression(C=np.inf if self.l2 == 0 else 1/self.l2, solver='newton-cg').fit(self.A, self.b)
        self.opt_x = sklearn_model.coef_
        self.opt_value = sklearn_log_loss(self.b, sklearn_model.predict_proba(self.A))

        self.loss_history = []
        self.iterations = 0
        
    def run(self, n_iterations=1000, batch_size=10, plot=True, plot_yscale='log'):
        if self.deterministic:
            self.batch_size = self.n_obj
        else:
            self.batch_size = batch_size
        for it_num in range(n_iterations):
            old_x = self.x
            self.step()
            if self.calculate_loss() < self.opt_value:
                break
            self.update_loss()
            if plot:
                self.plot_loss(yscale=plot_yscale)
            self.iterations = it_num
        return self
    
    def step(self):
        self.x = np.linalg.inv(np.mean(self.hessians, axis=0)) @ \
                                 np.mean([self.hessians[i] @ self.W[i] - self.grads[i] for i in range(self.n_obj)], axis=0)
        idxs_to_update = self.make_batch_idxs(self.batch_size)
        self.W[idxs_to_update] = [self.x.copy() for idx in idxs_to_update]
        self.update_grads(idxs_to_update)
        self.update_hessians(idxs_to_update)
        return None

    def sigmoid(self, x):
        x = np.clip(x, -self.exp_border, self.exp_border)
        return 1 / (1 + np.exp(-x))

    def logloss(self, p):
        p = np.clip(p, self.eps, 1 - self.eps)
        return - np.mean(self.b * np.log(p) + (1 - self.b) * np.log(1 - p))

    def calculate_loss(self):
        logits = self.A @ self.x
        p = self.sigmoid(logits)
        loss = self.logloss(p)
        return loss
    
    def update_loss(self):
        self.loss_history.append(self.calculate_loss())
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
        logits = np.array([self.A[i] @ self.W[i] for i in idxs])
        p = self.sigmoid(logits)
        self.grads[idxs] = self.A[idxs] * (p - self.b[idxs])[:, None]
        self.grads[idxs] += self.l2 * 2 * self.W[idxs]
        return self.grads

    def update_hessians(self, idxs=None):
        if idxs is None:
            idxs = range(self.n_obj)
        logits = np.array([self.A[i] @ self.W[i] for i in idxs])
        logits = np.clip(logits, -self.exp_border, self.exp_border)
        self.hessians[idxs] = np.array([np.outer(self.A[i], self.A[i]) * np.exp(-logits[j]) * self.sigmoid(logits[j])**2 \
                                        for j, i in enumerate(idxs)])
        self.hessians[idxs] += np.array([self.l2 * 2 * np.eye(self.dim) for i in idxs])
        return self.hessians
    
    def plot_loss(self, yscale='log'):
        if self.iterations % self.n_obj // self.batch_size == 0:
            clear_output(wait=True)
            plt.plot(np.array(self.loss_history) - self.opt_value, label='loss')
            plt.title('Loss history')
            plt.xlabel('Number of iterations')
            plt.ylabel('$f - f^*$')
            plt.yscale(yscale)
            plt.legend()
            plt.show()
