import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output


class StochasticNewtonLinReg:
    def __init__(self, A, b, deterministic=False, l2=None):
        
        self.A = A
        self.b = b
        self.n_obj, self.dim = A.shape
        self.deterministic = deterministic
        
        params_initialization_scale = np.sqrt((self.b**2).mean()) / np.sqrt((self.A**2).mean())
        self.x = np.random.rand(self.dim) * params_initialization_scale
        if self.deterministic: # Костыль для отладки в детерминированном режиме
            self.W = np.array([self.x.copy() for idx in range(self.n_obj)])
        else:
            self.W = np.random.rand(self.n_obj, self.dim) * params_initialization_scale
        
        self.l2 = 0 if l2 is None else l2
        
        self.grads = np.zeros((self.n_obj, self.dim))
        self.hessians = np.zeros((self.n_obj, self.dim, self.dim))
        self.initialize_derivatives()
        
        self.loss_history_labels = ['use_w, include_reg', 'use_x, include_reg', 'use_w, not_include_reg', 'use_x, not_include_reg']
        self.loss_history = {}
        for loss_history_label in self.loss_history_labels:
            self.loss_history[loss_history_label] = []
        
    def run(self, n_iterations=1000, batch_size=10, plot=True, plot_yscale='linear'):
        self.batch_size=batch_size
        for it_num in range(n_iterations):
            self.step()
            self.update_loss()
            if plot:
                self.plot_loss(yscale=plot_yscale)
        return self
    
    def step(self):
        self.x = np.linalg.inv(np.mean(self.hessians, axis=0)) @ \
                                 np.mean([self.hessians[i] @ self.W[i] - self.grads[i] for i in range(self.n_obj)], axis=0)
        if self.deterministic:
            idxs_to_update = range(self.n_obj)
        else:
            idxs_to_update = self.make_batch_idxs(self.batch_size)
        self.W[idxs_to_update] = [self.x.copy() for idx in idxs_to_update]
        self.update_grads(idxs_to_update)
        self.update_hessians(idxs_to_update)
        return None
    
    def calculate_loss(self, use_w=False, include_reg=False):
        if use_w:
            loss = 0.5 * np.mean((np.sum(self.A * self.W, axis=1) - self.b)**2)
            if include_reg:
                loss += self.l2 * np.mean(np.sum(self.W**2, axis=1))
        else:
            loss = 0.5 * np.mean((self.A @ self.x - self.b)**2)
            if include_reg:
                loss += self.l2 * np.sum(self.x**2)
        return loss
    
    def update_loss(self):
        loss_kind_num = 0
        for include_reg in [True, False]:
            for use_w in [True, False]:
                self.loss_history[self.loss_history_labels[loss_kind_num]].append(self.calculate_loss(use_w, include_reg))
                loss_kind_num += 1
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
        self.grads[idxs] = self.A[idxs] * (np.sum(self.A[idxs] * self.W[idxs], axis=1) - self.b[idxs])[:, None]
        self.grads[idxs] += self.l2 * np.sum(self.W[idxs]**2, axis=1)[:, None]
        return self.grads
        
    def update_hessians(self, idxs=None):
        if idxs is None:
            idxs = range(self.n_obj)
        self.hessians[idxs] = np.array([np.outer(self.A[i], self.A[i]) for i in idxs])
        self.hessians[idxs] += np.array([self.l2 * np.eye(self.dim) for i in idxs])
        return self.hessians
    
    def plot_loss(self, yscale='linear'):
        if len(self.loss_history[self.loss_history_labels[0]]) % 50 == 0:
            clear_output(wait=True)
            
            for loss_history_label in self.loss_history_labels:
                plt.plot(self.loss_history[loss_history_label], label=loss_history_label)
            plt.title('Loss history')
            plt.xlabel('Number of iterations')
            plt.ylabel('Loss')
            plt.yscale(yscale)
            plt.legend()
            plt.show()
            
            
class StochasticNewtonLogReg:
    def __init__(self, A, b, deterministic=False, l2=None):
        
        self.A = A
        self.b = b
        self.n_obj, self.dim = A.shape
        self.deterministic = deterministic
        
        params_initialization_scale = np.sqrt((self.b**2).mean()) / np.sqrt((self.A**2).mean())
        self.x = np.random.rand(self.dim) * params_initialization_scale
        if self.deterministic: # Костыль для отладки в детерминированном режиме
            self.W = np.array([self.x.copy() for idx in range(self.n_obj)])
        else:
            self.W = np.random.rand(self.n_obj, self.dim) * params_initialization_scale
        
        self.l2 = 0 if l2 is None else l2
        
        self.grads = np.zeros((self.n_obj, self.dim))
        self.hessians = np.zeros((self.n_obj, self.dim, self.dim))
        self.initialize_derivatives()
        
        self.loss_history_labels = ['use_w, include_reg', 'use_x, include_reg', 'use_w, not_include_reg', 'use_x, not_include_reg']
        self.loss_history = {}
        for loss_history_label in self.loss_history_labels:
            self.loss_history[loss_history_label] = []
        
    def run(self, n_iterations=1000, batch_size=10, plot=True, plot_yscale='linear'):
        self.batch_size=batch_size
        for it_num in range(n_iterations):
            self.step()
            self.update_loss()
            if plot:
                self.plot_loss(yscale=plot_yscale)
        return self
    
    def step(self):
        self.x = np.linalg.inv(np.mean(self.hessians, axis=0)) @ \
                                 np.mean([self.hessians[i] @ self.W[i] - self.grads[i] for i in range(self.n_obj)], axis=0)
        if self.deterministic:
            idxs_to_update = range(self.n_obj)
        else:
            idxs_to_update = self.make_batch_idxs(self.batch_size)
        self.W[idxs_to_update] = [self.x.copy() for idx in idxs_to_update]
        self.update_grads(idxs_to_update)
        self.update_hessians(idxs_to_update)
        return None
    
    # To be changed
    def calculate_loss(self, use_w=False, include_reg=False):
        if use_w:
            loss = 0.5 * np.mean((np.sum(self.A * self.W, axis=1) - self.b)**2)
            if include_reg:
                loss += self.l2 * np.mean(np.sum(self.W**2, axis=1))
        else:
            loss = 0.5 * np.mean((self.A @ self.x - self.b)**2)
            if include_reg:
                loss += self.l2 * np.sum(self.x**2)
        return loss
    
    def update_loss(self):
        loss_kind_num = 0
        for include_reg in [True, False]:
            for use_w in [True, False]:
                self.loss_history[self.loss_history_labels[loss_kind_num]].append(self.calculate_loss(use_w, include_reg))
                loss_kind_num += 1
        return None
    
    def make_batch_idxs(self, batch_size):
        idxs = np.random.choice(self.n_obj, size=batch_size, replace=False)
        return idxs
    
    def initialize_derivatives(self):
        self.update_grads()
        self.update_hessians()
        return self
    
    # To be changed
    def update_grads(self, idxs=None):
        if idxs is None:
            idxs = range(self.n_obj)
        self.grads[idxs] = self.A[idxs] * (np.sum(self.A[idxs] * self.W[idxs], axis=1) - self.b[idxs])[:, None]
        self.grads[idxs] += self.l2 * np.sum(self.W[idxs]**2, axis=1)[:, None]
        return self.grads
        
    # To be changed
    def update_hessians(self, idxs=None):
        if idxs is None:
            idxs = range(self.n_obj)
        self.hessians[idxs] = np.array([np.outer(self.A[i], self.A[i]) for i in idxs])
        self.hessians[idxs] += np.array([self.l2 * np.eye(self.dim) for i in idxs])
        return self.hessians
    
    def plot_loss(self, yscale='linear'):
        if len(self.loss_history[self.loss_history_labels[0]]) % 50 == 0:
            clear_output(wait=True)
            
            for loss_history_label in self.loss_history_labels:
                plt.plot(self.loss_history[loss_history_label], label=loss_history_label)
            plt.title('Loss history')
            plt.xlabel('Number of iterations')
            plt.ylabel('Loss')
            plt.yscale(yscale)
            plt.legend()
            plt.show()