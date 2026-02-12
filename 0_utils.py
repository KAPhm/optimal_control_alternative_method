import time
import os
import torch
import logging
from scipy.optimize import minimize_scalar
import sys

# Function to create a timestap
def timestamp():
    return time.strftime('%Y%m%d_%H%M%S')

# Function to create a directoy with a timestamp
def create_dir(basedir='./', dirname = "results", suffix = None):
    if suffix is None:
        suffix = "_" + timestamp()

    dir_path = os.path.join(basedir, f"{dirname}{suffix}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# Function to save a torch model, optimizer, sceduler and epoch
def save_checkpoint(model, optimizer = None, scheduler_step = None, scheduler_expo = None, 
                    epoch = None, basedir='models', suffix = None, additional_info = {}):
    if suffix is None:
        suffix = timestamp()
    checkpoint_path = os.path.join(basedir, f'checkpoint_{suffix}.pth')
    to_save = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if not optimizer is None else None,
        'scheduler_step_state_dict': scheduler_step.state_dict()if not scheduler_step is None else None,
        'scheduler_expo_state_dict': scheduler_expo.state_dict()if not scheduler_expo is None else None,
        'additional_info': additional_info
    }
    torch.save(to_save, checkpoint_path)
    return checkpoint_path

# Function to load a torch model, optimizer, sceduler and epoch
def load_checkpoint(checkpoint_path, model, optimizer = None, 
                    scheduler_step = None, scheduler_expo = None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if not optimizer is None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if not scheduler_step is None:
        scheduler_step.load_state_dict(checkpoint['scheduler_step_state_dict'])
    if not scheduler_expo is None:
        scheduler_expo.load_state_dict(checkpoint['scheduler_expo_state_dict'])
    return checkpoint['epoch'], checkpoint['additional_info']

# Function to set the logger
def setup_logging(root_path, log_level = 'INFO', fname='log.log'):
    level_dict = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'FATAL': logging.CRITICAL
    }
    level = level_dict.get(log_level, logging.INFO)

    format_ = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    filename = '{}/{}'.format(root_path, fname)

    logger = logging.getLogger('kim')
    logger.setLevel(level)
    fh = logging.FileHandler(filename)
    fh.setFormatter(logging.Formatter(format_))
    fh.setLevel(level)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(format_))
    ch.setLevel(level)

    logger.addHandler(ch)

    return logger


# Functions to handle gradient and hessian functions
# Here we define the operators we need to test that the value network is correct
# H^u_w and \cal H_w

def func_Huw(b, sigma, w):
    """
    Given a drift function b(t, x, u), a volatility function sigma(t, x, u) and a value function w(t, x),
    create and return the function
        H^u_w(t, x, u) = -w_t(t, x) + b(t, x, u) @ w_x(t, x) + 0.5 * Tr(sigma(t, x, u) @ sigma(t, x, u)^T @ w_xx(t, x))
    The returned object is a function of two variables (torch.Tensor), that outputs a scalar (torch.Tensor)

    b and s are expected to return a vector of the same size as x
    w is a scalar function
    """
    dw_t = torch.func.grad(w, 0)
    dw_x = torch.func.grad(w, 1)
    d2w_x = torch.func.hessian(w, 1)


    def aux(t, x, u):
        s = sigma(t, x, u).reshape(-1, 1)
        r = -dw_t(t, x) + b(t, x, u) @ dw_x(t, x) +  0.5 *torch.trace( (s @ s.T) @ d2w_x(t, x))
        return r[0]

    return aux

def func_calHw(Huw, bounds = (0,1)):
    """
    Given the function H^u_w(t, x, u), create and return the function
        cal H_w(t, x) = min_u H^u_w(t, x, u)
    The returned object is a function of two variables (torch.Tensor), that outputs a scalar (torch.Tensor)

    The parameter u is forced to lie in the interval `bounds`
    """
    def aux(t, x):
        f = lambda u: Huw(t, x, torch.tensor(u)).detach().numpy()
        res = minimize_scalar(f, bounds = bounds)
        return res.fun
    return aux

# Function to compute gradients and hessians in terms of space variable for the domain boundary value neural network
def compute_grad_hessian(w_NN, inputs):
    '''
        Inputs : 
            w_NN : w_nn to be evaluated
            inputs : a sample of (t,x) data points          - dim = (M, 1 + dim_state) where M = n_samples * n_periods
        Outputs :
            dw_t : gradient of w_nn w.r.t time dimension t  - dim = (M, 1)
            dw_x : gradient of w_nn w.r.t space dimension x - dim = (M, dim_state)
            dw_xx : hessian of w_nn w.r.t space dimension x - dim = (M, dim_state, dim_state)
    '''
    # prep inputs
    inputs = inputs.clone().detach().requires_grad_(True)
    t = inputs[:, 0:1]                      # dim = (M, 1)
    x = inputs[:, 1:]                       # dim = (M, dim_state)
    M, dim_state = x.shape

    # forward pass
    outputs = w_NN(inputs)        # dim = (M, )

    # gradients
    grads = torch.autograd.grad(outputs = outputs,
                                inputs = inputs,
                                grad_outputs = torch.ones_like(outputs),
                                create_graph = True
                                )[0]        # dim = (M, dim_state + 1)
        
    dw_t = grads[:, 0:1]                    # dim = (M, 1)
    dw_x = grads[:, 1:]                     # dim = (M, dim_state)

    # hessian
    dw_xx = torch.zeros((M, dim_state, dim_state))
    for i in range(dim_state):
        dw_x_i = dw_x[:, i]
        dw_xx_i = torch.autograd.grad(outputs = dw_x_i,
                                      inputs = inputs[:, 1:],
                                      grad_outputs = torch.ones_like(dw_x_i),
                                      retain_graph = True, create_graph = True
                                      )[0]   # dim = (M, dim_state)
        dw_xx[:, i, :] = dw_xx_i             # fill row i of hessian 
        
    return dw_t, dw_x, dw_xx


# The last one was not working, got the following error:
# RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.
# As I do not know how to fix this quickly, I will reimplement it using the functions already given in torch
# If needed, we can compute this one with the original version after being fixed
def compute_grad_hessian_2(w_NN, inputs):
    '''
        Inputs : 
            w_NN : w_nn to be evaluated
            inputs : a sample of (t,x) data points          - dim = (M, 1 + dim_state) where M = n_samples * n_periods
        Outputs :
            dw_t : gradient of w_nn w.r.t time dimension t  - dim = (M, 1)
            dw_x : gradient of w_nn w.r.t space dimension x - dim = (M, dim_state)
            dw_xx : hessian of w_nn w.r.t space dimension x - dim = (M, dim_state, dim_state)
    '''
    is_training = w_NN.training
    w_NN.eval()

    w_func = lambda t, x: w_NN(torch.cat((t, x), dim=0).reshape(1, -1))
    dw_t_func = torch.vmap(torch.func.grad(w_func, 0))
    dw_x_func = torch.vmap(torch.func.grad(w_func, 1))
    dw_xx_func = torch.vmap(torch.func.hessian(w_func, 1))

    dw_t = dw_t_func(inputs[:, 0:1], inputs[:, 1:])
    dw_x = dw_x_func(inputs[:, 0:1], inputs[:, 1:])
    dw_xx = dw_xx_func(inputs[:, 0:1], inputs[:, 1:])

    w_NN.train(is_training)
    
    return dw_t, dw_x, dw_xx
