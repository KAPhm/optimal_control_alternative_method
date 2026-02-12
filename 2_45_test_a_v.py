'''
This script aims to compute the error measure on the training of (u,a) and V 
similarly to the one used for the training of u and w

PLEASE READ 4_Training_ValueFunction_global.md, section 1 for detailed explanation
'''
# ---- Packages --------- # 
import os

    # pytorch
import torch
import torch.nn as nn
import torch.optim as optim

    # to import configs 
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import importlib
    # to import customized functions
pm = importlib.import_module('0_portfolio_model', package=None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

import shutil
from scipy.optimize import minimize, Bounds
from tqdm import tqdm

    # to compute
import numpy as np
from numpy import exp, log
import itertools

# ---- Support Funct ---- # 
def get_index_from_time(t, n_periods):
    if type(t) == torch.Tensor:
        t = t.item()
    return int(round(t*n_periods, 0))

# Generator of (t,X,M) sample given the initial states and the augmented control process 
def generate_data(x_0, p_0, dW, a_nn, w_nn, u_nn, T, boundary_update_func):
    '''
    This function aims to generate data in the interior, on the boundary, and at the terminal for the training of V.

    Inputs :
        x_0 : a sample of initial states            - dim = (n_samples, dim_state)
        p_0 : a sample of constraint limits at t=0  - dim = (n_samples, )
        dW : a sample of brownian increments paths  - dim = (n_samples, dim_sto, n_periods)
        a_nn : neural network for the control in the interior of the domain
        w_nn : neural network to estimate the boundary value function
        u_nn : neural network for the control on the boundary of the domain
        T : terminal of the time horizon
        boundary_update_func : the update function for the boundary (involving only the states X and control u)

    Outputs : 
        sample of interior data of the augmented states with their corresponding augmented controls 
        sample of boundary (before terminal) data
        sample of terminal data
    '''
    device = x_0.device                             # device
    n_periods = len(u_nn)                           # number of the time steps in the time horizon
    n_samples, dim_state = x_0.size(0), x_0.size(1) 
    dim_sto = dW.size(1)
    dt = T/n_periods

    # pass forward to get full history from initial states
    states, martingales, controls, martingale_increments, taus = a_nn.forward_merge(initial_states = x_0, constraint_limits = p_0, brownians = dW, w = w_nn, u_nn = u_nn, x_u_update_func = boundary_update_func)
    # concatenate t, X, and M for full states
    timeline = torch.arange(start = 0.0, end = T + dt, step = dt, device = device) 
    full_timeline = timeline.repeat((n_samples,1, 1))   # dim = (n_samples, 1, n_periods + 1)
    full_states = torch.cat([full_timeline, states, martingales.unsqueeze(1)], 
                            dim = 1)                    # dim = (n_samples, dim_state + 2, n_periods + 1)

    # concatenate u and a for full controls
    full_controls = torch.cat([controls, martingale_increments], 
                              dim = 1)                  # dim = (n_samples, dim_control + dim_sto, n_periods) 

    # Filtering data
    with torch.no_grad():
        terminal_data = full_states[:,:,-1].squeeze().clone()   # dim = (n_samples, dim_state + 2, )
        terminal_data = terminal_data.to(device) 
        
            # initiation
        interior_data = None
        boundary_data = None

        interior_controls = None    # for the interior, the control includes u and a

            # loop 
        for i in range(n_samples):
            temp_tau = taus[i].item() # scalar
            temp_states = full_states[i,:,:-1].to(device)       # dim = (dim_state + 2, n_periods)
            temp_controls = full_controls[i,:,:].to(device)     # dim.= (dim_control + dim_state, n_periods)
            
            if temp_tau == T :  # case 1 : the trajectory is completely inside the domain or only reaches the boundary at terminal    
                if interior_data is None : 
                    interior_data = temp_states.clone().to(device)
                else :
                    interior_data = torch.cat([interior_data, temp_states.clone()], dim = 1).to(device)

            else :              # case 2 : the trajectory tounches the domain boundary before the terminal
                    # slicing based on the stopping time
                temp_id_interior = timeline[:-1] < temp_tau     # timeline excluding the last time step and up to tau
                temp_id_boundary = timeline[:-1] >= temp_tau    # since the boundary is absobing, temp_tau is included in the boundary 

                    # interior 
                if interior_data is None : 
                    interior_data = temp_states[:, temp_id_interior].clone()    # dim = (dim_state + 2, len(temp_id_interior))
                else :
                    interior_data = torch.cat([interior_data, temp_states[:, temp_id_interior].clone()], dim = 1)   
                interior_data = interior_data.to(device)

                    # boundary
                if boundary_data is None : 
                    boundary_data = temp_states[:, temp_id_boundary].clone()    # dim = (dim_state + 2, len(temp_id_boundary))
                else :
                    boundary_data = torch.cat([boundary_data, temp_states[:, temp_id_boundary].clone()], dim = 1)  
                boundary_data = boundary_data.to(device)

                    # control in the interior
                if interior_controls is None :
                    interior_controls = temp_controls[:, temp_id_interior].clone()   # dim = (dim_control + dim_sto, len(temp_id_interior))
                else :
                    interior_controls = torch.cat([interior_controls, temp_controls[:, temp_id_interior].clone()], dim = 1)

            # transpose the data set
        interior_data = interior_data.transpose(1,0)            # dim = (n_samples_interior, dim_state + 2)
        interior_controls = interior_controls.transpose(1,0)    # dim = (n_samples_interior, dim_control + dim_sto)

        boundary_data = boundary_data.transpose(1,0)            # dim = (n_samples_boundary, dim_state + 2)

        return interior_data, interior_controls, boundary_data, terminal_data

# Supporting function to create a function out of the neural network V_NN
def create_V_func(V_NN):
    def V_func(t, x, m):
        # Concatenate t and x
        if len(x.size()) == 1: 
            z = torch.cat((t.unsqueeze(0), x.unsqueeze(0), m.unsqueeze(0)), dim=1)
        else:
            z = torch.cat((t, x, m), dim=1)
        # Pass through the value network
        return V_NN(z)
    return V_func

# Create a function representing the dynkin attached to V_nn taking any random control (u,a) as input
def Operator(V_NN, drift_func, vol_func, g):
    '''
    Inputs :
        V_NN : neural network estimation of the value function
        drift_func : function computing the drift of the state process X 
        vol_func : function computing the volatility of the state process X
        g : intermediate penalty presented in the training of u_NN
    Output : Hamiltonian-Bellman operator of function V 
    '''
    # compute gradients and hessian of w 
    V_func = create_V_func(V_NN)
    dV_t_func = torch.func.grad(V_func, 0)
    dV_x_func = torch.func.grad(V_func, 1)
    dV_p_func = torch.func.grad(V_func, 2)
    d2V_xp_func = torch.func.hessian(V_func, (1,2))

    def HV(t, x, m, u, a):
        '''
        Inputs:
            t and m are scalars
            x, u, a are tensors of shape (1, dim_state), (1, dim_control), and (1, dim_sto) respectively
        '''
        # compute drift and volatility for a given u
        drift = drift_func(x.unsqueeze(0), u.unsqueeze(0)).squeeze() 
        if len(a.unsqueeze(0).shape) < 3:
            vol = torch.cat([vol_func(x.unsqueeze(0)).squeeze(), a.unsqueeze(0)], dim=0)
        else:
            vol = torch.cat([vol_func(x.unsqueeze(0)).squeeze(), a], dim=0)
        variance = torch.matmul(vol, vol.T)

        # compute gradients and hessian at (t,x,m)
        dV_t = dV_t_func(t, x, m)
        dV_x = dV_x_func(t, x, m)
        dV_p = dV_p_func(t, x, m)
        d2V_xp = d2V_xp_func(t, x, m)

        # d2V_xp is a tuple of tuples of tensors, need to concatenate them to get the total matrix
        aux1 = torch.cat(d2V_xp[0], dim=1)
        aux2 = torch.cat(d2V_xp[1], dim=1)
        aux3 = torch.cat((aux1, aux2), dim=0)

        # compute the H operator of V at (t,x,m)
        operator = dV_t + torch.dot(drift, dV_x)
        operator += 0.5 * torch.trace(torch.matmul(variance, aux3))
        operator -= g(x.reshape(1,-1)).squeeze() * dV_p
        return operator

    return HV   

# Create a function representing the dynkin attached to V_nn and a_nn
def Operator_at_ann(V_NN, a_NN, drift_func, vol_func, g):
    
    HV = Operator(V_NN, drift_func, vol_func, g)
    n_periods = len(a_NN)

    def HV_nn(t, x, m):
        current_time_idx = get_index_from_time(t, n_periods)  # indicator of the subnetwork / time period
        if current_time_idx < 0:
            logger.warning(f"Warning: current_time_idx is {current_time_idx}, t={t}, n_periods={n_periods}.")
        u, a = a_NN[current_time_idx](x.reshape(1, -1), m)    # passing the current state through the corresponding subnetwork
        u, a = u.squeeze(), a.squeeze()
        return HV(t, x, m, u, a).item()     
    
    return HV_nn

# Create a function representing the dynkin of V_nn at optimal value of (u,a) within a given range
def Operator_optimal(V_NN, drift_func, vol_func, g, u_lower, u_upper, a_lower, a_upper):

    dim_control = len(u_upper)
    device = u_lower.device

    V_func = create_V_func(V_NN)
    d2V_xp_func = torch.func.hessian(V_func, (1,2)) # 2nd order deriv of V wrt x and p - dim = {{(1, dim_state, dim_state), (1, dim_state, 1)},{(1,1,dim_state), (1,1,1)}}
        
    def HV_optim(t,x,m):    
        # compute the optimal a which has an explicit formulation - check documentation
            # coefficients needed for the formula
        hessian_tuples = d2V_xp_func(t,x,m)
        d2V_dp_dx = hessian_tuples[1][0].squeeze(0)     # dim = (1, dim_state)
        d2V_dp_dp = hessian_tuples[1][1].item()         # scalar

        with torch.no_grad():
            vol = vol_func(x.reshape(1,-1))                               # dim = (dim_state, dim_sto)
            B = torch.matmul(d2V_dp_dx, vol)                # dim = (1, dim_sto)

            # optimal a 
            if d2V_dp_dp < 0 :      # concave quadratic case
                coeff_a = -1/d2V_dp_dp    
                a_optim = coeff_a * B         # dim = (1, dim_sto)
            else:                   # either convex quadratic or linear case
                # Since a_lower = - a_upper, if the function is convex quadratic, the maximizer is determined based on the linear term
                a_optim = torch.where(B >= 0, a_upper, a_lower)  
        
        # compute the optimal u which can be found at the upper or lower bounds of admissible range
        HV = Operator(V_NN, drift_func, vol_func, g)
        with torch.no_grad(): 
            # candidates for optimal u
            permutations = torch.Tensor(list(itertools.product([0,1], repeat=dim_control))).to(device)
            u_lower_extend = u_lower.unsqueeze(0).expand(2**dim_control, dim_control)
            u_upper_extend = u_upper.unsqueeze(0).expand(2**dim_control, dim_control)
            candidates = torch.where(permutations.bool(), u_lower_extend, u_upper_extend)

            # test different combinations
            values = torch.Tensor([HV(t,x,m,u,a_optim) for u in candidates])
            optim_max = torch.max(values).item()
        return optim_max         
    
    return HV_optim

# Create a function to compute the error for the interior points
def compute_error_interior(aug_sample_interior, V_NN, a_NN, drift_func, vol_func, g, u_lower, u_upper, a_lower, a_upper):
    '''
    Inputs : 
        aug_sample_interior : a sample of augmented state in the interior of the domain (t,x,p) - dim = (n_samples, dim_state + 2) 
        V_NN : value function network
        a_NN : augmented optimal control network
        drift_func, vol_func, g : functions for the portfolio dynamic with parameters already fixed (only taking x and u as arguments)
        u_upper, u_lower, a_upper, a_lower : upper and lower bounds for u and a
    Outputs : 
        _ Operators evaluated at each point in the sample with (1) controls estimated by a_nn and (2) optimizer of the operator
        _ Penalties at each point in the sample
    '''
    n_samples = aug_sample_interior.shape[0]    # number of data points

    # supporting functions
    HV_ann_func = Operator_at_ann(V_NN, a_NN, drift_func, vol_func, g)
    HV_optim_func = Operator_optimal(V_NN, drift_func, vol_func, g, u_lower, u_upper, a_lower, a_upper)

    # compute the penalty at each state (note that the time dimension is not needed here)
    X = aug_sample_interior[:, 1:-1]    # dim = (n_samples, dim_state)

    # error measure computation
        # initiation
    delta_a = 0         # sum of absolute value of the different between operators HV at a_nn and at optimal
    delta_v = 0         # sum of absolute value of operators HV at a_nn
    delta_o = 0         # sum of absolute value of operators HV at optimal

    HV_a = 0           
    HV_o = 0            
    
    for i in tqdm(range(n_samples)):
        # extract t, x, m at each data point
        t = aug_sample_interior[i, 0].unsqueeze(0)
        x = aug_sample_interior[i, 1:-1]
        m = aug_sample_interior[i, -1].unsqueeze(0)  

        # compute the operator at the data point
        HV_a = HV_ann_func(t,x,m)     # scalar
        HV_o = HV_optim_func(t,x,m)   # scalar

        # accumulate the absolute values
        delta_a += abs(HV_a - HV_o)
        delta_v += abs(HV_a)
        delta_o += abs(HV_o)
    
    return delta_a, delta_v, delta_o

if __name__ == "__main__":
    # alias for supportive function
    J = os.path.join

# ---- Parsers ----------- #
    # Argument parser to read config file
    parser = ArgumentParser(description='Train : Value function V', 
                            formatter_class=ArgumentDefaultsHelpFormatter)
        # for the main computation
    parser.add_argument('-s', '--seed', type=int, default=47, help='seed for randomization')
    parser.add_argument('-n', '--n_samples', type=int, default=1000, help='sample size for initial states')
    parser.add_argument('-v', '--config_v', type=str, required=True, help='path to the parameters of the trained network V_nn (value function on the entire viable domain)')
    
        # for the interior
    parser.add_argument('-a', '--config_a', type=str, required=True, help='path to the parameters of the trained network a_nn (augmented optimal control)')
      
        # for the boundary
    parser.add_argument('-vb', '--config_vb', type=str, required=True, help='path to the parameters of the trained network V_bound (value function at the boundary)')
    parser.add_argument('-u', '--config_u', type=str, required=True, help='path to the parameters of the trained network u_nn (optimal control at the boundary)')
    parser.add_argument('-w', '--config_w', type=str, required=True, help='path to the parameters of the trained network w_nn (boundary value function)')
        
        # for result keeping
    parser.add_argument('-o', '--results_dir', type=str, default='./results/test', help='directory to store the results')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='log level for the logger, e.g., INFO, DEBUG, WARN', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
    
    # Create parser 
    pm.parser_portfolio(parser)
    args = parser.parse_args()

    # Create a based directory to store results, checkpoints and figures, if any
    results_dir = utils.create_dir(basedir=args.results_dir, dirname = "test_a_v", suffix = None)
    logger = utils.setup_logging(results_dir, log_level = args.log_level, fname='test_a_v.log')

    # Device and seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # General parameters
        # sample size
    n_samples = args.n_samples      # number of trajectories to generate

        # portfolio parameters; if tensor then send to device
    n_assets, mu_S, sigma_S, coupon_rate, book_value, param_r, corr, param_L, u_lower, u_upper, n_periods, T, dt, K_below, K_above, scale_ind, dist_params = pm.helper_read_config_files(args)
    mu_S = mu_S.to(device)
    sigma_S = sigma_S.to(device)
    coupon_rate = coupon_rate.to(device)
    book_value = book_value.to(device)
    param_r = param_r.to(device)
    param_L = param_L.to(device)
    corr = corr.to(device)
    u_lower = u_lower.to(device)
    u_upper = u_upper.to(device)

        # dimension parameters
    dim_state = n_assets*2 + 3      # number of variables in the state process
    dim_control = n_assets + 1      # number of variables in the control process
    dim_sto = n_assets + 1          # number of variables in the stochastic factor (brownians)

        # absolute risk aversion parameters (for utility function) 
    config = json.load(open(args.config_portfolio, 'r'))
    alpha = config["alpha"]         # parameter for the untility function F

    # Fix parameters for supporting functions
    gn_update_func = lambda X, M, u, a, dW: pm.augmented_update(X=X, M=M, u=u, a=a, dt=dt, dW = dW, d=n_assets,
                                               mu_S=mu_S, sigma_S=sigma_S, param_r=param_r, 
                                               param_L= param_L, coupon_rate = coupon_rate, 
                                               book_value = book_value)
    
    u_update_func = lambda x, u, dW: pm.update(x=x, u=u, dt=dt, dW = dW, d=n_assets,
                                               mu_S=mu_S, sigma_S=sigma_S, param_r=param_r, 
                                               param_L= param_L, coupon_rate = coupon_rate, 
                                               book_value = book_value)

    drift_func = lambda x,u : pm.drift(x, u, n_assets, mu_S, param_r, param_L, coupon_rate, book_value)
    
    vol_func = lambda x : pm.vol(x, n_assets, sigma_S, param_r)
    
    utility_func = lambda x: pm.terminal_utility_exponential_2(x, alpha) 

    # Load a_nn 
        # load the configuration 
    a_dir = args.config_a
    config_ann = json.load(open(J(a_dir, "config_a.json"), 'r'))
    
    a_lower = config_ann['a_lower']
    a_upper = config_ann['a_upper']
    p_range = config_ann['p_range']

        # load the trained augmented optimal control neural network
    a_NN = custom_nn.AugGlobalNetworks(dim_input = dim_state + 1, 
                                       dim_output = dim_control + dim_sto, 
                                       dim_hidden = config_ann['n_neuron'], 
                                       n_hidden = config_ann['n_hidden_layers'], 
                                       n_subnetworks = n_periods,
                                       aug_update_func= gn_update_func,
                                       u_lower_bounds = u_lower, 
                                       u_upper_bounds = u_upper, 
                                       a_lower = a_lower, 
                                       a_upper = a_upper,
                                       T= T, 
                                       a_sigmoid_scale = 1.
                                       )

        # load the last checkpoint from the checkpoints directory
    dir_list = [os.path.join(a_dir, 'checkpoints', f) for f in os.listdir(os.path.join(a_dir, 'checkpoints'))]
    ann_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f'Load a_nn from {ann_checkpoint_path}')
    
    a_NN.to(device=device)
    a_NN.load_state_dict(torch.load(ann_checkpoint_path, map_location=device)['model_state_dict'])
    a_NN.float()
    a_NN.eval()

    # Load u_nn
        # load the configuration 
    u_dir = args.config_u
    config_unn = json.load(open(J(u_dir, "config_u.json"), 'r'))

        # fix the parameter of the penalty function related to training of u_nn
    margin = config_unn['margin']
    g_func = lambda x: pm.total_intermediate_penalty(x, margin = margin)
    
        # load the trained optimal neural network
    u_NN = custom_nn.GlobalNetworks(dim_input = dim_state,
                            dim_output = dim_control,
                            dim_hidden = config_unn['n_neuron'],
                            n_hidden = config_unn['n_hidden_layers'],
                            n_subnetworks = n_periods,
                            update_func=u_update_func,
                            lower_bounds = u_lower,
                            upper_bounds = u_upper
                            )

        # load the last checkpoint from the checkpoints directory
    dir_list = [os.path.join(u_dir, 'checkpoints', f) for f in os.listdir(os.path.join(u_dir, 'checkpoints'))]
    unn_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f'Load u_nn from {unn_checkpoint_path}')
    
    u_NN.to(device=device)
    u_NN.load_state_dict(torch.load(unn_checkpoint_path, map_location=device)['model_state_dict'])
    u_NN.float()
    u_NN.eval()

    # Load w_nn
        # load the configuration 
    w_dir = args.config_w
            
    config_wnn = json.load(open(J(w_dir, "config_w.json"), 'r'))
    shutil.copy(J(w_dir, "config_w.json"), J(results_dir,'config_w.json'))
            
    args_wnn = json.load(open(J(w_dir, "args.json"), 'r'))
    shutil.copy(J(w_dir, "args.json"), J(results_dir,'args_w.json'))

        # build the model
    if args_wnn['arch'] == 'mlp':

        w_NN = custom_nn.BoundaryValueNetwork(dim_input = dim_state + 1,
                                dim_hidden = config_wnn['n_neuron'],
                                n_hidden = config_wnn['n_hidden_layers'],
                                activation = torch.nn.ReLU())
    elif args_wnn['arch'] == 'residual':

        w_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_wnn['n_neuron'], num_layers=config_wnn['n_hidden_layers'], time_encoding='none', norm_type=args_wnn['norm_type'])

    elif args_wnn['arch'] == 'residual-sin':

        w_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_wnn['n_neuron'], num_layers=config_wnn['n_hidden_layers'], time_encoding='sinusoidal', norm_type=args_wnn['norm_type'])

    elif args_wnn['arch'] == 'residual-learnable':

        w_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_wnn['n_neuron'], num_layers=config_wnn['n_hidden_layers'], time_encoding='learnable', norm_type=args_wnn['norm_type'])
    
    else:
        raise ValueError(f"Unknown architecture {args_wnn['arch']}. Choose between 'mlp' and 'residual'.")
    
        # load the last checkpoint from the checkpoints directory
    dir_list = [J(w_dir, 'checkpoints', f) for f in os.listdir(J(w_dir, 'checkpoints'))]
    wnn_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f"Load w_nn from {wnn_checkpoint_path}")

    w_NN.to(device=device)
    w_NN.load_state_dict(torch.load(wnn_checkpoint_path, map_location=device)['model_state_dict'])
    w_NN.float()
    w_NN.eval()

    # Load V_bound network
        # load the configuration for the value network, create the model and load the weights
    V_bound_dir = args.config_vb

    config_v_bound = json.load(open(J(V_bound_dir, "config_vb.json"), 'r'))
    args_vbnn = json.load(open(J(V_bound_dir, "args.json"), 'r'))

        # build the model
    if args_vbnn['arch'] == 'mlp':
        V_bound_NN = custom_nn.BoundaryValueNetwork(dim_input = dim_state + 1,
                                dim_hidden = config_v_bound['n_neuron'],
                                n_hidden = config_v_bound['n_hidden_layers'],
                                activation = torch.nn.ReLU())
    elif args_vbnn['arch'] == 'residual':
        V_bound_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, 
                                                 state_dim=dim_state, 
                                                 hidden_dim=config_v_bound['n_neuron'], 
                                                 num_layers=config_v_bound['n_hidden_layers'], 
                                                 time_encoding='none', 
                                                 norm_type=args_vbnn['norm_type'])

    elif args_vbnn['arch'] == 'residual-sin':
        V_bound_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, 
                                                 state_dim=dim_state, 
                                                 hidden_dim=config_v_bound['n_neuron'], 
                                                 num_layers=config_v_bound['n_hidden_layers'], 
                                                 time_encoding='sinusoidal', 
                                                 norm_type=args_vbnn['norm_type'])

    elif args_vbnn['arch'] == 'residual-learnable':
        V_bound_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, 
                                                 state_dim=dim_state, 
                                                 hidden_dim=config_v_bound['n_neuron'], 
                                                 num_layers=config_v_bound['n_hidden_layers'], 
                                                 time_encoding='learnable', 
                                                 norm_type=args_vbnn['norm_type'])
    else:
        raise ValueError(f"Unknown architecture {args_vbnn['arch']} for V_bound_NN >> Choose between 'mlp' and 'residual'.")
    
    
        # load the last checkpoint from the checkpoints directory
    dir_list = [J(V_bound_dir, 'checkpoints', f) for f in os.listdir(J(V_bound_dir, 'checkpoints'))]
    V_bound_nn_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f"V_bound_nn loaded from {V_bound_nn_checkpoint_path}")

    V_bound_NN.to(device=device)
    V_bound_NN.load_state_dict(torch.load(V_bound_nn_checkpoint_path, map_location=device)['model_state_dict'])
    V_bound_NN.eval()


    # Load V_nn
        # load the configuration
    V_dir = args.config_v

    config_vnn = json.load(open(J(V_dir, "config_v.json"), 'r'))
    args_vnn = json.load(open(J(V_dir, "args.json"), 'r'))
    
        # build the model
    if args_vnn['arch'] == 'mlp':
        V_NN = custom_nn.BoundaryValueNetwork(dim_input = dim_state + 1,
                                dim_hidden = config_vnn['n_neuron'],
                                n_hidden = config_vnn['n_hidden_layers'],
                                activation = torch.nn.ReLU())
    elif args_vnn['arch'] == 'residual':
        V_NN = custom_nn.ValueNet(num_time_freqs=8, 
                                                 state_dim=dim_state, 
                                                 hidden_dim=config_vnn['n_neuron'], 
                                                 num_layers=config_vnn['n_hidden_layers'], 
                                                 time_encoding='none', 
                                                 norm_type=args_vnn['norm_type'])

    elif args_vnn['arch'] == 'residual-sin':
        V_NN = custom_nn.ValueNet(num_time_freqs=8, 
                                                 state_dim=dim_state, 
                                                 hidden_dim=config_vnn['n_neuron'], 
                                                 num_layers=config_vnn['n_hidden_layers'], 
                                                 time_encoding='sinusoidal', 
                                                 norm_type=args_vnn['norm_type'])

    elif args_vnn['arch'] == 'residual-learnable':
        V_NN = custom_nn.ValueNet(num_time_freqs=8, 
                                                 state_dim=dim_state, 
                                                 hidden_dim=config_vnn['n_neuron'], 
                                                 num_layers=config_vnn['n_hidden_layers'], 
                                                 time_encoding='learnable', 
                                                 norm_type=args_vnn['norm_type'])
    else:
        raise ValueError(f"Unknown architecture {args_vnn['arch']} for V_NN >> Choose between 'mlp' and 'residual'.")
    
    
        
        # load the last checkpoint from the checkpoints directory
    dir_list = [J(V_dir, 'checkpoints', f) for f in os.listdir(J(V_dir, 'checkpoints'))]
    Vnn_checkpoint_path = max(dir_list, key = os.path.getmtime)
    logger.info(f"V_bound_nn loaded from {Vnn_checkpoint_path}")

    V_NN.to(device = device)
    V_NN.load_state_dict(torch.load(Vnn_checkpoint_path, map_location = device, weights_only = False)['model_state_dict'])
    V_NN.float()
    V_NN.eval()       
    
    # Pre-training info
    logger.info(f"Using device: {device}")
    logger.info(f"Number of trajectories = {n_samples} -> Total number of data points = {n_samples * (n_periods+1)}")

    # Save config files in case needed for replication
    json.dump(json.load(open(args.config_portfolio, 'r')), open(f'{results_dir}/config_portfolio.json', 'w'), indent=4)
    json.dump(json.load(open(args.config_simulator, 'r')), open(f'{results_dir}/config_simulator.json', 'w'), indent=4)
    
    json.dump(config_unn, open(f'{results_dir}/config_u.json', 'w'), indent=4)

    json.dump(config_wnn, open(f'{results_dir}/config_w.json', 'w'), indent=4)
    json.dump(args_wnn, open(f'{results_dir}/args_w,json', 'w'), indent = 4)
    
    json.dump(config_ann, open(f'{results_dir}/config_a.json','w'), indent=4)

    json.dump(config_v_bound, open(f'{results_dir}/config_vb.json', 'w'), indent=4)
    json.dump(args_vbnn, open(f'{results_dir}/args_vb,json', 'w'), indent = 4)

    json.dump(config_vnn, open(f'{results_dir}/config_v.json', 'w'), indent=4)
    json.dump(args_vnn, open(f'{results_dir}/args_v,json', 'w'), indent = 4)


# ---- Computation ------ # 
    # Generate data
        # initial data 
    initial_x = pm.simulate_initial_state(n_samples, n_assets, dist_params, scale=True).to(device)  # dim = (samples_per_batch, dim_state)
    brownians = pm.simulate_brownians(n_samples, dim_sto, n_periods, dt, corr).to(device)           # dim = (samples_per_batch, dim_sto, n_periods)
    initial_p = pm.generate_p(initial_x, w_NN, p_range=p_range).to(device)
        
        # trajectories
    interior_data, interior_controls, boundary_data, terminal_data = generate_data(x_0 = initial_x, p_0 = initial_p, dW = brownians, 
                                                                    a_nn = a_NN, w_nn = w_NN, u_nn = u_NN, T = T, 
                                                                    boundary_update_func = u_update_func)
    n_interior = interior_data.size(0)      
    n_boundary = boundary_data.size(0)
    n_terminal = terminal_data.size(0)
    sample_size = n_interior + n_boundary + n_terminal

    logger.info(f"Partition : n_interior={n_interior} + n_boundary={n_boundary} + n_terminal={n_terminal} -> sample size={sample_size}")   
    assert sample_size == n_samples * (n_periods + 1), f"Error in total sample size: expecting n_samples * (n_periods + 1)={n_samples * (n_periods+1)} but receiving sample_size={sample_size}"
    
    # Compute the error measure
        # 1. Difference beween V_nn and Vb_nn at boundary data before terminal
    V_tau = V_NN(boundary_data)
    Vb_tau = V_bound_NN(boundary_data[:,:-1])

    d_b = V_tau - Vb_tau                    # dim = (n_boundary, )
    delta_b = torch.abs(d_b).sum().item()   # scalar

        # 2. Difference between V_nn and F at the terminal 
    F_T = utility_func(terminal_data[:, 1:-1])  
    V_T = V_NN(terminal_data)

    d_t = V_T - F_T                         # dim = (n_terminal, )
    delta_t = torch.abs(d_t).sum().item()   # scalar

        # 3. For other error measures, we need to compute the H operator and the optimal H operator
    delta_a, delta_v, delta_o = compute_error_interior(interior_data, V_NN, a_NN, drift_func, vol_func, g_func, 
                                        u_lower, u_upper, a_lower, a_upper)

        # Total error 
    total_error = ((delta_a + delta_v + delta_b)/n_periods + delta_t)/n_samples
    
        # Denominator of the error measure
    aug_initial_states = torch.concat([torch.zeros((n_samples, 1)).to(device), initial_x, initial_p.unsqueeze(1)],
                                       dim = 1) # dim = (n_samples, dim_state + 2)
    V_0 = V_NN(aug_initial_states)              # dim = (n_samples, )
    expected_V_0 = torch.mean(V_0).item()       # scalar

        # Error measure - Note that V is naturally negative, so we have to take absolute value
    error_measure = total_error / abs(expected_V_0)

    # Report result
    logger.info(f"Error measure : {error_measure} - in percentage : {100*error_measure} %")
    logger.info("Formula for the error measure = {[(delta_a + delta_v + delta_b) / n_periods + delta_t] / n_samples} / expected_V_0")
    logger.info(f" - Decomposition : delta_t={delta_t}, Delta_b={delta_b}, Delta_a={delta_a}, Delta_v={delta_v}, V_0={expected_V_0}")
    logger.info(f" - Extra info :")
    logger.info(f"   -- delta_o = {delta_o}")
    logger.info(f"   -- Boundary : mean = {d_b.mean().item()}, min = {d_b.min().item()}, max = {d_b.max().item()}, std = {d_b.std().item()}")
    logger.info(f"   -- Terminal : mean = {d_t.mean().item()}, min = {d_b.min().item()}, max = {d_t.max().item()}, std = {d_t.std().item()}")

