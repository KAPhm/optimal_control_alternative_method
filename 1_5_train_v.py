"""
This script is used to train the Value Function V.
Note that the training of V must be split into 2 parts : 
- V at the boundary of the domain, henceforth called V_bound, which takes only 2 arguments t and x as inputs
- V in the interior of the domain, which takes 3 arguments t, x, and p as inputs
We are going to use the merged version of the augmented neural network a_nn to complete this training.

In this module, we execute the first training.
PLEASE READ 4_Training_ValueFunction_global.md, section 2 for detailed explanation
"""
# Last update : 19/08/2025
# Author : Kim 

# ---- Package ----------- # 
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

    # to track time
import time
import shutil

# ---- Support Functions ---- #

# Generator of (t,X,M) sample given the initial states and the augmented control process 
def generate_data_V_rebounce(x_0, p_0, dW, a_nn, w_nn, u_nn, T, boundary_update_func):
    '''
    This function aims to generate data in the interior, on the boundary, and at the terminal for the training of V
    Note : we artificially rebounce any data point that falls on or outside of the boundary back into the interior.
        The data samples for the parabolic boundary (spacial and time-wise) are drawn with only the (t,X) component of the interior data
        This consequentially create a homogeneity in the sample sizes regardless of the randomization. 

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
        interior_data : a sample of data points (t,X,M) that are in the interior of the domain
        interior_countrols : a sample of (u,a) corresponding to the data points in the interior_data sample
        boundary_data : a sample of data points (t,X,w(t,X)) that are on the (spacial) boundary of the domain where t < T
        terminal_data : a sample of data points (T,X,p) that are on the terminal boundary of the domain where t = T and p >= w(T, X)
    '''
    # Parameters and device
    device = x_0.device             # device
    n_periods = len(u_nn)           # number of time steps in the time horizon
    n_samples = x_0.size(0)         # sample size of the initial states
    dim_state = x_0.size(1)         # dimension of the (spacial) state X
    dim_sto = dW.size(1)            # dimension of the brownians = dimension of the martingale increment a
    dim_control = u_nn.dim_output   # dimension of the control u (without martingale increments)

    # Initiation    
    interior_data = None            # should have dim = (n_samples * n_periods, dim_state + 2)
    interior_controls = None        # should have dim = (n_samples * n_periods, dim_control + dim_sto)

    boundary_data = None            # should have dim = (n_samples, dim_state + 2, n_periods)
    terminal_data = None            # should have dim = (n_samples, dim_state + 2, )

    # Generate the interior data with rebouncing - Note : there is no need for taus here
        # initiate 
    current_X = x_0.clone()             # dim = (n_samples, dim_state)
    current_M = p_0.clone()             # dim = (n_samples, )
    
        # loop through the time horizon
    for i, a_subnet in enumerate(a_nn.subnetworks):   # i = 0, ..., N-1
        t = i * T/n_periods                         # t_i = 0, T/N, ..., T(N-1)/N

            # save the current augmented state (t included) - note that this will not include t = T since i = 0, ..., N-1
        current_t = torch.full(size = (n_samples, 1), fill_value = t).to(device)
        current_state = torch.concat([current_t, current_X, current_M.unsqueeze(1)], 
                            dim = 1).to(device)     # dim = (n_samples, dim_state + 2)
        if interior_data is None: 
            interior_data = current_state.clone()
        else:
            interior_data = torch.concat([interior_data, current_state],
                                     dim = 0)       # dim = (n_samples * (i+1), dim_state + 2)  -- (i+1) instead of i because i starts at 0 instead of 1
        
            # compute the augmented controls (u, a) at the current time step
        u_t, a_t = a_subnet(current_X, current_M)   
        current_controls = torch.concat([u_t, a_t],
                            dim = 1).to(device)     # dim = (n_samples, dim_control + dim_sto)
        if interior_controls is None:
            interior_controls = current_controls.clone()
        else:
            interior_controls = torch.concat([interior_controls, current_controls],
                                        dim = 0)    # dim = (n_samples * (i+1), dim_control + dim_sto)             

            # update the state X and the martingale representation (not caring about whether the point is inside or outside the boundary yet)
        dW_t = dW[:, :, i]                          # dim = (n_samples, dim_sto)
        current_X, current_M = a_nn.aug_update_func(current_X, current_M, u_t, a_t, dW_t)
        
            # compute the boundary w(t,x) at the data point
        current_t_X = torch.concat([current_t, current_X], 
                                        dim = 1)    # dim = (n_samples, 1 + dim_state)
        current_w = w_nn(current_t_X).to(device)    # dim = (n_samples, )
        
            # check whether the data points are in the interior, and if not, rebounce
        distance = current_M - current_w            # distance of M from w - if M is out of the boundary then this is negative
        rebounce_distance = torch.where(distance < 0, - 2 * distance, 0.0).to(device)     
        current_M = current_M + rebounce_distance   # if M < w, replace M with M - 2*(M-w) = M + 2*(w - M); otherwise, no change to M
        
    # Generate the terminal data 
        # the last update of the previous step has created a sample that can be used as the terminal data
    terminal_data = torch.concat([torch.full(size = (n_samples, 1), fill_value = T).to(device),   
                                current_X, current_M.unsqueeze(1)],   
                                dim = 1)            # dim = (n_samples, dim_state + 2)

    # Generate the boundary data    
    t_X = interior_data[:, :-1].to(device)          # dim = (n_samples * n_periods, dim_state + 1) - selecting only t and X, not M
    w_t_X = w_nn(t_X).to(device).unsqueeze(1)       # dim = (n_samples * n_periods, 1)
    boundary_data = torch.cat([t_X, w_t_X], dim = 1)# dim = (n_samples * n_periods, dim_state + 2)

    return interior_data, interior_controls, boundary_data, terminal_data

# ---- Loss Functions ------- #

# Loss function at the terminal of the time horizon 
def LossFunction_terminal(V_NN, aug_terminal_states, F):
    '''
    At T, the function V_NN(T, x, p) = F(x) for any feasible state x and any constraint limit p >= w(T,x).
    Note that the condition p >= w(T,x) must be checked elsewhere outside of this loss function.

    Inputs : 
        V_NN : neural network model used to estimate the value function V
        terminal_states : a sample of (T, X, M)                            - dim = (n_samples, dim_state + 2)
        F : (wrapped) utility function, should be able to handle a sample of dim = (n_samples, dim_state)
    Output : 
        MSE loss between the predictions and the true terminal utilities from the data
            MSE = sum [ || terminal_prediction - F(terminal_state) ||^2 ] / n_samples
        
    Note : The Martingale representation process M of the constraint limit p is not needed here. 
    '''
    device = aug_terminal_states.device                     # device
    predictions = V_NN(aug_terminal_states).to(device)      # dim = (n_samples, )
    targets = F(aug_terminal_states[:, 1:-1]).to(device)    # dim = (n_samples, )

    return nn.MSELoss()(predictions, targets)

# Loss function at the boundary M = w(t,X)
def LossFunction_boundary(V_NN, aug_states, V_bound):
    '''
    At the boundary where M = w(t, X). .
    Inputs : 
        V_NN : neural network model used to estimate the value function V
        aug_states : a sample of (t, X, M)       - dim = (n_samples, dim_state + 2)
        V_bound : the boundary value function 
            _ should be able to handle a sample of dim = (n_samples, dim_state + 1)
        
    Output :
        MSE loss between the predictions by V and the value given by V_bound
            MSE = sum [ |V(t,X,w(t,X)) - V_bound(t,X)|^2 ] / n_samples 
    Note that we will not use M but use w(t,X) directly for the 3rd argument of V
    '''       
    device = aug_states.device                      # device  
    predictions = V_NN(aug_states).to(device)       # dim = (n_samples, )
    targets = V_bound(aug_states[:,:-1]).to(device) # dim = (n_samples, )

    return nn.MSELoss()(predictions, targets) 

# Loss function in the interior of the viable domain
def LossFunction_interior(V_NN, aug_states, aug_controls, drift_func, vol_func, g_func, dim_control): 
    '''
    Inputs: 
        V_NN : neural network model used to estimate the value function V
        aug_states : a sample of (t, X, M)              - dim = (n_samples, 1 + dim_state + 1)
        aug_controls : a sample of (u, a)               - dim = (n_samples, dim_control + dim_sto)
        drift_func : (wrapped) drift function for state process X 
        vol_func : (wrapped) volatility function for state process X
        g_func : intermediate penalty function (used for training w) 
    Output: 
        Total loss of the operator applied to V at each data point in the sample
            Loss = sum [ |HV(t,X,M)|^2 ]  with H being the operator applied on V according to the PDE
    Note : please consult the accompanying article for the derivation of the operator 
    '''

    device = aug_states.device          # device
    dim_state = aug_states.size(1) - 2  

    X = aug_states[:, 1: -1]            # state process X               - dim = (n_samples, dim_state)
    M = aug_states[:, -1]               # Martingale representation M   - dim = (n_samples, )
    u = aug_controls[:, :dim_control]   # control u                     - dim = (n_samples, dim_control)
    a = aug_controls[:, dim_control:]   # Martingale increment a        - dim = (n_samples, dim_sto)

    # elements of computation
        # gradients and hessians
    dV_t, dV_xp, dV_2_xp = utils.compute_grad_hessian_2(V_NN, aug_states)

    dV_t = dV_t.squeeze().to(device)                                    # dim = (n_samples, )
    dV_x = dV_xp[:, :-1].unsqueeze(2).to(device)                        # dim = (n_samples, dim_state, 1)
    dV_p = dV_xp[:, -1].to(device)                                      # dim = (n_samples, )
    dV_2_xp = dV_2_xp.to(device)                                        # dim = (n_samples, dim_state + 1, dim_state + 1)

        # drifts and vols
    drifts = drift_func(X, u).unsqueeze(2).to(device)                   # dim = (n_samples, dim_state, 1)
    drifts_T = torch.transpose(drifts, 1, 2).to(device)                 # dim = (n_samples, 1, dim_state)
    drift_term = torch.bmm(drifts_T, dV_x).squeeze()                    # dim = (n_samples, )

    vols = vol_func(X).to(device)                                       # dim = (n_samples, dim_state, dim_sto)
    a_extend = a.unsqueeze(2).to(device)                                # dim = (n_samples, dim_sto, 1)
    a_extend_T = torch.transpose(a_extend, 1, 2).to(device)             # dim = (n_samples, 1, dim_sto)
    aug_vols = torch.cat([vols, a_extend_T], dim = 1)                   # dim = (n_samples, dim_state + 1, dim_sto)
    aug_var = torch.bmm(aug_vols, aug_vols.transpose(1,2)).to(device)   # dim = (n_samples, dim_state + 1, dim_state + 1)
    diffusion_term = torch.bmm(aug_var, dV_2_xp).to(device)             # dim = (n_samples, dim_state + 1, dim_state + 1)

        # intermediate penalty
    penalties = g_func(X).to(device)                                    # dim = (n_samples, )
    penalty_term = torch.mul(dV_p, penalties)                           # dim = (n_samples, )

    # operator & loss
    operators = dV_t + drift_term + 0.5 * torch.vmap(torch.trace)(diffusion_term) - penalty_term
    operators = operators.to(device)
    losses = (operators)**2                                             # dim = (n_samples, )
    losses = losses.to(device)

    return torch.mean(losses)

if __name__ == '__main__':

    # alias for supportive function
    J = os.path.join

# ---- Parsers ----------- #
    # Argument parser to read config file
    parser = ArgumentParser(description='Train : Value function V', 
                            formatter_class=ArgumentDefaultsHelpFormatter)
        # for training
    parser.add_argument('-s', '--seed', type=int, default=47, help='seed for randomization')
    parser.add_argument('-v', '--config_v', type=str, required=True, help='path to the config files with hyper parameters for this training')
    
            # architecture of the network
    parser.add_argument('--arch', type=str, default='residual-sin', 
                        help='architecture for the neural network, must coincide with the architecture of the checkpoints to be loadded', choices=['mlp','residual','residual-sin'])
    parser.add_argument('--norm-type', type=str, default='identity', help='type of normalization layer used in the residual mlp', choices=['batch','layer','identity'])

            # weighting scheme 
    parser.add_argument('-e', type=int, default=100, help='number of epochs until the regularization coefficient lambda_V reaches the value of lambda_V_end starting to increase/decrease linearly from lambda_V_start; default = 100 ')
    parser.add_argument('-d', type=int, default=1, help='number of epochs during which the lambda_V coefficient remains constant between two modifications')
    parser.add_argument('--lambda_start', type=float, default = 1.0, help='initial value for lambda_V coefficient for the interior loss.')
    parser.add_argument('--lambda_end', type=float, default=1.0, help='final value for the lambda_V coefficient for the interior loss. This value will be reached at the e-th epoch')

            # learning rate
    parser.add_argument('--sched', type=str, default='multistep', help='scheduler for the learning rate', choices=['multistep','cosine'])
    parser.add_argument('--sched-args', type=str, help='Arguments for the scheduler in the format key1:value1, key2:value2... If passed, they replace the values in the config file')

            # optimizer
    parser.add_argument('--opt', type=str, default='adam', help='Optimizer.', choices=['adam', 'sgd'])
    parser.add_argument('--opt-args', type=str, help='Arguments for the optimizer in the format key1:value1,key2:value2.... If passed, they replaces the ones in the config file if present. See the pytorch docs for the parameters available for each optimizer.')

        # for the interior
    parser.add_argument('-a', '--config_a', type=str, required=True, help='path to the parameters of the trained network a_nn (augmented optimal control)')
        
        # for the boundary
    parser.add_argument('-vb', '--config_vb', type=str, required=True, help='path to the parameters of the trained netwrok V_bound (value function at the boundary)')
    parser.add_argument('-u', '--config_u', type=str, required=True, help='path to the parameters of the trained network u_nn (optimal control at the boundary)')
    parser.add_argument('-w', '--config_w', type=str, required=True, help='path to the parameters of the trained network w_nn (boundary value function)')
        
        # for result keeping
    parser.add_argument('-o', '--results_dir', type=str, default='./results/debug', help='directory to store the results')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='log level for the logger, e.g., INFO, DEBUG, WARN', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])

    # Load portfolio model parameters 
    pm.parser_portfolio(parser)
    args = parser.parse_args()

    # Create a based directory to store results, checkpoints and figures, if any
    results_dir = utils.create_dir(basedir=args.results_dir, dirname = "train_v", suffix = None)
    logger = utils.setup_logging(results_dir, log_level = args.log_level, fname='train_v.log')

        # device and seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Save configuration files for replication
    shutil.copy2(args.config_v, J(results_dir,'config_v.json'))
    shutil.copy2(args.config_portfolio, J(results_dir,'config_portfolio.json'))
    shutil.copy2(args.config_simulator, J(results_dir,'config_simulator.json'))

    logger.info(f"Device: {device}")

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
    alpha = config["alpha"]       

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
    with open(J(a_dir, "config_a.json"), 'r') as f:
        config_a = json.load(f)

    a_lower = config_a['a_lower']
    a_upper = config_a['a_upper']
    p_range = config_a['p_range']

        # load the trained augmented optimal control neural network
    a_NN = custom_nn.AugGlobalNetworks(dim_input = dim_state + 1, 
                                       dim_output = dim_control + dim_sto, 
                                       dim_hidden = config_a['n_neuron'], 
                                       n_hidden = config_a['n_hidden_layers'], 
                                       n_subnetworks = n_periods,
                                       aug_update_func= gn_update_func,
                                       u_lower_bounds = u_lower, 
                                       u_upper_bounds = u_upper, 
                                       a_lower = a_lower, 
                                       a_upper = a_upper,
                                       T= T
                                       )
        # load the last checkpoint from the checkpoints directory
    dir_list = [os.path.join(a_dir, 'checkpoints', f) for f in os.listdir(os.path.join(a_dir, 'checkpoints'))]
    ann_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f"Augmented optimal control network a_nn loaded from {ann_checkpoint_path}")
    
    a_NN.to(device=device)
    a_NN.float()
    a_NN.load_state_dict(torch.load(ann_checkpoint_path, map_location=device)['model_state_dict'])
    a_NN.eval()

    # Load u_nn
        # load the configuration 
    u_dir = args.config_u
    with open(J(u_dir, "config_u.json"), 'r') as f:
        config_u = json.load(f)

        # fix the parameter of the penalty function related to training of u_nn
    margin = config_u['margin']
    g_func = lambda x: pm.total_intermediate_penalty(x, margin = margin)
    
        # load the trained optimal neural network
    u_NN = custom_nn.GlobalNetworks(dim_input = dim_state,
                            dim_output = dim_control,
                            dim_hidden = config_u['n_neuron'],
                            n_hidden = config_u['n_hidden_layers'],
                            n_subnetworks = n_periods,
                            update_func=u_update_func,
                            lower_bounds = u_lower,
                            upper_bounds = u_upper
                            )

        # load the last checkpoint from the checkpoints directory
    dir_list = [os.path.join(u_dir, 'checkpoints', f) for f in os.listdir(os.path.join(u_dir, 'checkpoints'))]
    unn_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f"Optimal control network u_nn loaded from {unn_checkpoint_path}")

    u_NN.to(device=device)
    u_NN.float()
    u_NN.load_state_dict(torch.load(unn_checkpoint_path, map_location=device)['model_state_dict'])
    u_NN.eval()

    # Load w_nn
        # load the configuration 
    w_dir = args.config_w
    config_w = json.load(open(J(w_dir, 'config_w.json'), 'r'))  # config
    arg_wnn = json.load(open(J(w_dir, "args.json"), 'r'))                   # architecture
    
        # architecture of w_nn
    w_arch = arg_wnn["arch"]
    w_norm = arg_wnn["norm_type"]

    if w_arch == 'mlp':
        w_NN = custom_nn.BoundaryValueNetwork(dim_input=dim_state+1, dim_hidden=config_w['n_neuron'], n_hidden=config_w['n_hidden_layers'], activation=nn.ReLU())
    elif w_arch == 'residual-sin':
        w_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_w['n_neuron'], num_layers=config_w['n_hidden_layers'],
                                            time_encoding='sinusoidal', norm_type=w_norm)
    elif w_arch == 'residual':
        w_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_w['n_neuron'], num_layers=config_w['n_hidden_layers'],
                                            time_encoding='none', norm_type=w_norm)
    elif w_arch == 'residual-learnable':
        w_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_w['n_neuron'], num_layers=config_w['n_hidden_layers'],
                                            time_encoding='learnable', norm_type=w_norm)
    else:
        raise ValueError(f"Unknown architecture {w_arch} for w_nn. It should be among the options : 'mlp', 'residual', 'residual-sin', 'residual-learnable'.")
    
        # load the last checkpoint from the checkpoints directory
    dir_list = [J(w_dir, 'checkpoints', f) for f in os.listdir(J(w_dir, 'checkpoints'))]
    wnn_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f"Boundary value function w_nn loaded from {wnn_checkpoint_path}")

    w_NN.to(device=device)
    w_NN.float()
    w_NN.load_state_dict(torch.load(wnn_checkpoint_path, map_location=device, weights_only=False)['model_state_dict'])
    w_NN.eval()

    # Load V_bound_nn
        # load the configuration
    V_bound_dir = args.config_vb
    config_vb = json.load(open(J(V_bound_dir, "config_vb.json"), 'r')) # config
    arg_vb = json.load(open(J(V_bound_dir, "args.json"), 'r'))      

        # architecture of V_bound_nn

    if arg_vb['arch'] == 'mlp':

        V_bound_NN = custom_nn.BoundaryValueNetwork(dim_input = dim_state + 1,
                                dim_hidden = config_vb['n_neuron'],
                                n_hidden = config_vb['n_hidden_layers'],
                                activation = torch.nn.ReLU())
    elif arg_vb['arch'] == 'residual':
    
        V_bound_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_vb['n_neuron'], 
                                num_layers=config_vb['n_hidden_layers'], time_encoding='none', norm_type='identity')

    elif arg_vb['arch'] == 'residual-sin':
    
        V_bound_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_vb['n_neuron'], 
                                num_layers=config_vb['n_hidden_layers'], time_encoding='sinusoidal', norm_type='identity')

    elif arg_vb['arch'] == 'residual-learnable':

        V_bound_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_vb['n_neuron'], 
                                num_layers=config_vb['n_hidden_layers'], time_encoding='learnable', norm_type='identity')
    
    else:
        raise ValueError(f"Unknown architecture {arg_vb['arch']}. Choose between 'residual', 'residual-sin', 'residual-learnable'.")
    
        # load the last checkpoint from the checkpoints directory
    dir_list = [J(V_bound_dir, 'checkpoints', f) for f in os.listdir(J(V_bound_dir, 'checkpoints'))]
    Vbnn_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f"Value function at the boundary V_bound_nn loaded from {Vbnn_checkpoint_path}")
    
    V_bound_NN.to(device=device)
    V_bound_NN.float()
    V_bound_NN.load_state_dict(torch.load(Vbnn_checkpoint_path, map_location=device, weights_only=False)['model_state_dict'])
    V_bound_NN.eval()

    # Load config file to train V
    config = json.load(open(args.config_v, 'r'))

        # hyper-parameters for training
    n_hidden_layers = config["n_hidden_layers"]     # number of hidden layers 
    n_neuron = config["n_neuron"]                   # number of neuron per hidden layer
    n_samples = config["n_samples"]                 # number of data points in the sample of initial states X_0 = (x_0^j)^{j=1,...,n_samples}
    
    batch_size = config["batch_size"]               # batch size

    n_batches = n_samples // batch_size
    samples_per_batch = n_samples // n_batches

    n_epoch =  config["n_epoch"]                    # number of training epochs

        # resampling indicator
    ind_resample_initial_state = config["ind_resample_initial_state"]
    ind_resample_brownian = config["ind_resample_brownian"]

        # optimizer and scheduler
            # from config file 
    lr = config["lr"]                               # learning rate
    milestones = config["milestones"]               # milestones for the learning rate scheduler
    gamma_step = config["gamma_step"]               # decay factor for the learning rate step scheduler
    gamma_expo = config["gamma_expo"]               # decay factor the the learning rate exponential scheduler

            # from terminal command - if given, this overwrites the parameters in the config file
    lambda_V_start = args.lambda_start              # initial value 
    lambda_V_end = args.lambda_end                  # final value 
    lambda_V = lambda_V_start                       # initiation
    
    lambda_V_min = min(lambda_V_start, lambda_V_end)
    lambda_V_max = max(lambda_V_start, lambda_V_end)

        # set seeds
    torch.manual_seed(args.seed)                    # Set the seed to obtain the same batch seeds depending on input seed
    batch_seeds = torch.multinomial(torch.ones(10000), n_batches, replacement=False).to(torch.int16)
    batch_seeds_val = torch.multinomial(torch.ones(10000), n_batches, replacement=False).to(torch.int16)


# ---- Directory --------- #
    # Set a directory to store results
    checkpoint_dir = utils.create_dir(basedir=results_dir, dirname = "checkpoints", suffix = "")

    # Save the training script 
    script_name = os.path.basename(__file__)
    shutil.copy(os.path.abspath(__file__), J(results_dir, script_name))

    # Save config files in case needed for replication
    json.dump(json.load(open(args.config_portfolio, 'r')), open(f'{results_dir}/config_portfolio.json', 'w'), indent=4)
    json.dump(json.load(open(args.config_simulator, 'r')), open(f'{results_dir}/config_simulator.json', 'w'), indent=4)
    
    json.dump(config_a, open(f'{results_dir}/config_a.json', 'w'), indent=4)
    json.dump(config_u, open(f'{results_dir}/config_u.json', 'w'), indent=4)
    json.dump(config_w, open(f'{results_dir}/config_w.json', 'w'), indent=4)
    json.dump(config_vb, open(f'{results_dir}/config_vb.json', 'w'), indent=4)
    json.dump(config, open(f'{results_dir}/config_v.json', 'w'), indent=4)
    json.dump(arg_wnn, open(f'{results_dir}/args_w.json', 'w'), indent=4)
    json.dump(arg_vb, open(f'{results_dir}/args_vb.json', 'w'), indent=4)
    
    args_dict = vars(args)
    args_dict['config_u'] = os.path.abspath(args.config_u)
    args_dict['config_w'] = os.path.abspath(args.config_w)
    args_dict['config_a'] = os.path.abspath(args.config_a)
    args_dict['config_vb'] = os.path.abspath(args.config_vb)
    json.dump(args_dict, open(f'{results_dir}/args.json', 'w'), indent=4)

# ---- Training ---------- #
    
    # Create a frame for V_NN with chosen hyperparameters
    if args.arch == 'mlp':
        model = custom_nn.ValueFunctionNetwork(dim_input = dim_state + 2, dim_hidden = n_neuron, 
                                               n_hidden = n_hidden_layers, activation = nn.ReLU())

    elif args.arch == 'residual':
    
        model = custom_nn.ValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=n_neuron, num_layers=n_hidden_layers, time_encoding='none', norm_type=args.norm_type)

    elif args.arch == 'residual-sin':
    
        model = custom_nn.ValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=n_neuron, num_layers=n_hidden_layers, time_encoding='sinusoidal', norm_type=args.norm_type)

    elif args.arch == 'residual-learnable':

        model = custom_nn.ValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=n_neuron, num_layers=n_hidden_layers, time_encoding='learnable', norm_type=args.norm_type)
    
    else:
        raise ValueError(f"Unknown architecture {args.arch}. Choose between 'mlp', 'residual', 'residual-sin', 'residual-learnable'.")

    model.to(device)
    model.float()

    # Optimizer and scheduler
        # optimizer
    opt_args = {}
    if args.opt_args is not None:
        for arg in args.opt_args.split(','):
            key, value = arg.split(':')
            opt_args[key] = json.loads(value)
        if not 'lr' in opt_args:
            opt_args['lr'] = config["lr"]
        if not 'weight_decay' in opt_args:
            opt_args['weight_decay'] = config["weight_decay"]

    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), **opt_args)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), **opt_args)
    else:
        raise ValueError(f"Unknown optimizer {args.opt}. Choose between 'adam' and 'sgd'.")
    
        # scheduler 
    sched_args = {}
    if args.sched_args is not None :
        for arg in args.sched_args.split(','):
            key, value = arg.split(':')
            sched_args[key] = json.loads(value)

    if args.sched == 'multistep':
        if 'milestones' not in sched_args:
            sched_args['milestones'] = config["milestones"]
        if 'gamma' not in sched_args:
            sched_args['gamma'] = config["gamma"]

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sched_args)
    elif args.sched == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_args)
    else:
        raise ValueError(f"Unknown scheduler {args.sched}. Choose between 'multistep' and 'cosine'.")

    # Starter
    start_epoch = 0

        # training losses 
    losses = []
    interior_losses = []
    boundary_losses = []
    terminal_losses = []

    var_to_plot = {
        "losses": [],
        "interior_losses":[], "boundary_losses":[], "terminal_losses":[]
    }

    percent_interior = []
    percent_boundary = []
    percent_terminal = []

        # validation losses
    losses_val = []
    interior_losses_val = []
    boundary_losses_val = []
    terminal_losses_val = []

    best_loss_val = None

    # Load checkpoint if needed 
    if config['checkpoint_path'] is not None:
        start_epoch, additional_info = utils.load_checkpoint(config['checkpoint_path'], model, optimizer, scheduler)
            
        if "losses" in additional_info:
            losses = additional_info['losses']
        if "interior_losses" in additional_info:
            interior_losses = additional_info['interior_losses']
        if "boundary_losses" in additional_info:
            boundary_losses = additional_info['boundary_losses']
        if "terminal_losses" in additional_info:
            terminal_losses = additional_info['terminal_losses']

        
        if "losses_val" in additional_info:
            losses_val = additional_info['losses_val']
        if "interior_losses_val" in additional_info:
            interior_losses_val = additional_info['interior_losses_val']
        if "boundary_losses_val" in additional_info:
            boundary_losses_val = additional_info['boundary_losses_val']
        if "terminal_losses_val" in additional_info:
            terminal_losses_val = additional_info['terminal_losses_val']

        logger.info(f"Checkpoint loaded from {config['checkpoint_path']} at epoch {start_epoch}")
        epoch = start_epoch
    else :
        logger.info("No checkpoint loaded, starting from scratch")

    # Pre-training info
        # device
    logger.info(f"Using device: {device}")
        # architecture
    logger.info(f"Architecture : {args.arch} with {n_hidden_layers} hidden layers x {n_neuron} neurons/layers")
    if 'residual' in args.arch:
        logger.info(f'Using normalization type {args.norm_type} for the residual mlp')
        # data 
    logger.info(f"Sample size {n_samples} = {n_batches} batches x {samples_per_batch} samples per batch ")
    logger.info(f"Number of training points per interation : {n_samples * n_periods} interior points + {n_samples * n_periods} boundary points + {n_samples} terminal points")
    logger.info(f'sample size for : interior = {n_samples * n_periods}, boundary = {n_samples * n_periods}, terminal = {n_samples} >> total = {n_samples * (2 * n_periods + 1)} \n')
        # optimizer and scheduler
    logger.info(f"Optimizer = {args.opt} / Scheduler = {args.sched} with arguments : {sched_args}")
        # weighting scheme
    logger.info(f"Coefficient for interior loss : start with initial value = {lambda_V_start} and update every {args.d} epoch(s) until reaching final value = {lambda_V_end} at epoch {args.e}")
        # general info
    logger.info(f"Seed = {args.seed}")
    logger.info(f"Training method : PINN")
    logger.info(f'Independent resampling of initial states = {ind_resample_initial_state}')
    logger.info(f'Independent resampling of brownians = {ind_resample_brownian}')
    logger.info('Note that the total loss reported does NOT include lambda_V, which is used only in the loss for gradient descent')
    
    # Training loop
    start = time.time()

    best_loss_train = None        # this is for choosing the 'best' model

    for epoch in range(start_epoch, n_epoch):
        # header 
        logger.info(f"Epoch [{epoch+1}/{n_epoch}]:  Lambda w = {lambda_V} - Learning rate =  {scheduler.get_last_lr()[0]}")

        # initiate 
        model.train()

        epoch_loss = 0
        epoch_interior_loss = 0
        epoch_boundary_loss = 0
        epoch_terminal_loss = 0

        # Shuffle the batch seeds for each epoch
        idx = torch.randperm(batch_seeds.size(0))
        batch_seeds = batch_seeds[idx]  # Shuffle the seeds

        for j in range(n_batches):
            # header
            logger.debug(f"Batch {j+1}/{n_batches}, Seed = {batch_seeds[j].item()}")

            # generate data
            idx = torch.randperm(samples_per_batch)                         # shuffling generator

                # initial states
            if ind_resample_initial_state:
                temp_seed = torch.seed()                                    # random seed
                logger.debug(f"Changing seed to {temp_seed} for independent resampling of initial states")
            else:
                torch.manual_seed(batch_seeds[j].item())                    # seed from batch seed

            initial_x = pm.simulate_initial_state(samples_per_batch, n_assets, dist_params, 
                                                scale=True).to(device)      # dim = (samples_per_batch, dim_state)
            initial_x = initial_x[idx]                                      # shuffle data
            
                # constraint limit
            initial_p = pm.generate_p(initial_x, w_NN, p_range).to(device)  # dim = (samples_per_batch, )

                # brownian
            if ind_resample_brownian :
                temp_seed = torch.seed()                                    # random seed
                logger.debug(f"Changing seed to {temp_seed} for independent resampling of brownians")
            else:
                torch.manual_seed(batch_seeds[j].item())                    # seed from batch seed

            brownians = pm.simulate_brownians(samples_per_batch, dim_sto, n_periods, dt, 
                                              corr).to(device)              # dim = (samples_per_batch, dim_sto, n_periods)
            brownians = brownians[idx]                                      # shuffle data
               
                # split data into regions
            interior_data, interior_controls, boundary_data, terminal_data = generate_data_V_rebounce(x_0 = initial_x, p_0 = initial_p, dW = brownians, 
                                                                            a_nn = a_NN, w_nn = w_NN, u_nn = u_NN, T = T, boundary_update_func = u_update_func)
            
            # reset gradient 
            optimizer.zero_grad()

            # compute loss - the forward pass is included inside the definition of the function
            batch_interior_loss = LossFunction_interior(V_NN = model, 
                                aug_states = interior_data, aug_controls = interior_controls, 
                                drift_func = drift_func, vol_func = vol_func, g_func = g_func, dim_control = dim_control)
            batch_boundary_loss = LossFunction_boundary(V_NN = model, aug_states = boundary_data, V_bound = V_bound_NN)
            batch_terminal_loss = LossFunction_terminal(V_NN = model, aug_terminal_states = terminal_data, F = utility_func)
                
            loss = lambda_V * batch_interior_loss + batch_boundary_loss + batch_terminal_loss

            # back propagate (within each batch)
            loss.backward()
            optimizer.step()

            # update weight for interior loss, if necessary
            if (epoch+1) % args.d == 0 :
                lambda_V = min(lambda_V_max, max(lambda_V_min, (lambda_V_end - lambda_V_start)/args.e * (epoch+1) + lambda_V_start))

            # cumulate losses at epoch level
            epoch_interior_loss += batch_interior_loss.detach().item() 
            epoch_boundary_loss += batch_boundary_loss.detach().item()
            epoch_terminal_loss += batch_terminal_loss.detach().item()

        # save loss values (at epoch level)
        epoch_interior_loss /= n_batches     
        epoch_boundary_loss /= n_batches
        epoch_terminal_loss /= n_batches
        epoch_loss = epoch_interior_loss + epoch_boundary_loss + epoch_terminal_loss

        losses.append(epoch_loss)
        interior_losses.append(epoch_interior_loss)
        boundary_losses.append(epoch_boundary_loss)
        terminal_losses.append(epoch_terminal_loss)

        # adjust learning rate
        scheduler.step()

        # validation 
        model.eval()
        with torch.no_grad():
            # initiation
            epoch_interior_loss_val = 0
            epoch_boundary_loss_val = 0
            epoch_terminal_loss_val = 0

            for j_val in range(n_batches):
            # generate data
                torch.manual_seed(batch_seeds_val[j_val].item()+1)      # the +1 is used to obtain different data for validation
                initial_x_val = pm.simulate_initial_state(samples_per_batch, n_assets, dist_params, scale=True).to(device)
                initial_p_val = pm.generate_p(initial_x_val, w_NN, p_range).to(device)
                brownians_val = pm.simulate_brownians(samples_per_batch, dim_sto, n_periods, dt, corr).to(device)

                interior_data_val, interior_controls_val ,boundary_data_val, terminal_data_val = generate_data_V_rebounce(x_0 = initial_x_val, p_0 = initial_p_val, dW = brownians_val, 
                                                                                                    a_nn = a_NN, w_nn = w_NN, u_nn = u_NN, T = T, boundary_update_func = u_update_func)
            
            # compute loss
                batch_interior_loss_val = LossFunction_interior(V_NN = model, 
                                aug_states = interior_data_val, aug_controls = interior_controls_val, 
                                drift_func = drift_func, vol_func = vol_func, g_func = g_func, dim_control = dim_control)
                batch_boundary_loss_val = LossFunction_boundary(V_NN = model, aug_states = boundary_data_val, V_bound = V_bound_NN)
                batch_terminal_loss_val = LossFunction_terminal(V_NN = model, aug_terminal_states = terminal_data_val, F = utility_func)
                    
                lambda_V * batch_interior_loss_val + batch_boundary_loss_val + batch_terminal_loss_val
            
            # accumulate losses across batches
                epoch_interior_loss_val += batch_interior_loss_val.detach().item()
                epoch_boundary_loss_val += batch_boundary_loss_val.detach().item()
                epoch_terminal_loss_val += batch_terminal_loss_val.detach().item()

            # save loss values at epoch level
            epoch_interior_loss_val /= n_batches
            epoch_boundary_loss_val /= n_batches
            epoch_terminal_loss_val /= n_batches
            epoch_loss_val = epoch_interior_loss_val + epoch_boundary_loss_val + epoch_terminal_loss_val

            losses_val.append(epoch_loss_val)
            interior_losses_val.append(epoch_interior_loss_val)
            boundary_losses_val.append(epoch_boundary_loss_val)
            terminal_losses_val.append(epoch_terminal_loss_val)                

    # Track progress
        logger.info(f" --- Training loss (total) = {epoch_loss: 0.9f} : interior = {epoch_interior_loss: 0.9f} & boundary = {epoch_boundary_loss: 0.9f} & terminal = {epoch_terminal_loss: 0.9f}")
        logger.info(f" - Validation loss (total) = {epoch_loss_val: 0.9f} : interior = {epoch_interior_loss_val: 0.9f} & boundary = {epoch_boundary_loss_val: 0.9f} & terminal = {epoch_terminal_loss_val: 0.9f}")

        # check whether to save checkpoint based on validation loss
        if best_loss_val is None or best_loss_val > epoch_loss_val:
            best_loss_val = epoch_loss_val
            utils.save_checkpoint(model, optimizer = optimizer, scheduler_step = scheduler, epoch = epoch, basedir=checkpoint_dir, suffix = 'best',
                additional_info = {
                                    "losses": losses, "interior_losses": interior_losses, "boundary_losses": boundary_losses, "terminal_losses": terminal_losses,
                                    "losses_val": losses_val, "interior_losses_val": interior_losses_val, "boundary_losses_val": boundary_losses_val, "terminal_losses_val": terminal_losses_val,
                                    }
                                )
            logger.info(f">>> Found best model in Epoch {epoch+1}, saving in {checkpoint_dir}")
        
    # Finalize training
    end = time.time()
    time_exec = end - start
    logger.info(f"Execution time : {(time_exec)/60} minutes")    
