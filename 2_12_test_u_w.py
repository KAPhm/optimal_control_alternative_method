# ---- Packages ------ # 
import torch

import os
import json
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from tqdm import tqdm

import itertools

import importlib
    # to import customized functions
pm = importlib.import_module('0_portfolio_model', package=None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

# ---- Definitions --- # 
# time conversion 
def get_index_from_time(t, n_periods):
    if type(t) == torch.Tensor:
        t = t.item()
    return int(round(t*n_periods, 0))

# data generator 
def generate_data_error(x_0, dW, u_nn, T): 
    '''
    Inputs :
        x_0 : a sample of initial states                 - dim = (n_samples, dim_state)
        dW : a sample of brownian increment trajectories - dim = (n_samples, dim_state, n_periods)
        u_nn : the trained optimal control network with n_subnetworks = n_periods
    Outputs :
        a sample of training data in the form (t, x) of dim = (n_samples *n_periods , dim_state + 1)
    Notes :
        _ the sample is obtained by rolling forward x_0 and dW using control_network
    '''
    device = x_0.device

    # roll forward data for the control network
    u_nn.eval()
    with torch.no_grad():
        data, controls = u_nn(x_0.to(device), dW.to(device))
        data = data.to(device)
        n_samples, dim_state, n_periods = data.shape
        dt = T/n_periods 

        # inner data for t < T 
            # add initial states x_0
        inner_data = torch.cat([x_0.unsqueeze(2), data[:,:,:-1]], dim = 2)    # dim = (n_samples, dim_state, n_periods)
            # reshape
        inner_data = torch.transpose(inner_data, 1, 2)                      # dim = (n_samples, n_periods, dim_state)
        inner_data = inner_data.reshape((n_samples * n_periods, -1))        # dim = (n_samples * n_periods, dim_state)
            # add time dimension
        time_data = torch.linspace(0, T-dt, n_periods).reshape(-1,1).repeat(n_samples, 1).to(device)
                                                                            # dim = (n_samples * n_periods, 1)
        inner_data = torch.cat([time_data, inner_data], dim =1)             # dim = (n_samples * n_periods, 1 + dim_state)

        # terminal data for t = T
        tensor_T = torch.ones((n_samples, 1)).to(device) * T           
        terminal_data = torch.cat([tensor_T, data[:,:,-1].squeeze()], dim = 1)  # dim = (n_samples, 1 + dim_state)

    return inner_data.to(torch.float16), terminal_data.to(torch.float16)
    
# supporting functions :
    # function to compute the derivatives
def create_w_func(w_nn): # this function takes in (t,x) as input 
    def w_func(t, x):
        # Concatenate t and x
        if len(x.size()) == 1: 
            z = torch.cat((t.unsqueeze(0), x.unsqueeze(0)), dim=1)
        else:
            z = torch.cat((t, x), dim=1)
        # Pass through the value network
        return w_nn(z)
    return w_func

    # general operator Dynkin (for any arbitrary control u)
def Dynkin(w_NN, drift_func, vol_func):
    '''
    The Dynkin of a function w at point (t,x) given the control u is :
       H^u_w(t,x,u) = w_t(t, x) + b(t, x, u) @ w_x(t, x) + 0.5 * Tr(sigma(t, x, u) @ sigma(t, x, u)^T @ w_xx(t, x))

    Note that as of now, this function is point-wise ! 
    '''
    # compute gradients and hessian of w 
    w_func = create_w_func(w_NN)
    dw_t_func = torch.func.grad(w_func, 0)
    dw_x_func = torch.func.grad(w_func, 1)
    d2w_x_func = torch.func.hessian(w_func, 1)

    def Luw(t, x, u):
        # compute drift and volatility for a given u
        drift = drift_func(x.detach().unsqueeze(0), u.unsqueeze(0)).squeeze() 
        vol = vol_func(x.detach().unsqueeze(0)).squeeze()
        variance = torch.matmul(vol.detach(), vol.detach().T)

        # compute gradients and hessian at (t,x)
        dw_t = dw_t_func(t, x).detach()
        dw_x = dw_x_func(t, x).detach()
        d2w_x = d2w_x_func(t, x).detach()

        # compute the dynkin operator of w at (t,x)
        dynkin = dw_t + torch.dot(drift, dw_x)          # dim = (M, )
        dynkin += 0.5 * torch.trace(torch.matmul(variance, d2w_x))
        return dynkin

    return Luw   

    # operator Dynkin associated with u_nn
def Dynkin_u_NN(w_NN, u_NN, drift_func, vol_func):
    
    Luw = Dynkin(w_NN, drift_func, vol_func)
    n_periods = len(u_NN)

    def Luw_nn(t, x):
        current_time_idx = get_index_from_time(t, n_periods)                # indicator of the subnetwork / time period
        u = u_NN[current_time_idx](x.reshape(1, -1)).squeeze().detach()     # passing the current state through the corresponding subnetwork
        return Luw(t, x.detach(), u.detach())
    
    return Luw_nn

    # operator Dynkin at optimal points for w_nn : 
        # in our problem, the optimizer is found at the limit of the permissible range for the controls
def Dynkin_optimal(w_NN, drift_func, vol_func, u_lower, u_upper):
    '''
    Due to the specifity of our model, the optimizer of the Dynkin operator should be found at the limit of the admissible range,
        meaning either at u_lower or u_upper.
    u_lower and u_upper are of dimension (dim_control, )
    '''
    LuW = Dynkin(w_NN, drift_func, vol_func)

    def Luw_optim(t,x):
        # list of candidates for the optimizer
        dim_control = u_lower.shape[0]
        with torch.no_grad():
            permutations = torch.tensor(list(itertools.product([0,1], repeat=3))).to(u_lower.device)       # dim = (2**dim_control, dim_control)
            u_lower_extend = u_lower.unsqueeze(0).expand(2**dim_control, dim_control)   
            u_upper_extend = u_upper.unsqueeze(0).expand(2**dim_control, dim_control)

            candidates = torch.where(permutations.bool(), u_lower_extend, u_upper_extend)
            values = torch.tensor([LuW(t,x,u) for u in candidates])
            optimal = torch.min(values).item()
            optimizer_index = torch.argmin(values)  # retain the optimizers
            optimizer = candidates[optimizer_index]
        return optimal, optimizer
    
    return Luw_optim

# compute the components of the error measure
def compute_error(w_NN, u_NN, sample, 
                  drift_func, vol_func, penalty_func,
                  u_lower, u_upper, reg_coeff):
    '''
    Inputs : 
        sample = (t,x) : a sample of time-space variable for the "inner" part of the error measure, i.e. t < T
            _ dim = (n_samples, dim_state + 1)
        w_NN : boundary value neural network
        u_NN : control neural network
        drift_func, vol_func : drift and volatility functions from the portoflio model
        u_lower, u_upper : lower and upper limits of the permissible range for the control variable
            _ dim = (dim_control, )
        reg_coeff : regularization coefficient for the penalty function

    Outputs : the dynkins (optimal and u_NN) and penalties at each point (t,x) in the sample
    '''
    # dimension
    M = sample.shape[0]     # number of data points

    # supporting functions
    dynkin_u_NN_func = Dynkin_u_NN(w_NN, u_NN, drift_func, vol_func)
    dynkin_optim_func = Dynkin_optimal(w_NN, drift_func, vol_func, u_lower, u_upper)

    # penalty at each state (time dimension excluded here)
    X = sample[:, 1:]                       # dim = (M, dim_state)
    penalties = penalty_func(X).detach()    # dim = (M, )
    penalties_avg = torch.mean(penalties).item()

    # error measure computation
    e_u, e_w = 0.0, 0.0
    for i in tqdm(range(M)):
        t,x = sample[i,0].unsqueeze(0).detach(), sample[i,1:].detach()
        d_o, _ = dynkin_optim_func(t,x)
        d_u = dynkin_u_NN_func(t,x)
        e_u += abs(d_u - d_o)
        e_w += abs(d_u + reg_coeff * penalties[i])
    
    e_u = e_u / M
    e_w = e_w / M

    return e_u, e_w, penalties_avg

if __name__ == "__main__":
# ---- Parsers ------- #  
    # Create argument parser to read config file
    parser = ArgumentParser(description = 'Test : Optimal Control u and Domain Boundary w v4',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-u', '--config_u', type=str, required=True, help='Path to the directory containing the configuration for the optimal control network, and the checkpoint folder')
    parser.add_argument('-w', '--config_w', type=str, required=True, help='Path to the directory containing the configuration for the value network, and the checkpoint folder')
    parser.add_argument('-n', '--n_samples', type=int, default=1000, help='Number of samples at initial state')
    parser.add_argument('-o', '--results_dir', type=str, default='./results', help='directory to store the results')
    parser.add_argument('-s', '--seed', type=int, default=133, help='Random seed for reproducibility')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='log level for the logger, e.g., INFO, DEBUG, WARN', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
    parser.add_argument('--force-cpu', action='store_true', help='Force the use of CPU even if a GPU is available')

    J = os.path.join # alias for frequently used function 

    # Portfolio model parameters : --config_portfolio and --config_simulator
    pm.parser_portfolio(parser)
    args = parser.parse_args()

    # device and seed
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu")

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

        # fixed the portfolio parameters in necessary functions 
    gn_update_func = lambda x, u, dW: pm.update(x=x, u=u, dt=dt, dW = dW, d=n_assets,
                                                mu_S=mu_S, sigma_S=sigma_S, param_r=param_r, 
                                                param_L= param_L, coupon_rate = coupon_rate, 
                                                book_value = book_value)
    drift_func = lambda x,u : pm.drift(x, u, n_assets, mu_S, param_r, param_L, coupon_rate, book_value)
    vol_func = lambda x : pm.vol(x, n_assets, sigma_S, param_r)
    penalty_func = lambda x: pm.total_intermediate_penalty(x, margin)
    final_loss_func = lambda x: pm.terminal_capital_loss(x, K_below, K_above) 

    # Create a base directory to store results, checkpoints and figures
    results_dir = utils.create_dir(basedir=args.results_dir, dirname = "test_error", suffix = None)
    logger = utils.setup_logging(results_dir, log_level = args.log_level, fname='test_error.log')

    # Load the trained optimal control network u_nn
    unn_dir = args.config_u                                                           # locate
    config_unn = json.load(open(J(unn_dir, 'config_u.json'), 'r'))       # retrieve configuration

        # parameters for loss function (shared parameters from the training of u_nn)
    reg_coeff = config_unn['k']
    margin = config_unn['margin']

        # load the trained optimal neural network
    u_NN = custom_nn.GlobalNetworks(dim_input = dim_state,
                            dim_output = dim_control,
                            dim_hidden = config_unn['n_neuron'],
                            n_hidden = config_unn['n_hidden_layers'],
                            n_subnetworks = n_periods,
                            update_func=gn_update_func,
                            lower_bounds = u_lower,
                            upper_bounds = u_upper
                            )

        # load the last checkpoint from the checkpoints directory
    dir_list = [J(unn_dir, 'checkpoints', f) for f in os.listdir(os.path.join(unn_dir, 'checkpoints'))]
    unn_checkpoint_path = max(dir_list, key=os.path.getmtime)

    u_NN.to(device=device)
    u_NN.load_state_dict(torch.load(unn_checkpoint_path, map_location=device)['model_state_dict'])
    u_NN.eval()

    # Load the configuration for the value network, create the model and load the weights
    w_dir = args.config_w
    with open(J(w_dir, "config_w.json"), 'r') as f:
        config_wnn = json.load(f)

    shutil.copy(J(w_dir, "config_w.json"), J(results_dir,'config_w.json'))


    with open(J(w_dir, "args.json"), 'r') as f:
        args_wnn = json.load(f)

    shutil.copy(J(w_dir, "args.json"), J(results_dir,'args_w.json'))


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
    
    # Load the last checkpoint from the checkpoints directory
    dir_list = [J(w_dir, 'checkpoints', f) for f in os.listdir(J(w_dir, 'checkpoints'))]
    wnn_checkpoint_path = max(dir_list, key=os.path.getmtime)

    w_NN.to(device=device)
    w_NN.load_state_dict(torch.load(wnn_checkpoint_path, map_location=device)['model_state_dict'])
    logger.info(f"Value network loaded from {wnn_checkpoint_path}")
    w_NN.eval()

# ---- Computation --- # 
# Generate data
    # generate initial states and brownians 
    n_samples = args.n_samples
    x_0 = pm.simulate_initial_state(n_samples, n_assets, dist_params, scale=True).to(device)
    dW = pm.simulate_brownians(n_samples, n_assets + 1, n_periods, dt, corr).to(device)

    # roll forward with the control network u_NN
    inner_data, terminal_data = generate_data_error(x_0, dW, u_NN, T)
    inner_data, terminal_data = inner_data.to(device), terminal_data.to(device)

    logger.info(f"Sample size final: {inner_data.size(0)}") # should have dim = (n_samples * n_periods, 1 + dim_x) = (n_samples * n_periods, 2 * n_assets + 4)

# Compute components of error measure
    # set all networks and data to the same data type (float)
    w_NN = w_NN.float()
    u_NN = u_NN.float()
    inner_data = inner_data.float()
    terminal_data = terminal_data.float()   

    # error component for the inner data
    e_u, e_w, penalties_avg = compute_error(w_NN, u_NN, inner_data, 
                                            drift_func, vol_func, penalty_func, 
                                            u_lower, u_upper, reg_coeff)
    
    # error component for the terminal data
    with torch.no_grad():
        terminal_prediction = w_NN(terminal_data)   # dim = (n_samples,)
        e_T = torch.abs(terminal_prediction - final_loss_func(terminal_data[:, 1:])).mean()

    total_error = e_u + e_w + e_T.item()

    # denominator of the error measure
    extended_x_0 = torch.concat([torch.zeros((n_samples, 1)).to(device), x_0],
                                dim = 1) # dim = (n_samples, 1 + dim_state)
    with torch.no_grad():
        w_0 = w_NN(extended_x_0)
        avg_w_0 = torch.mean(w_0).item()

    # error measure
    error_measure = total_error / abs(avg_w_0)

    error_measure = error_measure.item()

    logger.info(f"Error measure = {error_measure}, equivalent to {100*error_measure} %")
    logger.info(f"Composition of errors (in average): e_u = {e_u.item()}, e_w = {e_w.item()}, g = {penalties_avg}, reg_coeff = {reg_coeff}, e_T = {e_T.item()}")
    logger.info(f"Denominator (in average) : w_0 = {torch.mean(w_0).item()} ")