# ---- Packages ------- # 
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


from tqdm import tqdm
import shutil

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
def create_Vb_func(Vb_nn): # this function takes in (t,x) as input 
    def Vb_func(t, x):
        # Concatenate t and x
        if len(x.size()) == 1: 
            z = torch.cat((t.unsqueeze(0), x.unsqueeze(0)), dim=1)
        else:
            z = torch.cat((t, x), dim=1)
        # Pass through the value network
        return Vb_nn(z)
    return Vb_func

# general operator Dynkin (for any arbitrary control u)
def Dynkin(Vb_NN, drift_func, vol_func):
    '''
    The Dynkin of a function Vb at point (t,x) given the control u is :
       H^u_Vb(t,x,u) = Vb_t(t, x) + b(t, x, u) @ Vb_x(t, x) + 0.5 * Tr(sigma(t, x, u) @ sigma(t, x, u)^T @ Vb_xx(t, x))

    Note that as of now, this function is point-wise ! 
    '''
    # compute gradients and hessian of Vb
    Vb_func = create_Vb_func(Vb_NN)
    dVb_t_func = torch.func.grad(Vb_func, 0)
    dVb_x_func = torch.func.grad(Vb_func, 1)
    d2Vb_x_func = torch.func.hessian(Vb_func, 1)

    def LuVb(t, x, u):
        # compute drift and volatility for a given u
        drift = drift_func(x.detach().unsqueeze(0), u.unsqueeze(0)).squeeze() 
        vol = vol_func(x.detach().unsqueeze(0)).squeeze()
        variance = torch.matmul(vol.detach(), vol.detach().T)

        # compute gradients and hessian at (t,x)
        dVb_t = dVb_t_func(t, x).detach()
        dVb_x = dVb_x_func(t, x).detach()
        d2Vb_x = d2Vb_x_func(t, x).detach()

        # compute the dynkin operator of Vb at (t,x)
        dynkin = dVb_t + torch.dot(drift, dVb_x)          # dim = (M, )
        dynkin += 0.5 * torch.trace(torch.matmul(variance, d2Vb_x))
        return dynkin

    return LuVb   

    # operator Dynkin associated with u_nn
def Dynkin_u_NN(Vb_NN, u_NN, drift_func, vol_func):
    
    LuVb = Dynkin(Vb_NN, drift_func, vol_func)
    n_periods = len(u_NN)

    def LuVb_nn(t, x):
        current_time_idx = get_index_from_time(t, n_periods)                # indicator of the subnetwork / time period
        u = u_NN[current_time_idx](x.reshape(1, -1)).squeeze().detach()     # passing the current state through the corresponding subnetwork
        return LuVb(t, x.detach(), u.detach())
    
    return LuVb_nn

# compute the components of the error measure
def compute_interior_error(Vb_NN, u_NN, sample, 
                  drift_func, vol_func):
    '''
    Inputs : 
        sample = (t,x) : a sample of time-space variable for the "inner" part of the error measure, i.e. t < T
            _ dim = (n_samples, dim_state + 1)
        Vb_NN : value function at the boundary neural network
        u_NN : control neural network
        drift_func, vol_func : drift and volatility functions from the portoflio model
        
    Outputs : the average absolute value of the dynkins (u_NN) at each point (t,x) in the sample
    '''
    M = sample.shape[0]                     # number of data points
    X = sample[:, 1:]                       # dim = (M, dim_state)

    # supporting functions
    dynkin_u_NN_func = Dynkin_u_NN(Vb_NN, u_NN, drift_func, vol_func)
    
    # error measure computation
    e_Vb = 0.0
    for i in tqdm(range(M)):
        t,x = sample[i,0].unsqueeze(0).detach(), sample[i,1:].detach()
        d_u = dynkin_u_NN_func(t,x).detach().item()
        e_Vb += abs(d_u)
        
    e_Vb = e_Vb / M

    return e_Vb

if __name__ == '__main__':

    # Small alias for this function 
    J = os.path.join

    # ---- Parser --------- #
    # Create argument parser to read config file
    parser = ArgumentParser(description='Train: Augmented Optimal Control Neural Network', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-u', '--config_u', type=str, required=True, help='path to the results folder of the optimal control network')
    parser.add_argument('-vb', '--config_vb', type=str, required=True, help='path to the results folder of the domain boundary value network')
    parser.add_argument('-o', '--results_dir', type=str, default='./results/debug', help='directory to store the results')
    parser.add_argument('-s', '--seed', type=int, default=21, help='seed for the random number generator')
    parser.add_argument('-n', '--n_samples', type=int, default=1000, help='Number of samples to simulate')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='log level for the logger, e.g., INFO, DEBUG, WARN', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
    parser.add_argument('--save_every', type=int, default=100, help='save the model every n epochs')

    # Load portfolio model parameters : --config_portfolio and --config_simulator
    pm.parser_portfolio(parser)
    args = parser.parse_args()

    # Create a base directory to store results, checkpoints and figures
    results_dir = utils.create_dir(basedir=args.results_dir, dirname = "test_vb", suffix = None)
    logger = utils.setup_logging(results_dir, log_level = args.log_level, fname='test_vb.log')

        # device and seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Save configuration files for replication
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

        # wrap the necessary functions with portfolio parameters
    gn_update_func = lambda x, u, dW: pm.update(x=x, u=u, dt=dt, dW = dW, d=n_assets,
                                                  mu_S=mu_S, sigma_S=sigma_S, param_r=param_r, 
                                                  param_L= param_L, coupon_rate = coupon_rate, 
                                                  book_value = book_value)
    utility_func = lambda x: pm.terminal_utility_exponential_2(x, alpha)
    drift_func = lambda x,u : pm.drift(x, u, n_assets, mu_S, param_r, param_L, coupon_rate, book_value)
    vol_func = lambda x : pm.vol(x, n_assets, sigma_S, param_r)
    
    # Load u network
    unn_dir = args.config_u                                                           # locate
    config_unn = json.load(open(os.path.join(unn_dir, 'config_u.json'), 'r'))       # retrieve configuration

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
    dir_list = [os.path.join(unn_dir, 'checkpoints', f) for f in os.listdir(os.path.join(unn_dir, 'checkpoints'))]
    unn_checkpoint_path = max(dir_list, key=os.path.getmtime)

    u_NN.to(device=device)
    u_NN.load_state_dict(torch.load(unn_checkpoint_path, map_location=device)['model_state_dict'])
    logger.info(f"u_nn loaded from {unn_checkpoint_path}")
    u_NN.eval()     

    # Load V_bound network
        # load the configuration for the value network, create the model and load the weights
    V_bound_dir = args.config_vb
    with open(J(V_bound_dir, "config_vb.json"), 'r') as f:
        config_v_bound = json.load(f)

    shutil.copy(J(V_bound_dir, "config_vb.json"), J(results_dir,'config_vb.json'))

        # select the correct architecture
    with open(J(V_bound_dir, "args.json"), 'r') as f:
        args_vbnn = json.load(f)

    shutil.copy(J(V_bound_dir, "args.json"), J(results_dir,'args_vb.json'))


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
        raise ValueError(f"Unknown architecture {args_vbnn['arch']}. Choose between 'mlp' and 'residual'.")
    
    
        # load the last checkpoint from the checkpoints directory
    dir_list = [J(V_bound_dir, 'checkpoints', f) for f in os.listdir(J(V_bound_dir, 'checkpoints'))]
    V_bound_nn_checkpoint_path = max(dir_list, key=os.path.getmtime)

    V_bound_NN.to(device=device)
    V_bound_NN.load_state_dict(torch.load(V_bound_nn_checkpoint_path, map_location=device)['model_state_dict'])
    logger.info(f"V_bound_nn loaded from {V_bound_nn_checkpoint_path}")
    V_bound_NN.eval()

# ---- Computation --- # 
# Generate data
    # generate initial states and brownians
    n_samples = args.n_samples
    x_0 = pm.simulate_initial_state(n_samples, n_assets, dist_params, scale=True).to(device)
    dW = pm.simulate_brownians(n_samples, n_assets + 1, n_periods, dt, corr).to(device)

    # roll forward with the control network u_NN
    inner_data, terminal_data = generate_data_error(x_0, dW, u_NN, T)
    inner_data, terminal_data = inner_data.to(device), terminal_data.to(device)

    logger.info(f"Sample size : interior = {inner_data.size(0)} - terminal = {terminal_data.size(0)}") 

# Calculate components of the error measure
    # set all networks and data to the same data type (float)
    V_bound_NN = V_bound_NN.float()
    u_NN = u_NN.float()
    inner_data = inner_data.float()
    terminal_data = terminal_data.float()   
   
    # error for the interior region
    e_Vb = compute_interior_error(V_bound_NN, u_NN, inner_data, drift_func, vol_func)

    # error for the terminal region
    with torch.no_grad():
        terminal_prediction = V_bound_NN(terminal_data)
        e_T = torch.abs(terminal_prediction - utility_func(terminal_data[:, 1:])).mean()
        e_T = e_T.item()

    total_error = e_Vb + e_T
   
    # determinator of the error measure
    x_0_extended = torch.concat([torch.zeros((n_samples, 1)).to(device), x_0], dim = 1)
    with torch.no_grad():
        Vb_0 = V_bound_NN(x_0_extended)
        avg_Vb_0 = torch.mean(Vb_0).item()
   
    # error measure
    error_measure = total_error / abs(avg_Vb_0)

    # Print results
    logger.info(f"Error measure for V_bound = {error_measure}, equivalent to {100*error_measure} %")
    logger.info(f"Composition of errors (in average): e_Vb = {e_Vb}, e_T = {e_T}")
    logger.info(f"Denominator (in average) : Vb_0 = {avg_Vb_0} ")