"""
This script is used to train the Value Function V.
Note that the training of V must be split into 2 parts : 
- V at the boundary of the domain, henceforth called V_bound, which takes only 2 arguments t and x as inputs
- V in the interior of the domain, which takes 3 arguments t, x, and p as inputs

In this module, we execute the first training.
PLEASE READ 3_Training_ValueFunction_at_DomainBoundary.md, section 1 for detailed explanation
"""
# ---- Packages --------- # 
import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import numpy as np
from numpy import exp, log
import matplotlib.pyplot as plt
import time
import shutil

import torch.nn as nn
import torch.optim as optim

import importlib
    # to import customized functions
pm = importlib.import_module('0_portfolio_model', package=None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

# ---- Loss Function ----- # 

# Loss function at the terminal of the time horizon 
def LossFunction_terminal(V_bound_NN, terminal_states, F):
    '''
    At T, the function V_bound(T, x) = F(x) for any feasible state x 
    Inputs :
        V_bound_NN : the estimate of V_bound in the form of neural network
        terminal_data : a sample of (T,x) to be passed into V_NN - dim = (n_samples, 1 + dim_state)
        F : utility function to be applied to the terminal state 
            _ must be wrapped with all the necessary parameters so that x is its only argument
            _ must be able to handle a sample of dim = (n_samples, dim_state)
    Output : 
        MSE loss between the predictions and the true terminal utilities from the data
            Loss = sum [ | terminal_predictions - F(terminal_data) |**2 ]/n_samples
    '''
    device = terminal_states.device                                    # device
    terminal_predictions = V_bound_NN(terminal_states).to(device)      # predictions from V_bound_NN
    terminal_targets = F(terminal_states[:, 1:]).to(device)            # time dimension is not needed
    return nn.MSELoss()(terminal_predictions, terminal_targets)

# Loss function at the parabolic boundary of the domain (excluding the terminal points)
def LossFunction_boundary(V_bound_NN, states, controls, 
                          drift_func, vol_func):
    '''
    For any pair (t,x) with t < T, we compute the PDE (L^u_X V_bound)(t,x) and uses its squares as the training loss
    Inputs :
        V_bound_NN : the estimate of V_bound in the form of neural network
        states : a sample of (t, x) to be passed into V_bound_NN                - dim = (n_samples, 1 + dim_state)
        controls : optimal controls (estimated by u_NN) for the sample of (t,x) - dim = (n_samples, dim_control)
        drift_func, vol_func : functions to compute the drift and the volatility (for Dynkin operator)
            _ must be already wrapped with parameters other than the state variable x
    Output : 
        Average PDE loss across sample : Loss = sum [ |(L^u w)(t,x)|^2 ] / n_samples
    '''
    X = states[:, 1:]           # extract the state variables
    device = states.device      # device

    # elements of computation
        # gradients and hessians
    dV_t, dV_x, dV_xx = utils.compute_grad_hessian_2(V_bound_NN, states)
    dV_t = dV_t.to(device)
    dV_x = dV_x.unsqueeze(2).to(device)    # dim = (n_samples, dim_state, 1) - additional dimension needed for tensor multiplication
    dV_xx = dV_xx.to(device)               # dim = (n_samples, dim_state, dim_state)

        # drifts and vols
    drifts = drift_func(X, controls).unsqueeze(2).to(device)        # dim = (n_samples, dim_state, 1)
    drifts_T = torch.transpose(drifts, 1, 2).to(device)             # dim = (n_samples, 1, dim_state)

    vols = vol_func(X).to(device)                                   # dim = (n_samples, dim_state, dim_sto)
    variances = torch.bmm(vols, torch.transpose(vols, 1, 2))        # dim = (n_samples, dim_state, dim_state)
    diffusion = torch.bmm(variances, dV_xx)                         # dim = (n_samples, dim_state, dim_state)

    # Dynkin operator - note that there is no penalty function attached to this Dynkin
    dynkins = dV_t + torch.bmm(drifts_T, dV_x)                      # dim = (n_samples, 1)
    dynkins += 0.5 * torch.vmap(torch.trace)(diffusion).unsqueeze(1)# dim = (n_samples, 1)
    dynkins = dynkins.to(device)

    errors = (dynkins.squeeze()) ** 2                               # dim = (n_samples)
    errors = errors.to(device)

    return torch.mean(errors)

if __name__ == '__main__':
# ---- Parser -------- # 
    # Create argument parser to read config file
    parser = ArgumentParser(description='Train: Value Function Network at the Domain Boundary with PINN',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-u', '--config_u', type=str, required=True, help='path to the results folder of the control network. This folder must contain a JSON file named `config_u.json` with the parameters used to train the control network, and a folder named checkpoints from which the weights of the optimal control network will be loaded. The loaded checkpoint will be the file that was last modified, obtained via `os.path.getmtime()`')
    parser.add_argument('-vb', '--config_vb', type=str, required=True, help='path to the configuration file with training parameters for the value function network at the boundary V_bound')
    parser.add_argument('-o', '--results_dir', type=str, default='./results', help='directory to store the results')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed for the random number generator')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='log level for the logger, e.g., INFO, DEBUG, WARN', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
    

    parser.add_argument('--arch', type=str, default='mlp', help='Architecture to use. Careful when loading checkpoint, they must coincide.', choices=['mlp', 'residual', 'residual-sin', 'residual-learnable'])

    parser.add_argument('--norm-type', type=str, default='batch', help='Type of normalization layer used in the residual mlp.', choices=['batch', 'layer', 'identity'])
    

    parser.add_argument('-e', type=int, default=100, help='number of epochs untill the regularization parameter lambda_w is set to lambda_w_end. It will increase/decrease linearly from the initial value lambda_w_start to lambda_w_end in the first e epochs. Default is 100')

    parser.add_argument('-d', type=int, default=1, help='number of epochs in which the lamnda_w remains constnt between two modifications')

    parser.add_argument('--lambda_start', type=float, default=1.0, help='initial regularization parameter lambda_w for the interior loss. It will increase/decrease linearly from the initial value lambda_start to lambda_end in the first e epochs, remaining constant for d epochs. Default is 1')

    parser.add_argument('--lambda_end', type=float, default=1.0, help='final regularization parameter lambda_w for the interior loss. It will increase/decrease linearly from the initial value lambda_start to lambda_end in the first e epochs, remaining constant for d epochs. Default is 1')


    parser.add_argument('--opt', type=str, default='adam', help='Optimizer.', choices=['adam', 'sgd'])

    parser.add_argument('--opt-args', type=str, help='Arguments for the optimizer in the format key1:value1,key2:value2.... If passed, they replaces the ones in the config file if present. See the pytorch docs for the parameters available for each optimizer.')

    parser.add_argument('--sched', type=str, default='multistep', help='Scheduler.', choices=['multistep', 'cosine'])

    parser.add_argument('--sched-args', type=str, help='Arguments for the scheduler in the format key1:value1,key2:value2.... If passed, they replaces the ones in the config file if present. See the pytorch docs for the parameters available for each scheduler.')

    # Portfolio model parameters : --config_portfolio and --config_simulator
    pm.parser_portfolio(parser)
    args = parser.parse_args()

        # device and seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        # fix the portfolio parameters in the update function
    gn_update_func = lambda x, u, dW: pm.update(x=x, u=u, dt=dt, dW = dW, d=n_assets,
                                                mu_S=mu_S, sigma_S=sigma_S, param_r=param_r, 
                                                param_L= param_L, coupon_rate = coupon_rate, 
                                                book_value = book_value)
    
    # Load the trained optimal control network u_nn
    unn_dir = args.config_u                                                           # locate
    config_u = json.load(open(os.path.join(unn_dir, 'config_u.json'), 'r'))       # retrieve configuration

        # load the trained optimal neural network
    u_NN = custom_nn.GlobalNetworks(dim_input = dim_state,
                            dim_output = dim_control,
                            dim_hidden = config_u['n_neuron'],
                            n_hidden = config_u['n_hidden_layers'],
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
    u_NN.eval()

    # Read config file for training V_bound
    config = json.load(open(args.config_vb, 'r'))  # retrieve configuration for training

        # hyper-parameters for training
    n_hidden_layers = config["n_hidden_layers"]     # number of hidden layers 
    n_neuron = config["n_neuron"]                   # number of neuron per hidden layer
    n_samples = config["n_samples"]                 # number of data points in the sample of initial states X_0 = (x_0^j)^{j=1,...,n_samples}



    n_epoch =  config["n_epoch"]                    # number of training epochs
    batch_size = config["batch_size"]               # batch size
    
    n_batches = n_samples // batch_size 
    torch.manual_seed(args.seed)                    # Set the seed to obtain the same batch seeds depending on input seed
    batch_seeds = torch.multinomial(torch.ones(10000), n_batches, replacement=False).to(torch.int16)
    batch_seeds_val = torch.multinomial(torch.ones(10000), n_batches, replacement=False).to(torch.int16)

        # optimizer and scheduler
    lr = config["lr"]                               # learning rate
    milestones = config["milestones"]               # milestones for the learning rate scheduler
    gamma = config["gamma"]                         # decay factor for the learning rate scheduler

    lambda_w_start = args.lambda_start                   # regularization coefficient for the terminal penalty function in the loss function of w_nn
    lambda_w_end = args.lambda_end

    lambda_w_min = min(lambda_w_start, lambda_w_end) # minimum value of lambda_w
    lambda_w_max = max(lambda_w_start, lambda_w_end) # maximum value of lambda_w
    lambda_w = lambda_w_start

        # Resampling indicator
    ind_resample_initial_state = config.get("ind_resample_initial_state", False)
    ind_resample_brownian = config.get("ind_resample_brownian", False)
    

# ---- Directory ---- #
    # Set a directory to store results
    results_dir = utils.create_dir(basedir=args.results_dir, dirname = "train_vb", suffix = None)
    checkpoint_dir = utils.create_dir(basedir=results_dir, dirname = "checkpoints", suffix = "")
    logger = utils.setup_logging(results_dir, log_level = args.log_level, fname='train_vb.log')

    # Pre-training info
    logger.info(f"Using device: {device}")
    logger.info(f"Optimal control network loaded from {unn_checkpoint_path}")
    logger.info(f"Number of batches = {n_batches}")

    # Save the config files in case needed for replication
    json.dump(config_u, open(f'{results_dir}/config_u.json', 'w'), indent=4)
    json.dump(json.load(open(args.config_portfolio, 'r')), open(f'{results_dir}/config_portfolio.json', 'w'), indent=4)
    json.dump(json.load(open(args.config_simulator, 'r')), open(f'{results_dir}/config_simulator.json', 'w'), indent=4)
    json.dump(config, open(f'{results_dir}/config_vb.json', 'w'), indent=4)

    # Copy the script file to the results directory

    script_name = os.path.basename(__file__)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(results_dir, script_name))

    # Save a json file with all the arguments in the parser
    args_dict = vars(args)
    args_dict['config_u'] = os.path.abspath(args.config_u)
    args_dict['config_vb'] = os.path.abspath(args.config_vb)
    args_dict['results_dir'] = os.path.abspath(args.results_dir)
    json.dump(args_dict, open(f'{results_dir}/args.json', 'w'), indent=4)

# ---- Loss Funct --- #
    # Wrap the drift, the volatility, the penalty (g), and the terminal loss (G) functions
    drift_func = lambda x,u : pm.drift(x, u, n_assets, mu_S, param_r, param_L, coupon_rate, book_value)
    vol_func = lambda x : pm.vol(x, n_assets, sigma_S, param_r)
    utility_func = lambda x: pm.terminal_utility_exponential_2(x, alpha, KF = -10e10)

# ---- Train V at the boundary --------- # 
    # Create a frame for V_bound_NN with chosen hyper-parameters

    if args.arch == 'mlp':
        model = custom_nn.ValueFunction_Boundary_Network(dim_input = dim_state + 1,
                                        dim_hidden = n_neuron,
                                        n_hidden = n_hidden_layers,
                                        activation = nn.ReLU())

    elif args.arch == 'residual':
    
        model = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=n_neuron, num_layers=n_hidden_layers, time_encoding='none', norm_type=args.norm_type)

    elif args.arch == 'residual-sin':
    
        model = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=n_neuron, num_layers=n_hidden_layers, time_encoding='sinusoidal', norm_type=args.norm_type)

    elif args.arch == 'residual-learnable':

        model = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=n_neuron, num_layers=n_hidden_layers, time_encoding='learnable', norm_type=args.norm_type)
    
    else:
        raise ValueError(f"Unknown architecture {args.arch}. Choose between 'mlp', 'residual', 'residual-sin', 'residual-learnable'.")

    model.to(device=device)

    model.float()



    # Optimizer
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
    

    # Scheduler


    sched_args = {}
    if args.sched_args is not None:
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

    # Initiate 
    losses = []
    boundary_losses = []
    terminal_losses = []

    losses_val = []
    boundary_losses_val = []
    terminal_losses_val = []

    best_loss_val = None
    best_epoch_val = None
    
    start_epoch = 0

    # Load checkpoint 
    if config['checkpoint_path'] is not None:
            start_epoch, additional_info = utils.load_checkpoint(config['checkpoint_path'], model, optimizer, scheduler)
            
            if "losses" in additional_info:
                losses = additional_info['losses']
            if "losses_val" in additional_info:
                losses_val = additional_info['losses_val']
            if "time_exec" in additional_info:
                time_exec = additional_info['time_exec']

            logger.info(f"Checkpoint loaded from {config['checkpoint_path']} at epoch {start_epoch}")
            # In case we load the checkpoint just to make the plots
            epoch = start_epoch
    else:
            logger.info("No checkpoint loaded, starting from scratch")

    logger.info(f'Number of training points per iteration = {n_samples*n_periods}, Number of Layers = {n_hidden_layers}, Number of Neurons per Layer = {n_neuron}')
    logger.info(f'Batch size = {batch_size}, Number of batches = {n_batches}, Learning rate = {lr}')
    logger.info(f'Optimizer = {args.opt}')
    logger.info(f'Scheduler = {args.sched}, Scheduler arguments = {sched_args}')
    logger.info(f'Seed = {args.seed}')
    logger.info(f'Independent resampling of initial states = {ind_resample_initial_state}')
    logger.info(f'Independent resampling of brownians = {ind_resample_brownian}')
    logger.info(f'Using architecture {args.arch} for the value network')
    if 'residual' in args.arch:
        logger.info(f'Using normalization type {args.norm_type} for the residual mlp')
    logger.info(f'Scheduler for lambda_w: increasing/decreasing linearly from {lambda_w_start} to {lambda_w_end} in the first {args.e} epochs, remaining constant for {args.d} epochs')

    # Training Loop and Validation
    start = time.time()
    for epoch in range(start_epoch, n_epoch):
        # clear settings
        model.train()

        epoch_boundary_loss = 0
        epoch_terminal_loss = 0

        logger.info(f"Epoch {epoch+1}/{n_epoch}, Lambda w: {lambda_w}, lr scheduler: {scheduler.get_last_lr()[0]}")

        # Shuffle the batch seeds for each epoch
        idx = torch.randperm(batch_seeds.size(0))
        batch_seeds = batch_seeds[idx]  # Shuffle the seeds

        # compute losses in batches
        for j in range(n_batches):

            if ind_resample_initial_state:
                # Resample initial states with a random seed
                temp_seed = torch.seed()
                logger.debug(f"Changing seed to {temp_seed} for independent resampling of initial states")
            else:
                # Seed with batch seed
                torch.manual_seed(batch_seeds[j].item())
            
            initial_states = pm.simulate_initial_state(batch_size, n_assets, dist_params, scale=True).to(device) # dim = (batch_size, 2*n_assets + 3)
            
            if ind_resample_brownian :
                # Resample brownians if needed with a random seed
                temp_seed = torch.seed()
                logger.debug(f"Changing seed to {temp_seed} for independent resampling of brownians")
                
            else:
                # Seed with batch seed
                torch.manual_seed(batch_seeds[j].item()) # set the seed for each batch

            brownians = pm.simulate_brownians(batch_size, dim_sto, n_periods, dt, corr).to(device)
            

            optimizer.zero_grad()

            # Shuffle the data for each batch
            idx = torch.randperm(initial_states.size(0))
            initial_states = initial_states[idx]
            brownians = brownians[idx]    


            # Generate data for PINN training, split into interior and terminal
            boundary_data, terminal_data, controls = pm.generate_data_PINN(initial_states, brownians, u_NN, T) 

            # compute loss - note that the forward pass is already included in the loss functions
            boundary_loss = torch.tensor(0.0, device=device)  # initiate boundary loss
            
            if lambda_w > 0:
                boundary_loss = LossFunction_boundary(model, boundary_data, controls, 
                                                    drift_func = drift_func, vol_func = vol_func)  # average loss over (n_periods * samples_per_batch) data points
                
            terminal_loss = LossFunction_terminal(model, terminal_data, F = utility_func)         # average loss over (samples_per_batch) data points
            loss = terminal_loss + boundary_loss

            # culmulative losses
            epoch_boundary_loss += boundary_loss.detach().cpu().item() 
            epoch_terminal_loss += terminal_loss.detach().cpu().item() 

            # back propagate
            loss.backward()
            optimizer.step()

        if (epoch+1) % args.d == 0:
            # increase/decrease linearly from lambda_w_start to lambda_end in the first e epochs, remaining constant for args.d epochs
            lambda_w  = min(max(lambda_w_min, (lambda_w_end - lambda_w_start)/args.e * (epoch+1) + lambda_w_start), lambda_w_max) 

        # save loss values
        epoch_boundary_loss /= n_batches    # average loss over n_batches * (n_periods * samples_per_batch) = n_periods * n_samples    - ok
        epoch_terminal_loss /= n_batches                  # average loss over n_batches * samples_per_batch = n_samples                - ok
        epoch_loss = epoch_boundary_loss + epoch_terminal_loss     

        losses.append(epoch_loss)
        boundary_losses.append(epoch_boundary_loss)
        terminal_losses.append(epoch_terminal_loss)

        # adjust learning rate
        scheduler.step()

        # evaluation (within the epoch)
        model.eval()
        with torch.no_grad():
            epoch_loss_val, epoch_boundary_loss_val, epoch_terminal_loss_val = 0, 0, 0   # initiation

            for j_val in range(n_batches):
                torch.manual_seed(batch_seeds_val[j_val].item())

                # data for evaluation
                initial_states_val = pm.simulate_initial_state(batch_size, n_assets, dist_params, scale=True).to(device) 
                brownians_val = pm.simulate_brownians(batch_size, dim_sto, n_periods, dt, corr).to(device)
                boundary_data_val, terminal_data_val, controls_val = pm.generate_data_PINN(initial_states_val, brownians_val, u_NN, T)

                epoch_boundary_loss_val += LossFunction_boundary(model, boundary_data_val, controls_val, 
                                                                                            drift_func=drift_func, vol_func=vol_func).detach().cpu().item() 
                epoch_terminal_loss_val += LossFunction_terminal(model, terminal_data_val, F = utility_func).detach().cpu().item()

            epoch_boundary_loss_val /= n_batches
            epoch_terminal_loss_val /= n_batches
            epoch_loss_val = epoch_boundary_loss_val + epoch_terminal_loss_val

            boundary_losses_val.append(epoch_boundary_loss_val)
            terminal_losses_val.append(epoch_terminal_loss_val)
            losses_val.append(epoch_loss_val)

        # track progress
        logger.info(f"Epoch [{epoch + 1}/{n_epoch}]: Training Loss = {epoch_loss: 0.9f} - Validation Loss = {epoch_loss_val: 0.9f}")
        logger.info(f" - Train : Int: {epoch_boundary_loss: 0.9f}, Term: {epoch_terminal_loss: 0.9f} - Val : Int: {epoch_boundary_loss_val: 0.9f}, Term: {epoch_terminal_loss_val: 0.9f}")

        # model selection based on progress
        if best_loss_val is None or best_loss_val > epoch_loss_val:
            best_loss_val = epoch_loss_val
            utils.save_checkpoint(model, optimizer = optimizer, scheduler_step = scheduler,
                            epoch = epoch, basedir=checkpoint_dir, suffix = 'best',
                            additional_info = {"losses": losses, "losses_val": losses_val,
                                                "boundary_losses": boundary_losses, "terminal_losses": terminal_losses})
            logger.info(f"Found best model in Epoch {epoch+1}, saving in {checkpoint_dir}")
            best_epoch_val = epoch + 1
        else:
            logger.info(f"Best model still in Epoch {best_epoch_val}")


    end = time.time()
    time_exec = end - start
    logger.info(f"Execution time : {(time_exec)/60} minutes")    





            


            


