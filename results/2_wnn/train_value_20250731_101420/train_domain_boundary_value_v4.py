'''
This script tries to integrate everything from the previous versions
The idea is to have the choice between the two training methods:
1. Training with the MC approach
2. Training with the PINN approach

When using the PINN approach, the regularization parameter will be used on the interior loss instead of the terminal loss.
This allows to train w with terminal loss only, which is not possible on version 3

'''
# ---- Packages ----- #
import torch
import torch.nn as nn 
import torch.optim as optim

import time

import json
import os, shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import utils
import portfolio_model as pm
import neural_networks as custom_nn

# Define the PDE-guided loss function for interior
def LossFunction_MC(w_NN, initial_states, brownians, 
                        lambda_coeff, u_NN, T, K_below, K_above, margin):
      
    data, target = pm.generate_full_data_with_target(initial_states, brownians, u_NN, T, lambda_coeff = lambda_coeff, K_below = K_below, K_above = K_above, margin = margin)
    
    device = data.device       # device

    loss_fn = nn.MSELoss()

    # Forward pass
    predict = w_NN(data.to(device=device).float())

    # Loss computation
    return torch.tensor([0.0]).to(device), loss_fn(predict, target.to(device=device).float()).mean()



def LossFunction_PINN(w_NN, initial_states, brownians, G, 
                        drift_func, vol_func, penalty_func,
                        lambda_coeff, u_NN, T, compute_interior):
    

    interior_data, terminal_data, controls = pm.generate_data_PINN(initial_states, brownians, u_NN, T)
    device = terminal_data.device

    # Terminal

    terminal_predictions = w_NN(terminal_data.to(device))         # prediction from w_nn
    terminal_targets = G(terminal_data[:,1:])       # this loss funct does not need the time dimension


    loss_terminal =  nn.MSELoss()(terminal_predictions.squeeze(), terminal_targets.squeeze())

    loss_interior = torch.tensor([0.0]).to(device)  # initialize interior loss to zero

    if  compute_interior:
    # Interior

        M = interior_data.shape[0]          # number of data points = n_samples * n_periods
        X = interior_data[:, 1:]            # separate the space variables
        device = interior_data.device       # device

        # elements of computation 
            # gradients and hessians
        # dw_t, dw_x, dw_xx = utils.compute_grad_hessian(w_NN, interior_data).to(device)
        dw_t, dw_x, dw_xx = utils.compute_grad_hessian_2(w_NN, interior_data)
        dw_t.to(device)
        dw_x = dw_x.to(device)               
        dw_xx = dw_xx.to(device)             
        dw_x = dw_x.unsqueeze(2)  # needed for tensor multiplication, dim = (M, dim_state, 1)
            
            # drifts and vols
        drifts = drift_func(X, controls).unsqueeze(2).to(device)    # dim = (M, dim_state, 1) 
        drifts_T = torch.transpose(drifts, 1, 2).to(device)         # dim = (M, 1, dim_state)
            
        vols = vol_func(X).to(device)                               # dim = (M, dim_state, dim_sto)     
        variances = torch.bmm(vols, torch.transpose(vols, 1, 2))    # dim = (M, dim_state, dim_state) 
        diffusion = torch.bmm(variances, dw_xx)                     # dim = (M, dim_state, dim_state)           
            
            # penalties
        penalties = penalty_func(X).unsqueeze(1).to(device)                 # dim = (M, 1)

        # Dynkin operator
        dynkins = dw_t + torch.bmm(drifts_T, dw_x)                          # dim = (M, 1)
        dynkins += 0.5 * torch.vmap(torch.trace)(diffusion).unsqueeze(1)    # dim = (M, 1)
        dynkins = dynkins.to(device)

        errors = dynkins + lambda_coeff * penalties                         # dim = (M, 1)
        errors = errors.squeeze()                                           # dim = (M,)
        errors = errors ** 2                                                # dim = (M,)
        errors = errors.to(device)

        loss_interior =  torch.mean(errors)

    return loss_interior, loss_terminal





if __name__ == '__main__':
# ---- Parser ------- # 
    # Create argument parser to read config file
    parser = ArgumentParser(description='Train: Domain Boundary Value Network V4',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-u', '--config_unn', type=str, required=True, help='path to the results folder of the control network. This folder must contain a JSON file named `config_train.json` with the parameters used to train the control network, and a folder named checkpoints from which the weights of the optimal control network will be loaded. The loaded checkpoint will be the file that was last modified, obtained via `os.path.getmtime()`')
    parser.add_argument('-w', '--config_train', type=str, default='./config/base_train_value.json', help='path to the configuration file with all the parameters for the training of the value network (w)')
    parser.add_argument('-o', '--results_dir', type=str, default='./results', help='directory to store the results')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed for the random number generator')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='log level for the logger, e.g., INFO, DEBUG, WARN', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])

    parser.add_argument('--method', type=str, default='pinn', help='Training method: pinn uses regularization on the derivatives, mc uses sampling of the trajectories and only regression', choices=['pinn', 'mc'])

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

        # fixed the portfolio parameters in the update function
    gn_update_func = lambda x, u, dW: pm.update(x=x, u=u, dt=dt, dW = dW, d=n_assets,
                                                mu_S=mu_S, sigma_S=sigma_S, param_r=param_r, 
                                                param_L= param_L, coupon_rate = coupon_rate, 
                                                book_value = book_value)
    
    # Load the trained optimal control network u_nn
    unn_dir = args.config_unn                                                           # locate
    config_unn = json.load(open(os.path.join(unn_dir, 'config_train.json'), 'r'))       # retrieve configuration

        # parameters for loss function (shared parameters from the training of u_nn)
    lambda_coeff = config_unn['k']
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
    dir_list = [os.path.join(unn_dir, 'checkpoints', f) for f in os.listdir(os.path.join(unn_dir, 'checkpoints'))]
    unn_checkpoint_path = max(dir_list, key=os.path.getmtime)

    u_NN.to(device=device)
    u_NN.load_state_dict(torch.load(unn_checkpoint_path, map_location=device)['model_state_dict'])
    u_NN.eval()

    # Read config file for training w_nn
    config = json.load(open(args.config_train, 'r'))  # retrieve configuration for training

        # hyper-parameters for training
    n_hidden_layers = config["n_hidden_layers"]     # number of hidden layers 
    n_neuron = config["n_neuron"]                   # number of neuron per hidden layer
    n_samples = config["n_samples"]                 # number of data points in the sample of initial states X_0 = (x_0^j)^{j=1,...,n_samples}


    

    n_epoch =  config["n_epoch"]                    # number of training epochs
    batch_size = config["batch_size"]               # batch size
    
    n_batches = n_samples // batch_size 
    torch.manual_seed(args.seed)                    # Set the seed to obtain the same batch seeds depending on input seed
    batch_seeds = torch.multinomial(torch.ones(10000), n_batches, replacement=False).to(torch.int16)

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
    ind_resample_initial_state = config["ind_resample_initial_state"]
    ind_resample_brownian = config["ind_resample_brownian"]

# ---- Directory ---- #
    # Set a directory to store results
    results_dir = utils.create_dir(basedir=args.results_dir, dirname = "train_value", suffix = None)
    checkpoint_dir = utils.create_dir(basedir=results_dir, dirname = "checkpoints", suffix = "")
    logger = utils.setup_logging(results_dir, log_level = args.log_level, fname='train_value.log')

    # Copy the script file to the results directory

    script_name = os.path.basename(__file__)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(results_dir, script_name))


    # Pre-training info
    logger.info(f"Using device: {device}")
    logger.info(f"Optimal control network loaded from {unn_checkpoint_path}")
    logger.info(f"Number of batches = {n_batches}, samples per batch = {batch_size}")

    # Save the config files in case needed for replication
    json.dump(config_unn, open(f'{results_dir}/config_train_optimal_control.json', 'w'), indent=4)
    json.dump(json.load(open(args.config_portfolio, 'r')), open(f'{results_dir}/config_portfolio.json', 'w'), indent=4)
    json.dump(json.load(open(args.config_simulator, 'r')), open(f'{results_dir}/config_simulator.json', 'w'), indent=4)
    json.dump(config, open(f'{results_dir}/config_train_value.json', 'w'), indent=4)

    # Save a json file with all the arguments in the parser
    args_dict = vars(args)
    args_dict['config_unn'] = os.path.abspath(args.config_unn)
    args_dict['config_train'] = os.path.abspath(args.config_train)
    args_dict['results_dir'] = os.path.abspath(args.results_dir)
    json.dump(args_dict, open(f'{results_dir}/args.json', 'w'), indent=4)

# ---- Loss Funct --- #
    # Wrap the drift, the volatility, the penalty (g), and the terminal loss (G) functions
    drift_func = lambda x,u : pm.drift(x, u, n_assets, mu_S, param_r, param_L, coupon_rate, book_value)
    vol_func = lambda x : pm.vol(x, n_assets, sigma_S, param_r)
    penalty_func = lambda x: pm.total_intermediate_penalty(x, margin)
    final_loss_func = lambda x: pm.terminal_capital_loss(x, K_below, K_above)    
                        
# ---- Training ----- # 
    # Create a frame for w_NN with chosen hyper-parameters
    # model = custom_nn.BoundaryValueNetwork(dim_input = dim_state + 1,
    #                                 dim_hidden = n_neuron,
    #                                 n_hidden = n_hidden_layers,
    #                                 activation = nn.ReLU())


    if args.arch == 'mlp':
        model = custom_nn.BoundaryValueNetwork(dim_input = dim_state + 1,
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
    interior_losses = []
    terminal_losses = []

    losses_val = []
    interior_losses_val = []
    terminal_losses_val = []

    best_loss_val = None

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
    logger.info(f'Using method {args.method} for training the value network')
    logger.info(f'Using architecture {args.arch} for the value network')
    if 'residual' in args.arch:
        logger.info(f'Using normalization type {args.norm_type} for the residual mlp')
    logger.info(f'Scheduler for lambda_w: increasing/decreasing linearly from {lambda_w_start} to {lambda_w_end} in the first {args.e} epochs, remaining constant for {args.d} epochs')


    # Training Loop and Validation
    start = time.time()
    
    for epoch in range(start_epoch, n_epoch):
        
        # clear settings
        model.train()

        epoch_interior_loss = 0
        epoch_terminal_loss = 0

        logger.info(f"Epoch {epoch+1}/{n_epoch}, Lambda w: {lambda_w}, lr scheduler: {scheduler.get_last_lr()[0]}")

        # Shuffle the batch seeds for each epoch
        idx = torch.randperm(batch_seeds.size(0))
        batch_seeds = batch_seeds[idx]  # Shuffle the seeds

        for j in range(n_batches):
            logger.debug(f"Batch {j+1}/{n_batches}, Seed: {batch_seeds[j].item()}")

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


            if args.method == 'mc':
                interior_loss, terminal_loss = LossFunction_MC(model, initial_states, brownians, 
                                                lambda_coeff=lambda_coeff, u_NN=u_NN, T=T, 
                                                K_below=K_below, K_above=K_above, margin=margin)
            elif args.method == 'pinn':
                if lambda_w > 0:
                    compute_interior = True
                else:
                    compute_interior = False

                interior_loss, terminal_loss = LossFunction_PINN(model, initial_states, brownians, 
                                                G=final_loss_func, drift_func=drift_func, vol_func=vol_func, 
                                                penalty_func=penalty_func, lambda_coeff=lambda_coeff, u_NN=u_NN, T=T,
                                                compute_interior=compute_interior)
            else:
                raise ValueError(f"Unknown method {args.method}. Choose between 'mc' and 'pinn'.")

            
            
            loss = terminal_loss + lambda_w * interior_loss



            # back propagate
            loss.backward()
            optimizer.step()

            # For logging and reporting
            epoch_interior_loss += interior_loss.detach().item()
            epoch_terminal_loss += terminal_loss.detach().item()


        if (epoch+1) % args.d == 0:
            # increase/decrease linearly from lambda_w_start to lambda_end in the first e epochs, remaining constant for args.d epochs
            lambda_w  = min(max(lambda_w_min, (lambda_w_end - lambda_w_start)/args.e * (epoch+1) + lambda_w_start), lambda_w_max) 

            
        # save loss values
        epoch_interior_loss /= n_batches
        epoch_terminal_loss /= n_batches
        epoch_loss = epoch_interior_loss + epoch_terminal_loss
        losses.append(epoch_loss)
        interior_losses.append(epoch_interior_loss)
        terminal_losses.append(epoch_terminal_loss)
        
        # adjust learning rate    
        scheduler.step()

        # Evaluation
        model.eval()
        with torch.no_grad():

            
            epoch_interior_loss_val = 0
            epoch_terminal_loss_val = 0

            for j_val in range(n_batches):

                torch.manual_seed(batch_seeds[j_val].item()+1) # set the seed for each batch. The +1 is used to obtain different data for validation

                initial_states_val = pm.simulate_initial_state(batch_size, n_assets, dist_params, scale=True).to(device) # dim = (n_samples, 2*n_assets + 3)
                brownians_val = pm.simulate_brownians(batch_size, dim_sto, n_periods, dt, corr).to(device)               # dim = (n_samples, n_assets + 1, n_periods)
                
                
                # We always compute pinn loss for validation, as it is the one that is more linked to the error measure
                interior_loss_val, terminal_loss_val = LossFunction_PINN(model, initial_states_val, brownians_val, 
                                                G=final_loss_func, drift_func=drift_func, vol_func=vol_func, 
                                                penalty_func=penalty_func, lambda_coeff=lambda_w, u_NN=u_NN, T=T, compute_interior=True)
               
                
                epoch_terminal_loss_val += terminal_loss_val.item()
                epoch_interior_loss_val += interior_loss_val.item()
        
            epoch_terminal_loss_val /= n_batches
            epoch_interior_loss_val /= n_batches
            epoch_loss_val = epoch_terminal_loss_val + epoch_interior_loss_val

            terminal_losses_val.append(epoch_terminal_loss_val)
            interior_losses_val.append(epoch_interior_loss_val)
            losses_val.append(epoch_terminal_loss_val + epoch_interior_loss_val)

        # Track 
        logger.info(f"Epoch [{epoch + 1}/{n_epoch}], Training Loss: {epoch_loss: 0.9f}, Validation Loss: {epoch_loss_val: 0.9f}")
        logger.info(f" - Train : Int: {epoch_interior_loss: 0.9f}, Term: {epoch_terminal_loss: 0.9f} - Val : Int: {epoch_interior_loss_val: 0.9f}, Term: {epoch_terminal_loss_val: 0.9f}")
            
        
        # Always keep a checkpoint with the best model
        if best_loss_val is None or best_loss_val > epoch_loss_val:
            best_loss_val = epoch_loss_val
            utils.save_checkpoint(model, optimizer = optimizer, scheduler_step = scheduler,
                            epoch = epoch, basedir=checkpoint_dir, suffix = 'best',
                            additional_info = {"losses": losses, "losses_val": losses_val,
                                                "interior_losses": interior_losses, "terminal_losses": terminal_losses})
            logger.info(f"Found best model in Epoch {epoch+1}, saving in {checkpoint_dir}")


    end = time.time()
    time_exec = end - start
    logger.info(f"Execution time : {(time_exec)/60} minutes")    

# ---- Save model --- #
#    utils.save_checkpoint(model, optimizer = optimizer, scheduler_step = scheduler,
#                        epoch = epoch, basedir=checkpoint_dir, suffix = 'final',
#                        additional_info = {"losses": losses, "losses_val": losses_val,
#                                        "interior_losses": interior_losses, "terminal_losses": terminal_losses,
#                                        "time_exec": time_exec})
