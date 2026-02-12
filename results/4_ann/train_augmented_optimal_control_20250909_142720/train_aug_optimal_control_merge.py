'''
This is to train the augmented optimal control which will estimate (u,a) where u is the control for 
state variable X and a is the stochastic increment of the Martingale representation M of the constraint limit p.

PLEASE READ 4_Training_ValueFunction_global.md, section 1 for detailed explanation
'''
# ---- Packages ------- # 
import os
    # pytorch
import torch
import torch.nn as nn
import torch.optim as optim

    # to import configs 
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    # to import customized functions
import portfolio_model as pm
import neural_networks as custom_nn
import utils 

    # to track time
import time
import shutil

# ---- Loss Func ------ #
def LossFunction_onlyF(full_states, F):
    '''
    Formula : N := n_periods
        Given a sample of full history of states X and stopping times tau, the loss is computed conditionally on tau. 
        For one trajectory (X_0, ... X_N) and its corresponding tau, loss = V_bound(tau, X_at_tau)
    Note: tau is in the form of timestamp, taking value in {0, T/N, 2T/N, ..., T}

    Inputs : 
        states : a sample of full history of state variable (X_0), ... X_N) - dim = (n_samples, dim_state, n_periods)    
        taus : a sample of stopping time corresponding to the states       - dim = (n_samples, )
        V_bound : value function at the boundary, taking a sample of (t,x) as its input
    Output : 
        Average Loss across the sample of the loss defined above
    '''
    
    losses = -F(full_states[:, :, -1])   # dim = (n_samples, dim_state)   
    avg_loss = torch.mean(losses)       # dim = (1, )
    return avg_loss

def stats_taus(taus, n_periods, T):
    taus_ind = torch.round(taus * n_periods / T).long().cpu()  # convert tau to index in the range [0, n_periods]
    min_tau = taus_ind.min().item()
    max_tau = taus_ind.max().item()
    mean_tau = taus_ind.float().mean().item()
    taus_quantiles = torch.quantile(taus_ind.float(), torch.tensor([0.25, 0.5, 0.75]))
    return min_tau, max_tau, mean_tau, taus_quantiles.tolist()

# function to generate p based on initial states x_0 and boundary value function w
def generate_p(x0, w, p_range=0.5):
    x0_extended = torch.cat((torch.zeros(x0.size(0), 1).to(x0.device), x0), 
                            dim=1)                                  # dim = (n_samples, dim_state + 1)
    p_min = w(x0_extended).squeeze()                                # dim = (n_samples, )
    return p_range/10 + p_min + torch.rand(x0.size(0)).to(x0.device) * p_range   # dim = (n_samples, )


if __name__ == '__main__':

    # Small alias for this function 
    J = os.path.join

    # ---- Parser --------- #
    # Create argument parser to read config file
    parser = ArgumentParser(description='Train: Augmented Optimal Control Neural Network', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-a', '--config_train', type=str, required=True, help='path to the configuration file with training parameters for augmented optimal control neural network')
    parser.add_argument('-w', '--w_config', type=str, required=True, help='path to the results folder of the domain boundary value network')
    parser.add_argument('-u', '--u_config', type=str, required=True, help='path to the results folder of the optimal control network (u_nn)')
    parser.add_argument('-o', '--results_dir', type=str, default='./results/debug', help='directory to store the results')
    parser.add_argument('-s', '--seed', type=int, default=21, help='seed for the random number generator')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='log level for the logger, e.g., INFO, DEBUG, WARN', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
    parser.add_argument('--save_every', type=int, default=100, help='save the model every n epochs')

    # Load portfolio model parameters : --config_portfolio and --config_simulator
    pm.parser_portfolio(parser)
    args = parser.parse_args()
    
    # Device and seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a base directory to store results, checkpoints and figures
    results_dir = utils.create_dir(basedir=args.results_dir, dirname = "train_augmented_optimal_control", suffix = None)
    logger = utils.setup_logging(results_dir, log_level = args.log_level, fname='augmented_optimal_control.log')

    # Directory and Storage of results : create a base directory to store results, checkpoints and figures
    checkpoint_dir = utils.create_dir(basedir=results_dir, dirname = "checkpoints", suffix = "")

    logger.info(f"Device: {device}")

    # Import general portfolio
        # portfolio parameters
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

    utility_function = lambda x: pm.terminal_utility_exponential_2(x, alpha)

    # Load w network
        # load the configuration for the value network, create the model and load the weights
    w_dir = args.w_config
    with open(J(w_dir, "config_train_value.json"), 'r') as f:
        config_wnn = json.load(f)

    shutil.copy(J(w_dir, "config_train_value.json"), J(results_dir,'config_w.json'))

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
    
        # load the last checkpoint from the checkpoints directory
    dir_list = [J(w_dir, 'checkpoints', f) for f in os.listdir(J(w_dir, 'checkpoints'))]
    wnn_checkpoint_path = max(dir_list, key=os.path.getmtime)

    w_NN.to(device=device)
    w_NN.load_state_dict(torch.load(wnn_checkpoint_path, map_location=device, weights_only=False)['model_state_dict'])
    logger.info(f"Value network w_nn loaded from {wnn_checkpoint_path}")
    w_NN.eval()

    # Load V_bound network
        # load the configuration for the value network, create the model and load the weights
    u_dir = args.u_config
    with open(J(u_dir, "config_train.json"), 'r') as f:
        config_unn = json.load(f)

    shutil.copy(J(u_dir, "config_train.json"), J(results_dir,'config_train_u.json'))

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
    logger.info(f"Optimal control network u_nn loaded from {unn_checkpoint_path}")

    u_NN.to(device=device)
    u_NN.load_state_dict(torch.load(unn_checkpoint_path, map_location=device)['model_state_dict'])
    u_NN.eval()

    # Load training parameters for the augmented control network
        # read training config file
    config = json.load(open(args.config_train, 'r'))

        # hyper parameters and training info
    n_hidden_layers = config["n_hidden_layers"]  # number of hidden layers in each sub-network used to estimate the optimal control
    n_neuron = config["n_neuron"]      # number of neuron per hidden layer
    n_samples = config["n_samples"]      # total number of samples for training
    batch_size = config["batch_size"]      # batch size for each iteration (epoch)
    n_epoch = config["n_epoch"]        # number of training iterations

        # resampling indicator
    ind_resample_brownian = config["ind_resample_brownian"]

        # optimizer and scheduler
    lr = config["lr"]                  # learning rate
    milestones = config["milestones"]  # milestones for the learning rate step scheduler
    gamma_step = config["gamma_step"]  # decay factor for the learning rate step scheduler
    gamma_expo = config["gamma_expo"]  # decay factor the the learning rate exponential scheduler

    a_lower = config['a_lower']        # lower limit for sto. increment a
    a_upper = config['a_upper']        # upper limit for sto. increment a

    p_range = config["p_range"]        # range for the martingale process

    ind_resample_initial_state = config.get("ind_resample_initial_state", False)
    ind_resample_brownian = config.get("ind_resample_brownian", False)
    ind_resample_p = config.get("ind_resample_p", False)


    # Save the config files in case needed for replication
    json.dump(config_unn, open(f'{results_dir}/config_u.json', 'w'), indent=4)
    json.dump(config_wnn, open(f'{results_dir}/config_w.json', 'w'), indent=4)
    json.dump(json.load(open(args.config_portfolio, 'r')), open(f'{results_dir}/config_portfolio.json', 'w'), indent=4)
    json.dump(json.load(open(args.config_simulator, 'r')), open(f'{results_dir}/config_simulator.json', 'w'), indent=4)
    json.dump(config, open(f'{results_dir}/config_train_a.json', 'w'), indent=4)


    # Copy the script file to the results directory

    script_name = os.path.basename(__file__)
    shutil.copyfile(os.path.abspath(__file__), os.path.join(results_dir, script_name))

    # Save a json file with all the arguments in the parser
    args_dict = vars(args)
    args_dict['p_range'] = p_range
    args_dict['a_limit'] = a_lower
    args_dict['lr'] = lr 

    args_dict['config_unn'] = os.path.abspath(args.u_config)
    args_dict['config_wnn'] = os.path.abspath(args.w_config)
    args_dict['config_train'] = os.path.abspath(args.config_train)
    args_dict['results_dir'] = os.path.abspath(args.results_dir)
    json.dump(args_dict, open(f'{results_dir}/args.json', 'w'), indent=4)

    ###############################################
    # ------- Training -------------------------- #

    # Pre-training info
    logger.info(f"p_range = {p_range}, lr = {lr}") 

    # Create Model
    model = custom_nn.AugGlobalNetworks(dim_input = dim_state + 1,
                            dim_output = dim_control + dim_sto,
                            dim_hidden = n_neuron,
                            n_hidden = n_hidden_layers,
                            n_subnetworks = n_periods,
                            aug_update_func=gn_update_func, 
                            u_lower_bounds = u_lower, u_upper_bounds= u_upper,
                            a_lower = a_lower, a_upper = a_upper
                            )
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler_step = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma_step)
    scheduler_expo = optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma_expo)

    # Starter
    start_epoch = 0

    # TODO : check this - loss or losses ?
    var_to_plot = {
        "loss": [],
        "n_paths_full": [],
        "percentage_paths_full": []
    }


    # Load checkpoint if needed
    if config['checkpoint_path'] is not None:
        start_epoch, additional_info = utils.load_checkpoint(config['checkpoint_path'], model, optimizer, scheduler_step, scheduler_expo)

        if "losses" in additional_info:
            losses = additional_info['losses']
        if "n_paths_full" in additional_info:
            n_paths_full = additional_info['n_paths_full']
        if "percentage_paths_full" in additional_info:
            percentage_paths_full = additional_info['percentage_paths_full']

        logger.info(f"Checkpoint loaded from {config['checkpoint_path']} at epoch {start_epoch}")
        # In case we load the checkpoint just to make the plots
        epoch = start_epoch
    else:
        logger.info("No checkpoint loaded, starting from scratch")

    # Set batch size (in case running the entire sample is too heavy)
    n_batches = n_samples // batch_size
    samples_per_batch = n_samples // n_batches
    
    torch.manual_seed(args.seed)                    # Set the seed to obtain the same batch seeds depending on input seed
    batch_seeds = torch.multinomial(torch.ones(10000), n_batches, replacement=False).to(torch.int16)

    # Info before training
    logger.info('Loss Function version : F for all data points, merge a and u control networks')
    logger.info(f'N_samples = {n_samples}, Batch size = {samples_per_batch}, Number of Assets = {n_assets}, Number of Periods/Subnetworks = {n_periods}')
    logger.info(f'Number of Layers per Subnetworks = {n_hidden_layers}, Number of Neurons per Layer = {n_neuron}')
    logger.info(f'Initial learning rate = {lr}, Milestones = {milestones}, Gamma Step = {gamma_step}, Gamma Exponential = {gamma_expo}')

    # Training loop
    start = time.time()

    best_loss_val = None  

    for epoch in range(start_epoch, n_epoch):
        model.train()
        
        epoch_loss = 0
        epoch_n_paths_full = 0

        # Shuffle the batch seeds for each epoch
        idx = torch.randperm(batch_seeds.size(0))
        batch_seeds = batch_seeds[idx]  # Shuffle the seeds

        for j in range(n_batches):

            logger.debug(f"Batch {j+1}/{n_batches} starting")

            optimizer.zero_grad()

            if ind_resample_initial_state:
                # Resample initial states with a random seed
                temp_seed = torch.seed()
                logger.debug(f"Changing seed to {temp_seed} for independent resampling of initial states")
            else:
                # Seed with batch seed
                torch.manual_seed(batch_seeds[j].item())

            initial_x = pm.simulate_initial_state(samples_per_batch, n_assets, dist_params, scale=True).to(device)  # dim = (samples_per_batch, dim_state)


            if ind_resample_brownian :
                # Resample brownians if needed with a random seed
                temp_seed = torch.seed()
                logger.debug(f"Changing seed to {temp_seed} for independent resampling of brownians")
                
            else:
                # Seed with batch seed
                torch.manual_seed(batch_seeds[j].item()) # set the seed for each batch

            brownians = pm.simulate_brownians(samples_per_batch, dim_sto, n_periods, dt, corr).to(device)           # dim = (samples_per_batch, dim_sto, n_periods)


            if ind_resample_p :
                # Resample p if needed with a random seed
                temp_seed = torch.seed()
                logger.debug(f"Changing seed to {temp_seed} for independent resampling of p")
                
            else:
                # Seed with batch seed
                torch.manual_seed(batch_seeds[j].item()) # set the seed for each batch
        
            initial_p = generate_p(initial_x, w_NN, p_range=p_range).to(device)                                     # dim = (samples_per_batch, )

            # get full history and controls from u_NN
                # IN: (initial_states, constraint_limits, brownians, w)
                    # OUT: (states, martingales, controls, sto_increments, taus)
            full_states, martingales, controls, sto_increments, taus = model.forward_merge(initial_x, initial_p, brownians, w_NN, u_NN, u_update_func)             
            
            # compute losses - note that the forward pass is already included in the loss functions 
            batch_loss = LossFunction_onlyF(full_states, utility_function)

            # back propagate
            batch_loss.backward()

            # for logging and reporting
            epoch_loss += batch_loss.item()

        optimizer.step()
            
        # save values for plotting
        epoch_loss /= n_batches
        
        # adjust learning rate    
        scheduler_expo.step()
        scheduler_step.step()

        # progress tracking
        logger.info(f"Epoch [{epoch + 1}/{n_epoch}], Loss = {epoch_loss}, n_paths_full: {epoch_n_paths_full}, Percentage paths full: {epoch_n_paths_full*100/n_samples}")

        min_tau, max_tau, mean_tau, taus_quantiles = stats_taus(taus, n_periods, T)
        logger.info(f"Epoch [{epoch + 1}/{n_epoch}], min_tau: {min_tau}, max_tau: {max_tau}, mean_tau: {mean_tau}, taus_quantiles: {taus_quantiles}")


        # Always keep a checkpoint with the best model. Here, we use the training loss as the metric
        if best_loss_val is None or best_loss_val > epoch_loss:
            best_loss_val = epoch_loss
            utils.save_checkpoint(model, optimizer = optimizer, 
                            scheduler_step = scheduler_step, scheduler_expo = scheduler_expo,
                            epoch = epoch, basedir=checkpoint_dir, suffix = 'best')
            logger.info(f"Epoch [{epoch + 1}/{n_epoch}]: Found best model, saving")
            
    # # Saving the final model 
    # utils.save_checkpoint(model, optimizer = optimizer, 
    #                       scheduler_step = scheduler_step, scheduler_expo = scheduler_expo,
    #                       epoch = epoch, basedir=checkpoint_dir, suffix = f'final',
    #                       additional_info = dict(**var_to_plot))

    # Report on exec time
    end = time.time()
    time_exec = end - start
    logger.info(f"Training finished, total time: {time_exec/60} minutes")
