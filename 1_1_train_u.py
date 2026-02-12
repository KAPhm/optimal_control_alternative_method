##########################################################
# ----------------- Packages --------------------------- #
import os
import shutil
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import time

import torch.nn as nn
import torch.optim as optim

import importlib
    # to import customized functions
pm = importlib.import_module('0_portfolio_model', package=None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

############################################################
# ------- Parameters for training ------------------------ # 


# Parsers
    # Create argument parser to read the configuration file
parser = ArgumentParser(description='Train: Optimal Control with Boundary Value Problem: Optimal control network', 
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-u', '--config_u', type=str, required=True, 
                    help='path to the configuration file with all the parameters for the training of the optimal control network (u)')
parser.add_argument('-o', '--results_dir', type=str, default='./results/debug', 
                    help='directory to store the results')
parser.add_argument('-s', '--seed', type=int, default=0, 
                    help='seed for the random number generator')
parser.add_argument('-l', '--log_level', type=str, default='INFO', help='log level for the logger, e.g., INFO, DEBUG, WARN', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
parser.add_argument('-c', '--save_every', type=int, default=100, 
                    help='save the model every n epochs')
parser.add_argument('--no-save', action='store_true', 
                    help='when testing, if this option is passed, the model will not be saved')

parser.add_argument('--new-lr', action='store_true', 
                    help='if passed and there is a checkpoint, the learning rate scheduler will be set to the new scheduler built from the config file, thus ignoring the learning rate scheduler in the checkpoint')


    # Add the portfolio model parameters : --config_portofolio and --config_simulator
pm.parser_portfolio(parser)
args = parser.parse_args()



# Set device for torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Portfolio parameters

    # general parameters
n_assets, mu_S, sigma_S, coupon_rate, book_value, param_r, corr, param_L, u_lower, u_upper, n_periods, T, dt, K_below, K_above, scale_ind, dist_params = pm.helper_read_config_files(args)
    
# dimension parameters
dim_state = n_assets*2 + 3      # number of variables in the state process
dim_control = n_assets + 1      # number of variables in the control process
dim_sto = n_assets + 1          # number of variables in the stochastic factor (brownians)

    # Create an update function for the control global network 

# Send tensors to device
mu_S = mu_S.to(device)
sigma_S = sigma_S.to(device)
coupon_rate = coupon_rate.to(device)
book_value = book_value.to(device)
param_r = param_r.to(device)
param_L = param_L.to(device)
corr = corr.to(device)
u_lower = u_lower.to(device)
u_upper = u_upper.to(device)

gn_update_func = lambda x, u, dW: pm.update(x=x, u=u, dt=dt, dW = dW, d=n_assets,
                                               mu_S=mu_S, sigma_S=sigma_S, param_r=param_r, 
                                               param_L= param_L, coupon_rate = coupon_rate, 
                                               book_value = book_value)


# Train parameters
    # Read the configuration file for the training
config = json.load(open(args.config_u, 'r'))

    # Hyper parameters and training info
n_hidden_layers = config["n_hidden_layers"]  # number of hidden layers in each sub-network used to estimate the optimal control
n_neuron = config["n_neuron"]      # number of neuron per hidden layer
n_samples = config["n_samples"]      # batch size for each iteration (epoch)
n_epoch = config["n_epoch"]        # number of training iterations

batch_size = config["batch_size"]               # batch size
    
n_batches = n_samples // batch_size 

torch.manual_seed(args.seed)       # Set the seed to obtain the same batch seeds depending on input seed
batch_seeds = torch.multinomial(torch.ones(10000), n_batches, replacement=False).to(torch.int16)

    # Resampling indicator
ind_resample_initial_state = config["ind_resample_initial_state"]
ind_resample_brownian = config["ind_resample_brownian"]

    # Loss function parameters
lambda_coeff = config["k"]         # coefficient of penalty for state constraints
margin = config["margin"]          # margin for intermediate losses (playing the role of zero)

    # Optimizer and scheduler
lr = config["lr"]                  # learning rate
milestones = config["milestones"]  # milestones for the learning rate step scheduler
gamma_step = config["gamma_step"]  # decay factor for the learning rate step scheduler
gamma_expo = config["gamma_expo"]  # decay factor the the learning rate exponential scheduler


###############################################
# ------- Training -------------------------- #

# Directory and Storage of results
    # Create a base directory to store results, checkpoints and figures
results_dir = utils.create_dir(basedir=args.results_dir, dirname = "train_u", suffix = None)
checkpoint_dir = utils.create_dir(basedir=results_dir, dirname = "checkpoints", suffix = "")

    # Save configuration files for replication
json.dump(config, open(f'{results_dir}/config_u.json', 'w'), indent=4)
json.dump(json.load(open(args.config_portfolio, 'r')), open(f'{results_dir}/config_portfolio.json', 'w'), indent=4)
json.dump(json.load(open(args.config_simulator, 'r')), open(f'{results_dir}/config_simulator.json', 'w'), indent=4)


# Copy the script file to the results directory

script_name = os.path.basename(__file__)
shutil.copyfile(os.path.abspath(__file__), os.path.join(results_dir, script_name))

# Save a json file with all the arguments in the parser
args_dict = vars(args)
args_dict['config_u'] = os.path.abspath(args.config_u)
args_dict['results_dir'] = os.path.abspath(args.results_dir)
json.dump(args_dict, open(f'{results_dir}/args.json', 'w'), indent=4)

logger = utils.setup_logging(results_dir, log_level = args.log_level, fname='train_u.log')


logger.info(f"Device: {device}")

# Create Model
model = custom_nn.GlobalNetworks(dim_input = dim_state,
                           dim_output = dim_control,
                           dim_hidden = n_neuron,
                           n_hidden = n_hidden_layers,
                           n_subnetworks = n_periods,
                           update_func=gn_update_func, 
                           lower_bounds = u_lower,
                           upper_bounds = u_upper
                           )
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler_step = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma_step)
scheduler_expo = optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma_expo)

# Starter
start_epoch = 0
losses = []
terminal_losses =[]

losses_to_plot = {
    "bankruptcy_loss": [],
    "negative_liab_loss": [],
    "negative_cash_loss": [],
    "terminal_loss": [],
    "total_loss": []
}


# Load checkpoint if needed
if config['checkpoint_path'] is not None:

    if args.new_lr:
        # Dont load the scheduler from the checkpoint, but keep the one that has already been created
        logger.info("Ignoring optimizer and learning rate scheduler in the checkpoint, loading only the model")
        start_epoch, additional_info = utils.load_checkpoint(config['checkpoint_path'], model)
    else:
        start_epoch, additional_info = utils.load_checkpoint(config['checkpoint_path'], model, optimizer, scheduler_step, scheduler_expo)
    
    if "losses" in additional_info:
        losses = additional_info['losses']
    if "terminal_losses" in additional_info:
        terminal_losses = additional_info['terminal_losses']
    if "time_exec" in additional_info:
        time_exec = additional_info['time_exec']

    logger.info(f"Checkpoint loaded from {config['checkpoint_path']} at epoch {start_epoch}")
    # In case we load the checkpoint just to make the plots
    epoch = start_epoch
else:
    logger.info("No checkpoint loaded, starting from scratch")

    # Info before training
logger.info(f'N samples = {n_samples}, Batch size = {batch_size}, N Batches = {n_batches}')
logger.info(f'Number of Assets = {n_assets}, Number of Periods/Subnetworks = {n_periods}')
logger.info(f'Number of Layers per Subnetworks = {n_hidden_layers}, Number of Neurons per Layer = {n_neuron}')
logger.info(f'Initial learning rate = {lr}, Milestones = {milestones}, Gamma Step = {gamma_step}, Gamma Exponential = {gamma_expo}')
logger.info(f"schedulers: {scheduler_step.state_dict()}, {scheduler_expo.state_dict()}")
logger.info(f'Optimizer lr: {[d['lr'] for d in optimizer.state_dict()['param_groups']]}')
logger.info(f'Margin = {margin}, lambda = {lambda_coeff}')
logger.info(f'Seed = {args.seed}')
logger.info(f'Independent resampling of initial states = {ind_resample_initial_state}')
logger.info(f'Independent resampling of brownians = {ind_resample_brownian}')


# Training loop
start = time.time()

for epoch in range(start_epoch, n_epoch):

    model.train()

    epoch_bankruptcy_loss = 0.0
    epoch_negative_liab_loss = 0.0
    epoch_negative_cash_loss = 0.0
    epoch_terminal_loss = 0.0
    epoch_loss = 0.0

    # Data for training
    for j in range(n_batches):

        logger.debug(f"Batch {j+1}/{n_batches}, Seed: {batch_seeds[j].item()}")

        if ind_resample_initial_state:
            # Resample initial states with a random seed
            temp_seed = torch.seed()
            logger.debug(f"Changing seed to {temp_seed} for independent resampling of initial states")
        else:
            # Seed with batch seed
            torch.manual_seed(batch_seeds[j].item())

        initial_states = pm.simulate_initial_state(batch_size, n_assets, dist_params, scale=True) # dim = (batch_size, 2*n_assets + 3)


        if ind_resample_brownian :
            # Resample brownians if needed with a random seed
            temp_seed = torch.seed()
            logger.debug(f"Changing seed to {temp_seed} for independent resampling of brownians")
        else:
            # Seed with batch seed
            torch.manual_seed(batch_seeds[j].item()) # set the seed for each batch
        
        brownians = pm.simulate_brownians(batch_size, dim_sto, n_periods, dt, corr)
        
        optimizer.zero_grad()

        # Forward pass
        list_state, _ = model(initial_states.to(device), brownians.to(device)) # we don't need the control at this state
        
        # Loss
        # bankruptcy_loss = sum(pm.avg_penalty_bankruptcy(list_state[:,:,t], margin = margin) for t in range(n_periods))/n_periods
        # negative_liab_loss = sum(pm.avg_penalty_negative_liab(list_state[:,:,t], margin = margin) for t in range(n_periods))/n_periods
        # negative_cash_loss = sum(pm.avg_penalty_negative_cash(list_state[:,:,t], margin = margin) for t in range(n_periods))/n_periods
        bankruptcy_loss, negative_cash_loss, negative_liab_loss = pm.avg_multi_period_penalty(list_state, margin)
        terminal_loss = torch.mean(pm.terminal_capital_loss(list_state[:,:,-1], K_below, K_above))

        loss = lambda_coeff * (bankruptcy_loss + negative_liab_loss + negative_cash_loss) + terminal_loss

        # Backward propagation
        logger.debug("Before backward")
        loss.backward()
        logger.debug("Before optimizer.step()")
        optimizer.step()

        epoch_bankruptcy_loss += bankruptcy_loss.item() / n_batches
        epoch_negative_liab_loss += negative_liab_loss.item() / n_batches
        epoch_negative_cash_loss += negative_cash_loss.item() / n_batches
        epoch_terminal_loss += terminal_loss.item() / n_batches
        epoch_loss += loss.item() / n_batches

    

    losses_to_plot["bankruptcy_loss"].append(epoch_bankruptcy_loss)
    losses_to_plot["negative_liab_loss"].append(epoch_negative_liab_loss)
    losses_to_plot["negative_cash_loss"].append(epoch_negative_cash_loss)
    losses_to_plot["terminal_loss"].append(epoch_terminal_loss)
    losses_to_plot["total_loss"].append(epoch_loss)

    scheduler_expo.step()
    scheduler_step.step()

    logger.info(f"Epoch [{epoch + 1}/{n_epoch}], Loss: Total = {epoch_loss}, Term = {epoch_terminal_loss}")
    logger.info(f"      bankruptcy = {epoch_bankruptcy_loss*lambda_coeff}, negLiab = {epoch_negative_liab_loss*lambda_coeff}, negCash = {epoch_negative_cash_loss*lambda_coeff}")
    
    # Progress tracking
    if (epoch + 1) % args.save_every == 0 and not args.no_save:
        
        # Save model
        utils.save_checkpoint(model, optimizer = optimizer, 
                              scheduler_step = scheduler_step, scheduler_expo = scheduler_expo,
                              epoch = epoch, basedir=checkpoint_dir, suffix = f'epoch_{epoch+1}',
                              additional_info = dict(**losses_to_plot))

    end = time.time()
    time_exec = end - start

# - Save the model
utils.save_checkpoint(model, optimizer = optimizer, 
                        scheduler_step = scheduler_step, scheduler_expo = scheduler_expo,
                        epoch = epoch, basedir=checkpoint_dir, suffix = f'final',
                        additional_info = dict(**losses_to_plot))


