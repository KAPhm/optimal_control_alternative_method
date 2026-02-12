'''
This scripts allow for creating 2 graphics of trajectories at the same time, one for u_nn and the other a_nn. 
'''

# ---- Packages ------------- # 
    # pytorch 
import torch
import torch.nn as nn

    # to import configs & export results
import json, os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    # to graph
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

    # to import customized functions 
import importlib
pm = importlib.import_module('0_portfolio_model', package=None)
custom_nn = importlib.import_module('0_neural_networks', package=None)
utils = importlib.import_module('0_utils', package=None)

# ---- Supporting functions - #
# Function to generate p (same function as in training a_nn script)
def generate_p(x_0, w, p_range = 1.0):
    '''
    Inputs : 
        x_0 : sample of initial states - dim = (n_samples, dim_state)
        w : trained neural network to estimate the domain boundary
        p_range : range of distribution of p above the minimum w(0, x_0)
    Out : a sample of p
    Note : we give a buffer of 10% * p_range to p 
    '''
    x_0_extended = torch.cat((torch.zeros(x_0.size(0), 1).to(x_0.device), x_0),
                             dim = 1)       # dim = (n_samples, dim_state + 1)
    p_min = w(x_0_extended).squeeze()       # dim = (n_samples, )
    return p_range/10 + + p_min + torch.rand(x_0.size(0)).to(x_0.device) * p_range   

# Function to translate time stamp to index
def get_index_from_time(t, n_periods):
    return torch.round(t * n_periods)


if __name__ == '__main__':
# ---- Parsers -------------- #
    J = os.path.join # alias for frequently used function

    # Config file 
    parser = ArgumentParser(description='Visualize : Trajectories of state processes by u_nn and a_nn', 
                            formatter_class=ArgumentDefaultsHelpFormatter)
        # trained networks 
    parser.add_argument('-a', '--config_a', type=str, required=True, help='path to the results folder of the augmented optimal control network a_nn')
    parser.add_argument('-u', '--config_u', type=str, required=True, help='path to the results folder of the optimal control network u_nn')
    parser.add_argument('-w', '--config_w', type=str, required=True, help='path to the results folder of the domain boundary value network w_nn')

        # parameters for results keeping and generator
    parser.add_argument('-o', '--results_dir', type=str, default='./results/debug', help='directory to store the results')
    parser.add_argument('-s', '--seed', type=int, default=75, help='seed for the random number generator')
    parser.add_argument('-l', '--log_level', type=str, default='INFO', help='log level for the logger, e.g., INFO, DEBUG, WARN', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
    # parser.add_argument('-k', '--scenario', type=int, default=0, help='index of the scenario to be plot. The sample of scenarios are given in the script of visual_u_v4c.py; default=0')
    parser.add_argument('-n', '--n_paths', type=int, default=1000, help='number of brownian trajectories to simulate; default = 1000')
    parser.add_argument('-i', '--descale_ind', type=int, default=1, help='indicator for whether the values of the state process are going to be descaled; must be either =1 (yes to descaling) or =0 (no descaling)')
    # Portfolio model parameters : -p and -z 
    pm.parser_portfolio(parser)
    args = parser.parse_args()

    # Base directory to store results, checkpoints, and figures
    results_dir = utils.create_dir(basedir=args.results_dir, dirname = "visuals_a", suffix = None)
    logger = utils.setup_logging(results_dir, log_level = args.log_level, fname='visual_control_a_multi_paths.log')

    # Device and seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # General parameters
        # portfolio model
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
    logger.info(f'Book value = {book_value}')

        # dimension parameters
    dim_state = n_assets*2 + 3      # number of variables in the state process
    dim_control = n_assets + 1      # number of variables in the control process
    dim_sto = n_assets + 1          # number of variables in the stochastic factor (brownians)

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

    drift_extra_func = lambda x, u : pm.drift_extra(x=x, u=u, d=n_assets, param_r=param_r, param_L=param_L, coupon_rate=coupon_rate, book_value=book_value)


    # Load a_nn
        # load config
    a_dir = args.config_a
    config_ann = json.load(open(J(a_dir, 'config_train_a.json'), 'r'))

    a_lower = config_ann['a_lower']
    a_upper = config_ann['a_upper']
    p_range = config_ann['p_range']

        # load the trained augmented optimal control network
    a_NN = custom_nn.AugGlobalNetwork_extra(dim_input = dim_state + 1, dim_output = dim_control + dim_sto, 
                                       dim_hidden = config_ann['n_neuron'], n_hidden = config_ann['n_hidden_layers'], 
                                       n_subnetworks = n_periods, 
                                       aug_update_func= gn_update_func, update_extra_func = drift_extra_func, 
                                       u_lower_bounds = u_lower, u_upper_bounds = u_upper, 
                                       a_lower = a_lower, a_upper = a_upper,
                                       T= T, a_sigmoid_scale = 1.
                                       )
        # load check point
    dir_list = [os.path.join(a_dir, 'checkpoints', f) for f in os.listdir(os.path.join(a_dir, 'checkpoints'))]
    ann_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f"Augmented control network a_NN loaded from {ann_checkpoint_path}")
    
    a_NN.to(device=device)
    a_NN.load_state_dict(torch.load(ann_checkpoint_path, map_location=device)['model_state_dict'])
    a_NN.eval()

    # Load w_nn
        # load config 
    w_dir = args.config_w
    config_wnn = json.load(open(J(w_dir, 'config_train_value.json'), 'r'))    # config
    arg_wnn = json.load(open(J(w_dir, 'args.json'), 'r'))         # architecture

        # architecture of w_nn
    w_arch = arg_wnn["arch"]
    w_norm = arg_wnn["norm_type"]

    if w_arch == 'mlp':
        w_NN = custom_nn.BoundaryValueNetwork(dim_input=dim_state+1, dim_hidden=config_wnn['n_neuron'], n_hidden=config_wnn['n_hidden_layers'], activation=nn.ReLU())
    elif w_arch == 'residual-sin':
        w_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_wnn['n_neuron'], num_layers=config_wnn['n_hidden_layers'],
                                            time_encoding='sinusoidal', norm_type=w_norm)
    elif w_arch == 'residual':
        w_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_wnn['n_neuron'], num_layers=config_wnn['n_hidden_layers'],
                                            time_encoding='none', norm_type=w_norm)
    elif w_arch == 'residual-learnable':
        w_NN = custom_nn.PortfolioValueNet(num_time_freqs=8, state_dim=dim_state, hidden_dim=config_wnn['n_neuron'], num_layers=config_wnn['n_hidden_layers'],
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

    # Load u_nn
        # load config
    u_dir = args.config_u
    config_unn = json.load(open(J(u_dir, "config_train.json"), 'r'))

    margin = config_unn['margin']
    g_func = lambda x: pm.total_intermediate_penalty(x, margin = margin)

        # create the model and load the weights
    u_NN = custom_nn.GlobalNetworks_extra(dim_input = dim_state, dim_output = dim_control, update_extra_func = drift_extra_func, 
                        dim_hidden = config_unn['n_neuron'], n_hidden = config_unn['n_hidden_layers'], 
                        n_subnetworks = n_periods, update_func = u_update_func, lower_bounds = u_lower, upper_bounds = u_upper)

        # load the last checkpoint from directory
    dir_list = [os.path.join(u_dir, 'checkpoints', f) for f in os.listdir(os.path.join(u_dir, 'checkpoints'))]
    unn_checkpoint_path = max(dir_list, key=os.path.getmtime)
    logger.info(f"Optimal control u_nn loaded from {unn_checkpoint_path}")

    u_NN.to(device=device)
    u_NN.float()
    u_NN.load_state_dict(torch.load(unn_checkpoint_path, map_location=device)['model_state_dict'])
    u_NN.eval()

# ---- Data simulation ------ #
    # Generate trajectories 
    n_paths = args.n_paths  # number of trajectories to simulate
    epsilon = 1e-10         # tolerance threshold (considered as zero)

        # initial states 
    #x_0 = pm.simulate_initial_state(1, n_assets, dist_params, scale=True).to(device)    
    x_0 = torch.tensor([0.106, 0.217, 0.045, 0.138, 1.64, 1.01, 0.371707])
    logger.info(f'Initial state x_0 = {x_0}')
    X_0 = x_0.repeat(n_paths, 1)                    # dim = (n_paths, dim_state)
    
    #X_0 = pm.simulate_initial_state(n_paths, n_assets, dist_params, scale=True).to(device)
        # brownians 
    dW = pm.simulate_brownians(n_paths, n_assets + 1, n_periods, dt, corr).to(device)       # dim = (n_paths, dim_sto, n_periods)
        # initial constraints (p)
    p_0 = generate_p(X_0, w_NN, p_range).to(device)

        # hedging trajectories by u_nn
    states_unn, controls_unn, lapses_unn, fin_prod_unn, wealths_unn = u_NN(X_0, dW)
    states_unn = states_unn.detach().cpu()          # dim = (n_paths, dim_state, n_periods + 1)
    controls_unn = controls_unn.detach().cpu()      # dim = (n_paths, dim_control, n_periods)
    lapses_unn = lapses_unn.detach().cpu()          # dim = (n_paths, n_periods)
    fin_prod_unn = fin_prod_unn.detach().cpu()      # dim = (n_paths, n_periods)
    wealths_unn = wealths_unn.detach().cpu()        # dim = (n_paths, n_periods + 1)
            
        # augmented trajectories by a_nn
    states, martingales, controls, sto_increments, taus, lapses, fin_prod, wealths, boundaries = a_NN.forward(initial_states = X_0, constraint_limits = p_0, 
                                                                brownians = dW, u_nn = u_NN, x_u_update_func = u_update_func, w_nn = w_NN)
    states = states.detach().cpu()                  # dim = (n_paths, dim_state, n_periods + 1)
    martingales = martingales.detach().cpu()        # dim = (n_paths, n_periods + 1)
    controls = controls.detach().cpu()              # dim = (n_paths, dim_control, n_periods)
    sto_increments = sto_increments.detach().cpu()  # dim = (n_paths, dim_sto, n_periods)
    taus = taus.detach().cpu()                      # dim = (n_paths, )
    lapses = lapses.detach().cpu()                  # dim = (n_paths, n_periods)
    fin_prod = fin_prod.detach().cpu()              # dim = (n_paths, n_periods) 
    wealths = wealths.detach().cpu()                # dim = (n_paths, n_periods)
    boundaries = boundaries.detach().cpu()          # dim = (n_paths, n_periods + 1)

    # Trajectories selection - we select 3 samples based on the hitting time tau : one reaching the boundary early, another late, and last not reaching at all
        # case 1 : tau = T 
    M_T = martingales[:, -1]                        # martingales at terminal
    w_T = w_NN(torch.cat([torch.full((n_paths, 1), T).to(device), states[:, :, -1].squeeze()],
                          dim = 1))                 # dim = (n_paths, )
    id_tau_at_T = torch.logical_and(wealths[:, -1] > wealths_unn[:, -1], taus==T)

    if id_tau_at_T.sum() > 0 :
        indices_at_T = id_tau_at_T.nonzero().int()
        shuffle_tau_T = torch.randperm(len(indices_at_T))[:1]
        id_selected = indices_at_T[shuffle_tau_T]           
        n_case_1 = id_tau_at_T.sum()
    else : 
        indices_at_T = None
        n_case_1 = 0
        id_selected = torch.tensor([])

        # case 2 : tau < T
    id_tau_before_T = taus < T
    n_case_2 = id_tau_before_T.sum()
    
    id_tau_before_T = torch.logical_and(taus < T/2, taus > T/3)
    indices_tau_early = id_tau_before_T.nonzero().int()
    shuffle_tau_early = torch.randperm(len(indices_tau_early))[:8]
    id_selected = torch.cat([id_selected, indices_tau_early[shuffle_tau_early]])

    id_tau_late = torch.logical_and(taus > T/2, taus < 5*T/6)
    indices_tau_late = id_tau_late.nonzero().int()
    shuffle_tau_late = torch.randperm(len(indices_tau_late))[:2]
    id_selected = torch.cat([id_selected, indices_tau_late[shuffle_tau_late]])

    id_selected = id_selected.squeeze().int()

        # trajectory selection
    states = torch.index_select(states, 0, id_selected)
    martingales = torch.index_select(martingales, 0, id_selected)
    controls = torch.index_select(controls, 0, id_selected)
    sto_increments = torch.index_select(sto_increments, 0, id_selected)
    taus = torch.index_select(taus, 0, id_selected)
    p_0 = torch.index_select(p_0, 0, id_selected)
    dW = torch.index_select(dW, 0, id_selected)
    lapses = torch.index_select(lapses, 0, id_selected)
    fin_prod = torch.index_select(fin_prod, 0, id_selected)
    wealths = torch.index_select(wealths, 0, id_selected)  
    boundaries = torch.index_select(boundaries, 0, id_selected) 

    states_unn = torch.index_select(states_unn, 0, id_selected)
    controls_unn = torch.index_select(controls_unn, 0, id_selected)
    lapses_unn = torch.index_select(lapses_unn, 0, id_selected)
    wealths_unn = torch.index_select(wealths_unn, 0, id_selected)
    fin_prod_unn = torch.index_select(fin_prod_unn, 0, id_selected)

    # Descaling, if needed
    if args.descale_ind == 1 : 
        descale_factor_X = [1e3, 1e3, 1, 1e6, 1e3, 1e3, 1e6]
        descale_factor_M = 1e6                  # M has the same scale as total asset/total liability
        descale_factor_u = [1e3, 1e3, 1]
        
        for i in range(dim_state):
            descale_factor_i = descale_factor_X[i]
            states_unn[:, i, :] = torch.mul(states_unn[:, i, :], descale_factor_i)
            states[:, i, :] = torch.mul(states[:, i, :], descale_factor_i)
                
        if scale_ind == True : # if book value has been scaled at the reading config step
            book_value = torch.mul(book_value, descale_factor_X[0])        

        for i in range(dim_control):
            descale_factor_i = descale_factor_u[i]
            controls_unn[:, i, :] = torch.mul(controls_unn[:,i,:], descale_factor_i)
            controls[:, i, :] = torch.mul(controls[:,i,:], descale_factor_i)
            u_lower[i] = u_lower[i] * descale_factor_i
            u_upper[i] = u_upper[i] * descale_factor_i
    
        wealths_unn = torch.mul(wealths_unn, descale_factor_X[-1])  # wealth is on the same scale as total asset/total liability
        fin_prod_unn = torch.mul(fin_prod_unn, descale_factor_X[-1])# financial production is on the same scale as total asset/total liability 

        wealths = torch.mul(wealths, descale_factor_X[-1])  # wealth is on the same scale as total asset/total liability
        fin_prod = torch.mul(fin_prod, descale_factor_X[-1])# financial production is on the same scale as total asset/total liability 

        # adjustment for variable in percentage 
        lapses_unn = torch.mul(lapses_unn, 100)                     # lapse rate
        lapses = torch.mul(lapses, 100)
        states_unn[:, 2, :] = torch.mul(states_unn[:, 2, :], 100)   # interest rate
        states[:, 2, :] = torch.mul(states[:, 2, :], 100)           
        controls_unn[:, 2, :] = torch.mul(controls_unn[:, 2, :],100)# profit sharing rate
        controls[:, 2, :] = torch.mul(controls[:, 2, :],100)
        u_lower[-1] = u_lower[-1]*100
        u_upper[-1] = u_upper[-1]*100

        # wealth decomposition
        values_asset_1_unn = torch.mul(states_unn[:, 0, :], states_unn[:, 4, :])
        values_asset_2_unn = torch.mul(states_unn[:, 1, :], states_unn[:, 5, :])
        values_cash_unn = states_unn[:, 3, :]
        values_assets_unn = values_asset_1_unn + values_asset_2_unn + values_cash_unn
        values_liability_unn = states_unn[:, -1, :]

        values_asset_1 = torch.mul(states[:, 0, :], states[:, 4, :])
        values_asset_2 = torch.mul(states[:, 1, :], states[:, 5, :])
        values_cash = states[:, 3, :]
        values_assets = values_asset_1 + values_asset_2 + values_cash
        values_liability = states[:, -1, :]

# ---- Graphing ------------- #
    # Common formating 
    colors = ['mediumblue','crimson','purple', 'green']         # asset 1, asset 2, liability, cash
    stack_colors = ['green', 'blue', 'red']                     # cash, asset 1, asset 2   

    x_axis_state = np.arange(0, n_periods + 1)                  # 0 to N
    x_axis_control = np.arange(0, n_periods)                    # 0 to N-1

    fs_axis = 'medium'
    fs_label = 'small'
    fs_title = 'large'

    # Titles and units
        # for u_nn graph
    plot_u_titles = [[r'Price $S^1$', r'Price $S^2$', r'Interest rate $r$'],
                    [r'Quantity $\phi^1$', r'Quantity $\phi^2$',  r'Liability $L$ and Cash $\beta$' ],
                    [r'Trade Intensity $\dot\phi^1$',  r'Trade Intensity $\dot\phi^2$', r'Profit Sharing rate $\pi$'],
                    [r'Financial Production $\dot \eta$', r'Lapse rate $\dot \gamma$', r'Wealth composition $\phi \cdot S + \beta - L$']]
    if args.descale_ind == 1 :
        plot_u_units = [['\N{euro sign}', '\N{euro sign}','%'],
                      [ 'units', 'units', '\N{euro sign}'],
                      ['units', 'units', '%'],
                      ['\N{euro sign}', '%', '\N{euro sign}']]
    else : # without descaling
        plot_u_units = [['1e3 \N{euro sign}', '1e3 \N{euro sign}', '%'],
                      ['1e3 units', '1e3 units', '1e6 \N{euro sign}'],
                      ['1e3 units', '1e3 units','%'],
                      ['1e6 \N{euro sign}', '%', '1e6 \N{euro sign}']]
        
    for k in range(len(id_selected)):
    # Data selection 
        M_to_plot = martingales[k, :].squeeze().detach().numpy()            # dim = (n_periods + 1, )
        w_to_plot = boundaries[k, :].squeeze().detach().numpy()             # dim = (n_periods + 1, )
        X_to_plot = states[k, :, :].squeeze().detach().numpy()              # dim = (dim_state, n_periods + 1)
        tau_to_plot = get_index_from_time(taus[k], n_periods) + 1 
        u_to_plot = controls[k, :].squeeze().detach().numpy()               # dim = (dim_control, n_periods)

    # Frame
        fig = plt.figure(figsize=(15,20))   # a_nn graph
        gs = gridspec.GridSpec(nrows=5, ncols=3, figure=fig, height_ratios=[1.5, 1, 1, 1, 1])
        ax = []
            
        ax.append(fig.add_subplot(gs[0, :]))    # first subplot for a_nn graph is different
        ax[0].set_title(r"Martingale $M$ and domain boundary $\varpi$", fontsize = fs_title)
        ax[0].set_xlabel('Timestep', fontsize = fs_axis, loc = 'center')
        ax[0].axvline(x=tau_to_plot, color='red', ls='solid', lw = 1.25, label=r'Hitting time $\tau$'+f'={int(tau_to_plot)}')
        if args.descale_ind == 1: 
            ax[0].set_ylabel('\N{euro sign}', fontsize=fs_axis)
        else : 
            ax[0].set_ylabel('1e6 \N{euro sign}', fontsize=fs_axis)
        ax[0].grid(visible = True, axis = 'both', ls = '--', lw = 0.5)

        fig_u, ax_u = plt.subplots(nrows = 4, ncols = 3, figsize = (14,15))

        for i in range(4):      # id for row
            for j in range(3):  # id for column
                    # frame for a_nn graph
                id_temp = 3*i + j + 1   # id for subplot in a_nn graph
                ax.append(fig.add_subplot(gs[i+1, j]))
                if id_temp < 12 :
                    ax[id_temp].set_title(plot_u_titles[i][j], fontsize = fs_title, pad = 10)
                    ax[id_temp].axvline(x=tau_to_plot, color='red', ls='solid', lw = 0.75)
                else : 
                    ax[id_temp].set_title('Wealth comparison', fontsize = fs_title, pad = 10)
                ax[id_temp].set_xlabel('Time step', fontsize = fs_axis, loc = 'right')
                ax[id_temp].set_ylabel(plot_u_units[i][j], fontsize = fs_axis, loc = 'top')
                ax[id_temp].grid(visible = True, axis = 'both', ls = '--', lw = 0.5)
                if j == 1 and i < 3:     # subplot for asset 2
                    ax[id_temp].sharey(ax[id_temp - 1]) # same y-axis as asset 1

                    # frame for u_nn graph
                ax_u[i][j].set_title(plot_u_titles[i][j], fontsize = fs_title, pad = 10)
                ax_u[i][j].set_xlabel('Time step', fontsize = fs_axis, loc = 'right')
                ax_u[i][j].set_ylabel(plot_u_units[i][j], fontsize = fs_axis, loc = 'top')
                ax_u[i][j].grid(visible = True, axis = 'both', ls = '--', lw = 0.5)
                if id_temp < 12 :
                    ax_u[i][j].sharey(ax[id_temp])          # same y-axis as its corresponding subplot in a_nn graph

    # Graphing

        l = 0   # counter for subplots in the a_nn graph
        i = 0   # counter for row

        # Martingale vs domain boundary - only in a_nn graph
        ax[l].plot(x_axis_state, M_to_plot , color='seagreen', ls='solid', 
                   label=r'Martingale $M$')                                 # Martingale representation M
        ax[l].plot(x_axis_state, w_to_plot, color='darkorange', ls='dotted', 
                   label=r'Domain boundary $\varpi$')                       # domain boundary w    
        ax[l].legend(fontsize = fs_label)

        # Prices - same for a_nn and u_nn graphs
        for j in range(2):
            # a_nn graph
            l += 1  # l = 1,2
            ax[l].plot(x_axis_state, X_to_plot[j,:], color=colors[j], ls='solid', lw=1.5, label='Price')
            ax[l].axhline(book_value[j], color=colors[j], ls='dotted', lw=1.0, label='Book value')
            ax[l].legend(fontsize=fs_label)

            # u_nn graph
            y_axis = states_unn[k, j, :].squeeze().detach().numpy()         # dim = (n_periods + 1, )
            ax_u[i][j].plot(x_axis_state, y_axis, color = colors[j], ls = 'solid', lw = 1.25, label = 'Price')
            ax_u[i][j].axhline(book_value[j], color = colors[j], ls = 'dotted', lw = 1, label = 'Book value')
            ax_u[i][j].legend(fontsize=fs_label)

        # Interest rate - same for a_nn and u_nn graphs
            # a_nn graph
        j += 1      # j = 3
        l += 1      # l = 3
        ax[l].plot(x_axis_state, X_to_plot[2,:], color = colors[-1], ls='solid', lw=1.5)
            # u_nn graph
        y_axis = states_unn[k, 2, :].squeeze().detach().numpy()             # dim = (n_periods + 1, )
        ax_u[i][2].plot(x_axis_state, y_axis, color = colors[-1], ls = 'solid', lw = 1.25)

        i = 1
        # Quantity 
        for j in range(2):  
            l += 1  # l = 4,5
            ax[l].plot(x_axis_state, X_to_plot[4+j, :], color=colors[j], ls='solid', lw=1.5)
            
            y_axis = states_unn[k, 4 + j, :].squeeze().detach().numpy()     # dim = (n_periods + 1, )
            ax_u[i][j].plot(x_axis_state, y_axis, color = colors[j], ls = 'solid', lw = 1.25)

        # Liability and Cash
        j += 1      # j = 2
        l += 1      # l = 6
        ax[l].plot(x_axis_state, X_to_plot[3,:], color = colors[-1], ls='solid', lw=1.5, label = r'Cash $\beta$')
        ax[l].plot(x_axis_state, X_to_plot[-1,:], color = colors[2], ls='solid', lw=1.5, label = r'Liability $L$')
        ax[l].axhline(y=0, color='red', ls='dotted',lw=0.75, alpha=0.7)
        ax[l].legend(fontsize=fs_label)
        
        y_axis_1 = states_unn[k, 3, :].squeeze().detach().numpy()    # cash # dim = (n_periods + 1, )
        y_axis_2 = states_unn[k, -1, :].squeeze().detach().numpy()   # liab # dim = (n_peridos + 1, )
        ax_u[i][j].plot(x_axis_state, y_axis_1, color = colors[-1], ls = 'solid', lw = 1.25, label = r'Cash $\beta$')
        ax_u[i][j].plot(x_axis_state, y_axis_2, color = colors[2], ls = 'solid', lw = 1.25, label = r'Liability $L$')
        ax_u[i][j].axhline(y=0,color='red',ls='dotted',lw=0.5, alpha=0.7)
        ax_u[i][j].legend(fontsize=fs_label)

        i = 2
        # Control variables (trade intensities and profit sharing rate)
        for j in range(3):
            l += 1  # l = 7,8,9
            ax[l].scatter(x_axis_control, u_to_plot[j, :], color=colors[j], edgecolors=None, marker='o', alpha=1, s=5)
            #ax[l].text(-20, u_lower[j], f'{u_lower[j]}', color='red', va='center', ha='right', fontsize=fs_axis, fontweight='bold')
            #ax[l].text(-20, u_upper[j], f'{u_upper[j]}', color='red', va='center', ha='right', fontsize=fs_axis, fontweight='bold')
            ax[l].axhline(y = u_upper[j], lw=0.75, color='red', ls='dotted')
            ax[l].axhline(y = u_lower[j], lw=0.75, color='red', ls='dotted')

            y_axis = controls_unn[k, j, :].squeeze().detach().numpy()       # dim = (n_periods, )
            ax_u[i][j].scatter(x_axis_control, y_axis, color = colors[j], marker ='o', edgecolors = None, alpha = 1, s=4)
            ax_u[i][j].axhline(y=u_lower[j],color='red',ls='dotted',lw=0.5, alpha=0.7)  # lower limit for control
            ax_u[i][j].axhline(y=u_upper[j],color='red',ls='dotted',lw=0.5, alpha=0.7)  # upper limit for control
            #ax_u[i][j].text(-20, u_lower[j], f'{u_lower[j]}', color='red', va='center', ha='right', fontsize=fs_axis, fontweight='bold')
            #ax_u[i][j].text(-20, u_upper[j], f'{u_upper[j]}', color='red', va='center', ha='right', fontsize=fs_axis, fontweight='bold')

        i = 3
        # Financial production
        l += 1  # l = 10
        ax[l].scatter(x_axis_control, fin_prod[k,:], color=colors[-1], edgecolors=None, marker='o', alpha=1, s=5)

        j = 0
        y_axis = fin_prod_unn[k, :].squeeze().detach().numpy()              # dim = (n_periods, )
        ax_u[i][j].scatter(x_axis_control, y_axis, color = colors[-1], marker='o', edgecolors = None, alpha=1, s=4)
        
        # Lapse rate
        l += 1  # l = 11
        ax[l].scatter(x_axis_control, lapses[k,:], color=colors[2], edgecolors=None, marker='o', alpha=1, s=5)

        j += 1  # j = 1
        y_axis = lapses_unn[k, :].squeeze().detach().numpy()                # dim = (n_periods, )
        ax_u[i][j].scatter(x_axis_control, y_axis, color = colors[2], marker='o', edgecolors = None, alpha=1, s=4)
        
        # Wealth comparison - a_nn 
        l += 1  # l = 12
        ax[l].stackplot(x_axis_state, values_cash[k,:], values_asset_1[k, :], values_asset_2[k, :],
                           labels = [r'Cash $\beta$', r'Asset 1 $\phi^1 S^1$', r'Asset 2 $\phi^2 S^2$'], 
                           colors = stack_colors, baseline='zero', alpha=0.2)       # asset decomposition 
        ax[l].plot(x_axis_state, values_assets[k,:], label='Asset', 
                                   color='black', ls='solid', lw=1.5)                 # total asset
        ax[l].plot(x_axis_state, values_liability[k,:], label='Liability',
                                   color='black', ls='dashed', lw=1.5)                # liability
        ax[l].legend(reverse=True, fontsize = fs_label, bbox_to_anchor=(1, .8))
        #ax[l].plot(x_axis_state, wealths[k, :], label=r'Strategy by $\upsilon^V$', color = 'red',ls='dotted', lw = 1.25)
        #ax[l].plot(x_axis_state, wealths_unn[k, :], label=r'Strategy by $\nu^\varpi$', color = 'blue',ls='dotted', lw = 1.25)
        #ax[l].legend(fontsize=fs_label)

        # Wealth decomposition - u_nn    
        j += 1  # j = 2
        ax_u[i][j].stackplot(x_axis_state, values_cash_unn[k,:], values_asset_1_unn[k, :], values_asset_2_unn[k, :],
                           labels = [r'Cash $\beta$', r'Asset 1 $\phi^1 S^1$', r'Asset 2 $\phi^2 S^2$'], 
                           colors = stack_colors, baseline='zero', alpha=0.2)       # asset decomposition 
        ax_u[i][j].plot(x_axis_state, values_assets_unn[k,:], label='Asset', 
                                   color='black', ls='solid', lw=1)                 # total asset
        ax_u[i][j].plot(x_axis_state, values_liability_unn[k,:], label='Liability',
                                   color='black', ls='dashed', lw=1)                # liability
        ax_u[i][j].legend(fontsize=fs_label, reverse=True, bbox_to_anchor=(1, .8)) # add legend  

    # Save the graphs
        fig.tight_layout()
        file_k = f'graph_a_seed_{args.seed}_trajectory_{id_selected[k]}.pdf'
        fig.savefig(J(results_dir, file_k))
        # plt.close()
   
        fig_u.tight_layout()
        file_u_k = f'graph_u_seed_{args.seed}_trajectory_{id_selected[k]}.pdf'
        fig_u.savefig(J(results_dir, file_u_k))
        plt.close()



            
            