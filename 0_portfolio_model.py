# This script contains all functions concerning the portfolio modeling
# Last update : 13/05/2025
# Author : Kim

import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
from torch import vmap

from math import sqrt

# --------------------------------------- #
######## Parsing fixed parameters #########
def parser_portfolio(parser):
    ''' Add a group of arguments corresponding to the portfolio model to a parser.

    The arguments are:
    - config_portfolio : path to the configuration file with parameters for portfolio modeling
    - config_simulator : path to the configuration file with parameters for simulation of initial states
    '''
    # Create parser 
    group = parser.add_argument_group('Portfolio model parameters', 
                                      'Parameters for the portfolio model and initial distribution simulation')
    group.add_argument('-p', '--config_portfolio', type=str, required=True, 
                        help="path to the configuration file with parameters for portfolio modeling")
    group.add_argument('-z', '--config_simulator', type=str, required=True, 
                        help="path to the configuration file with parameters for simulation of initial states")


# --------------------------------------- #
######## Portfolio Roll-forwards ##########

# Drift function
def drift(x, u, d, mu_S, param_r, param_L, coupon_rate, book_value) :
    """
    This function is b(x,u) where x is the current state and u is the control
    Inputs :
        x: state variable (tensor)
            _ x.shape = (n_samples, 2 * n_assets + 3)
            _ the 2nd dimension of x includes the following variables in the exact order
                    price               - dim = n_assets
                    interest rate       - dim = 1
                    cash                - dim = 1
                    quantity            - dim = n_assets
                    liability           - dim = 1
        u: control variable (tensor)
            _ u.shape = (n_samples, n_assets + 1)
            _ the 2nd dimension of u includes the following variables in the exact order
                    trade quantity      - dim = n_assets
                    profit-sharing %    - dim = 1
        d: number of assets in the portfolio

        mu_S: parameter for the drift of prices - dim = (n_assets,)
        param_r: parameters for the drift of the interst rate - dim = (3,)
        param_L: parameters for the lapse  model - dim = (2,)

        coupon_rate: the coupon (or dividend) rate for each asset - dim = (n_assets, )
        book_value: the book value for each asset in the portfolio - dim = (n_assets, )
    Output : a tensor of dim = (n_samples, 2*n_assets + 3) representing the drift 
    """

    device = x.device
    param_r = param_r.to(device)
    param_L = param_L.to(device)
    # check dimension 
    assert x.size(1) == 2 * d + 3, f'Dimension mismatch for state variable x : x.size(1) = {x.size(1)} <> 2d+3= {2*d+3}'
    assert u.size(1) == d + 1, f'Dimension mismatch for control variable u : u.size(1) = {u.size(1)} <> d+1 = {d+1}'

    # drifts for exogenous variables (S and r)
    drift_S = mu_S.to(device) * x[:,:d]
    drift_r = param_r[0] * (param_r[1] - x[:,d])

    # profit intensity
        # realized capital gain/loss from asset sale
    latent_capital_gain = x[:,:d] - book_value.to(device)                                   # unit = 1e3
    sale_quantity = torch.where(u[:,:d] < 0, torch.abs(u[:,:d]),0)                          # unit = 1e3
    profit_intensity = torch.sum(sale_quantity * latent_capital_gain, dim = 1)              # unit = 1e3 * 1e3 = 1e6
    
    coupon_intensity = coupon_rate.to(device) * book_value.to(device)                       # unit = % * 1e3 = 1e3
    coupons_revenue = torch.sum(x[:, d+2:2*d+2] * coupon_intensity, dim = 1)                # unit = 1e3 * 1e3 = 1e6
    cash_interest = x[:,d+1] * x[:,d]                                                       # unit = 1e6 * % = 1e6    
    profit_intensity +=  cash_interest + coupons_revenue 

    # lapse intensity
    lapse_intensity = param_L[0] + param_L[1] * (x[:, d] - u[:, -1] * profit_intensity / x[:, -1]) # unit = %

    # drifts for endogenous variables (beta, phi, L)
    drift_phi = u[:, :d]                                                       # trade quantity

    drift_beta = cash_interest + coupons_revenue   # inflows from cash's interest, bond's coupon, and stock's dividend
    drift_beta += - lapse_intensity * x[:, -1]                                 # outflows from paying out claims
    drift_beta += - torch.sum(u[:, :d] * x[:, :d], dim=1)                      # cash as residual for asset reallocation

    drift_L = u[:, -1] * profit_intensity                                      # sharing of profit
    drift_L += - lapse_intensity * x[:, -1]                                    # lapse reduces the obligation


    return torch.cat([t.unsqueeze(1) if len(t.size())==1 else t for t in (drift_S, drift_r, drift_beta, drift_phi, drift_L)], dim = 1)

def drift_extra(x, u, d, param_r, param_L, coupon_rate, book_value):
    '''
    This function behaves similarly to drift but it only returns the lapse intensity and profit intensity
    
    Note : 
    - this function is used for graphing only, not for computation purposes
    - this is a deterministic function that should be computed before the stochastic update
    '''
    device = x.device
    param_r = param_r.to(device)
    param_L = param_L.to(device)

    # profit intensity
        # from realizing capital gain/loss
    latent_capital_gain = x[:,:d] - book_value.to(device)                                      # S - S_til = market value - book value
    sale_quantity = torch.where(u[:,:d] < 0, torch.abs(u[:,:d]),0)
    profit_intensity = torch.sum(sale_quantity * latent_capital_gain, dim = 1)              # realized capital gain/loss from asset sale    
        
        # from cash's interest, bond's coupon, and stock's dividend
    coupon_intensity = coupon_rate.to(device) * book_value.to(device)
    coupons_revenue = torch.sum(x[:, d+2:2*d+2] * coupon_intensity, dim = 1)
    cash_interest = x[:,d+1] * x[:,d]
    profit_intensity +=  cash_interest + coupons_revenue   # revenue from cash's interest, bond's coupon, and stock's dividend
    
    # lapse intensity
    lapse_intensity = param_L[0] + param_L[1] * (x[:, d] - u[:, -1] * profit_intensity / x[:, -1])

    return profit_intensity, lapse_intensity

# Volatilty function
def vol(x, d, sigma_S, param_r):
    """
    This function is sigma(x, u) where x is the current state and u is the control
    Inputs : 
        x: state variable (tensor) - the same as in the drift function
            x.shape = (n_samples, 2*n_assets + 3)
        d: number of assets in the portfolio

        sigma_S: parameters for the drift of each asset - dim = (n_assets,) 
        param_r: parameters for the interest rate model - dim = (3,)
    Outputs : volatility matrix of dim = (n_samples, 2 * n_assets + 3, n_assets + 1) = (n_samples, dim_state, dim_sto)
        
    Note : For each data point, the volatility matrix is of size (dim_state, dim_sto) = (2*n_assets + 3, n_assets + 1), meaning that it is NOT a square matrix
    In fact, the matrix is composed of a (n_assets + 1, n_assets +1) diagonal matrix with diagonal elements (sigma_S * x[:,:d] (dim=d), param_r[2] (dim=1)), and 0s elsewhere.    
    """
    # check dimension
    assert x.size(1) == 2 * d + 3, f'Dimension mismatch for state variable x : x.size(1) = {x.size(1)} <> 2d+3= {2*d+3}'
    n_samples = x.size(0)
    device = x.device
    param_r = param_r.to(device)

    # volatility for exogenous variables (S and r)
    vol_S = sigma_S.to(device) * x[:, :d]                      # dim = (n_samples, n_assets)
    vol_r = param_r[2] * torch.ones(n_samples, 1).to(device)   # dim = (n_samples, 1)
    vol_vector = torch.cat([vol_S, vol_r], dim = 1)            # dim = (n_samples, n_assets + 1)

    # vol matrix for all states 
    vol = torch.zeros((n_samples, 2*d + 3, d + 1)).to(device)
    vol[:, :d+1, :d+1] = torch.diag_embed(vol_vector) # only the exogenous states have volatility 

    return vol

def update(x, u, 
           dt, dW, 
           d, 
           mu_S, sigma_S, param_r, param_L,
           coupon_rate, book_value):
    
    '''
    This function updates the portfolio state x after applying the control u
    Inputs :
        x : a sample of portfolio states 
            _ dim = (n_samples, 2*n_assets + 3)
            _ the 2nd dimension of x includes the following variables in the exact order
                    price               - dim = n_assets
                    interest rate       - dim = 1
                    cash                - dim = 1
                    quantity            - dim = n_assets
                    liability           - dim = 1
        u : a sample of control          
            _ dim = (n_samples, n_assets + 1)
            _  the 2nd dimension of u includes the following variables in the exact order
                    trade quantity      - dim = n_assets
                    profit-sharing %    - dim = 1

        d = n_assets number of assets in the portfolio
        dt : time step
        dW : brownian motions for 1 time step - dim = (n_samples, n_assets + 1)

        mu_S, sigma_S : parameters for the (Black-Scholes) dynamic for assets - dim = (n_assets,) each
        param_r : parameters for the (Vasicek) dynamic of the interest rate   - dim = (3,)
        param_L : parameters for the (linear) dynamic of the lapse intensity  - dim = (2,)
        coupon_intensity, book_value : fixed value for portfolio, same for every data point - dim = (n_assets,) each
        
    Output : a new state for the portfolio, x_new with dim = (n_samples, 2 * n_assets + 3)
    '''
    # calculate the next state using Euler method
    x_next = x + dt * drift(x, u, d, mu_S, param_r, param_L, coupon_rate, book_value)        # dim = (n_samples, 2 * n_assets + 3)
    n_samples = x.size(0)
    x_next += torch.bmm(vol(x, d, sigma_S, param_r), dW.reshape(n_samples, -1, 1)).squeeze() # dim = (n_samples, 2 * n_assets + 3)
    return x_next


def augmented_update(X, M, u, a, 
           dt, dW, 
           d, 
           mu_S, sigma_S, param_r, param_L,
           coupon_rate, book_value):
    '''
    This function updates the portfolio state x AND the Martingale representation of the constraint limit p after applying a control u and a stochastic move a
    Inputs :
        X : a sample of portfolio states X                      - dim = (n_samples, dim_state)
        M : a sample of Martingale representation M of p        - dim = (n_samples, )
        u : a sample of control u for X                         - dim = (n_samples, dim_control)
        a : a sample of stochastic increment a for M            - dim = (n_samples, dim_sto)
    
        dt : time step
        dW : brownian trajectories for X                        - dim = (n_samples, dim_sto)
            _ note that M uses the same brownians as X
        d = n_assets : number of assets in the portfolio
        mu_S, sigma_S, param_r, param_L, coupon_rate, book_value : portfolio parameters
    Outputs:
        a sample of X after applying the control u              - dim = (n_samples, dim_state) 
        a sample of M after applying the stochastic increment a - dim = (n_samples, )
    '''
    device = X.device   

    # update state variable X
    X_next = update(X, u, dt, dW, d, mu_S, sigma_S, param_r, param_L, 
                    coupon_rate, book_value).to(device)     # dim = (n_samples, dim_state)

    # update Martingale representation M
    M_next = M + torch.bmm(a.unsqueeze(1), dW.unsqueeze(2)).squeeze() # dim = (n_samples, 1, 1)
    M_next = M_next.squeeze().to(device)                    # dim = (n_samples, )            

    return X_next, M_next

# ------------------------------------ #
####### Checks and constraints #########

# Functions for an one-period sample X with dim = (n_samples, dim_state)
def penalty_bankruptcy(X, margin = 0.0):
    ''' 
    Intermediate loss - applied to all time steps t = 1, ..., n_periods
    Inputs :
        X : sample of portfolio states - dim = (n_samples, 2*n_assets + 3)
    '''
    d = int((X.shape[1]-3)/2)   # number of assets 

    net_debts =  X[:,-1] - torch.sum(X[:, :d] * X[:, d+2:2*d+2], dim=1) - X[:,d+1]     # (liability - total_assets) for each sample point
    # positive_net_debts = torch.where(net_debts > 0, torch.maximum(net_debts, 0.5 * net_debts**2),0)
    positive_net_debts = torch.where(net_debts > - margin, net_debts + margin, 0)
    return positive_net_debts   # dim = (n_samples, )

def penalty_negative_liab(X, margin = 0.0):
    ''' 
    Intermediate loss - applied to all time steps t = 1, ..., n_periods
    Input : X - sample of portfolio states - dim = (n_samples, 2*n_assets + 3)
    '''
    # negative_liab = torch.where(X[:,-1] < 0, torch.maximum(torch.abs(X[:, -1]), 0.5 * X[:, -1]**2), 0)
    negative_liab = torch.where(X[:,-1] < margin, -X[:, -1] + margin, 0)      
    return negative_liab        # dim = (n_samples, )

def penalty_negative_cash(X, margin = 0.0):
    '''
    Intermediate loss - applied to all time steps t = 1, ..., n_periods
    Inputs :
        X : sample of portfolio states - dim = (n_samples, 2*n_assets + 3)
    '''
    d = int((X.shape[1]-3)/2)    # number of assets 

    # negative_cash = torch.where(X[:, d+1] < 0, torch.maximum(torch.abs(X[:, d+1]), 0.5 * X[:, d+1]**2), 0)
    negative_cash = torch.where(X[:, d+1] < margin, -X[:, d+1] + margin, 0)
    return negative_cash         # dim = (n_samples, )

def total_intermediate_penalty(X, margin = 0.0):
    '''
    This is the sum of all intermediate losses WITHOUT averaging across the sample 
    Inputs :
        X : sample of portfolio states - dim = (n_samples, 2 * n_assets + 3)
    Outputs :
        a matrix of dim = (n_samples, ) 
            each entry = penalties (_bankruptcy + _negative_cash + _negative_liab) at the corresponding state
    '''
    return penalty_bankruptcy(X, margin) + penalty_negative_cash(X, margin) + penalty_negative_liab(X, margin) # dim = (n_samples, )

def terminal_capital_loss(X, K_below, K_above):
    '''
    Terminal loss - applied uniquely at the end of the horizon 
    Inputs : 
        X : sample of portfolio states - dim = (n_samples, 2*n_assets + 3)
        K_below, K_above : lower and upper bounds for the values
    '''
    d = int((X.shape[1]-3)/2)    # number of assets 
    net_loss = X[:, -1] - torch.sum(X[:, :d] * X[:, d+2:2*d+2], dim=1) - X[:,d+1]
    return torch.clamp(net_loss, min=K_below, max=K_above) # dim = (n_samples, )

# Total penalties for multiple-period sample X_hist = (n_samples, dim_state, n_periods)
def multi_period_total_penalty(X_hist, margin):
    '''
    Inputs : 
        X_hist : multi-period sample of states - dim = (n_samples, dim_state, n_periods)
        margin 
    Outputs :
        a matrix of dim = (n_samples, n_periods)
            each entry = penalty(_bankruptcy + _negative_PM + _negative_cash) of (corresponding state)
    '''
    # dimension
    n_samples, dim_state, n_periods = X_hist.shape

    # transpose the sample so that the n_periods dimension becomes the "batch" dimension
    X_hist = X_hist.permute(2, 0 , 1)                     # dim = (n_periods, n_samples, dim_state)
    
    # apply the function using vmap
    def total_penalty_wrapped(x_single):                  # dim(x_single) = (n_samples, dim_state)
        return total_intermediate_penalty(x_single,margin)# fixing margin into the function
    
    penalties = vmap(total_penalty_wrapped)(X_hist)       # should be of dim = (n_periods, n_samples)

    # reshape for the right dimension
    penalties = penalties.permute(1, 0)                   # should be of dim = (n_samples, n_periods)
    assert penalties.shape == torch.Size([n_samples, n_periods]),  f"Dimension mistmatch, please check" 

    return penalties

def multi_period_compute(one_period_penalty_func, X_hist, margin):
    '''
    Inputs :    
        single_period_func : function to be apply to one-period data sample of dim = (n_samples, dim_state)
        X_hist : sample of state variable across multiple periods - dim = (n_samples, dim_state, n_periods)
        margin : margin for penalty computation
    Output :
        a matrix of dim = (n_samples, n_periods), each entry corresponding to the penalty computed at the corresponding state
    '''
    n_samples, dim_state, n_periods = X_hist.shape              # original dimension
    X_hist = X_hist.permute(2,0,1)          # dim = (n_periods, n_samples, dim_state) >> n_periods becomes the "batch" dimension 

    def func_wrapped(x_single):             # x_single is one-period sample of dim = (n_samples, dim_state)
        return one_period_penalty_func(x_single, margin)
    
    multi_period_penalty = vmap(func_wrapped)(X_hist)           # should be of dim = (n_periods, n_samples)
    multi_period_penalty = multi_period_penalty.permute(1,0)    # should be of dim = (n_samples, n_periods)
    assert multi_period_penalty.shape == torch.Size([n_samples, n_periods]), f"Dimension mismatch, please check"

    return multi_period_penalty             

def avg_multi_period_penalty(X_hist, margin):
    '''
    Inputs : 
        X_hist : multi-period sample of states - dim = (n_samples, dim_state, n_periods)
        margin 
    Outputs : 
        average of these following quantities across data points and periods : bankruptcy penalty, negative cash penalty, negative liability penalty
    '''
    device = X_hist.device            # device

    # compute each average penalty separately 
    avg_bankruptcy_penalty = torch.mean(multi_period_compute(penalty_bankruptcy, X_hist, margin).to(device))
    avg_negative_cash_penalty = torch.mean(multi_period_compute(penalty_negative_cash, X_hist, margin).to(device))
    avg_negative_liab_penalty = torch.mean(multi_period_compute(penalty_negative_liab, X_hist, margin).to(device))

    return avg_bankruptcy_penalty, avg_negative_cash_penalty, avg_negative_liab_penalty
    

# Cumulative loss =[ terminal loss + lambda * intermediate penalties ] (sample of dim = (n_samples, dim_state, n_periods))
def composite_loss(X_hist, K_below, K_above, lambda_coeff, margin):
    '''
    Inputs : 
        X_hist : multiple-period sample of states - dim = (n_samples, dim_state, n_periods)
        K_below, K_above : lower and upper bounds for the values
        lambda_coeff : coefficient for the intermediate penalties
        margin : margin for the penalties
    Outputs :
        a matrix of dim = (n_samples, )
            each entry = [ terminal loss + lambda * penalty(_bankruptcy + _negative_PM + _negative_cash) ] (corresponding state)
    '''

    # compute the terminal loss
    terminal_loss = terminal_capital_loss(X_hist[:,:,-1], K_below, K_above) # dim = (n_samples, )

    # compute the intermediate penalties
    penalties = multi_period_total_penalty(X_hist, margin)                  # dim = (n_samples, n_periods)

    # create the target for training and reshape
    loss = terminal_loss + lambda_coeff * torch.sum(penalties, dim=1)       # dim = (n_samples, )
    
    return loss      

# ------------------------------------ #
####### Utility funct (to maximize) ####

def terminal_utility_exponential(X, alpha = 0.5):
    '''
    Terminal utility applied uniquely at the end of the horizon. 
    Formula : U(w) = - alpha * exp(- alpha * w) when w = terminal weath = (total assets - liabilty) @ T
    Inputs : 
        X : sample of portfolio (terminal) states - dim = (n_samples, dim_state) = (n_samples, 2*n_assets + 3)
        alpha : absolute risk aversion (for exponential utility function)
    Output :
        the corresponding terminal utility of the sample - dim = (n_samples,)
    '''     
    d = int((X.shape[1] - 3)/2)     # n_assets : number of assets
    device = X.device               # device 

    net_gains = torch.sum(X[:, :d] * X[:, d+2:2*d+2], dim=1) + X[:,d+1] - X[:, -1] # terminal wealth = total assets - liability
    net_gains = net_gains.to(device).squeeze()        # dim = (n_samples, )
    return - alpha * torch.exp(-alpha * net_gains)    # dim = (n_samples, )             

def terminal_utility_exponential_2(X, alpha = 0.5, KF=-10e9):
    '''
    Terminal utility applied uniquely at the end of the horizon. 
    Formula : U(w) = - alpha * exp(- alpha * w) when w = terminal weath = (total assets - liabilty) @ T
    Inputs : 
        X : sample of portfolio (terminal) states - dim = (n_samples, dim_state) = (n_samples, 2*n_assets + 3)
        alpha : absolute risk aversion (for exponential utility function)
    Output :
        the corresponding terminal utility of the sample - dim = (n_samples,)
    '''     
    d = int((X.shape[1] - 3)/2)     # n_assets : number of assets
    device = X.device               # device 

    net_gains = torch.sum(X[:, :d] * X[:, d+2:2*d+2], dim=1) + X[:,d+1] - X[:, -1] # terminal wealth = total assets - liability
    net_gains = net_gains.to(device).squeeze()        # dim = (n_samples, )
    return torch.maximum(-torch.exp(-alpha * net_gains), torch.tensor([KF], device=device))    # dim = (n_samples, )                       
    
# ------------------------------------ #
####### Simulators #####################

def simulate_initial_state(n_samples, n_assets, dist_params, scale=True):
    '''
    This function generates a sample of portfolio initial states based on a given distribution (uniform)
    There is an option to scale the sample or not (scale = 1 or scale = 0)

    Recall : X is a sample of state variables - X.shape = (n_samples, 2*n_assets + 3)
        >> 2nd dimension of X = (price - dim = n_assets, rate - dim=1, cash - dim=1, quantity - dim=n_assets, liability - dim=1)
    '''
    dim_x = 2 * n_assets + 3

    # From dist_params dictionary, create the tensors of lower and upper bounds, as well as the scaling factor
    lower_bound = torch.zeros(size = (1, dim_x))
    upper_bound = torch.zeros(size = (1, dim_x))
    scale_factor = torch.zeros(size = (1, dim_x))
    j = 0

    # Order is given in the documentation of the function: price, rate, cash, quantity, liability
    for field in ['price', 'interest_rate', 'cash', 'quantity', 'liability_ratio']:
        value = dist_params[field]
        if isinstance(value, list):
            for dic in value:
                lower_bound[0,j] = dic['range'][0]
                upper_bound[0,j] = dic['range'][1]
                scale_factor[0,j] = dic['scale']
                j += 1
        elif isinstance(value, dict):
            lower_bound[0,j] = value['range'][0]
            upper_bound[0,j] = value['range'][1]
            scale_factor[0,j] = value['scale']
            j += 1
        else:
            raise ValueError(f'Invalid type for field {field}: Found {type(value)}, expected list or dict.')

    assert j == dim_x, f'Error in the number of fields in dist_params: {j} <> {dim_x}'

    # generate random factors
    d = n_assets # Simplify the notation
    random_draw = torch.rand(size = (n_samples, 2*d + 3))    
    random_in_range = random_draw * (upper_bound - lower_bound) + lower_bound   # note that here, liability L is still in ratio

    # compute the total asset (cash + price * quantity)
    random_total_asset = torch.sum(random_in_range[:, :d] * random_in_range[:, d+2:2*d+2], dim=1) # (price * quantity)
    random_total_asset += random_in_range[:,d+1]                                                  # + cash
    random_in_range[:,-1] = random_in_range[:,-1] * random_total_asset          # liability = liability_ratio * total_asset

    if scale:
        sample = random_in_range * scale_factor
    else :
        sample = random_in_range

    return sample

def simulate_brownians(n_samples, dim_sto, n_periods, dt, corr):
    '''
    This function generates brownians of dimension (n_samples, dim_sto, n_periods) for the update function
    Inputs : 
        n_samples : number of data points in the sample
        dim_sto : dimension of the stochastic factor; as of now, dim_sto = n_assets + 1
        n_periods : number of periods in the time horizon
        dt : time step
        corr : correlation matrix between the brownians 
    Output : a sample of brownian motions with dim = (n_samples, dim_sto, n_periods)
    
    Note : the Brownians are independent of one another as of now (corr = identity matrix)
    '''
    # Get the device from the correlation matrix, use it a reference device
    device = corr.device
    generator = torch.distributions.MultivariateNormal(torch.zeros((dim_sto,)).to(torch.device(device)), corr * sqrt(dt))
    dW = generator.sample((n_samples,n_periods)).permute(0,2,1)     # dim = (n_samples, dim_sto, n_periods)
    return dW.to(device)

# Generator of (t,X) sample with their corresponding target (for the training of w_nn)
    # condition : there must be a control process neural network ready to employ
def generate_full_data_with_target(x_0, dW, u_NN, T, 
                                   lambda_coeff, K_below, K_above, margin):
    '''
    Inputs :
        x_0 : a sample of initial states                 - dim = (n_samples, dim_state)
        dW : a sample of brownian increment trajectories - dim = (n_samples, dim_state, n_periods)
        control_network : the trained optimal control network with n_subnetworks = n_periods
    Outputs :
        a sample of training data in the form (t, x) of dim = (n_samples *(n_periods + 1), dim_state + 1)
        a sample of training target of dim = (n_samples * (n_periods + 1), )
    Notes :
        _ the sample is obtained by rolling forward x_0 and dW using control_network
        _ the target is the accumulative future loss of the form G + lambda * g (more detailed definition in the documentation) 
    
    >> need a way to parse the value of the regularization lambda/k here
    >> need a way to parse the terminal loss and the penalties as inputs (instead of hardcoding as of now)
    '''
    # declare device
    device = x_0.device 

    # use the control network to roll forward data
    u_NN.eval()
    with torch.no_grad():
        data_roll_forward, _ = u_NN(x_0.to(device), dW.to(device))                            # dim = (n_samples, dim_state, n_periods)
        data_roll_forward = data_roll_forward.to(device)

        n_samples, dim_state, n_periods = data_roll_forward.shape

        # compute final loss
        final_states = data_roll_forward[:, :, -1].squeeze()                                  # dim = (n_samples, dim_state)
        final_losses = terminal_capital_loss(final_states, K_below, K_above).to(device)       # dim = (n_samples, )
        
        # create a matrix of accumulated future intermediate loss
        intermediate_losses = multi_period_total_penalty(data_roll_forward, margin).to(device)# dim = (n_samples, n_periods)
            # flip backward along the time dimension : (n_samples ,1 -> N) --> (n_samples, N -> 1)
        intermediate_losses = torch.flip(intermediate_losses, dims = [1])                     # dim = (n_samples, n_periods)
            # cumulative sum of the intermediate losses
        intermediate_losses = torch.cumsum(intermediate_losses, dim = 1)                      # dim = (n_samples, n_periods)
            # flip again to get the correct time order (n_samples, 1 -> N)
                # note that at entry (i, j) we have the sum_{k=i}^N g(\hat X^j_{t_k})
        intermediate_losses = torch.flip(intermediate_losses, dims = [1])                     # dim = (n_samples, n_periods)
            # add a column of 0 at the end (for target of t_N) - check documentation for detailed description
        intermediate_losses = torch.cat([intermediate_losses, torch.zeros((n_samples,1)).to(device)],
                                         dim = 1)                                             # dim = (n_samples, n_periods + 1)
       
        # create the target for training and reshape
        target = final_losses.unsqueeze(1) + lambda_coeff * intermediate_losses               # dim = (n_samples, n_periods + 1)
        target = target.reshape(n_samples*(n_periods+1), -1) # should be dim = (n_samples * (n_periods + 1),) and (t_{0 to N}, X^j_{0 to N})^{j=1 to M}

        # create the training data set WITH the initial state x_0 and ADD time dimension
            # add initial states x_0
        training_data = torch.cat([x_0.unsqueeze(2), data_roll_forward], dim = 2)             # dim = (n_samples, dim_state, n_periods + 1)
            # reshape
        training_data = torch.transpose(training_data, 1, 2)                                  # dim = (n_samples, n_periods + 1, dim_state)
        training_data = training_data.reshape((n_samples * (n_periods + 1), -1))              # dim = (n_samples*(n_periods + 1), dim_state)
            # add time dimension
        time_data = torch.linspace(0, T, n_periods + 1).reshape(-1, 1).repeat(n_samples, 1)   # dim = (n_samples*(n_periods + 1), 1)
        training_data = torch.cat([time_data.to(device), training_data], dim = 1)             # dim = (n_samples*(n_periods + 1), dim_state + 1)

        # final check of dimension
        assert training_data.shape[0] == n_samples * (n_periods + 1), f"Dimension in training sample : expect {n_samples} (samples) * {n_periods + 1} (periods) = {n_samples *(n_periods + 1)} data points but receive {training_data.shape[0]} instead"
        assert training_data.shape[1] == dim_state + 1, f"Dimension in training sample : each data point should have 1 (time) + {dim_state} (space) dimension but the state dimension is {training_data.shape[1]} in the training data"
        assert target.shape[0] == n_samples * (n_periods + 1), f"Dimension in target: expect {n_samples} (samples) * {n_periods + 1} (periods) = {n_samples *(n_periods + 1)} data points but receive {target.shape[0]} instead"
        
        return training_data.float(), target.squeeze().float()

# Generator of (t,X) sample WITHOUT target (for the error measure computation)
def generate_data_error(x_0, dW, u_NN, T):
    '''
    Inputs :
        x_0 : a sample of initial states                 - dim = (n_samples, dim_state)
        dW : a sample of brownian increment trajectories - dim = (n_samples, dim_state, n_periods)
        u_NN : the trained optimal control network with n_subnetworks = n_periods
        T : terminal of the time horizon
    Outputs :
        a sample of training data in the form (t, x) of dim = (n_samples *n_periods, dim_state + 1) 
    Notes :
        _ the sample is obtained by rolling forward x_0 and dW using control_network
    '''
    device = x_0.device  # device
    
    u_NN.eval()          # use the control network to roll forward data
    with torch.no_grad():
        data_roll_forward, controls = u_NN(x_0.to(device), dW.to(device))         # dim = (n_samples, dim_state, n_periods)
        data_roll_forward = data_roll_forward.to(device)
        controls = controls.to(device)

        n_samples, dim_state, n_periods = data_roll_forward.shape
        dt = T/n_periods

        # create the training data set WITH the initial state x_0 and ADD time dimension
            # add initial states x_0 and exclude the final states X_{t_N}
        training_data = torch.cat([x_0.unsqueeze(2), data_roll_forward[:,:,:-1]], dim = 2)   # dim = (n_samples, dim_state, n_periods)
            # reshape
        training_data = torch.transpose(training_data, 1, 2)                                 # dim = (n_samples, n_periods, dim_state)
        training_data = training_data.reshape((n_samples * (n_periods), -1))                 # dim = (n_samples * n_periods, dim_state)
            # add time dimension
        time_data = torch.linspace(0, T-dt, n_periods).reshape(-1, 1).repeat(n_samples, 1)   # dim = (n_samples * n_periods, 1)
        training_data = torch.cat([time_data.to(device), training_data], dim = 1)            # dim = (n_samples * n_periods, dim_state + 1)

        # final check of dimension
        assert training_data.shape[0] == n_samples * n_periods, f"Dimension in training sample : expect {n_samples} (samples) * {n_periods} (periods) = {n_samples * n_periods } data points but receive {training_data.shape[0]} instead"
        assert training_data.shape[1] == dim_state + 1, f"Dimension in training sample : each data point should have 1 (time) + {dim_state} (space) dimension but the state dimension is {training_data.shape[1]} in the training data"
      
        return training_data.float(), controls
    
# Generator of (t,X) sample WITHOUT target for training w_nn using PINN
def generate_data_PINN(x_0, dW, u_NN, T):
    '''
    Inputs :
        x_0 : a sample of initial states                 - dim = (n_samples, dim_state)
        dW : a sample of brownian increment trajectories - dim = (n_samples, dim_state, n_periods)
        u_NN : the trained optimal control network with n_subnetworks = n_periods
        T : terminal of the time horizon
    Outputs :
        a sample of interior data (t, x) with 0 =< t < T - dim = (n_samples * n_periods, 1 + dim_state)
        a sample of terminal data (T, x)                 - dim = (n_samples, 1 + dim_state)
    Notes :
        _ the sample is obtained by rolling forward x_0 and dW using control_network
    '''
    device = x_0.device   # device

    u_NN.eval()          # use the control network to roll forward data
    with torch.no_grad():
        # roll forward
        data_roll_forward, controls = u_NN(x_0.to(device), dW.to(device))                     # dim(data) = (n_samples, dim_state, n_periods) and dim(controls) = (n_samples, dim_sto, n_periods)
        data_roll_forward = data_roll_forward.to(device)                                  
        controls = controls.to(device)                      # move to device
        n_samples, dim_state, n_periods = data_roll_forward.shape                             # extract dimension
        dim_control = controls.shape[1]
        dt = T/n_periods

        # create the sample of interior data
            # include x_0, exclude final states X_T
        interior_data = torch.cat([x_0.unsqueeze(2), data_roll_forward[:, :, :-1]], dim = 2)       
            # reshape                                   
        interior_data = torch.transpose(interior_data, 1, 2)                                # dim = (n_samples, n_periods, dim_state)
        interior_data = interior_data.reshape((n_samples * n_periods, -1))                  # dim = (n_samples * n_periods, dim_state)
            # add time dimension
        time_data = torch.linspace(0, T-dt, n_periods).reshape(-1, 1).repeat(n_samples, 1)  # dim = (n_samples * n_periods, 1)
        interior_data = torch.cat([time_data.to(device), interior_data], dim = 1)           # dim = (n_samples * n_periods, 1 + dim_state)

        # reshape the control (to accompany the interior data)
        controls = torch.transpose(controls, 1, 2 )                                         # dim = (n_samples, n_periods, dim_control)
        controls = controls.reshape((n_samples * n_periods, -1))                            # dim = (n_samples * n_periods, dim_control)

        # create the sample of terminal data
        
        terminal_data = data_roll_forward[:,:,-1]                                           # dim = (n_samples, dim_state)
        terminal_data = torch.cat([torch.ones((n_samples,1)).to(device), terminal_data], 
                                  dim=1)                                                    # dim = (n_samples, 1 + dim_state)
        
        # check dimension
        assert controls.shape == torch.Size([n_samples * n_periods, dim_control]), f"Dimension mismatch for controls, please check"
        assert interior_data.shape == torch.Size([n_samples * n_periods, 1 + dim_state]), f"Dimension mismatch for interior_data, please check"
        assert terminal_data.shape == torch.Size([n_samples, 1 + dim_state]), f"Dimension mismatch for terminal_data, please check"

        return interior_data.float().to(device), terminal_data.float().to(device), controls

def generate_p(x0, w, p_range=0.5):
    x0_extended = torch.cat((torch.zeros(x0.size(0), 1).to(x0.device), x0), 
                            dim=1)                                  # dim = (n_samples, dim_state + 1)
    p_min = w(x0_extended).squeeze()                                # dim = (n_samples, )
    return p_range/10 + p_min + torch.rand(x0.size(0)).to(x0.device) * p_range   # dim = (n_samples, )

# ------------------------------------ #
######### Visualisers ##################

# Function to compute 1 trajectory from a given initial state and a given brownian 
def create_trajectory_with_control(x_0, dW, u_NN):
    '''
    Inputs : 
        x_0 : a single initial state                - dim = (dim_state, ) 
        dW : a single brownian increment trajectory - dim = (dim_sto, n_periods) 
        u_NN : a trained optimal control network
    Outputs : 
        the corresponding state trajectory and control process
    '''
    # device 
    device = x_0.device 

    # dimension 
    dim_state = x_0.shape[0]
    dim_sto, n_periods = dW.shape
    n_assets = int((dim_state - 3)/2)   # number of assets
    assert dim_sto == n_assets + 1 , f"Dimension mismatch : initial state x_0 is of dim = {dim_state} implying that n_assets = {n_assets} but browmian dim = {dim_sto} <> {n_assets+1} as expected"
    assert len(u_NN) == n_periods, f"Dimension mismatch : control network u_NN has {len(u_NN)} subnetworks but there are {n_periods} periods in each brownian trajectories"

    control = torch.zeros((n_assets + 1, n_periods)).to(device)

    # roll forward 
    u_NN.eval()
    with torch.no_grad():
        X_hist, control = u_NN(x_0.to(device).unsqueeze(0), dW.to(device).unsqueeze(0))
        X_hist = X_hist.squeeze(0).to(device)                  # dim(X_hist) = (dim_state, n_periods)
        control = control.squeeze(0).to(device)                # dim(control) = (dim_control, n_periods)

        X_hist = torch.concat([x_0.reshape(-1,1), X_hist], dim = 1)                          # dim = (dim_state, 1 + n_periods)

        return X_hist.permute(1,0), control.permute(1,0)        # dim(X_hist) = (n_periods + 1, dim_state) and dim(control) = (n_periods, n_assets + 1)     
            

# Function to compute trajectories stemming from the same initial state with one or multiple brownian trajectories
def create_trajectories(x_0, dW, u_NN):
    '''
    Inputs :
        x_0 : a SINGLE initial state - dim = (dim_state,)
        dW : a sample of brownian increment trajectories - dim = (n_samples, dim_sto, n_periods)
        u_NN : a trained optimal control network
    Outputs :
        the corresponding trajectory, and if requested, a visualization for it
    '''
    # device 
    device = x_0.device

    # dimension
    dim_state = x_0.shape[0]
    n_samples, dim_sto, n_periods = dW.shape
    n_assets = int((dim_state - 3)/2)   # number of assets
    assert dim_sto == n_assets + 1 , f"Dimension mismatch : initial state x_0 is of dim = {dim_state} implying that n_assets = {n_assets} but browmian dim = {dim_sto} <> {n_assets+1} as expected"
    assert len(u_NN) == n_periods, f"Dimension mismatch : control network u_NN has {len(u_NN)} subnetworks but there are {n_periods} periods in each brownian trajectories"

    if n_samples > 1:
        x_0_extend = x_0.repeat((n_samples, 1))
    else:
        x_0_extend = x_0.reshape(1, -1)

    # roll forward
    trajectory, _ = u_NN(x_0_extend, dW) # dim = (n_sample, dim_state, n_periods)


    trajectory = torch.cat((x_0_extend.unsqueeze(2), trajectory.to(device)), dim=2) # dim = (n_samples, dim_state, n_periods + 1)
    
    return trajectory




# ---------------------------------------- #
####### Supporting functions ###############

def helper_read_config_files(args):
    ''' Read the configuration files for the portfolio model and the initial distribution simulation. 
    This will return all the variables needed. It is ugly, but useful for the moment.
    '''
    # Read the portfolio model file
    config = json.load(open(args.config_portfolio, 'r'))

        # portfolio parameters 
    n_assets = config['n_assets']
    mu_S = torch.Tensor(config['mu_S'])
    sigma_S = torch.Tensor(config['sigma_S'])
    coupon_rate = torch.Tensor(config['coupon_rate'])
    book_value = torch.Tensor(config['book_value'])
    param_r = torch.Tensor(config['param_r'])
    corr = torch.Tensor(config['corr'])
    param_L = torch.Tensor(config['param_L'])
    u_lower = torch.Tensor(config['u_lower'])
    u_upper = torch.Tensor(config['u_upper'])

        # time parameters
    n_periods = config['n_periods']
    T = config['T']
    dt = T/n_periods

        # limit for capital loss
    K_below, K_above = config['K_below'], config['K_above']
    
        # indicator for scaling
    scale_ind = config['scale_ind']
    
    # Read the initial distribution file
    dist_params = json.load(open(args.config_simulator, 'r'))
    
    # Scaling book value : Note that if scale indicator is 1, we have to scale the book value manually here
    if scale_ind == 1 :     
        # import the scaling factor
        scale_factor_price = torch.zeros((len(book_value),))
        assert len(dist_params['price']) == len(book_value), f"Dimension mismatch between book value (len={len(book_value)}) and scaling factor for price (len={len(dist_params['price'])})"
        for i in range(len(dist_params['price'])):
            scale_factor_price[i] = dist_params['price'][i]['scale']
            
        # scale book value
        book_value = scale_factor_price * book_value
        
    return n_assets, mu_S, sigma_S, coupon_rate, book_value, param_r, corr, param_L, u_lower, u_upper, n_periods, T, dt, K_below, K_above, scale_ind, dist_params

if __name__ == '__main__':

    # Create blank parser to test the parser_portfolio function
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser_portfolio(parser)

    args = parser.parse_args()

    n_assets, mu_S, sigma_S, c, S_til, param_r, corr, param_L, n_periods, T, dt, K_below, K_above, scale_ind, dist_params = helper_read_config_files(args)


    # Test the simulation function
    n_samples = 4
    sample = simulate_initial_state(n_samples, n_assets, dist_params, scale=False)
    print("Test sample without scaling:")
    print("Sample shape:", sample.shape)
    print("Sample:", sample)

    sample = simulate_initial_state(n_samples, n_assets, dist_params, scale=True)
    print("Test sample with scaling:")
    print("Sample shape:", sample.shape)
    print("Sample:", sample)

    # Test penalty functions
    print("Test penalty functions:")
    bankruptcy_penalty = penalty_bankruptcy(sample, n_assets)
    negative_liab_penalty = penalty_negative_liab(sample)
    terminal_loss_penalty = terminal_capital_loss(sample, n_assets, K_below, K_above)

    print("Bankruptcy penalty:", bankruptcy_penalty)
    print("Negative liab penalty:", negative_liab_penalty)
    print("Terminal capital loss penalty:", terminal_loss_penalty)


    # Test the update function
    control = torch.rand(n_samples, n_assets + 1)
    updated_sample = update(sample, control, dt, n_assets, mu_S, sigma_S, param_r, param_L, c, S_til, corr)
    print("Updated sample shape:", updated_sample.shape)
    print("Updated sample:", updated_sample)
