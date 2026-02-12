# This script contains the definitions of the neural networks used in other training

import torch
import torch.nn as nn
import math


'------- NEURAL NETWORKS FOR OPTIMAL CONTROL ---------------------------------'

# Subnetwork for control at each time step
class Subnetwork(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, lower_bounds, upper_bounds, 
                 activation=nn.ReLU(), sigmoid_scale=[1e-4, 1e-4, 1e-1]):
        '''
        Inputs :
            dim_input = dimension of the variable x = 2 * n_assets + 3
            dim_output = dimension of the control variable u = n_assets + 1
            dim_hidden = number of neurons in each hidden layer
            n_hidden = number of hidden layers in each subnetwork
            lower_bounds : lower bounds for the outputs of the network - dim = (n_assets + 1,)
            upper_bounds : upper bounds for the outputs of the network - dim = (n_assets + 1,)
        '''
        super(Subnetwork, self).__init__()

        # Create the neural network
            # Initiate a list of layers
        layers = []

            # First hidden layer
        layers.append(nn.BatchNorm1d([dim_input]))
        layers.append(nn.Linear(dim_input, dim_hidden))
        layers.append(activation)

            # In-between hidden layers
        for _ in range(n_hidden-1):
            layers.append(nn.BatchNorm1d([dim_hidden]))
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activation)

            # Last layer (output layer)
        layers.append(nn.Linear(dim_hidden, dim_output))

            # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)
        
        # Set the parameters for the boundary of the control variables (which are outputs of the neural network)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.sigmoid_scale = sigmoid_scale

    def forward(self, X):
        '''
        Input : X is a sample of the state variables            - dim = (n_samples, 2 * n_assets + 3)
        Output : u is the corresponding control variables for X - dim = (n_samples, n_assets + 1)
        '''
        # Go through the hidden layers
        scale = torch.Tensor(self.sigmoid_scale).to(X.device) # scale the sigmoid function to the range of the outputs
        u = self.model(X) * scale

        # Set the outputs of the neural network into admissible range (between lower bounds and upper bounds)
        device = u.device
        u = self.lower_bounds.to(device) + (self.upper_bounds.to(device) - self.lower_bounds.to(device))*torch.sigmoid(u)

        return u

# Global Network (Nested network) for the entire control process
class GlobalNetworks(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, n_subnetworks, 
                 update_func, lower_bounds, upper_bounds, 
                 activation = nn.ReLU(), sigmoid_scale=[1e-4, 1e-4, 1e-1]):
        '''
        Inputs :
            n_subnetworks : number of subnetwork inside the global network
                each of which requires the following inputs:
                    dim_input = dimension of one single state x = 2 * n_assets + 3
                    dim_output = dimension of one single control u = n_assets + 1
                    dim_hidden = number of neurons in each hidden layer
                    n_hidden = number of hidden layers in each subnetwork
                    activation = activation function 
                    lower_bounds, upper_bounds : the permissible limits for the outputs of the subnetwork
                    sigmoid_scale = scale for the sigmoid function in the 'scaling' of the control values (to make sure they stay inside the permissible range)
            update_func is the update function that rolls the portfolio state x forward after a control u is given
            
        Output : 
            a global network whose each subnetwork represents the strategy at its corresponding time step
        '''
        super(GlobalNetworks, self).__init__()
        self.n_subnetworks = n_subnetworks
        self.update_func = update_func
        self.subnetworks = nn.ModuleList([
                                            Subnetwork(dim_input, dim_output, dim_hidden, n_hidden, 
                                                       lower_bounds, upper_bounds, activation, sigmoid_scale)
                                            for _ in range(n_subnetworks)
                                        ])
        
        self.dim_output = dim_output
        
    def __len__(self): # Returns the number of subnetwork in the global network
        return len(self.subnetworks)
    
    def __getitem__(self, index): # Returns the subnetwork at index i
        return self.subnetworks[index]
    
    def forward(self, initial_states, brownians):
        '''
        Input : 
            initial_states : a sample of X at t=0               - dim = (n_samples, 2 * n_assets + 3)
            brownians : a sample of all the brownian motions dW - dim = (n_samples, n_assets + 1, n_periods) 
        Outputs : 
            a tensor of dim = (n_samples, 2 * n_assets + 3, n_periods) representing a list of all states X_1, ..., X_N (N = n_periods) 
            a tensor of dim = (n_samples, n_assets + 1) reprenting a list of all controls u_1, ..., u_N (N = n_periods)
        '''
        # dimensions 
        n_samples = initial_states.shape[0] # number of data points in the sample
        dim_state = initial_states.shape[1] # dimension of 1 single state variable 
        n_periods = self.n_subnetworks      # number of periods

        dim_control = int((dim_state - 1)/2)# dimension of 1 single control variable
        
        # Initiation
        current_state = initial_states.clone()
        states = torch.empty(size=(n_samples, dim_state, n_periods))
        controls = torch.empty(size=(n_samples, dim_control, n_periods))

        # Loop through the time horizon
        for t, subnet in enumerate(self.subnetworks):

            # compute the control for the period using the corresponding subnetwork
            control_t = subnet(current_state)   # dim = (n_samples, dim_control)
            controls[:,:,t] = control_t
            
            # brownian for the period
            dW_t = brownians[:,:,t]             # dim = (n_samples, n_assets + 1)

            # apply the control to update the state
            current_state = self.update_func(x=current_state, u=control_t, dW=dW_t)

            # store the new state
            states[:,:,t] = current_state

        return states, controls
    
class GlobalNetworks_extra(GlobalNetworks):
    '''
    The same network as GlobalNetworks but also returns extra information 
    that is used for graphing only (not used for training)
    '''
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, n_subnetworks, 
                 update_func, update_extra_func, 
                 lower_bounds, upper_bounds, activation=nn.ReLU(), sigmoid_scale=[0.0001, 0.0001, 0.1]):
        super().__init__(dim_input, dim_output, dim_hidden, n_hidden, n_subnetworks, update_func, lower_bounds, upper_bounds, activation, sigmoid_scale)
        self.update_extra_func = update_extra_func
    
    def forward(self, initial_states, brownians):
        '''
        Note that this function also produces the following extra information:
        - lapse rate (%)
        - financial production (%)
        - wealth (monetary unit)
        '''
        # dimensions 
        n_samples = initial_states.shape[0] # number of data points in the sample
        dim_state = initial_states.shape[1] # dimension of 1 single state variable 
        n_periods = self.n_subnetworks      # number of periods

        dim_control = int((dim_state - 1)/2)# dimension of 1 single control variable
        n_assets = dim_control - 1          # number of assets in the portfolio
        
        # Initiation
        current_state = initial_states.clone()
        states = torch.empty(size=(n_samples, dim_state, n_periods))
        controls = torch.empty(size=(n_samples, dim_control, n_periods))

        lapses = torch.empty(size=(n_samples, n_periods))           # lapse rate
        fin_prod = torch.empty(size=(n_samples, n_periods))         # financial production
        wealths = torch.empty(size=(n_samples, n_periods))          # wealth

        # Loop through the time horizon
        for t, subnet in enumerate(self.subnetworks):

            # compute the control for the period using the corresponding subnetwork
            control_t = subnet(current_state)   # dim = (n_samples, dim_control)
            controls[:,:,t] = control_t
            
            # brownian for the period
            dW_t = brownians[:,:,t]             # dim = (n_samples, n_assets + 1)

            # apply the control to update the state
            current_state = self.update_func(x=current_state, u=control_t, dW=dW_t)
            states[:,:,t] = current_state

            # additional information
            fin_prod_t, lapse_t = self.update_extra_func(x=current_state, u=control_t)  # compute lapse rate and financial production
            lapses[:,t] = lapse_t.squeeze()  
            fin_prod[:,t] = fin_prod_t.squeeze()  

            wealth_t = torch.sum(current_state[:, :n_assets] * current_state[:, n_assets+2:n_assets+2+n_assets], dim=1) + current_state[:, n_assets+1] - current_state[:, -1] # wealth = sum of asset values + cash - liability
            wealths[:,t] = wealth_t

        # Add the initial state at t=0 for full history
        states = torch.cat((initial_states.unsqueeze(2), states), dim=2)  # dim = (n_samples, dim_state, n_periods + 1)
 
        wealth_0 = torch.sum(initial_states[:, :n_assets] * initial_states[:, n_assets+2:n_assets+2+n_assets], dim=1) + initial_states[:, n_assets+1] - initial_states[:, -1]  # wealth at t=0
        wealths = torch.cat((wealth_0.unsqueeze(1), wealths), dim=1)  # dim = (n_samples, n_periods + 1)

        return states, controls, lapses, fin_prod, wealths


'------- NEURAL NETWORK FOR DOMAIN BOUNDARY VALUE ----------------------------'

class BoundaryValueNetwork(nn.Module):
    def __init__(self, dim_input, dim_hidden, n_hidden, activation=nn.ReLU(), print_layers = False):
        '''
        Inputs :
            dim_input = dimension of one data point (t, x) = 1 for t + dim_state for x = 1 + (2 * n_assets + 3)
            dim_hidden : number of neurons in each hidden layer
            n_hidden : number of hidden layers
        '''
        super(BoundaryValueNetwork, self).__init__()

        # Initiate a list to store layers
        layers = []

        # First hidden layer
        layers.append(nn.BatchNorm1d(dim_input))
        layers.append(nn.Linear(dim_input, dim_hidden))
        layers.append(activation)

        # In-between hidden layers
        for _ in range(n_hidden-1):
            
            layers.append(nn.BatchNorm1d(dim_hidden))
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activation)

        # Last layer (output layer) - output is a scalar

        layers.append(nn.BatchNorm1d(dim_hidden))
        layers.append(nn.Linear(dim_hidden, 1))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, Z):
        '''
        Input : Z - a sample of (t,x) - dim = (n_samples, dim_state + 1) 
        Output : an estimate for w(t,x) - dim = (n_samples, )
        '''
        z = self.model(Z)
        z = torch.squeeze(z)
        return z # should have dim = (n_samples, )
    

'------- NEURAL NETWORK FOR VALUE FUNCTION AT THE BOUNDARY --------------------'
# Value function on the viable domain (whether on the boundary or in the interior)
class ValueFunction_Boundary_Network(nn.Module):
    def __init__(self, dim_input, dim_hidden, n_hidden, activation=nn.ReLU(), print_layers = False):
        '''
        Inputs :
            dim_input = dimension of one data point (t, x) 
                    = 1 for t + dim_state for x = 1 + (2 * n_assets + 3)
            dim_hidden : number of neurons in each hidden layer
            n_hidden : number of hidden layers

        Note : the Value Function is at the spatial boundary where p = w(t,x), so in fact the input
            is only (t,x) instead of (t,x,p) since p is technically known.
        '''
        super(ValueFunction_Boundary_Network, self).__init__()

        # Initiate a list to store layers
        layers = []

        # First hidden layer
        layers.append(nn.BatchNorm1d(dim_input))
        layers.append(nn.Linear(dim_input, dim_hidden))
        layers.append(activation)

        # In-between hidden layers
        for _ in range(n_hidden-1):
            
            layers.append(nn.BatchNorm1d(dim_hidden))
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activation)

        # Last layer (output layer) - output is a scalar

        layers.append(nn.BatchNorm1d(dim_hidden))
        layers.append(nn.Linear(dim_hidden, 1))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, Z):
        '''
        Input : Z - a sample of (t,x) - dim = (n_samples, dim_state + 1) 
        Output : an estimate for V_bound(t,x) - dim = (n_samples, )
        '''
        z = self.model(Z)
        z = torch.squeeze(z)
        return z # should have dim = (n_samples, )
    

'---- NEURAL NETWORK FOR AUGMENTED OPTIMAL CONTROL -----------------------'

# Subnetwork for control for each time step
class AugSubnetwork(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, 
                 u_lower_bounds, u_upper_bounds, a_lower, a_upper,
                 activation = nn.ReLU(),
                 u_sigmoid_scale = [1e-4, 1e-4, 1e-1], a_sigmoid_scale = 1e-1):
        '''
        Inputs : 
            dim_input = dim of x + 1 for p = dim_state + 1 = (2 * n_assets + 3) + 1 = 2 * n_assets + 4
            dim_output = dim of u + dim of a = dim_control + dim_sto = (n_assets + 1) + (n_assets + 1) = 2 * n_assets + 2
            dim_hidden = number of neurons per hidden layer
            n_hidden = number of hidden layers in each subnetwork
            u_lower_bounds, u_upper_bounds : lower and upper limit for the control u  
                _ dim = dim_control = n_assets + 1
            a_lower, a_upper: lower and upper limit for each sto. increment a 
                _ dim = dim_sto = n_assets + 1
            u_sigmoid_scale, a_sigmoid_scale : scaling factors for the sigmoid function that handle the scaling of the output
                    to make sure that the output stay within authorized bounds
        '''
        # Initiation
        super(AugSubnetwork, self).__init__()

        # Create the neural network structure
            # create a list of layers
        layers = []

            # first hidden layer
        layers.append(nn.BatchNorm1d([dim_input]))
        layers.append(nn.Linear(dim_input, dim_hidden))
        layers.append(activation)

            # in-between hidden layers
        for _ in range(n_hidden-1):
            layers.append(nn.BatchNorm1d([dim_hidden]))
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activation)

            # last layer (output layer)
        layers.append(nn.Linear(dim_hidden, dim_output))

            # combine layers into a sequential model
        self.model = nn.Sequential(*layers)

        # Set parameters for the boundary for the output
            # dimension 
        self.n_assets = int((dim_input - 3)/2)       
        self.dim_control = self.n_assets + 1    # + 1 dimension for a
        self.dim_sto = self.n_assets + 1        # + 1 dimension for M


            # boundary for a 
        device = u_lower_bounds.device
        self.aug_lower_bounds = torch.concat([u_lower_bounds, torch.full(size=(self.dim_sto,), fill_value = a_lower).to(device)])
        self.aug_upper_bounds = torch.concat([u_upper_bounds, torch.full(size=(self.dim_sto,), fill_value = a_upper).to(device)])
        self.aug_sigmoid_scale = torch.concat([torch.tensor(u_sigmoid_scale).to(device), torch.full(size=(self.dim_sto,), fill_value = a_sigmoid_scale).to(device)])

    def forward(self, X, M):
        '''
        Input : 
            X : a sample of the state variable X                 - dim = (n_samples, dim_state)   = (n_samples, 2 * n_assets + 3)
            M : a sample of the Martingale representation M of p - dim = (n_sample, )
        Output : 
            a sample of control u for X                          - dim = (n_samples, dim_control) = (n_samples, n_assets + 1)
            a sample of stochastic increment a for M             - dim = (n_samples, dim_sto)     = (n_samples, n_assets + 1)            
        '''
        device = X.device                                               # device            
        scale = torch.Tensor(self.aug_sigmoid_scale).to(device)         # scale for the sigmoid function

        # pass forward and scale : here u and a are together (in the same tensor)
        aug_X = torch.concat([X, M.unsqueeze(1).to(device)], dim = 1)   # dim = (n_samples, dim_state + 1)         = (n_samples, 2 * n_assets + 4)      
        aug_u = self.model(aug_X) * scale                               # dim = (n_samples, dim_control + dim_sto) = (n_samples, 2 * n_assest + 2)   
        aug_u = aug_u.to(device)    

        # set output to be between the boundaries
        aug_u = self.aug_lower_bounds.to(device) + (self.aug_upper_bounds.to(device) - self.aug_lower_bounds.to(device)) * torch.sigmoid(aug_u)
    
        # separate u and a
        u = aug_u[:, :self.dim_control].to(device)                      # dim = (n_samples, dim_control)
        a = aug_u[:, self.dim_control:].squeeze().to(device)            # dim = (n_samples, dim_sto)

        return u, a
    
# Augmented Global Network for the entire process
class AugGlobalNetworks(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, n_subnetworks,
                 aug_update_func, u_lower_bounds, u_upper_bounds, a_lower, a_upper,
                 T= 1.0, activation = nn.ReLU(), 
                 u_sigmoid_scale = [1e-4, 1e-4, 1e-1], a_sigmoid_scale = 1e-1):
        '''
        Inputs :
            n_subnetworks : number of subnetwork inside the global network
                each of which is an AugSubnetwork with dim_input, dim_output, dim_hidden, n_hidden, 
                                aug_lower_bounds, aug_upper_bounds, activation, aug_sigmoid_scale
            aug_update_func : an update function that outputs the next state by taking into account
                the augmented state (X,M), the augmented control (u,a), and an augmented brownian (dW_X, dW_p)
            T : terminus of the time horizon
        Output : 
            a global network whose subnetworks represent the strategy for their corresponding time step
        '''
        super(AugGlobalNetworks, self).__init__()
        self.n_subnetworks = n_subnetworks
        self.aug_update_func = aug_update_func
        self.subnetworks = nn.ModuleList([
                                            AugSubnetwork(dim_input, dim_output, dim_hidden, n_hidden, 
                                                        u_lower_bounds, u_upper_bounds, a_lower, a_upper,
                                                        activation, u_sigmoid_scale, a_sigmoid_scale)
                                            for _ in range(n_subnetworks)
                                        ])
        self.T = T
        self.dim_output = dim_output
    
    def __len__(self): # Returns the number of subnetwork in the global network
        return len(self.subnetworks)
    
    def __getitem__(self, index): # Returns the subnetworks at a specified index
        return self.subnetworks[index]
    
    def forward(self, initial_states, constraint_limits, brownians, w):
        ''' 
        Inputs : where N := n_periods,
            initial_states : a sample of X at t = 0                 - dim = (n_samples, dim_state)
            constraint_limits : a sample of p at t = 0              - dim = (n_samples, )
            brownians : a sample of dW_x at t=1,...,N               - dim = (n_samples, dim_sto, n_periods)
            w : the function which compute the boundary of the domain at any given point (t,x)
                _ should be able to handle a sample of dim = (n_samples, dim_state + 1)

        Output : 
            a full history of states and Martingale representation of all the sample points (X_i, M_i)_{i=0,...,N}  
                _ dim = (n_samples, dim_state + 1, n_periods + 1)
            a full list of augmented controls applied to each data point at each time step (u_i, a_i)_{i=1,...,N} 
                _ dim = (n_samples, dim_control + 1, n_periods)
            a list of the stopping time tau for the first time arriving at the boundary for each trajectory 
                _ dim = (n_samples, )
                _ tau is in the form of timestamp (0, T/N, 2T/N, ..., T)
                _ check documentation 4_Training_ValueFunction_global.md for definition and explanation of tau
        
        Note : with this forward, we compute the full history even though we know that once the boundary is reached,
            we switch to the domain boundary case and no longer need any history after this event. We count on tau to 
            send signal of whether the boundary is reach and consequentially, whether the regime (interior or boundary)
            is switched.
        '''
        # Dimensions and device
        device = initial_states.device 

        n_samples = initial_states.shape[0]    # number of data points in the sample / trajectories in the history

        dim_state = initial_states.shape[1]    # dimension of 1 single state variable X
        dim_control = int((dim_state - 1)/2)   # dimension of 1 single control variable u
        dim_sto = int((dim_state - 1)/2)       # dimension of 1 single sto. increment a

        n_periods = self.n_subnetworks         # number of time steps in the time horizon

        # Initiation
        current_X = initial_states.clone()
        current_M = constraint_limits.clone()
        
        states = torch.empty(size=(n_samples, dim_state, n_periods)).to(device)         # history of X
        martingales = torch.empty(size=(n_samples, n_periods)).to(device)               # history of M
        controls = torch.empty(size=(n_samples, dim_control, n_periods)).to(device)     # history of u
        sto_increments = torch.empty(size=(n_samples, dim_sto, n_periods)).to(device)   # history of a
        taus = torch.full(size=(n_samples,), fill_value=self.T).to(device)  # at t=0, default value is T

        # Loop through the time horizon
        for i, subnet in enumerate(self.subnetworks):   # i = 0 ,..., N-1
            # update the stopping time tau 
                # compute the boundary w(t,x) at each data point (t,x)
            t = i * self.T/n_periods                    # t_i = 0, T/N, ..., T(N-1)/N 
            current_coordinate = torch.concat([torch.full(size=(n_samples, 1), fill_value=t).to(device),
                                               current_X], 
                                               dim = 1) # dim = (n_samples, 1 + dim_state) - merge t and X to pass through w
            current_boundary = w(current_coordinate)    # dim = (n_samples, )
                
                # find the indices of the stopping time that need to be update in the sample
                    # where the boundary was NOT reached before the current time AND it IS reached now
            id_update_tau = torch.logical_and(taus == self.T, current_M <= current_boundary)
            taus[id_update_tau] = t                     # note that tau IS a time stamp

            # brownians for the time period
            dW = brownians[:,:,i]                       # dim = (n_samples, dim_sto)

            # compute the augmented control for the period using the current subnetwork
            u_t, a_t = subnet(current_X, current_M)     
            controls[:,:,i] = u_t                       # dim = (n_samples, dim_control)
            sto_increments[:,:,i] = a_t                 # dim = (n_samples, )

            # update the state variable X and the Martingale representation M
            current_X, current_M = self.aug_update_func(current_X, current_M, u_t, a_t, dW) 
            states[:,:,i] = current_X                   # dim = (n_samples, dim_state + 1)
            martingales[:,i] = current_M                # dim = (n_samples, )
        
        # Add the initial states into the full history
        states = torch.concat([initial_states.unsqueeze(2), states], 
                              dim = 2)                  # dim = (n_samples, dim_state, n_periods + 1)
        martingales = torch.concat([constraint_limits.unsqueeze(1), martingales],
                              dim = 1)                  # dim = (n_samples, n_periods + 1)

        return states, martingales, controls, sto_increments, taus
    

    def forward_freeze(self, initial_states, constraint_limits, brownians, w):
        ''' 
        Inputs : where N := n_periods,
            initial_states : a sample of X at t = 0                 - dim = (n_samples, dim_state)
            constraint_limits : a sample of p at t = 0              - dim = (n_samples, )
            brownians : a sample of dW_x at t=1,...,N               - dim = (n_samples, dim_sto, n_periods)
            w : the function which compute the boundary of the domain at any given point (t,x)
                _ should be able to handle a sample of dim = (n_samples, dim_state + 1)

        Output : 
            a full history of states and Martingale representation of all the sample points (X_i, M_i)_{i=0,...,N}  
                _ dim = (n_samples, dim_state + 1, n_periods + 1)
            a full list of augmented controls applied to each data point at each time step (u_i, a_i)_{i=1,...,N} 
                _ dim = (n_samples, dim_control + 1, n_periods)
            a list of the stopping time tau for the first time arriving at the boundary for each trajectory 
                _ dim = (n_samples, )
                _ tau is in the form of timestamp (0, T/N, 2T/N, ..., T)
                _ check documentation 4_Training_ValueFunction_global.md for definition and explanation of tau
        
        Note : with this forward, we compute the full history even though we know that once the boundary is reached,
            we switch to the domain boundary case and no longer need any history after this event. We count on tau to 
            send signal of whether the boundary is reach and consequentially, whether the regime (interior or boundary)
            is switched.
        '''
        # Dimensions and device
        device = initial_states.device 

        n_samples = initial_states.shape[0]    # number of data points in the sample / trajectories in the history

        dim_state = initial_states.shape[1]    # dimension of 1 single state variable X
        dim_control = int((dim_state - 1)/2)   # dimension of 1 single control variable u
        dim_sto = int((dim_state - 1)/2)       # dimension of 1 single sto. increment a

        n_periods = self.n_subnetworks         # number of time steps in the time horizon

        # Initiation 
        current_X = initial_states.clone()
        current_M = constraint_limits.clone()
        
        states = torch.empty(size=(n_samples, dim_state, n_periods)).to(device)         # history of X
        martingales = torch.empty(size=(n_samples, n_periods)).to(device)               # history of M
        controls = torch.empty(size=(n_samples, dim_control, n_periods)).to(device)     # history of u
        sto_increments = torch.empty(size=(n_samples, dim_sto, n_periods)).to(device)   # history of a
        taus = torch.full(size=(n_samples,), fill_value=self.T).to(device)  # at t=0, default value is T

        # Loop through the time horizon
        for i, subnet in enumerate(self.subnetworks):   # i = 0 ,..., N-1
            # update the stopping time tau 
                # compute the boundary w(t,x) at each data point (t,x)
            t = i * self.T/n_periods                    # t_i = 0, T/N, ..., T(N-1)/N 
            current_coordinate = torch.concat([torch.full(size=(n_samples, 1), fill_value=t).to(device),
                                               current_X], 
                                               dim = 1) # dim = (n_samples, 1 + dim_state) - merge t and X to pass through w
            current_boundary = w(current_coordinate)    # dim = (n_samples, )
                
                # find the indices of the stopping time that need to be update in the sample
                    # where the boundary was NOT reached before the current time AND it IS reached now
            id_tau_at_t_i = torch.logical_and(taus == self.T, current_M <= current_boundary)
            taus[id_tau_at_t_i] = t                     # note that tau IS a time stamp

            # Conditional update
                # If tau = T, we update the state and Martingale representation
            id_update = taus == self.T
            if torch.sum(id_update) != 0:  # if there is at least one data point to update
                X_to_be_updated = current_X[id_update]              # dim = (sum(id_update), dim_state)
                M_to_be_updated = current_M[id_update]              # dim = (sum(id_update), )
                dW_to_be_updated = brownians[id_update,:,i]         # dim = (sum(id_update), dim_sto)
                
                # compute the control for the period using the corresponding subnetwork
                u_t, a_t = subnet(X_to_be_updated, M_to_be_updated) # dim = (sum(id_update), dim_control + dim_sto)
                controls[id_update,:,i] = u_t                       # dim = (sum(id_update), dim_control)
                sto_increments[id_update,:,i] = a_t                 # dim = (sum(id_update), dim_sto)

                # update the state variable X and the Martingale representation M
                current_X[id_update], current_M[id_update] = self.aug_update_func(
                        X_to_be_updated, M_to_be_updated, u_t, a_t, dW_to_be_updated)

            # Store the state variable X and the Martingale representation M into history
            states[:,:,i] = current_X                   # dim = (n_samples, dim_state + 1)
            martingales[:,i] = current_M                # dim = (n_samples, )
        
        # Add the initial states into the full history
        states = torch.concat([initial_states.unsqueeze(2), states], 
                              dim = 2)                  # dim = (n_samples, dim_state, n_periods + 1)
        martingales = torch.concat([constraint_limits.unsqueeze(1), martingales],
                              dim = 1)                  # dim = (n_samples, n_periods + 1)

        return states, martingales, controls, sto_increments, taus
    

    def forward_merge(self, initial_states, constraint_limits, brownians, w, u_nn, x_u_update_func):
        ''' 
        Inputs : where N := n_periods,
            initial_states : a sample of X at t = 0                 - dim = (n_samples, dim_state)
            constraint_limits : a sample of p at t = 0              - dim = (n_samples, )
            brownians : a sample of dW_x at t=1,...,N               - dim = (n_samples, dim_sto, n_periods)
            w : the function which compute the boundary of the domain at any given point (t,x)
                _ should be able to handle a sample of dim = (n_samples, dim_state + 1)
            x_u_updat_func : update function at the boundary (only update X using u)

        Output : 
            a full history of states and Martingale representation of all the sample points (X_i, M_i)_{i=0,...,N}  
                _ dim = (n_samples, dim_state + 1, n_periods + 1)
            a full list of augmented controls applied to each data point at each time step (u_i, a_i)_{i=1,...,N} 
                _ dim = (n_samples, dim_control + 1, n_periods)
            a list of the stopping time tau for the first time arriving at the boundary for each trajectory 
                _ dim = (n_samples, )
                _ tau is in the form of timestamp (0, T/N, 2T/N, ..., T)
                _ check documentation 4_Training_ValueFunction_global.md for definition and explanation of tau
        
        Note : with this forward, we compute the full history even though we know that once the boundary is reached,
            we switch to the domain boundary case and no longer need any history after this event. We count on tau to 
            send signal of whether the boundary is reach and consequentially, whether the regime (interior or boundary)
            is switched.
        '''
        # Dimensions and device
        device = initial_states.device 

        n_samples = initial_states.shape[0]    # number of data points in the sample / trajectories in the history

        dim_state = initial_states.shape[1]    # dimension of 1 single state variable X
        dim_control = int((dim_state - 1)/2)   # dimension of 1 single control variable u
        dim_sto = int((dim_state - 1)/2)       # dimension of 1 single sto. increment a

        n_periods = self.n_subnetworks         # number of time steps in the time horizon

        # Initialization of the history
        current_X = initial_states.clone()             # dim = (n_samples, dim_state)
        current_M = constraint_limits.clone()          # dim = (n_samples, )
        
        states = torch.empty(size=(n_samples, dim_state, n_periods)).to(device)         # history of X
        martingales = torch.empty(size=(n_samples, n_periods)).to(device)               # history of M
        controls = torch.empty(size=(n_samples, dim_control, n_periods)).to(device)     # history of u
        sto_increments = torch.empty(size=(n_samples, dim_sto, n_periods)).to(device)   # history of a
        taus = torch.full(size=(n_samples,), fill_value=self.T).to(device)  # at t=0, default value is T

        # Loop through the time horizon
        for i, subnet in enumerate(self.subnetworks):   # i = 0 ,..., N-1
            # update the stopping time tau 
                # compute the boundary w(t,x) at each data point (t,x)
            t = i * self.T/n_periods                    # t_i = 0, T/N, ..., T(N-1)/N 
            current_coordinate = torch.concat([torch.full(size=(n_samples, 1), fill_value=t).to(device),
                                               current_X], 
                                               dim = 1) # dim = (n_samples, 1 + dim_state) - merge t and X to pass through w
            current_boundary = w(current_coordinate)    # dim = (n_samples, )
                
                # find the indices of the stopping time that need to be update in the sample
                    # where the boundary was NOT reached before the current time AND it IS reached now
            id_tau_at_t_i = torch.logical_and(taus == self.T, current_M <= current_boundary)
            taus[id_tau_at_t_i] = t                     # note that tau IS a time stamp

            # Conditional update
                # If tau = T, we update the state and Martingale representation
            id_interior = taus == self.T
            id_boundary = taus < self.T

            # Updating states in the interior with a_nn
            if torch.sum(id_interior) != 0:  # if there is at least one data point to update
                X_int = current_X[id_interior]              # dim = (sum(id_interior), dim_state)
                M_int = current_M[id_interior]              # dim = (sum(id_interior), )
                dW_int = brownians[id_interior,:,i]         # dim = (sum(id_interior), dim_sto)
                
                # compute the control for the period using the corresponding subnetwork
                u_t, a_t = subnet(X_int, M_int) # dim = (sum(id_interior), dim_control + dim_sto)
                controls[id_interior,:,i] = u_t                       # dim = (sum(id_interior), dim_control)
                sto_increments[id_interior,:,i] = a_t                 # dim = (sum(id_interior), dim_sto)

                # Fix shape of u and a in the case in which there was only one sample
                if len(a_t.size())<2:
                    a_t = a_t.reshape(1,-1)
                    u_t = u_t.reshape(1,-1)
                    
                # update the state variable X and the Martingale representation M
                current_X[id_interior], current_M[id_interior] = self.aug_update_func(
                        X_int, M_int, u_t, a_t, dW_int)
                
            # Updating states in the boundary with u_nn
            if torch.sum(id_boundary) != 0:  # if there is at least one data point to update
                X_bound = current_X[id_boundary]              # dim = (sum(id_boundary), dim_state)
                M_bound = current_boundary[id_boundary]       # dim = (sum(id_boundary), )
                dW_bound = brownians[id_boundary,:,i]         # dim = (sum(id_boundary), dim_sto)
                
                # compute the control for the period using the corresponding subnetwork
                u_t = u_nn[i](X_bound)                           # dim = (sum(id_boundary), dim_control + dim_sto)
                controls[id_boundary,:,i] = u_t                       # dim = (sum(id_boundary), dim_control)

                # Fix shape of u and a in the case in which there was only one sample
                if len(u_t.size())<2:
                    u_t = u_t.reshape(1,-1)

                # update the state variable X and the Martingale representation M
                current_X[id_boundary] = x_u_update_func(X_bound, u_t, dW_bound)
                current_M[id_boundary] = M_bound

            # Store the state variable X and the Martingale representation M into history
            states[:,:,i] = current_X                   # dim = (n_samples, dim_state + 1)
            martingales[:,i] = current_M                # dim = (n_samples, )
        
        # Add the initial states into the full history
        states = torch.concat([initial_states.unsqueeze(2), states], 
                              dim = 2)                  # dim = (n_samples, dim_state, n_periods + 1)
        martingales = torch.concat([constraint_limits.unsqueeze(1), martingales],
                              dim = 1)                  # dim = (n_samples, n_periods + 1)

        return states, martingales, controls, sto_increments, taus

# AugGlobalNetwork for graphing
class AugGlobalNetwork_extra(AugGlobalNetworks):
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, n_subnetworks,
                 aug_update_func, update_extra_func, u_lower_bounds, u_upper_bounds, a_lower, a_upper,
                 T= 1.0, activation = nn.ReLU(), u_sigmoid_scale = [1e-4, 1e-4, 1e-1], a_sigmoid_scale = 1e-1):
        super().__init__(dim_input, dim_output, dim_hidden, n_hidden, n_subnetworks,
                 aug_update_func, u_lower_bounds, u_upper_bounds, a_lower, a_upper,
                 T, activation, u_sigmoid_scale, a_sigmoid_scale)
        self.update_extra_func = update_extra_func      # additional function to compute the extra info 
        self.T = T

    def forward(self, initial_states, constraint_limits, brownians, w_nn, u_nn, x_u_update_func):
        '''
        Besides rolling forward the state process (X, M) like the forward_merge in its superclass, 
            this function also provides the following extra info :
            - lapse rate (%)
            - financial production (%)
            - wealth (monetary unit)
            - boundary constructed by w_nn
        '''
        # Dimensions and device
        device = initial_states.device 

        n_samples = initial_states.shape[0]     # number of data points in the sample / trajectories in the history

        dim_state = initial_states.shape[1]     # dimension of 1 single state variable X
        dim_control = int((dim_state - 1)/2)    # dimension of 1 single control variable u
        dim_sto = int((dim_state - 1)/2)        # dimension of 1 single sto. increment a
        n_assets = dim_sto - 1                  # number of assets in the portfolio 

        n_periods = self.n_subnetworks          # number of time steps in the time horizon

        # Initialization of the history
        current_X = initial_states.clone()             # dim = (n_samples, dim_state)
        current_M = constraint_limits.clone()          # dim = (n_samples, )
        
        states = torch.empty(size=(n_samples, dim_state, n_periods)).to(device)         # history of X
        martingales = torch.empty(size=(n_samples, n_periods)).to(device)               # history of M
        controls = torch.empty(size=(n_samples, dim_control, n_periods)).to(device)     # history of u
        sto_increments = torch.empty(size=(n_samples, dim_sto, n_periods)).to(device)   # history of a
        taus = torch.full(size=(n_samples,), fill_value=self.T).to(device)  # at t=0, default value is T

        lapses = torch.empty(size=(n_samples, n_periods))       # lapse rate
        fin_prod = torch.empty(size=(n_samples, n_periods))     # financial production
        wealths = torch.empty(size=(n_samples, n_periods))      # wealth
        boundaries = torch.empty(size=(n_samples, n_periods+1)) # boundary built by w(t,X)

        # Loop through the time horizon
        with torch.no_grad():
            for i, subnet in enumerate(self.subnetworks):   # i = 0 ,..., N-1
                t = i * self.T/n_periods                    # t_i = 0, T/N, ..., T(N-1)/N 
                dW_t = brownians[:, :, i]

                # Save the current state before update - needed for computing financial production and lapse rate    
                if i == 0:
                    X_before_update = current_X.clone()         # dim = (n_samples, dim_state)    
                else :
                    X_before_update = states[:, :, i-1].clone() # dim = (n_samples, dim_state)
                
                # Boundary
                current_coordinate = torch.concat([torch.full(size=(n_samples, 1), fill_value=t).to(device),
                                                current_X], 
                                                dim = 1) # dim = (n_samples, 1 + dim_state) - merge t and X to pass through w
                current_boundary = w_nn(current_coordinate)     # dim = (n_samples, )
                boundaries[:,i] = current_boundary.clone()   

                # Updating trajectories that are in the interior with a_nn
                id_interior_at_t = taus == self.T           # trajectories in the interior at the beginning of t_i, before updating    
                id_interior_at_t_X = id_interior_at_t.unsqueeze(1).repeat(1,dim_state)
                id_interior_at_t_W = id_interior_at_t.unsqueeze(1).repeat(1,dim_sto)
                
                if torch.sum(id_interior_at_t) != 0 :        # if there is at least one data point in the interior
                    X_int_temp = torch.where(id_interior_at_t_X, current_X, 0)      # dim = (n_samples, dim_state)           
                    M_int_temp = torch.where(id_interior_at_t, current_M, 0)        # dim = (n_samples, )
                    dW_int_temp = torch.where(id_interior_at_t_W, dW_t, 0)          # dim = (n_samples, dim_sto)
                    
                    # compute candidate for the augmented control
                    u_t_temp, a_t_temp = subnet(X_int_temp, M_int_temp)         # dim = (n_samples, dim_control + dim_sto)
                    
                    # update with the candidate for the augmented control
                    X_temp, M_temp = self.aug_update_func(X_int_temp, M_int_temp, u_t_temp, a_t_temp, dW_int_temp)

                    # check whether the augmented state is still in the interior - if not then have to redo the update for 
                    next_t = (i+1) * self.T/n_periods       # t_(i+1) = T/N, 2T/N, ... T
                    test_state = torch.concat([torch.full(size=(X_temp.size(0),1), fill_value = next_t).to(device), X_temp],
                                            dim = 1)      # dim = (n_samples, dim_state + 1)
                    test_boundary = w_nn(test_state)        # dim = (n_samples, )    
                    
                    # commit the changes for trajectories which are still in the interior after the update with a_nn
                    id_interior = torch.logical_and(id_interior_at_t, M_temp > test_boundary)

                        # update the augmented state process 
                    current_X[id_interior] = X_temp[id_interior]        # update the state variable X
                    current_M[id_interior] = M_temp[id_interior]        # update the martingale M
                        
                        # keep record of the control used
                    controls[id_interior, :, i] = u_t_temp[id_interior]     # dim = (sum(id_interior), dim_control) 
                    sto_increments[id_interior,:,i] = a_t_temp[id_interior] # dim = (sum(id_interior), dim_sto)    
                                            
                    # update the stopping time for trajectories that reach the boundary at t
                    id_reach_boundary_t = torch.logical_and(id_interior_at_t, M_temp <= test_boundary)
                    taus[id_reach_boundary_t] = t           # effectively switching tau = T to tau = t for these trajectories      
                    
                # Updating trajectories that are on the boundary (including those reaching the boundary at t)
                id_boundary = taus <=t 
                assert sum(id_boundary == ~ id_interior).item() == n_samples, f"Problem in filtering at time period {i+1}/{n_periods}: the set of interior points (tau == T) is NOT mutually exclusive with the set of boundary points (tau <=t)"
                assert sum(id_boundary).item() + sum(id_interior).item() == n_samples, f"Problem in filtering at time period {i+1}/{n_periods} : {sum(id_interior).item()} points in the interior (tau == T) + {sum(id_boundary).item()} points on the boundary (tau <= t_{i+1}) but there are {n_samples} in total"
                
                if torch.sum(id_boundary) != 0 :            # if there is at least one data point on the boundary
                    X_boundary = current_X[id_boundary]
                    t_X_boundary = torch.concat([torch.full(size=(X_boundary.size(0),1), fill_value = t).to(device), X_boundary],
                                                dim = 1)    # (t,X) at the boundary - input for w_nn
                    dW_boundary = dW_t[id_boundary]

                    u_t_boundary = u_nn[i](X_boundary)      # controls generated by the u_nn network (since we are on the boundary)
                    controls[id_boundary, :, i] = u_t_boundary

                    if len(u_t_boundary.size()) < 2 :       # fix dimension if there is only 1 sample in this case
                        u_t_boundary = u_t_boundary.reshape(1, -1)
                    
                    current_X[id_boundary] = x_u_update_func(X_boundary, u_t_boundary, dW_boundary)
                    current_M[id_boundary] = current_boundary[id_boundary]
                
                # Keep record of the trajectories (post update)
                states[:, :, i] = current_X     # X at t_i for i = 1,...,N
                martingales[:,i] = current_M    # M at t_i for i = 1,...,N

                # Compute the financial production and lapse rate (does not depend on neither M nor tau)
                fin_prod_t, lapse_t = self.update_extra_func(x=X_before_update.clone(), u=controls[:,:,i].clone())
                lapses[:,i] = lapse_t.squeeze()
                fin_prod[:,i] = fin_prod_t.squeeze()

            #  Add the initial states into the full history
            states = torch.concat([initial_states.unsqueeze(2), states], 
                                    dim = 2)    # dim = (n_samples, dim_state, n_periods + 1)
            martingales = torch.concat([constraint_limits.unsqueeze(1), martingales],
                                    dim = 1)    # dim = (n_samples, n_periods + 1)
             
            # Add the last state to the boundary (not computed in the loop)
            boundaries[:, -1] = w_nn(torch.concat([torch.full(size=(n_samples, 1), fill_value=self.T).to(device),
                                                        states[:, :, -1]], dim = 1))
                
            # Compute wealth trajectories
            wealths = torch.sum(states[:,:n_assets,:] * states[:,n_assets+2:2*n_assets+2, :], dim=1) 
            wealths = wealths + states[:,n_assets+1, :] - states[:, -1, :]  # dim = (n_samples, n_periods + 1) 

        return states, martingales, controls, sto_increments, taus, lapses, fin_prod, wealths, boundaries

class AugSubnetwork_single_model(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, 
                 u_lower_bounds, u_upper_bounds, a_lower, a_upper,
                 activation = nn.ReLU(),
                 u_sigmoid_scale = [1e-4, 1e-4, 1e-1], a_sigmoid_scale = 1e-1):
        '''
        Inputs : 
            dim_input = dim of x + 1 for p = dim_state + 1 = (2 * n_assets + 3) + 1 = 2 * n_assets + 4
            dim_output = dim of u + dim of a = dim_control + dim_sto = (n_assets + 1) + (n_assets + 1) = 2 * n_assets + 2
            dim_hidden = number of neurons per hidden layer
            n_hidden = number of hidden layers in each subnetwork
            u_lower_bounds, u_upper_bounds : lower and upper limit for the control u  
                _ dim = dim_control = n_assets + 1
            a_lower, a_upper: lower and upper limit for each sto. increment a 
                _ dim = dim_sto = n_assets + 1
            u_sigmoid_scale, a_sigmoid_scale : scaling factors for the sigmoid function that handle the scaling of the output
                    to make sure that the output stay within authorized bounds
        '''
        # Initiation
        super(AugSubnetwork_single_model, self).__init__()

        # Create the neural network structure
            # create a list of layers
        layers = []

            # first hidden layer
        layers.append(nn.BatchNorm1d([dim_input]))
        layers.append(nn.Linear(dim_input, dim_hidden))
        layers.append(activation)

            # in-between hidden layers
        for _ in range(n_hidden-1):
            layers.append(nn.BatchNorm1d([dim_hidden]))
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activation)

            # last layer (output layer)
        layers.append(nn.Linear(dim_hidden, dim_output))

            # combine layers into a sequential model
        self.model = nn.Sequential(*layers)

        # Set parameters for the boundary for the output
            # dimension 
        self.n_assets = int((dim_input - 1 - 3)/2) # now has time       
        self.dim_control = self.n_assets + 1    # for u
        self.dim_sto = self.n_assets + 1        # for a


            # boundary for a 
        device = u_lower_bounds.device
        self.aug_lower_bounds = torch.concat([u_lower_bounds, torch.full(size=(self.dim_sto,), fill_value = a_lower).to(device)])
        self.aug_upper_bounds = torch.concat([u_upper_bounds, torch.full(size=(self.dim_sto,), fill_value = a_upper).to(device)])
        self.aug_sigmoid_scale = torch.concat([torch.tensor(u_sigmoid_scale).to(device), torch.full(size=(self.dim_sto,), fill_value = a_sigmoid_scale).to(device)])

    def forward(self, X, M):
        '''
        Input : 
            X : a sample of the state variable X                 - dim = (n_samples, dim_state)   = (n_samples, 2 * n_assets + 3)
            M : a sample of the Martingale representation M of p - dim = (n_sample, )
        Output : 
            a sample of control u for X                          - dim = (n_samples, dim_control) = (n_samples, n_assets + 1)
            a sample of stochastic increment a for M             - dim = (n_samples, dim_sto)     = (n_samples, n_assets + 1)            
        '''
        device = X.device                                               # device            
        scale = torch.Tensor(self.aug_sigmoid_scale).to(device)         # scale for the sigmoid function

        # pass forward and scale : here u and a are together (in the same tensor)
        aug_X = torch.concat([X, M.unsqueeze(1).to(device)], dim = 1)   # dim = (n_samples, dim_state + 1)         = (n_samples, 2 * n_assets + 4)      
        aug_u = self.model(aug_X) * scale                               # dim = (n_samples, dim_control + dim_sto) = (n_samples, 2 * n_assest + 2)   
        aug_u = aug_u.to(device)    

        # set output to be between the boundaries
        aug_u = self.aug_lower_bounds.to(device) + (self.aug_upper_bounds.to(device) - self.aug_lower_bounds.to(device)) * torch.sigmoid(aug_u)
    
        # separate u and a
        u = aug_u[:, :self.dim_control].to(device)                      # dim = (n_samples, dim_control)
        a = aug_u[:, self.dim_control:].squeeze().to(device)            # dim = (n_samples, dim_sto)

        return u, a

class AugGlobalNetworks_single_model(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, n_subnetworks,
                 aug_update_func, u_lower_bounds, u_upper_bounds, a_lower, a_upper,
                 T= 1.0, activation = nn.ReLU(), 
                 u_sigmoid_scale = [1e-4, 1e-4, 1e-1], a_sigmoid_scale = 1e-1):
        '''
        Inputs :
            n_subnetworks : number of subnetwork inside the global network
                each of which is an AugSubnetwork with dim_input, dim_output, dim_hidden, n_hidden, 
                                aug_lower_bounds, aug_upper_bounds, activation, aug_sigmoid_scale
            aug_update_func : an update function that outputs the next state by taking into account
                the augmented state (X,M), the augmented control (u,a), and an augmented brownian (dW_X, dW_p)
            T : terminus of the time horizon
        Output : 
            a global network whose subnetworks represent the strategy for their corresponding time step
        '''
        super(AugGlobalNetworks_single_model, self).__init__()
        self.n_subnetworks = n_subnetworks
        self.aug_update_func = aug_update_func

        # Only one model across all time steps
        self.base_model = AugSubnetwork_single_model(dim_input, dim_output, dim_hidden, n_hidden, 
                                        u_lower_bounds, u_upper_bounds, a_lower, a_upper,
                                        activation, u_sigmoid_scale, a_sigmoid_scale)

        self.subnetworks = nn.ModuleList([
                                            self.base_model
                                            for _ in range(n_subnetworks)
                                        ])
        self.T = T
    
    def __len__(self): # Returns the number of subnetwork in the global network
        return len(self.subnetworks)
    
    def __getitem__(self, index): # Returns the subnetworks at a specified index
        return self.subnetworks[index]
    
    def forward(self, initial_states, constraint_limits, brownians, w):
        ''' 
        Inputs : where N := n_periods,
            initial_states : a sample of X at t = 0                 - dim = (n_samples, dim_state)
            constraint_limits : a sample of p at t = 0              - dim = (n_samples, )
            brownians : a sample of dW_x at t=1,...,N               - dim = (n_samples, dim_sto, n_periods)
            w : the function which compute the boundary of the domain at any given point (t,x)
                _ should be able to handle a sample of dim = (n_samples, dim_state + 1)

        Output : 
            a full history of states and Martingale representation of all the sample points (X_i, M_i)_{i=0,...,N}  
                _ dim = (n_samples, dim_state + 1, n_periods + 1)
            a full list of augmented controls applied to each data point at each time step (u_i, a_i)_{i=1,...,N} 
                _ dim = (n_samples, dim_control + 1, n_periods)
            a list of the stopping time tau for the first time arriving at the boundary for each trajectory 
                _ dim = (n_samples, )
                _ tau is in the form of timestamp (0, T/N, 2T/N, ..., T)
                _ check documentation 4_Training_ValueFunction_global.md for definition and explanation of tau
        
        Note : with this forward, we compute the full history even though we know that once the boundary is reached,
            we switch to the domain boundary case and no longer need any history after this event. We count on tau to 
            send signal of whether the boundary is reach and consequentially, whether the regime (interior or boundary)
            is switched.
        '''
        # Dimensions and device
        device = initial_states.device 

        n_samples = initial_states.shape[0]    # number of data points in the sample / trajectories in the history

        dim_state = initial_states.shape[1]    # dimension of 1 single state variable X
        dim_control = int((dim_state - 1)/2)   # dimension of 1 single control variable u
        dim_sto = int((dim_state - 1)/2)       # dimension of 1 single sto. increment a

        n_periods = self.n_subnetworks         # number of time steps in the time horizon

        # Initiation
        current_X = initial_states.clone()
        current_M = constraint_limits.clone()
        
        states = torch.empty(size=(n_samples, dim_state, n_periods)).to(device)         # history of X
        martingales = torch.empty(size=(n_samples, n_periods)).to(device)               # history of M
        controls = torch.empty(size=(n_samples, dim_control, n_periods)).to(device)     # history of u
        sto_increments = torch.empty(size=(n_samples, dim_sto, n_periods)).to(device)   # history of a
        taus = torch.full(size=(n_samples,), fill_value=self.T).to(device)  # at t=0, default value is T

        # Loop through the time horizon
        for i, subnet in enumerate(self.subnetworks):   # i = 0 ,..., N-1
            # update the stopping time tau 
                # compute the boundary w(t,x) at each data point (t,x)
            t = i * self.T/n_periods                    # t_i = 0, T/N, ..., T(N-1)/N 
            current_coordinate = torch.concat([torch.full(size=(n_samples, 1), fill_value=t).to(device),
                                               current_X], 
                                               dim = 1) # dim = (n_samples, 1 + dim_state) - merge t and X to pass through w
            current_boundary = w(current_coordinate)    # dim = (n_samples, )
                
                # find the indices of the stopping time that need to be update in the sample
                    # where the boundary was NOT reached before the current time AND it IS reached now
            id_update_tau = torch.logical_and(taus == self.T, current_M <= current_boundary)
            taus[id_update_tau] = t                     # note that tau IS a time stamp

            # brownians for the time period
            dW = brownians[:,:,i]                       # dim = (n_samples, dim_sto)

            # compute the augmented control for the period using the current subnetwork
            u_t, a_t = subnet(current_coordinate, current_M)     
            controls[:,:,i] = u_t                       # dim = (n_samples, dim_control)
            sto_increments[:,:,i] = a_t                 # dim = (n_samples, )

            # update the state variable X and the Martingale representation M
            current_X, current_M = self.aug_update_func(current_X, current_M, u_t, a_t, dW) 
            states[:,:,i] = current_X                   # dim = (n_samples, dim_state + 1)
            martingales[:,i] = current_M                # dim = (n_samples, )
        
        # Add the initial states into the full history
        states = torch.concat([initial_states.unsqueeze(2), states], 
                              dim = 2)                  # dim = (n_samples, dim_state, n_periods + 1)
        martingales = torch.concat([constraint_limits.unsqueeze(1), martingales],
                              dim = 1)                  # dim = (n_samples, n_periods + 1)

        return states, martingales, controls, sto_increments, taus
    
    

    
'------- NEURAL NETWORK FOR VALUE FUNCTION ON THE ENTIRE DOMAIN ---------------'
class ValueFunctionNetwork(nn.Module):
    def __init__(self, dim_input, dim_hidden, n_hidden, activation=nn.ReLU(), print_layers = False):
        '''
        Inputs :
            dim_input = dimension of one data point (t,x,p) 
                    = 1 for t + dim_state for x + 1 for p = 1 + (2 * n_assets + 3) + 1
            dim_hidden : number of neurons in each hidden layer
            n_hidden : number of hidden layers
        '''
        super(ValueFunctionNetwork, self).__init__()

        # Initiate a list of layers
        layers = []

        # First hiddden layer
        layers.append(nn.BatchNorm1d(dim_input))
        layers.append(nn.Linear(dim_input, dim_hidden))
        layers.append(activation)

        # In-between hidden layers
        for _ in range(n_hidden-1):
            
            layers.append(nn.BatchNorm1d(dim_hidden))
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(activation)

        # Last layer (output layer) - output is a scalar

        layers.append(nn.BatchNorm1d(dim_hidden))
        layers.append(nn.Linear(dim_hidden, 1))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, Z):
        '''
        Input : Z - a sample of (t,x,p)  - dim = (n_samples, 1 + dim_state + 1) = (n_samples, 2 * n_assets + 5)
        Output : an estimate of V(t,x,p) - dim = (n_samples, )
        '''
        device = Z.device
        z = self.model(Z)
        z = z.squeeze().to(device) # should have dim = (n_samples, )
        return z                    


' -------------- FANCY MODEL ---------------------------------'
### Fancy network for the domain boundary function w


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, num_frequencies=8):
        super(SinusoidalTimeEncoding, self).__init__()
        self.num_frequencies = num_frequencies

    def forward(self, t):
        # t: (batch_size, 1)
        freqs = 2 ** torch.arange(self.num_frequencies, device=t.device) * math.pi
        sin = torch.sin(freqs * t)  # (batch_size, num_frequencies)
        cos = torch.cos(freqs * t)
        return torch.cat([sin, cos], dim=-1)  # (batch_size, 2 * num_frequencies)
    

class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super(LearnableEmbedding, self).__init__()
        self.size = size
        self.linear = nn.Linear(1, size)
 
    def forward(self, x: torch.Tensor):
        return self.linear(x / self.size)
 
    def __len__(self):
        return self.size

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, output_dim=1, norm_type='batch'):
        super(ResidualMLP, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),  
                nn.Linear(hidden_dim, hidden_dim)
            ))
            if norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))
            elif norm_type == 'identity':
                self.norms.append(nn.Identity(hidden_dim))
            elif norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                raise ValueError(f"Unknown normalization type: {norm_type}")

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))  # Residual + LayerNorm
        return self.output_layer(x)  # Output shape: (batch_size, output_dim)

class PortfolioValueNet(nn.Module):
    def __init__(self, num_time_freqs=8, state_dim=7, hidden_dim=128, num_layers=4, time_encoding='sinusoidal', norm_type='batch'):
        super(PortfolioValueNet, self).__init__()
        if time_encoding == 'learnable':
            self.time_encoder = LearnableEmbedding(size= 2 * num_time_freqs)
            input_dim = 2 * num_time_freqs + state_dim
        elif time_encoding == 'sinusoidal':
            self.time_encoder = SinusoidalTimeEncoding(num_frequencies=num_time_freqs)
            input_dim = 2 * num_time_freqs + state_dim
        elif time_encoding == 'none':
            self.time_encoder = nn.Identity()
            input_dim = 1 + state_dim 
        
        self.mlp = ResidualMLP(input_dim=input_dim,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers,
                               norm_type=norm_type)

    def forward(self, Z):
        time_feat = self.time_encoder(Z[:, 0:1])
        x = torch.cat([time_feat, Z[:, 1:]], dim=-1)
        return torch.squeeze(self.mlp(x))  # Output shape: (batch_size,)

class ValueNet(nn.Module):
    def __init__(self, num_time_freqs=8, state_dim=7, hidden_dim=128, num_layers=4, time_encoding='sinusoidal', norm_type='identity'):
        super(ValueNet, self).__init__()
        if time_encoding == 'learnable':
            self.time_encoder = LearnableEmbedding(size = 2 * num_time_freqs)
            input_dim = 2 * num_time_freqs + state_dim + 1           # add 1 for the martingale representation
        elif time_encoding == 'sinusoidal':
            self.time_encoder = SinusoidalTimeEncoding(num_frequencies=num_time_freqs)
            input_dim = 2 * num_time_freqs + state_dim + 1
        elif time_encoding == 'none':
            self.time_encoder = nn.Identity()
        
        self.mlp = ResidualMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, norm_type=norm_type)

    def forward(self, Z):
        '''
        Input : 
            Z : a sample of (t,x,p) - dim = (n_samples, dim_state + 2)
        Output : 
            the corresponding value function V(t,x,p) - dim = (n_samples, )
        '''
        time_feat = self.time_encoder(Z[:, 0:1])
        x = torch.cat([time_feat, Z[:, 1:]], dim = -1)
        return torch.squeeze(self.mlp(x))   # dim = (n_samples, )

class ControlPolicyNet(nn.Module):
    def __init__(self, num_time_freqs=8, state_dim=7, control_dim=3, hidden_dim=128, num_layers=4, lower_bounds=None, upper_bounds=None, 
                 sigmoid_scale=[1e-4, 1e-4, 1e-1]):
        super().__init__()
        self.time_encoder = SinusoidalTimeEncoding(num_frequencies=num_time_freqs)
        self.mlp = ResidualMLP(input_dim=2 * num_time_freqs + state_dim,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers, output_dim=control_dim)
        
        self.lower_bounds = lower_bounds if lower_bounds is not None else torch.tensor([-1.0, -1.0, -1.0])
        self.upper_bounds = upper_bounds if upper_bounds is not None else torch.tensor([1.0, 1.0, 1.0])
        self.sigmoid_scale = sigmoid_scale

    def forward(self, t, X):
        time_feat = self.time_encoder(t)  # (batch_size, 2*num_freqs)

        x = torch.cat([time_feat, X], dim=-1)
        hidden = self.mlp(x)

        scale = torch.Tensor(self.sigmoid_scale).to(X.device) # scale the sigmoid function to the range of the outputs
        
        u = hidden * scale

        # Set the outputs of the neural network into admissible range (between lower bounds and upper bounds)
        device = u.device
        u = self.lower_bounds.to(device) + (self.upper_bounds.to(device) - self.lower_bounds.to(device))*torch.sigmoid(u)

        return torch.squeeze(u)  # control output
   


class GlobalFancy(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, n_hidden, n_subnetworks, 
                 update_func, lower_bounds, upper_bounds, 
                 sigmoid_scale=[1e-4, 1e-4, 1e-1]):

        super(GlobalFancy, self).__init__()
        self.n_subnetworks = n_subnetworks
        self.update_func = update_func
        self.subnetworks = ControlPolicyNet(num_time_freqs=8, state_dim=dim_input, control_dim=dim_output, hidden_dim=dim_hidden, num_layers=n_hidden, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                                            sigmoid_scale=sigmoid_scale)
        
        self.dim_output = dim_output
        
    def __len__(self): # Returns the number of subnetwork in the global network
        return self.n_subnetworks
    
    def __getitem__(self, index): # Returns the subnetwork at index i
        return self.subnetworks
    
    def forward(self, initial_states, brownians):
        '''
        Input : 
            initial_states : a sample of X at t=0               - dim = (n_samples, 2 * n_assets + 3)
            brownians : a sample of all the brownian motions dW - dim = (n_samples, n_assets + 1, n_periods) 
        Outputs : 
            a tensor of dim = (n_samples, 2 * n_assets + 3, n_periods) representing a list of all states X_1, ..., X_N (N = n_periods) 
            a tensor of dim = (n_samples, n_assets + 1) reprenting a list of all controls u_1, ..., u_N (N = n_periods)
        '''
        # dimensions 
        n_samples = initial_states.shape[0] # number of data points in the sample
        dim_state = initial_states.shape[1] # dimension of 1 single state variable 
        n_periods = self.n_subnetworks      # number of periods

        dim_control = int((dim_state - 1)/2)# dimension of 1 single control variable
        
        # Initiation
        current_state = initial_states.clone()
        states = torch.empty(size=(n_samples, dim_state, n_periods))
        controls = torch.empty(size=(n_samples, dim_control, n_periods))

        # Loop through the time horizon
        for t in range(self.n_subnetworks):

            # compute the control for the period using the corresponding subnetwork
            t_input = torch.tensor(float(t/self.n_subnetworks), device=initial_states.device).unsqueeze(0).repeat(n_samples, 1)

            control_t = self.subnetworks(t_input, current_state)   # dim = (n_samples, dim_control)
            controls[:,:,t] = control_t
            
            # brownian for the period
            dW_t = brownians[:,:,t]             # dim = (n_samples, n_assets + 1)

            # apply the control to update the state
            current_state = self.update_func(x=current_state, u=control_t, dW=dW_t)

            # store the new state
            states[:,:,t] = current_state

        return states, controls
    
