# II. Estimation of the Optimal Control Process and the Domain Boundary
## 1. Training the Optimal Control Process Neural Network $\hat u^\theta = (\hat u^\theta_{t_i})_{i = 0,...,N-1}$ 


Our goal is to use one neural network to estimate the entire optimal control process. To achieve this objective, we create a "nested" neural network consisting of $N$ consequential sub-networks, each being a fully connected network with `n_hidden` hidden layers and `n_neuron` neurons per layer. Note that for each time period $t_i$, the optimal control $u^*_{t_i}$ is estimated by the sub-network $\hat u^\theta_{t_i}$ which takes as input the current state $X_{t_i}$ and outputs the control to be applied $u_{t_i}$. This implies that our nested neural network $\hat u^\theta$ takes as input the initial states of the portfolio $X_0$, estimates the control taken at each time step $t_i$ in the (discretized) timeline, and rolls forwards the portfolio states until the end of the time horizon. Note that each subnetwork outputs the estimate of the (1-period) optimal control $\hat u^\theta_{t_i}$, but the nested global network $\hat u^\theta$ outputs the trajectory of the state variable across the time horizon $(\hat X_{t_i})_{i=1, ...N} = (X^{0,X_0, \hat u^\theta}_{t_i})_{i=1,...N}$ after applying the control estimated by the subnetworks. 

The loss function for this training is a composite loss defined by 
$$
\mathcal{L}^u (u^\theta|X_0, \Delta W) := \frac{1}{M} \sum_{j=1}^M \left[ G(\hat X^j_T) + \sum^n_{i=1} \sum_{k=1}^3 \left( \lambda_k g^k(\hat X^j_{t_i})\right) \right]
$$
where $M$ = `n_samples` is the number of data points in the sample of initial states $X_0 = (X_0^j)^{j=1,...M}$ and in the sample of Brownian increments $\Delta W = (\Delta W^j_{t_i})^{j=1,...M}_{i=1,...N}$ while $\lambda_k$ is the regularization coefficient corresponding to the penalty function $g^k$ (as defined above in the **Portfolio modelization** Section).

The training proceeds as follow : 
1. Initiate the global neural network with chosen hyper-parameters.
2. Generate a training sample of initial states $X_0 = (x_0^j)^{j=1 \rightarrow M}$ and a training sample of Brownian trajectories $\Delta W = (\Delta W^j_{t_i})^{j=1 \rightarrow M}_{i=1 \rightarrow N}$. 
3. Pass the training samples $X_0$ and $\Delta W$ through the global neural network to get the rolled-forward trajectories of the state variables $\hat X =(\hat X^j_{t_i})^{j=1 \rightarrow M}_{i=1 \rightarrow N}$.
4. Compute the loss function upon $\hat X$ and take a gradient descend step; adjust the learning step if necessary.
5. Repeat steps 3 and 4 until reaching the fixed number of epoch `n_epoch`.

Note that since this is unsupervised learning, there is no inherent validation method for such training. We defer the error measuring to a later section. 

## 2. Training the Domain Boundary Value Neural Network $\hat w_\theta$
*Corresponding modules :* `neural_networks.py`, `train_domain_boundary_value_v3.py`, `utils.py`

We employ the Physics-Informed Neural Network for this estimation of $w$. 

More conscretely, we know that the true domain boundary $w: (t,x) \mapsto w(t,x)$ must satisfy 
$$
w(T, x) = G(x) \forall x \in \mathbb R^q
$$
and
$$
\inf_{u \in U} \mathcal L^u w(t,x) + g(x) = 0, \forall (t,x) \in [0,T) \times \mathbb R^q
$$ 
where $q$ = `dim_state` and $\mathcal L^u \varphi$ is the Dynkin operator defined as
$$
\mathcal L^u \varphi (t,x) := \partial_t \varphi(t,x) + \mu(x,u)^\top \partial_X \varphi(t,x) + \frac{1}{2} Tr [(\sigma \sigma^\top)(x) \partial_X^2 \varphi(t,x)]
$$
for any $\varphi$ smooth. This implies that if estimated optimal control network $\hat u^\theta = (\hat u^\theta_{t_i})_{i=0,...,N-1}$ is well-trained, then we want our estimate $\hat w_\theta$ to satisfy
$$
\mathcal L^{\hat u^\theta_i} \hat w_\theta (t_i, x) + g(x) \approx 0
$$ 
for any $i = 0,...,N-1$ and $x \in \mathbb R$ possible. Note that in our specific model, the penalty only depends on the space variable $x$ and not on the control $u$. 

This means that the loss function for a neural network $w_\theta$ is defined as 
$$
\mathbf L^w(w_\theta|\mathcal T,\hat{\mathcal X}) := \frac{1}{M} \sum_{j=1}^M \frac{1}{N} \sum_{i=0}^{N-1} \left| \mathcal L^{\hat u^\theta_i} w_\theta (t_i, \hat X^j_{t_i}) + g(\hat X^j_{t_i}) \right|^2 + \frac{1}{M} \sum_{j=1}^M \left| w_\theta(T, \hat X^j_T) - G(\hat X^j_T) \right|^2
$$
for a set of data $(\mathcal T,\hat{\mathcal X}) = (t_i, \hat X^j_{t_i})_{i=1,...N}^{j=1,...M}$, which is obtained by passing $(x_0^j, (\Delta W_{t_i})_{i=1,...N})^{j=1,...M}$ through $\hat u^\theta$. The closer to 0 the loss function is, the better. Note that this loss function requires a computation of the gradients and the hessian of a neural network, which is not readily available in `pytorch`. A function is hence defined in `utils.py` to handle this calculation.

The training process is as follow :
1. Generate a sample of initial states and Brownian increment trajectories $(\mathcal X_0, \Delta \mathcal W) = (x_0^j, (\Delta W_{t_i})_{i=1,...N})^{j=1,...M}$.
2. Pass the sample through $\hat u^\theta$ and transform the result to a sample of the form $(\mathcal T,\hat{\mathcal X}) = (t_i, \hat X^j_{t_i})_{i=1,...N}^{j=1,...M}$.
3. Separate the sample into 2 data sets : `interior_data` with data points whose time dimension $t < T$ and `terminal_data` with data points whose time dimension $t = T$. 
4. For each training epoch:
- Compute the PDE-guided loss (Dynkin plus penalty) for the `interior_data`  
- Compute the MSE loss between predictions from $w_\theta$ and $G(\hat X_T)$ for the `terminal_data`
- Add the losses
- Take gradient descent step and adjust the learning rate if necessary. 
- Periodically compute the loss over the validation sample.

## 3. Error measures for the estimations

Observe that theoretically, the true optimal control process $\nu^* = (\nu^*_t)_{t \in [0,T]}$ and the true domain boundary value $w$ must satisfy :
$$w(0,\cdot) = \mathbb E \left[ G(X^{0, \cdot, \nu^*}_T) + \int_0^T g(X^{0, \cdot, \nu^*}_s) ds \right]$$
on $\mathbb R^d$, or in Stochastic Differential Equation (SDE) form, 
$$
0 = \mathcal L^{u^*} w(t,x) + g(x)
$$ 
holds for all $(t,x) \in [0,T] \times \mathbb R^d$ and $u^* = \nu^*_t(t,x)$ where $ \mathcal L^{u} w(t,x)$ is the Dynkin operator as defined above. However, in practice, since our neural networks $\hat u^\theta$ and $w_\theta$ are estimations, there exists some additional error term $h$ such that
$$
\hat w_\theta (0,x) = \mathbb E \left[ G(X^{0,x,\nu^*}_T) + \int_0^T g(X^{0, \cdot, \nu^*}_s) ds + \int^T_0 h(s,X^{s,x, \hat u^\theta_{s-}},\hat u^\theta)dt \right]
$$ 
From the SDE point of view, we consider the decomposition  
$$
\mathcal L^* \hat w_\theta + g := \inf_{u \in \mathcal U} \mathcal L^u \hat w_{\theta} + g = \left(\inf_{u \in \mathcal U} \mathcal L^u \hat w_\theta - \mathcal L^{\hat u^\theta} \hat w_\theta \right) + \left( \mathcal L^{\hat u^\theta} \hat w_\theta + g \right)
$$
which naturally leads to the definition of the following error measures: the optimal control error $\varepsilon^u$ and the domain boundary value error $\varepsilon^w$ which are defined as 
$$\varepsilon^u_w(t,x) := \Bigl|\mathcal L^* \hat w_\theta (t,x) - \mathcal L^{\hat u^\theta} \hat w_\theta(t,x) \Bigr| \\
\varepsilon^w_w(t,x) := \Bigl| \mathcal L^{\hat u^\theta} \hat w_\theta (t,x) + g(t,x)\Bigr|\\
\varepsilon^T_w(x) := \left| \hat w_\theta(T, x) - G(x)\right| $$
for any $(t,x)$. Thus, for a sample of $\hat \chi = (t_i, \hat X_{t_i}^j)^{j = 1 \rightarrow M}_{i = 1 \rightarrow N}$, the joint error measure for the estimations of $\hat u^\theta$ and $\hat w_\theta$ can be defined as
$$
\mathcal E^w(\hat u^\theta, \hat w_\theta|\hat \chi) := \frac{\frac{1}{M}\sum^M_{j=1} \left[ \frac{1}{N}\sum^{N-1}_{i=0}(\varepsilon^u_w + \varepsilon^w_w)(t_i, \hat X_{t_i}^j) + \varepsilon^T_w(\hat X^j_T)\right]}{\frac{1}{M}\sum^M_{j=1} \hat w_\theta(0,\hat X_{t_0}^j)} 
$$

Accordingly, the computation process of this joint measure is descibed below :
1. Generate a sample of initial states and Brownian trajectories $(x_0^j, (\Delta W^j_i)_{i=1 \rightarrow N})^{j = 1 \rightarrow M}$.
2. Pass this sample through $\hat u^\theta$ to get a sample of trajectories $\hat \chi = (t_i, \hat X^j_{t_i})^{j = 1 \rightarrow M}_{i = 0 \rightarrow N}$ with the associated control sample $(\hat u^j_{t_i})^{j = 1 \rightarrow M}_{i = 0 \rightarrow N-1}$
3. Pass the trajectory sample through $\hat w_\theta$ to compute the gradients $\partial_t, \partial_X$ and the hessian $\partial_{XX}$. 
3. Compute $\varepsilon^u_w, \varepsilon^w_w, \varepsilon^T_w$ at each data point in $\hat \chi$ and $\hat w_\theta(0, \cdot)$ at each data point in $X_0$.
4. Compute the error measure $\mathcal E^w$.
