# III. Estimate of the Value Function at the Domain Boundary

## 1. Training of the Value Function $\hat{\mathcal V}_\theta$ at the boundary of the viable domain


This algorithm uses PINN to estimate the value function $\mathcal V$ at the boundary of the viable domain. Note that this training requires only the optimal control $\hat u^\theta$ and not the domain boundary value $\hat w_\theta$.

Generally, the value function takes 3 arguments : the time variable $t$, the space variable $x$, and the constraint limit $p$ (we refer to the article for more detailed description of the mathematic framework). 
The viable domain is defined as 
$$\mathcal D := \{(t,x,p) \in [0,T] \times \mathbb R^d \times \mathbb R : p \geq w(t,x) \} $$
whose (parabolic spacial) boundary is 
$$\partial _P \mathcal D := \{(t,x,w(t,x)) : (t,x) \in [0,T] \times \mathbb R^d \}$$

Since the constraint limit variable $p$ is completely deterministic based on $(t,x)$ at the boundary $\partial \mathcal D$, the value function $\mathcal V$ at the domain boundary is thus a function of only $t$ and $x$. Furthermore, under the assumptions that 
1. The optimal control at each point $(t,x)$ is unique; in other words, $\mathrm U(t,x)$ is singleton for all $(t,x)$ in the domain,
2. The trained $\hat u^\theta$ is a good estimate of the true optimal control process,
3. (referring to the article for more explanation) $y=0$ and hence, omitted in the training,

the value function $\mathcal V$ should satisfy the PDE 
$$
\mathcal L^{\hat u_i}_X \mathcal V(t_i, x) = 0
$$
for any $t_i \in \{t_0=0, t_1, ...,t_{N-1}\}$, $x \in \mathbb R^d$, and $\hat u_i = \hat u^\theta[i] (t_i,x)$ which is the estimate of the optimal control at time step $t_i$ computed by the neural network $\hat u^\theta$ at the point $(t_i,x)$. Additionaly, $\mathcal V$ must also satisfy 
$$
\mathcal V(T, \cdot) = F(\cdot) \text{ on } \mathbb R^d
$$
where $F$ is the utility function to be maximized at the end of the time horizon.

Following this theoretical requirements, we define the loss function for any value network $\mathcal V_\theta$ as follow

$$\mathbf L^{\mathcal V}(\mathcal V_\theta| \hat u^\theta, \mathcal X_0, \Delta \mathcal W) := \frac{1}{M} \sum_{j = 1}^M \frac{1}{N} \sum^{N-1}_{i=0} \bigl| \mathcal L^{\hat u_i^j}_X \mathcal V_\theta(t_i, \hat X^j_{t_i}) \bigr|^2 + \frac{1}{M} \sum_{j=1}^M \bigl| \mathcal V_\theta (T, \hat X^j_T) - F(\hat X^j_T) \bigr|^2
$$ 
for a sample of initial states and brownian increment trajectories $(\mathcal X_0, \Delta \mathcal W) = (x_0^j, (\Delta W_{t_i}^j)_{i=1\rightarrow N})^{j=1\rightarrow M}$ which is passed through the optimal control neural networks $\hat u^\theta$ to obtain the state trajectories $\left( (t_i, \hat X^j_{t_i})_{i=1\rightarrow N}\right)^{j=1,..M}$ and their corresponding optimal control processes $\left((\hat u^j_i)_{i=0\rightarrow N -1}\right)^{j=1\rightarrow M}$. As in the previous step, $\mathcal L^u_X $ is the Dynkin operator.

The training process for $\hat{\mathcal V}_\theta$ is as follow :
1. Generate a sample of initial states and Brownian increment trajectories $(\mathcal X_0, \Delta \mathcal W) = (x_0^j, (\Delta W_{t_i}^j)_{i=1\rightarrow N})^{j=1\rightarrow M}$.
2. Pass the sample through $\hat u^\theta$ to obtain $\left( (t_i, \hat X^j_{t_i})_{i=0\rightarrow N}\right)^{j=1\rightarrow M}$ and their corresponding optimal control processes $\left((\hat u^j_i)_{i=0\rightarrow N -1}\right)^{j=1\rightarrow M}$.
3. For each training epoch :
- Compute the loss function $\mathbf L^{\mathcal V}$ as defined above
- Take gradient descent step and adjust the learning rate if necessary
- Periodically compute the loss over a validation sample

## 2. Validation for the estimation
Following the same logic as in Section II.3, we propose the following error measure for $\hat{\mathcal V}_\theta$. More specifically, given a sample of trajectories $\hat \chi := (t_i, \hat X^j_{t_i})^{j = 1 \rightarrow M}_{i=0 \rightarrow N}$, we define 
$$
\mathcal E^{\mathcal V} (\hat u^\theta, \hat{\mathcal V}_\theta| \hat \chi) := \frac{\frac1M \sum^M_{j=1}\left[\frac1N\sum^{N-1}_{i=0} \varepsilon^{\mathcal V}_{\mathcal V}(t_i, \hat X^j_{t_i}) + \varepsilon^T_{\mathcal V}(\hat X^j_T)\right]}{\frac1M\sum^M_{j=1}\hat{\mathcal V}_\theta(t_0, \hat X_{t_0}^j)}
$$
where for any $(t,x)\in \{t_0,t_1, ...t_N\}\times \R^d$, we have 
$$\varepsilon^{\mathcal V}_{\mathcal V}(t,x) := \Bigl|\mathcal L^{\hat u^\theta} \hat{\mathcal V}_\theta(t,x) \Bigr| $$
$$ \varepsilon^T_{\mathcal V}(x) := \Bigl| \hat{\mathcal V}_\theta (T,x)- F(x)\Bigr|$$

Then, the computation process of this joint measure is descibed below :
1. Generate a sample of initial states and Brownian trajectories $(x_0^j, (\Delta W^j_i)_{i=1 \rightarrow N})^{j = 1 \rightarrow M}$.
2. Pass this sample through $\hat u^\theta$ to retrieve the extended sample of $\hat \chi = (t_i, \hat X_{t_i}^j)^{j = 1 \rightarrow M}_{i = 1 \rightarrow N}$ and pass this extended sample through $\hat w_\theta$ to compute the gradients $\partial_t, \partial_X$ and the hessian $\partial_{XX}$. 
3. Compute $\varepsilon^{\mathcal V}_{\mathcal V}, \varepsilon^T_{\mathcal V}$ at each trajectory in $\hat \chi$ and $\hat{\mathcal V}_\theta(0, \cdot)$ at each data point in $X_0$.
4. Compute the error measure $\mathcal E^{\mathcal V}$.


