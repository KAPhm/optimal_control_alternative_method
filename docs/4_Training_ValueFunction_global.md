# IV. Estimate of the Value Function $V$ over the entire viable domain

## 1. Training the Augmented Optimal Control Process Neural Network $(\tilde u^\theta, \tilde a^\theta)$

Unlike the case of the optimal control at the boundary of the viable domain, the control process for the interior of the viable domain (which will be called the augmented control process from now on) includes the stochastic increment process $a = (a_t)_{t \in [0,T]}$ for the Martingale representation $M^a_{t,p}$ for the constratint limit $p$. This necessitates a new control network with an augmented structure to include the new control $a$ besides the control $u$. In our discretized framework, the dynamic for the Martingale representation becomes
$$
M_{t_{i+1}} = M_{t_i} + a_{t_i} \Delta W^M_{t_{i+1}}
$$
Note that the Martingale representation $M$ uses the a 1-dimensional Brownian movements $W^M$ which is independent of the multi-dimensional Brownian $W^X$ for $X$. Additionally, we know that in theory, each stochastic increment $a_t$ is unbounded which poses problem to our numerical simulation. However, as describe in the accompanying theoretical paper, an asymptotical bounded estimation $a^{\mathcal N} = - \mathcal N \vee a \wedge \mathcal N$ for some finite $\mathcal N > 0$ can be used to replace the true unbounded $a$ with little compromise of accuracy. Henceforth, for this reason, we will assume this bounded version in our training; for ease of notation, we denote the stochastic increment as $a$ in our algorithm description, but it should be understood implicitly that it is $a^{\mathcal N}$.

Understanding that at the boundary $\partial_P \mathcal D$, the value function $V$ behaves differently (different PDE characterization) and that $\partial_P \mathcal D$ is absorbant (meaning once reaching the boundary, the trajectory of the augmented state variable $(X, M)$ remains on the boundary until the end of the time horizon), it is crucial for us to keep track of the stopping time $\tau$ which is the first moment of reaching (or crossing) the boundary $\partial_P \mathcal D$; more concretely, for any given trajectory of the augmented state variable $(X,M)$ starting out from $(x,p)$ at time $t$ under the augmented control $(u, a)$, we define the boundary hitting time as 
$$ 
\tau := \inf \{s : M^a_{t,p} (s) \leq w(s, X^u_{t,x}(s))\} \wedge T
$$ 
The discrete equivalence $\tilde \tau$ for our framework is 
$$
\tilde \tau := \inf \{t_i : M_{t_i} \leq \hat w_\theta(t_i, X_{t_i})\} \wedge T
$$

To guarantee the domain boundary is absorbing, as noted in the associated paper, we impose that if a trajectory $(\tilde X^j, \tilde M^j)$ reaches the domain boundary at $\tilde \tau < T$, i.e the trajectory reaches the boundary before the end of the time horizon, then from this point onward, it must be that $\tilde M^j_{t_i} = \hat w_\theta(t_i, \tilde X^j_{t_i})$ and $\tilde u^\theta_{t_i} = \hat u^\theta_{t_i}$. Practically, we generate the trajectory for $(\tilde X, \tilde M)$ as follow
- if $\tilde \tau = T$, meaning that the trajectory is inside the viable domain during the entire time horizon
$$
(1) \hspace{.2cm} \tilde X_{t_{i+1}} = \tilde X_{t_i} + b(\tilde X_{t_i}, \tilde u^\theta_{t_i}) \Delta t + \sigma(\tilde X_{t_i}) \Delta W^X_{t_{i+1}} \\
(2) \hspace{.2cm} \tilde M_{t_{i+1}} = \tilde M_{t_i} + \tilde a^\theta_{t_i} \Delta W^M_{t_{i+1}} 
$$

- if $\tilde \tau < T$, meaning that the trajectory reaches the domain boundary in the middle of the time horizon, then for any $t_i < \tau$, we follow the same Euler scheme as in $(1)-(2)$. On the other hand, if $t_i \geq \tilde \tau$, then the controls becomes $\tilde u^\theta_{t_i} = \hat u^\theta_{t_i}(t_i, \tilde X_{t_i})$, the state trajectory for $\tilde X$ still follows the update scheme as in $(1)$, and the martingale representation becomes $\tilde M_{t_i} = \hat w^\theta(t_i, \tilde X_{t_i})$. This implies that for any $t_i > \tilde \tau$, the state process $(\tilde X, \tilde M)$ is on the boundary. 

Additionally, given a sample of initial states $X_0 = (x_0^j)^{j=1 \rightarrow M}$, brownian trajectories $\Delta W = (\Delta W^j_{t_i})^{j=1 \rightarrow M}_{i=1 \rightarrow N}$, and contraint limits $p = (p^j)^{j=1 \rightarrow M}$, let $(\tilde X^j_{t_i}, \tilde M^j_{t_i})^{j=1\rightarrow M}_{i=1\rightarrow N}$ denote the augmented state process starting under the control  from the data point $(x_0^j, p^j)$ at $t_0 = 0$  as follow :

$$
\mathbf L^{u, a} \left((u_\theta, a_\theta)| X_0, \Delta W, p, \right) := \frac{1}{M} \sum^M_{j=1} \left[ F(\tilde X_T^j) \right]
$$


The training proceeds similiar to that of $\hat u^\theta$ with the additional step of checking for the hitting time $\tilde \tau$ :
1. Generate a sample of initial states $X_0 = (x_0^j)^{j=1 \rightarrow M}$ and brownian trajectories $\Delta W = (\Delta W^j_{t_i})^{j=1 \rightarrow M}_{i=1 \rightarrow N}$. Using the trained boundary value network $\hat w_\theta$, generate a corresponding sample of constraint limits $p = (p^j)^{j=1 \rightarrow M}$ such that for each data point $j$, $p^j \geq \hat w(t_0, x_0^j)$.
2. Pass the sample through the proposed network $(\tilde u^\theta, \tilde a^\theta)$ to retrieve the full trajectories $(X^j_{t_i}, M^j_{t_i})^{j=1 \rightarrow M}_{i=1 \rightarrow N}$ of the sample. We compute $\tilde \tau^j$ accordingly for each trajectory $j$.
3. For each epoch :
- Compute the loss $\mathbf L^{u,a}$
- Take gradient descent step and adjust the learning rate if necessary
- Periodically compute the loss over a validation sample

## 2. Training the Value function $V$ over the entire viable domain

We are going to apply PINN once again for this training. We recall that for a given augmented control $(u_t, a_t)$ at time $t \in [0,T]$, the Dynkin operator $\mathcal L^{u,a}_{X, M}$ for any smooth function $\phi : (t,x,m) \in [0,T] \times \mathbf R^{d+1} \mapsto \phi(t,x,m) \in \mathbf R$ is defined as follow : 
$$
\mathcal L^{u,a}_{X, M} \phi(t,x,m) := \partial_t \phi(t,x,m) + \mu(x,u)^\top \partial_X \phi(t,x,m) + \frac{1}{2} \text{Tr} \left[(\tilde \sigma \tilde \sigma^\top) (x,u,a) \partial^2_{X,M} \phi(t,x,m) \right] 
$$
where $\partial_t, \partial_X, \partial_M$ are respectively gradients of $\phi$ in terms of its first, second, and last argument, and $\partial^2_{X,p}$ its Hessian with regards to its second and last arguments. Moreover, we have $\tilde \sigma (x,u,a) := \left[\begin{array}{c}\sigma(x,u) \\ a\end{array}\right]$. Note that there is no drift term for the process $M$ because it is a Martingale. 

Then, assuming that we have well-trained $(\tilde u^\theta, \tilde a^\theta)$ to be the optimal augmented control process in the viable domain's interior and $w_\theta$ to be the boundary value of the viable domain, given the PDE of the theoretical $V$, we deduce the discretized equivalence $\tilde V_\theta$ should be satisfying as follow :
- in $int \mathcal D$, for any $(t_i, \tilde X_{t_i}, \tilde M_{t_i}) \in \{ (t,x,m) \in \{t_0,...t_{N-1}\} \times \mathbf R^d \times \mathbf R : m > w_\theta(t,x) \}$ and $(\tilde u_i, \tilde a_i) = (\tilde u^\theta_{t_i}, \tilde a^\theta_{t_i})(t_i,\tilde X_{t_i}, \tilde M_{t_i})$ :
$$\mathcal L^{\tilde u_i, \tilde a_i}_{X,M} \tilde V_\theta(t_i, \tilde X_{t_i}, \tilde M_{t_i}) - g(\tilde X_{t_i},\tilde u_i) \partial_M \tilde V_\theta(t_i, \tilde X_{t_i}, \tilde M_{t_i}) = 0 
$$
- on the boundary $\partial \mathcal D$ where $(t_i, \tilde X_{t_i}, \tilde M_{t_i}) \in \{ (t,x,m) \in \{t_0,...t_{N-1}\} \times \mathbf R^d \times \mathbf R : m = w_\theta(t,x) \}$ : $$\tilde V_\theta (t_i, \tilde X_{t_i}, \tilde M_{t_i}) = \tilde{\mathcal V}_\theta(t_i, \tilde X_{t_i})$$
- at the terminal where $t_i = t_N = T$ and $\tilde M_T \geq p$:
$$\tilde V_\theta (T, \tilde X_T, \tilde M_T) = F(\tilde X_T)$$

Note that the data points $(t_i, \tilde X_{t_i}, \tilde M_{t_i})$ are generated by following the augmented control process given by $(\tilde u^\theta, \tilde a^\theta)$ starting from some given initial state $(x_0,p)$ as long as the process stays within the interior of the domain. As known from the theoretical framework of the problem, once the boundary is reached, the state process fully absorbed into the boundary, meaning that Martingal representation $M$ is always equal to the boundary value $w$, and the state variable $X$ is under the optimal control process $\hat u^\theta$ until the end of the time horizon. Please consult with the related article for further explanation.

Normally, we would generate data by applying the augmented control process $(\tilde u^\theta, \tilde a^\theta)$ from a randomly generated sample of initial states $\mathcal X_0 = (x_0^j)^{j=1,...M}$, constraint limits $\mathcal P_0 = (p^j)^{j=1,...M}$, and brownian increments $\Delta \mathcal W = (\Delta W_{t_i})_{i=1,...N}^{j=1...M}$ to obtain a sample of full trajectories $(t_i, \tilde X^j_{t_i}, \tilde M^j_{t_i})$ and the associated control process $(\tilde u^j_{t_i}, \tilde a^j_{t_i})^{j=1\rightarrow M}_{i=0\rightarrow N-1} = ((\tilde u^\theta_{t_i}, \tilde a^\theta_{t_i})(t_i, \tilde X^j_{t_i}, \tilde M^j_{t_i})_{i=0\rightarrow N-1})^{j=1\rightarrow M} $. We recall that by design, the trajectories generated by $(\cdot, \tilde X, \tilde M)$ is in the viable domain. The loss function is hence 

$$
\mathbf L^V(V_\theta|\tilde {\mathcal V}_\theta, \hat w_\theta, (\tilde u, \tilde a), \mathcal X_0, \Delta \mathcal W_0, \mathcal P_0) :=
\frac1M\sum^{M}_{j=1} \frac1N \sum^{N-1}_{i=0} \left|\mathcal L^{\tilde u^j_i, \tilde a^j_i}_{X,M} \tilde V_\theta(t_i, \tilde X^j_{t_i}, \tilde M^j_{t_i}) - g(\tilde X^j_{t_i},\tilde u^j_i) \partial_M \tilde V_\theta(t_i, \tilde X^j_{t_i}, \tilde M^j_{t_i})\right|^2  \\ +\frac1M \sum^M_{j=1} \frac1N \sum^{N-1}_{i=0} \left|V_\theta(t_i, \tilde X^j_{t_i}, \hat w_\theta(t_i, \tilde X^j_{t_i})) - \tilde {\mathcal V}_\theta(t_i, \tilde X^j_{t_i})\right|^2  
+ \frac{1}{M} \sum^{M}_{j=1} |V_\theta(T, \tilde X^j_T, \tilde M^j_T) - F(\tilde X^j_T)|^2
$$

The training process is executed as follow :
1. Generate a sample of initial states and Brownian increment trajectories $(\mathcal X_0, \Delta \mathcal W) = (x_0^j, (\Delta W_{t_i}^j)_{i=1\rightarrow N})^{j=1\rightarrow M}$ and a sample of initial constraints $\mathcal P_0=(p_0^j)^{j=1\rightarrow M}$ such that for each $j$, $p_0 \geq \hat w_\theta(0, x^j_0)$. 
2. Pass the sample through $(\tilde u^\theta, \tilde a^\theta)$ to obtain the trajectories $\left( (t_i, \tilde X^j_{t_i}, \tilde M^j_{t_i})_{i=0 \rightarrow N}\right)^{j=1\rightarrow M}$ and their corresponding optimal control processes $\left((\hat u^j_i)_{i=0,...,N-1}\right)^{j=1,...M}$.
3. For each training epoch :
- Compute the loss function $\mathbf L^{\mathcal V}$ as defined above
- Take gradient descent step and adjust the learning rate if necessary
- Periodically compute the loss over a validation sample


## 3. Error measure for the training of $V$

This is a joint error measure for the value function $\hat V_\theta$ and the optimal control network $(\tilde u^\theta,\tilde a^\theta)$. Effectively, we aim to measure the error of estimation in the same manner as in Section II. More concretely, let us define for any $(t, x, m) \in [0,T]\times \R^d \times \R$ such that $m \geq \hat w^\theta(t,x)$ and $(u, a) \in U \times [-\mathcal N, \mathcal N]$ the following
$$
\varepsilon^{u,a}_V(t,x,m) := \left|\sup_{(u,a) \in U \times [-\mathcal N, \mathcal N]} \mathcal L^{u,a}_{X,M} \hat V_\theta(t,x,m) - \mathcal L_{X,M}^{\tilde u^\theta, \tilde a^\theta} \hat V_\theta(t,x,m)\right| \\
\varepsilon^V_V(t,x,m):= \left|\mathcal L_{X,M}^{\tilde u^\theta, \tilde a^\theta} \hat V_\theta(t,x,m)\right| \\
\varepsilon^{\mathcal V}_V(t,x,m):= \left|\hat V_\theta(t,x,w(t,x,m)) - \hat{\mathcal V}_\theta(t,x) \right|\\
\varepsilon^T_V (x,m):= \left|\hat V_\theta(T,x,m) - F(x)\right|. 
$$
Then, the error measure for a sample $\tilde \chi = (t_i, \tilde X^j_{t_i}, \tilde M^j_{t_i})^{j = 1\rightarrow M}_{i = 0 \rightarrow N-1}$ is defined by
$$
\mathcal E^V(\tilde u^\theta,  \tilde a^\theta, \hat V^\theta|\tilde \chi) := \frac{\frac1M\sum^M_{j=1}\left[\frac1N\sum^{N-1}_{i=0}(\varepsilon_V^{u,a}+\varepsilon^V_V+ \varepsilon_V^{\mathcal V})(t_i, \tilde X^j_{t_i}, \tilde M^j_{t_i}) + \varepsilon_V^T(\tilde X^j_{t_i}, \tilde M^j_{t_i})\right]}{\frac1M\sum^M_{j=1} \hat V_\theta(0, \tilde X^j_{t_0}, \tilde M^j_{t_0})} 
$$


Accordingly, the computation process of this joint measure is descibed below :
1. Generate a sample of initial states and Brownian trajectories $(x_0^j, (\Delta W^j_i)_{i=1 \rightarrow N})^{j = 1 \rightarrow M}$ and a sample of the initial constraint $(p_0^j)^{j=1\rightarrow M}$ such that for any $j$, $p_0^j \geq \hat w_\theta(0, x_0^j)$.
2. Pass the sample through $(\tilde u^\theta, \tilde a^\theta)$ to retrieve the sample of trajectory $\tilde \chi = (t_i, \tilde X_{t_i}^j, \tilde M^j_{t_i})^{j = 1 \rightarrow M}_{i = 0 \rightarrow N}$ as well as the corresponding control process $(\tilde u_{t_i}^j, \tilde a_{t_i}^j)^{j = 1 \rightarrow M}_{i = 0 \rightarrow N-1}$
3. Pass the sample of trajectories through $\hat V_\theta$ to compute the gradients $\partial_t, \partial_X$ and the hessian $\partial_{XX}$. 
3. Compute $\varepsilon^{u,a}_V, \varepsilon^V_V, \varepsilon^{\mathcal V}_V$ and $\varepsilon^T_V$ for each trajectory and $\hat V_\theta$ for each initial state.
4. Compute the error measure $\mathcal E^V$.
