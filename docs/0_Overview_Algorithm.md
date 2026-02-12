This is an overview for the algorithm developed in this repository. It provides pointers to detailed and up-to-date description of each part of the algorithm. Note that we put a focus on the algorithm itself rather than the model being employed for the illustration of the algorithm.

# *Introduction to the mathematical problem*

We provide a brief introduction to the mathematical framework of the problem that the algorithm seeks to solve numerically.  A detailed explanation of the model is provided in the associated article.

Let $T$ be the end of the time horizon and let $X$ be a $d$-dimensional state variable which, under the control process $u = (u_t)_{t \in [t_0, T]}$, follows a known dynamic 
$$
dX^{u}_{t_0, x_0}(t) = b(X^{u}_{t_0, x_0}(t), u_t) dt + \sigma(X^{u}_{t_0, x_0}(t), u_t) dW_t
$$
where $dW_t$ is the (multi-dimensional) Brownian (increment) representing the stochasticity of our problem. Assuming that we are at the starting point of the time horizon $t_0 = 0 < T$, we seek to estimate a value function $V$ which is the optimal expectation of some reward function $F$ computed at $T$ while respecting a terminal constraint whose threshold is given as $p$. Mathematically, our problem is presented as
$$ 
V(t_0, x_0, p) := \sup_{u \in \mathcal U(t_0, x_0, p)} \mathbb E\left[ F(X^{u}_{t_0, x_0}(T))\right]
$$
where $x_0$ is the initial state at $t_0$ of $X$ and the feasible control space is defined as
$$
\mathcal U(t_0, x_0, p) := \{ u : u_t \in U \text{ compact and } \mathbb E \left[G(X^{u}_{t_0, x_0}(T)) + \int_{t_0}^T g(X^{u}_{t_0, x_0}(s))ds \right] \leq p \}
$$
To solve this problem, we introduce the Martingale representation $M_{t_0, p_0}^{a}$ of $p$ and augment our control to encouple $u$ with the Brownian increment process $a = (a_t)_{t \in [t_0, T]}$ of the Martingale representation. Then, we derive the Partial Differential Equation (PDE) characterization of the value function $V$ on the viable domain 
$$
\mathcal D := \{(t,x,p) \in [t_0,T] \times \mathbf R^d \times \mathbf R : \mathcal U(t,x,p) \neq \empty\}
$$ 
which can also be defined in terms of the augmented state process $(X, M)$ as 
$$
\mathcal D := \{(t,x,p) \in [t_0,T] \times \mathbf R^d \times \mathbf R : \exists (u, a) \text{ such that } X^u_{t,x}(s) \leq M^a_{t,p(s)} \forall s \in [t,T]\}
$$
We define the (spacial) boundary of the viable domain 
$$
\partial_P \mathcal D := \{ (t,x,p) \in \mathcal D : p = w(t,x) \text{ and } t < T\}
$$
where 
$$
w:= \inf_{u \in \mathcal U} \mathbb E \left[ G(X^{u}_{t_0, x_0}(T)) + \int_{t_0}^T g(X^{u}_{t_0, x_0}(s))ds\right]
$$ 
is the threshold for $p$ underwhich $\mathcal U(t,x,p) \neq \empty$. Note that the characterization of the value function $V$ hence follows the partition the viable domain into 3 parts : at the terminal $T$ of the time horizon, at the boundary  $\partial_P \mathcal D$, and in the interior $int \mathcal D$ of the viable domain. 

We seek to develop an algorithm which can estimate the value function $V$ using its PDE characterization. The project can be roughly divided into 4 parts : 
1. Pre-training modelization which includes notably setting parameters and constructing the dynamic of the portfolio (notably its drift and volatility when a control is applied) -> `1_PreTraining_Modelization.md`
2. Estimation of the optimal control process $u^*$ and the domain boundary value function $w$ using neural networks, which includes training the neural networks and validation with an error measure -> `2_Training_OptimalControl_DomainBoundary.md`
3. Estimation of the value function $\mathcal V$ at the viable domain boundary -> `3_Training_ValueFunction_at_DomainBoundary.md`
4. Estimation of the value function $V$ on the entire viable domain -> `4_Training_ValueFunction_global.md`


# *Simplified Table of Content for the algorithm*


## I. Portfolio modelization
*Corresponding modules :* `0_portfolio_model.py` 

We present a simplified description of the modeling of the portfolio here. Notably, we define the portfolio drift, the portfolio volatility, the penalties (which represent the constraints), the terminal loss function, and the utility function of the problem. 

## II. Estimation of the Optimal Control Process  and the Boundary Domain Value 
### 1. Training the Optimal Control Process Neural Network $\hat u^\theta = (\hat u^\theta_{t_i})_{i = 0,...,N-1}$ 
*Corresponding modules :* `0_neural_networks.py`, `1_1_train_u.py`

In this documentation, we introduce the consequential structure chosen for a control process and define the loss function as well as the training algorithm. This is an unsupervised training process which seeks to minimize the expected terminal loss. 

### 2. Training the Domain Boundary Value Neural Network $\hat w_\theta$
*Corresponding modules :* `0_neural_networks.py`, `1_2_train_w.py`

We employ Physics-Informed Neural Network (PINN) for the estimation of the Domain Boundary Value Neural Network. More concretely, we define a loss function that resembles the Partial Differential Equaition (PDE) that $w$ should satisfy. 

### 3. Joint Error Measure for the estimations of $\hat u^\theta$ and $\hat w_\theta$
*Corresponding modules :* `2_12_test_u_and_w.py`

We define the joint error measure so that it can be interpretted as an error margin in percentage. Additionally, we provide some visualization on the behavior of the state process when the estimate control process is applied and on the distribution of the state process across a wide range of simulation (from the same initial state).

## III. Estimation of the Value Function Network at the Domain Boundary

### 1. Training the Value Function Network $\hat{\mathcal V}_\theta$
*Corresponding module :* `0_neural_networks.py`, `1_3_train_vb.py`

Similar to the training of $\hat w_\theta$, we also use PINN for this estimation. Note that since we are on the domain boundary, the constraint limit variable $p$ is deterministic, meaning that $p=w(t,x)$, so the value function $\mathcal V$ at the boundary only need to take two variables $t$ and $x$. 

### 2. Validation for the estimation
*Corresponding module :* `2_30_test_vb.py`

We define the error measure for the estimation of the value function at the boundary, similar to the error measure in Section II. 

## IV. Estimation of the Value Funnction $V$ over the entire viable domain

We highly recommend a consultation with the source paper for more detailed explanation of the following section of the algorithm.

### 1. Training the Augmented Optimal Control Process Neural Network $(\tilde u_{t_i}, \tilde a_{t_i})_{i = 0,...,N-1}$ 
*Corresponding modules :* `0_neural_networks.py`, `1_4_train_a.py`

Note that for the value function $V$ in the viable domain, the PDE involves the optimal values of both control $u$ and the stochastic coefficient $a$ of the Martingal representation of $p$. This requires us to estimate a new process, so-called the augmented optimal control process $(\tilde u_t, \tilde a_t)_{t \in \{t_0, ..., t_{N-1}\}}$ which seeks to maximize the expected terminal utility under the condition of absorbing boundary. 

### 2. Training the Value Function $\hat V_\theta$
*Corresponding modules :* `0_neural_networks.py`, `1_5_train_v.py`

The training of the value function $\hat V_\theta$ uses the augmented optimal control process  $(\tilde u_{t_i}, \tilde a_{t_i})_{i = 0,...,N-1}$ and the value function at the boundary $\hat{\mathcal V}_\theta$. 

### 3. Validation of the estimation 
*Corresponding modules:* `2_45_test_a_v.py`

Similar to the joint error measure introduced in Section II, we will also validate the estimation of the augmented optimal control process and the value function together.

