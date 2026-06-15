# Reinforcement-Learning-Pendulum-From-Scratch-SAC-NumPy

A pure NumPy implementation of the Soft Actor-Critic (SAC) algorithm from scratch. Solves the non-linear Pendulum-v1 swing-up control problem by implementing explicit matrix calculus, Bellman optimality target updates, and analytical backpropagation pipelines without deep learning frameworks.

---

## 🏗️ System & Architectural Pipeline

The algorithm treats neural network blocks as explicit matrix operators mapping non-linear physical system states to bounded continuous control torques.



1. **State Observation:** The environment yields a 3D state vector.
2. **Policy Inference:** The Actor Network computes distribution limits ($\mu, \sigma$).
3. **Reparameterization Trick:** Stochastic action mapping via Gaussian noise vectors.
4. **Environment Step:** Bounded motor torque execution via $a = \tanh(u) \cdot 2.0$.
5. **Memory Storage:** Variables are stored inside the Replay Buffer Vault.
6. **Matrix Optimization Core:** Dual Critics and the Actor undergo manual matrix backpropagation.

---

## 🧮 Mathematical Foundations

### 1. State and Action Space Representation
The `Pendulum-v1` continuous control environment operates on the following vector spaces:

| Variable | Physical Meaning | Range | NumPy Matrix Shape |
| :--- | :--- | :--- | :--- |
| $\cos(\theta)$ | Cosine of the pendulum angle | $[-1.0, 1.0]$ | Element 0 of State Vector |
| $\sin(\theta)$ | Sine of the pendulum angle | $[-1.0, 1.0]$ | Element 1 of State Vector |
| $\dot{\theta}$ | Angular velocity of the joint | $[-8.0, 8.0]$ | Element 2 of State Vector |
| $a$ | Continuous motor torque | $[-2.0, 2.0]$ | Output Action Dimension |

$$s = \begin{bmatrix} \cos(\theta) & \sin(\theta) & \dot{\theta} \end{bmatrix} \in \mathbb{R}^{1 \times 3}$$

---

### 2. The Forward Pass Engine



#### A. Dense Layer Transformations
For any single hidden layer $l$ within the networks, the forward linear matrix combination is defined as:

$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$

Where $A^{[l-1]}$ is the incoming feature matrix, $W^{[l]}$ is the layer weight matrix, and $b^{[l]}$ is the bias row vector. This is followed by a element-wise Rectified Linear Unit (ReLU) non-linear activation tracking mask:

$$A^{[l]} = \max(0, Z^{[l]})$$

#### B. Stochastic Policy Reparameterization
The Actor network maps state features to Gaussian parameter matrices:

$$\mu = Z_{\text{mean}} = A^{[2]} W_{\text{mean}} + b_{\text{mean}}$$

$$\log \sigma = Z_{\text{log , std}} = A^{[2]} W_{\text{log , std}} + b_{\text{log ,std}}$$

To safely compute the standard deviation without numerical overflow, the log standard deviation is clamped inside steady boundaries before scaling:

$$\log \sigma_{\text{clipped}} = \text{clip}(\log \sigma, -20, 2)$$

$$\sigma = \exp(\log \sigma_{\text{clipped}})$$

Continuous actions are sampled using an independent noise vector $z \sim \mathcal{N}(0, I)$ to maintain differentiable training properties (the reparameterization trick):

$$u = \mu + \sigma \odot z$$

$$a = \tanh(u) \cdot 2.0$$

---

### 3. The Bellman Optimality Target for Q-Values
The Critic Networks model the state-action value function $Q(s,a)$. To suppress overestimation bias during matrix parameter evaluation loops, we construct a Clipped Twin-Critic framework using the off-policy Bellman expectation equation:

$$y = r + \gamma \min \left( Q_{\phi_1^{-}}(s', a'), Q_{\phi_2^{-}}(s', a') \right)$$

Where:
* $r$ is the scalar environment reward.
* $\gamma = 0.99$ is the temporal discount factor.
* $\phi_1^{-}, \phi_2^{-}$ represent the parameter matrices of the slow-moving target critic networks.
* $s'$ and $a'$ represent the sampled next states and next actions respectively.

The Mean Squared Error (MSE) loss minimized across a batch size of $N=64$ transitions evaluates to:

$$L(\phi_i) = \frac{1}{2N} \sum_{k=1}^{N} \left( Q_{\phi_i}(s_k, a_k) - y_k \right)^2$$

---

### 4. Explicit Matrix Backpropagation Equations

#### A. Critic Network Gradient Flow
Gradients move backwards across the layers of the Critic by evaluating the loss with respect to the output predictions:

$$\delta^{[3]} = \frac{\partial L}{\partial Q_{\phi}} = \frac{Q_{\phi}(s, a) - y}{N}$$

Using the chain rule, we map parameter updates across Layer 3, Layer 2, and Layer 1 sequentially:

$$\nabla_{W^{[3]}} L = (A^{[2]})^T \delta^{[3]}, \quad \nabla_{b^{[3]}} L = \sum_{\text{batch}} \delta^{[3]}$$

$$\delta^{[2]} = \left( \delta^{[3]} (W^{[3]})^T \right) \odot \mathbb{I}(A^{[2]} > 0)$$

$$\nabla_{W^{[2]}} L = (A^{[1]})^T \delta^{[2]}, \quad \nabla_{b^{[2]}} L = \sum_{\text{batch}} \delta^{[2]}$$

$$\delta^{[1]} = \left( \delta^{[2]} (W^{[2]})^T \right) \odot \mathbb{I}(A^{[1]} > 0)$$

$$\nabla_{W^{[1]}} L = (X)^T \delta^{[1]}, \quad \nabla_{b^{[1]}} L = \sum_{\text{batch}} \delta^{[1]}$$

Where 
$$X = \begin{bmatrix} s & a \end{bmatrix}$$ is the horizontally concatenated input matrix, and $\mathbb{I}$ 
represents the element-wise binary indicator function tracking positive activations.

#### B. Stochastic Policy (Actor) Gradient Flow
The Actor changes its weights to maximize the Q-values predicted by Critic 1. The objective gradient flows from Critic 1 backward into the action parameters:

$$\delta_{\text{action}} = -\frac{\partial Q_1(s, a)}{\partial a}$$

Because actions pass through a bounded hyperbolic tangent activation function, gradients must propagate through the derivative of $\tanh$:
$$\delta_{\text{mean}} = \delta_{\text{action}} \odot \left( 1 - \tanh^2(u) \right) \cdot 2.0$$

The parameter gradients for the output layer of the policy network evaluate to:
$$\nabla_{W_{\text{mean}}} L = (A^{[2]})^T \delta_{\text{mean}}, \quad \nabla_{b_{\text{mean}}} L = \sum_{\text{batch}} \delta_{\text{mean}}$$

---

### 5. Stochastic Gradient Descent (SGD) Parameter Update Rules
Once network parameter gradients are evaluated, weights and biases are modified via standard first-order optimization steps modulated by the structural learning rate ($\alpha = 0.001$):

$$\theta \leftarrow \theta - \alpha \cdot \nabla_{\theta} L$$

$$W^{[l]} \leftarrow W^{[l]} - \alpha \cdot \nabla_{W^{[l]}} L$$
$$b^{[l]} \leftarrow b^{[l]} - \alpha \cdot \nabla_{b^{[l]}} L$$

<h2 align="center">📊 Experimental Results & Agent Performance</h2>

<p align="center">
  <img width="420" height="270" alt="learning curve" src="https://github.com/user-attachments/assets/5253d23e-f8b5-4156-9b78-120e29a5dd8c" />
  <br>
  <em>Figure 1: The learning curve while training.</em>
</p>

<br>

<p align="center">
  <img width="400" height="330" alt="after 500 episodes" src="https://github.com/user-attachments/assets/c8ae6f93-dcb1-4adf-b921-266e14a1e08c" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img width="350" height="330" alt="pendulum_top" src="https://github.com/user-attachments/assets/3adf1322-62ea-42dd-aa71-e366d69514d8" />
  <br>
  <em>Figure 2: Final moving average convergence (left) and the trained SAC agent holding the pendulum steady at the top position (right).</em>
</p>

---

## 💻 Source Code Location

The complete, self-contained Python implementation—including network initializations, explicit matrix calculus backpropagation loops, and live rendering logic—is available in the primary repository file:

🔗 **[`PS-up With SAC Method.ipynb`](PS-up%20With%20SAC%20Method.ipynb)**

*Note: You can run this Jupyter Notebook directly within an Anaconda environment or using Google Colab to train the agent and view the physical pendulum interface window.*

