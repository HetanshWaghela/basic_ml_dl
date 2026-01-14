# Gradient Descent

## 1.1 The Core Intuition

Imagine you're blindfolded on a hilly landscape, trying to find the lowest valley. You can only feel the slope beneath your feet. **Gradient descent** is the strategy of always taking a step in the direction that goes downhill.

In machine learning:
- **The landscape** = the loss function surface
- **Your position** = the current parameter values
- **The slope** = the gradient (derivative) of the loss
- **The valley** = the optimal parameters that minimize loss

## 1.1.5 Quick Reference: Essential Formulas for Interviews

**The core formula** (memorize this!):
$$\theta_{t+1} = \theta_t - \alpha \cdot \nabla_\theta L(\theta_t)$$

**What each symbol means**:
- $\theta_t$: current parameters (weights) at iteration $t$
- $\alpha$: learning rate (step size, typically 0.001-0.01)
- $\nabla_\theta L$: gradient of loss function w.r.t. parameters
- $\theta_{t+1}$: updated parameters

**For single variable**: $x_{t+1} = x_t - \alpha \cdot f'(x_t)$

**Gradient definition**:
$$\nabla_\theta L = \begin{bmatrix} \frac{\partial L}{\partial \theta_1} \\ \frac{\partial L}{\partial \theta_2} \\ \vdots \\ \frac{\partial L}{\partial \theta_n} \end{bmatrix}$$

**Key insight**: Negative gradient = direction of steepest descent (downhill)

## 1.2 The Mathematics 

### Single Variable Case

**Step 1: Understand the derivative**
For a function $f(x)$, the **derivative** $f'(x)$ (or $\frac{df}{dx}$) tells us the slope:
- If $f'(x) > 0$: function is increasing → move left (decrease $x$)
- If $f'(x) < 0$: function is decreasing → move right (increase $x$)
- If $f'(x) = 0$: we're at a critical point (possibly minimum)

**Step 2: The basic update rule (easy to write on whiteboard)**

$$x_{t+1} = x_t - \alpha \cdot f'(x_t)$$


1. Start with current point: $x_t$
2. Compute the gradient: $f'(x_t)$ (tells us direction)
3. Multiply by learning rate: $\alpha$ (controls step size)
4. Subtract to move downhill: $x_t - \alpha \cdot f'(x_t)$
5. Get new point: $x_{t+1}$

**Alternative notation** (also common):
$$x_{\text{new}} = x_{\text{old}} - \alpha \cdot \frac{df}{dx}\bigg|_{x_{\text{old}}}$$

Where $\alpha$ (alpha) is the **learning rate** - a small positive number like 0.01 or 0.001.

### Multivariable Case (The Gradient)

**Step 1: Define the gradient**
For a function $f(x_1, x_2, ..., x_n)$ with $n$ variables, the **gradient** $\nabla f$ is a vector of all partial derivatives:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**How to explain**: "The gradient is just a vector where each component is the partial derivative with respect to that variable."

**Step 2: The update rule**

$$\theta_{t+1} = \theta_t - \alpha \cdot \nabla_\theta L(\theta_t)$$

**Breaking it down:**
- $\theta_t$: current parameter vector (weights/biases)
- $\nabla_\theta L(\theta_t)$: gradient of loss function with respect to parameters
- $\alpha$: learning rate (step size)
- $\theta_{t+1}$: updated parameters

**Common notation variations** (all mean the same thing):
- $\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \cdot \nabla f(\mathbf{x}_t)$
- $\theta = \theta - \alpha \cdot \frac{\partial L}{\partial \theta}$
- $w_{t+1} = w_t - \alpha \cdot \frac{\partial L}{\partial w}\bigg|_{w_t}$

**Key insight**: The gradient points in the direction of **steepest ascent** (uphill). We use the **negative gradient** to go **downhill** (steepest descent).

## 1.3 Why the Negative Gradient? 

**Question**: "Why do we subtract the gradient instead of adding it?"

**Answer** (explain step by step):

1. **What the gradient tells us**: 
   $$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$
   The gradient points in the direction where the function **increases fastest** (steepest ascent).

2. **What we want**: We want to **minimize** the loss function, so we need to go in the direction where it **decreases fastest**.

3. **The solution**: Use the **negative gradient** $-\nabla f$ which points in the direction of **steepest descent** (downhill).

**Geometric intuition**:
- Gradient $\nabla f$ → points **uphill** ⬆️
- Negative gradient $-\nabla f$ → points **downhill** ⬇️
- We want to minimize → walk **downhill** ⬇️

**Mathematical proof** (if asked): The directional derivative in direction $\mathbf{v}$ is:
$$D_{\mathbf{v}} f = \nabla f^T \mathbf{v} = \|\nabla f\| \|\mathbf{v}\| \cos(\theta)$$
This is minimized when $\cos(\theta) = -1$, i.e., when $\mathbf{v} = -\nabla f$.

## 1.4 The Learning Rate $\alpha$ 

**Definition**: The learning rate $\alpha$ (also written as $\eta$ or `lr`) controls **how big a step** we take in each iteration.

**The formula again**:
$$\theta_{t+1} = \theta_t - \alpha \cdot \nabla_\theta L(\theta_t)$$

**How to explain learning rate**:

| Learning Rate | Effect | Analogy |
|---------------|--------|---------|
| **Too small** ($\alpha = 0.0001$) | Converges very slowly, may get stuck in flat regions | Taking tiny steps - safe but takes forever |
| **Too large** ($\alpha = 10$) | Overshoots minimum, oscillates, may diverge | Taking giant leaps - might jump over the valley |
| **Just right** ($\alpha = 0.01$ or $0.001$) | Smooth convergence to minimum | Balanced steps - efficient and safe |

**Visual intuition**: 
- Small $\alpha$ = tiny careful steps 🐌 (slow but safe)
- Large $\alpha$ = giant leaps 🦘 (fast but risky - might overshoot)

**Common values**:
- Deep learning: $\alpha = 0.001$ to $0.01$
- Traditional ML: $\alpha = 0.01$ to $0.1$
- Sometimes adaptive: starts large, decreases over time

**Tip**: Always mention that choosing the right learning rate is crucial and often requires experimentation or adaptive methods like Adam.

## 1.5 Types of Gradient Descent 

**Key question**: "How much data do we use to compute the gradient?"

### Batch Gradient Descent (Vanilla GD)

**Formula** (easy to write):
$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta L(\theta_t; x_i, y_i)$$

**How to explain**:
- Use **entire dataset** ($N$ samples) to compute gradient
- Average gradients: $\frac{1}{N} \sum_{i=1}^{N} \nabla_\theta L(\theta; x_i, y_i)$
- Update once per epoch (one pass through all data)

**Pros & Cons**:
- ✅ Stable convergence (low variance)
- ✅ Guaranteed to decrease loss (for convex functions)
- ❌ Slow for large datasets (must process all $N$ samples)
- ❌ Requires all data in memory
- ❌ Can't update until seeing all data

**When to use**: Small datasets, when you need exact gradients

### Stochastic Gradient Descent (SGD)

**Formula** (simplest form):
$$\theta_{t+1} = \theta_t - \alpha \cdot \nabla_\theta L(\theta_t; x_i, y_i)$$

**How to explain**:
- Use **one random sample** at a time
- Update after each sample (not averaged)
- Process all $N$ samples = $N$ updates per epoch

**Pros & Cons**:
- ✅ Fast updates (immediate feedback)
- ✅ Memory efficient (only need one sample)
- ✅ Can escape local minima (noise helps exploration)
- ❌ Noisy, high variance updates
- ❌ May not converge smoothly (oscillates around minimum)

**When to use**: Very large datasets, online learning

### Mini-Batch Gradient Descent (Most Common!)

**Formula** (best balance):
$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta L(\theta_t; x_i, y_i)$$

**How to explain**:
- Use **small batch** of $B$ samples (typically 32, 64, 128, or 256)
- Average gradients over the batch: $\frac{1}{B} \sum_{i=1}^{B} \nabla_\theta L(\theta; x_i, y_i)$
- Update after each batch
- One epoch = $\lceil N/B \rceil$ updates

**Pros & Cons**:
- ✅ Balanced speed and stability
- ✅ Leverages GPU parallelism (process batch simultaneously)
- ✅ Reduces variance compared to SGD
- ✅ Most commonly used in practice
- ⚠️ Need to choose batch size $B$ (hyperparameter)

**Typical batch sizes**:
- Small: $B = 32$ or $64$ (more updates, more noise)
- Medium: $B = 128$ or $256$ (balanced)
- Large: $B = 512$ or $1024$ (more stable, fewer updates)

**Tip**: Always mention mini-batch is the standard in deep learning because it balances efficiency and stability.

## 1.6 Derivative Review (The Chain Rule)

The **chain rule** is the foundation of all gradient computation:

If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**Example**: Let $y = (3x + 2)^2$

Let $u = 3x + 2$, so $y = u^2$

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = 2u \cdot 3 = 6(3x + 2)$$

This generalizes to arbitrarily long chains:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

## 1.7 Formal Mathematical Derivation

### First-Order Taylor Expansion

Gradient descent can be derived from the **first-order Taylor expansion** of the loss function around the current point $\mathbf{x}_t$:

$$f(\mathbf{x}_{t+1}) \approx f(\mathbf{x}_t) + \nabla f(\mathbf{x}_t)^T (\mathbf{x}_{t+1} - \mathbf{x}_t)$$

We want to choose $\mathbf{x}_{t+1}$ to minimize $f(\mathbf{x}_{t+1})$. Since $\nabla f(\mathbf{x}_t)$ is fixed, we minimize the dot product:

$$\min_{\mathbf{x}_{t+1}} \nabla f(\mathbf{x}_t)^T (\mathbf{x}_{t+1} - \mathbf{x}_t)$$

By the **Cauchy-Schwarz inequality**, this is minimized when:

$$\mathbf{x}_{t+1} - \mathbf{x}_t = -\alpha \nabla f(\mathbf{x}_t)$$

for some $\alpha > 0$, which gives us the standard update rule.

### Directional Derivative

The **directional derivative** of $f$ in direction $\mathbf{v}$ is:

$$D_{\mathbf{v}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{v}) - f(\mathbf{x})}{h} = \nabla f(\mathbf{x})^T \mathbf{v}$$

To minimize $f$, we want the direction $\mathbf{v}$ that minimizes $D_{\mathbf{v}} f(\mathbf{x})$. Since:

$$D_{\mathbf{v}} f(\mathbf{x}) = \|\nabla f(\mathbf{x})\| \|\mathbf{v}\| \cos(\theta)$$

where $\theta$ is the angle between $\nabla f$ and $\mathbf{v}$, the minimum occurs when $\cos(\theta) = -1$, i.e., when $\mathbf{v} = -\nabla f(\mathbf{x})$.

## 1.8 Convergence Analysis

### Lipschitz Continuity

A function $f$ is **$L$-Lipschitz continuous** if:

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|$$

for all $\mathbf{x}, \mathbf{y}$ in the domain, where $L$ is the **Lipschitz constant**.

### Convergence Theorem

**Theorem**: If $f$ is $L$-Lipschitz continuous and convex, then gradient descent with learning rate $\alpha \leq \frac{1}{L}$ converges to the global minimum.

**Proof sketch**: For a convex function with $L$-Lipschitz gradient:

$$f(\mathbf{x}_{t+1}) \leq f(\mathbf{x}_t) - \alpha \|\nabla f(\mathbf{x}_t)\|^2 + \frac{L\alpha^2}{2} \|\nabla f(\mathbf{x}_t)\|^2$$

$$= f(\mathbf{x}_t) - \alpha\left(1 - \frac{L\alpha}{2}\right) \|\nabla f(\mathbf{x}_t)\|^2$$

If $\alpha \leq \frac{1}{L}$, then $1 - \frac{L\alpha}{2} \geq \frac{1}{2}$, so:

$$f(\mathbf{x}_{t+1}) \leq f(\mathbf{x}_t) - \frac{\alpha}{2} \|\nabla f(\mathbf{x}_t)\|^2$$

This guarantees that $f$ decreases at each step, and the algorithm converges.

### Convergence Rate

For a **strongly convex** function with parameter $\mu$ (i.e., $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2$), gradient descent achieves:

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \leq (1 - \mu\alpha)^t (f(\mathbf{x}_0) - f(\mathbf{x}^*))$$

This is **linear convergence** (also called geometric convergence).

### Convergence Conditions

For convergence, we need:

1. **Learning rate bound**: $\alpha < \frac{2}{L}$ (where $L$ is the Lipschitz constant)
2. **Function properties**: $f$ should be differentiable and have bounded gradients
3. **Initialization**: Starting point should be in the domain of convergence

## 1.9 Second-Order Information: Newton's Method

### The Hessian Matrix

The **Hessian matrix** $\mathbf{H}$ contains second-order derivatives:

$$\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

For a function $f: \mathbb{R}^n \to \mathbb{R}$:

$$\mathbf{H} = \begin{bmatrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}$$

### Newton's Method

**Newton's method** uses second-order information:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{H}^{-1}(\mathbf{x}_t) \nabla f(\mathbf{x}_t)$$

**Derivation**: Using second-order Taylor expansion:

$$f(\mathbf{x}_{t+1}) \approx f(\mathbf{x}_t) + \nabla f(\mathbf{x}_t)^T (\mathbf{x}_{t+1} - \mathbf{x}_t) + \frac{1}{2}(\mathbf{x}_{t+1} - \mathbf{x}_t)^T \mathbf{H}(\mathbf{x}_t) (\mathbf{x}_{t+1} - \mathbf{x}_t)$$

Taking the gradient with respect to $\mathbf{x}_{t+1}$ and setting it to zero:

$$\nabla f(\mathbf{x}_t) + \mathbf{H}(\mathbf{x}_t)(\mathbf{x}_{t+1} - \mathbf{x}_t) = 0$$

Solving for $\mathbf{x}_{t+1}$ gives Newton's update.

**Advantages**:
- ✅ Quadratic convergence rate (much faster than gradient descent)
- ✅ Automatically adapts step size based on curvature

**Disadvantages**:
- ❌ Requires computing and inverting the Hessian ($O(n^3)$ complexity)
- ❌ May not converge if Hessian is not positive definite
- ❌ Computationally expensive for large $n$

### Quasi-Newton Methods

**Quasi-Newton methods** (e.g., BFGS, L-BFGS) approximate the Hessian without computing it explicitly:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha_t \mathbf{B}_t^{-1} \nabla f(\mathbf{x}_t)$$

where $\mathbf{B}_t$ approximates $\mathbf{H}(\mathbf{x}_t)$ and is updated using gradient information.

## 1.10 Adaptive Learning Rate Methods

### Momentum

**Momentum** accumulates past gradients to smooth updates:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \nabla f(\mathbf{x}_t)$$

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \mathbf{v}_t$$

where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9).

**Physical analogy**: Like a ball rolling downhill, momentum helps overcome small bumps and accelerates in consistent directions.

**Mathematical intuition**: Momentum reduces oscillations in narrow valleys by averaging gradients over time.

### Nesterov Accelerated Gradient (NAG)

**Nesterov momentum** looks ahead before computing the gradient:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \nabla f(\mathbf{x}_t - \beta \mathbf{v}_{t-1})$$

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \mathbf{v}_t$$

This "anticipates" where the momentum will take us, leading to better convergence.

### AdaGrad

**AdaGrad** adapts learning rates per parameter:

$$G_t = G_{t-1} + \nabla f(\mathbf{x}_t) \odot \nabla f(\mathbf{x}_t)$$

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot \nabla f(\mathbf{x}_t)$$

where $\odot$ denotes element-wise multiplication and $\epsilon$ is a small constant (e.g., $10^{-8}$).

**Intuition**: Parameters with large gradients get smaller learning rates, and vice versa.

**Problem**: $G_t$ accumulates indefinitely, causing learning rates to vanish over time.

### RMSProp

**RMSProp** fixes AdaGrad's vanishing learning rate by using exponential moving average:

$$G_t = \beta G_{t-1} + (1 - \beta) \nabla f(\mathbf{x}_t) \odot \nabla f(\mathbf{x}_t)$$

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot \nabla f(\mathbf{x}_t)$$

where $\beta$ is typically 0.9.

### Adam (Adaptive Moment Estimation)

**Adam** combines momentum and RMSProp:

**First moment** (biased estimate of gradient):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla f(\mathbf{x}_t)$$

**Second moment** (biased estimate of squared gradient):
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla f(\mathbf{x}_t) \odot \nabla f(\mathbf{x}_t)$$

**Bias correction**:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update**:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Typical values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\alpha = 0.001$.

## 1.11 Worked Examples

### Example 1: Quadratic Function

Minimize $f(x) = x^2$ using gradient descent.

**Gradient**: $f'(x) = 2x$

**Update rule**: $x_{t+1} = x_t - \alpha \cdot 2x_t = x_t(1 - 2\alpha)$

**Analytical solution**: Starting from $x_0$:

$$x_t = x_0(1 - 2\alpha)^t$$

For convergence, we need $|1 - 2\alpha| < 1$, i.e., $0 < \alpha < 1$.

**Optimal learning rate**: For fastest convergence, minimize $|1 - 2\alpha|$, giving $\alpha = 0.5$, which converges in one step!

### Example 2: Multivariable Quadratic

Minimize $f(x, y) = x^2 + 2y^2$ using gradient descent.

**Gradient**: $\nabla f = \begin{bmatrix} 2x \\ 4y \end{bmatrix}$

**Update rule**:
$$\begin{bmatrix} x_{t+1} \\ y_{t+1} \end{bmatrix} = \begin{bmatrix} x_t \\ y_t \end{bmatrix} - \alpha \begin{bmatrix} 2x_t \\ 4y_t \end{bmatrix} = \begin{bmatrix} x_t(1 - 2\alpha) \\ y_t(1 - 4\alpha) \end{bmatrix}$$

**Convergence condition**: Need $|1 - 2\alpha| < 1$ and $|1 - 4\alpha| < 1$, so $0 < \alpha < 0.5$.

**Note**: Different parameters require different learning rates for optimal convergence. This motivates adaptive methods.

### Example 3: Linear Regression

For linear regression with loss $L(\theta) = \frac{1}{2N}\sum_{i=1}^N (h_\theta(x_i) - y_i)^2$ where $h_\theta(x) = \theta^T x$:

**Gradient**:
$$\nabla_\theta L = \frac{1}{N}\sum_{i=1}^N (h_\theta(x_i) - y_i) x_i = \frac{1}{N}\sum_{i=1}^N (\theta^T x_i - y_i) x_i$$

**In matrix form** (with $\mathbf{X} \in \mathbb{R}^{N \times d}$, $\mathbf{y} \in \mathbb{R}^N$):

$$\nabla_\theta L = \frac{1}{N}\mathbf{X}^T(\mathbf{X}\theta - \mathbf{y})$$

**Update rule**:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{N}\mathbf{X}^T(\mathbf{X}\theta_t - \mathbf{y})$$

## 1.12 Common Problems and Solutions

### Problem 1: Vanishing Gradients

**Symptom**: Gradients become extremely small, causing slow or stalled learning.

**Causes**:
- Deep networks with many layers
- Activation functions with small derivatives (e.g., sigmoid)
- Small learning rates

**Solutions**:
- Use ReLU or other activation functions with non-vanishing gradients
- Residual connections (skip connections)
- Batch normalization
- Gradient clipping

### Problem 2: Exploding Gradients

**Symptom**: Gradients become extremely large, causing unstable updates.

**Causes**:
- Deep networks
- Large weights
- Large learning rates

**Solutions**:
- Gradient clipping: $\mathbf{g} \leftarrow \frac{\mathbf{g}}{\max(1, \|\mathbf{g}\|/c)}$ where $c$ is a threshold
- Weight initialization (e.g., Xavier, He initialization)
- Smaller learning rates

### Problem 3: Local Minima

**Symptom**: Algorithm converges to a suboptimal solution.

**Solutions**:
- Stochastic gradient descent (noise helps escape)
- Multiple random initializations
- Simulated annealing
- Better initialization strategies

### Problem 4: Saddle Points

**Symptom**: Gradient is zero but not at a minimum (Hessian has both positive and negative eigenvalues).

**Mathematical definition**: A point $\mathbf{x}$ is a saddle point if:
- $\nabla f(\mathbf{x}) = 0$
- $\mathbf{H}(\mathbf{x})$ has both positive and negative eigenvalues

**Solutions**:
- Momentum methods help escape saddle points
- Second-order methods (Newton's method) can identify and escape
- Noise in SGD helps escape

## 1.13 Advanced Topics

### Line Search

Instead of fixed learning rate, **line search** finds optimal step size:

$$\alpha_t = \arg\min_{\alpha > 0} f(\mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t))$$

**Exact line search**: Solve the minimization problem exactly (expensive).

**Backtracking line search**: Start with large $\alpha$, reduce until sufficient decrease:

$$f(\mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t)) \leq f(\mathbf{x}_t) - c \alpha \|\nabla f(\mathbf{x}_t)\|^2$$

where $c \in (0, 1)$ is a constant (typically 0.1).

### Gradient Descent with Constraints

For constrained optimization:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to} \quad g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m$$

**Projected gradient descent**:

$$\mathbf{x}_{t+1} = \text{Proj}_C(\mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t))$$

where $\text{Proj}_C$ projects onto the feasible set $C = \{\mathbf{x} : g_i(\mathbf{x}) \leq 0\}$.

### Subgradient Methods

For non-differentiable functions, use **subgradients**. A vector $\mathbf{g}$ is a subgradient of $f$ at $\mathbf{x}$ if:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \mathbf{g}^T(\mathbf{y} - \mathbf{x})$$

for all $\mathbf{y}$ in the domain.

**Subgradient descent**:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha_t \mathbf{g}_t$$

where $\mathbf{g}_t$ is any subgradient at $\mathbf{x}_t$.

**Note**: Unlike gradient descent, subgradient methods require diminishing step sizes ($\alpha_t \to 0$) for convergence.

### Coordinate Descent

**Coordinate descent** updates one coordinate at a time:

$$x_i^{(t+1)} = \arg\min_{x_i} f(x_1^{(t)}, \ldots, x_{i-1}^{(t)}, x_i, x_{i+1}^{(t)}, \ldots, x_n^{(t)})$$

Useful when:
- The problem is separable
- Computing full gradient is expensive
- Parallelization is needed

## 1.14 Convergence in Practice

### Stopping Criteria

Common stopping criteria:

1. **Gradient norm**: Stop when $\|\nabla f(\mathbf{x}_t)\| < \epsilon$
2. **Function value change**: Stop when $|f(\mathbf{x}_{t+1}) - f(\mathbf{x}_t)| < \epsilon$
3. **Parameter change**: Stop when $\|\mathbf{x}_{t+1} - \mathbf{x}_t\| < \epsilon$
4. **Maximum iterations**: Stop after $T$ iterations

### Monitoring Convergence

**Loss curve**: Plot $f(\mathbf{x}_t)$ vs. iteration $t$. Should decrease monotonically (for batch GD) or with noise (for SGD).

**Gradient norm**: Plot $\|\nabla f(\mathbf{x}_t)\|$ vs. iteration. Should approach zero.

**Learning rate scheduling**: Reduce learning rate over time:

- **Step decay**: $\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/s \rfloor}$ where $s$ is step size
- **Exponential decay**: $\alpha_t = \alpha_0 \cdot \gamma^t$
- **Polynomial decay**: $\alpha_t = \alpha_0 \cdot (1 + \gamma t)^{-p}$

## 1.15 Mathematical Properties

### Convexity and Gradient Descent

For a **convex function** $f$:

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})$$

for all $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0, 1]$.

**Key property**: For convex functions, any local minimum is a global minimum.

**Gradient descent on convex functions**:
- Guaranteed to converge to global minimum (with appropriate learning rate)
- Convergence rate: $O(1/t)$ for general convex, $O((1-\mu/L)^t)$ for strongly convex

### Non-Convex Optimization

Most neural network loss functions are **non-convex**. Gradient descent:
- May converge to local minima
- May get stuck at saddle points
- No global convergence guarantees

However, in practice:
- Local minima are often "good enough"
- Saddle points are more common than local minima in high dimensions
- SGD noise helps escape poor critical points

### Smoothness and Strong Convexity

A function is **$\mu$-strongly convex** if:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

A function has **$L$-Lipschitz gradient** if:

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$$

**Condition number**: $\kappa = \frac{L}{\mu}$ measures the "difficulty" of optimization. Larger $\kappa$ means slower convergence.

For gradient descent on strongly convex functions with $L$-Lipschitz gradient:

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \leq \left(1 - \frac{\mu}{L}\right)^t (f(\mathbf{x}_0) - f(\mathbf{x}^*))$$

## 1.16 Implementation Considerations

### Numerical Stability

**Gradient computation**: Use automatic differentiation or numerical differentiation carefully.

**Numerical differentiation** (finite differences):
$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x})}{h}$$

where $h$ is small (e.g., $10^{-5}$) and $\mathbf{e}_i$ is the $i$-th unit vector.

**Automatic differentiation**: More accurate and efficient, used in frameworks like PyTorch, TensorFlow.

### Computational Complexity

**Time complexity per iteration**:
- Gradient computation: $O(n)$ for $n$ parameters
- Update: $O(n)$
- Total: $O(n)$ per iteration

**Space complexity**: $O(n)$ to store parameters and gradients.

**For neural networks**: 
- Forward pass: $O(W)$ where $W$ is number of weights
- Backward pass: $O(W)$
- Total: $O(W)$ per iteration

### Parallelization

**Data parallelism**: Split data across multiple workers, average gradients.

**Model parallelism**: Split model across multiple devices (for very large models).

**Gradient accumulation**: Accumulate gradients over multiple mini-batches before updating (useful for large batch sizes that don't fit in memory).