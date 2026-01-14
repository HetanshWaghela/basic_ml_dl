# Backpropagation in Matrix Form

In practice, neural networks operate on **vectors and matrices**, not scalars.
Matrix form backpropagation is simply the **vectorized version of the scalar chain rule**.

Nothing new conceptually is introduced — only notation changes.


## Step 0: One Fully Connected Layer

Assume a single dense layer.

Input vector:

$$
x \in \mathbb{R}^{n \times 1}
$$

Weight matrix:

$$
W \in \mathbb{R}^{m \times n}
$$

Bias vector:

$$
b \in \mathbb{R}^{m \times 1}
$$

---


## Forward Pass

Linear transformation:

$$
z = Wx + b
$$

Activation:

$$
a = \sigma(z)
$$

Loss:

$$
L = L(a)
$$

The dependencies are:

- $L$ depends on $a$
- $a$ depends on $z$
- $z$ depends on $W$, $b$, and $x$

---


## What Are We Trying to Compute?

We want gradients with respect to the parameters:

$$
\frac{\partial L}{\partial W}
\quad \text{and} \quad
\frac{\partial L}{\partial b}
$$

To compute them, gradients must flow **backward** through the graph.

---


## Step 1: Start From the Loss

Backpropagation always starts at the output.

Assume we already have the upstream gradient:

$$
\frac{\partial L}{\partial a}
$$

This comes from:
- the loss function (MSE, Cross-Entropy), or
- the next layer during backpropagation

---


## Step 2: Backprop Through the Activation

Given:

$$
a = \sigma(z)
$$

By the chain rule:

$$
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\odot
\sigma'(z)
$$

Where:
- $\odot$ denotes element-wise (Hadamard) multiplication
- $\sigma'(z)$ is the derivative of the activation function

This step explains:
- vanishing gradients for sigmoid and tanh
- blocked gradients for ReLU when $z < 0$

---



## Step 3: Backprop Through the Linear Layer

Recall the linear operation:

$$
z = Wx + b
$$

We now compute gradients with respect to the parameters.

---

### Gradient With Respect to Bias

Each component of $z$ depends directly on the corresponding bias term.

Therefore:

$$
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial z}
$$

The bias receives the gradient unchanged.

---

### Gradient With Respect to Weights

Each weight $W_{ij}$ connects input $x_j$ to output $z_i$.

The gradient is given by an outer product:

$$
\frac{\partial L}{\partial W}
=
\frac{\partial L}{\partial z}
\cdot
x^T
$$

Shape check:

$$
\frac{\partial L}{\partial z} \in \mathbb{R}^{m \times 1}
$$

$$
x^T \in \mathbb{R}^{1 \times n}
$$

$$
\Rightarrow
\frac{\partial L}{\partial W} \in \mathbb{R}^{m \times n}
$$

Each weight update corresponds to:

> error at the output multiplied by the value at the input

---

## Step 4: Gradient Passed to the Previous Layer

Backpropagation does not stop at the parameters.

We must compute the gradient with respect to the input:

$$
\frac{\partial L}{\partial x}
$$

Using the chain rule:

$$
\frac{\partial L}{\partial x}
=
W^T
\frac{\partial L}{\partial z}
$$

This gradient is passed backward to the previous layer.

---

## Why the Transpose Appears

Each input dimension contributes to multiple output neurons.

The transpose:

$$
W^T
$$

correctly aggregates these contributions during backpropagation.

This follows directly from matrix calculus.

---

## Final Summary (Lock This In)

For a single fully connected layer:

Forward pass:

$$
z = Wx + b
$$

$$
a = \sigma(z)
$$

Backward pass:

$$
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\odot
\sigma'(z)
$$

$$
\frac{\partial L}{\partial W}
=
\frac{\partial L}{\partial z}
\cdot
x^T
$$

$$
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial z}
$$

$$
\frac{\partial L}{\partial x}
=
W^T
\frac{\partial L}{\partial z}
$$

---

## One-Line Mental Model

Backpropagation moves error signals backward through the network,
scaling them at each step by local derivatives.