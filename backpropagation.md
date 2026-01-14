Used to calculate the gradient of the loss function with respect to every weight in the network.

Forward pass: Data flows from input to output. We compute the activations and finally the Loss

Backward pass: We propagate the error signal from the output layer back to the input layer to determine how muchh blame each weight bears for the total error.

### its a recursive application of the chain rule.

Consider a single neuron where

$$
z = wx + b
$$

and the activation is

$$
a = \sigma(z)
$$

The gradient with respect to the weight is computed using the chain rule:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w}
$$

Here is exactly where the previous topic fits in.

The term

$$
\frac{\partial L}{\partial a}
$$

describes how the loss changes as the neuron's output changes.  
This is propagated from the next layer during backpropagation.

The term

$$
\frac{\partial a}{\partial z}
$$

is the derivative of the activation function  
(e.g. $\sigma'(z)$ for sigmoid, or $1$ for ReLU when $z > 0$).  
This is where the vanishing gradient problem arises.

The term

$$
\frac{\partial z}{\partial w}
$$

comes from the linear function. Since

$$
z = wx + b
$$

we have

$$
\frac{\partial z}{\partial w} = x
$$


## Computational Efficiency

Instead of calculating these derivatives one by one, backprop uses Dynamic Programming. It calculates the gradient for the final layer and caches it. 
the layer before it uses that stored value to calculate its own gradient, avoiding redundant calculations.




Backpropagation is **not an algorithm separate from gradient descent**.  
It is simply an **efficient way to compute gradients** of the loss with respect to every parameter.

At its core, backprop answers one question repeatedly:

> If I change this parameter slightly, how much does the final loss change?




## The Core Idea

Every neural network is just a **composition of functions**.

Example (single neuron):

z = wx + b  
a = σ(z)  
L = loss(a)

The loss does **not** depend directly on w.  
It depends on a → which depends on z → which depends on w.

So gradients must flow **backward through these dependencies**.

This is exactly what the **chain rule** does.


## Why Gradients Flow Backward

During the forward pass:
- Information flows from input → output
- We compute activations and final loss

During the backward pass:
- Influence flows from loss → parameters
- We ask: *which parameters caused this loss?*

Each layer receives a gradient from the layer ahead of it and:
1. Uses it to compute its own parameter gradients
2. Passes a new gradient further backward