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

