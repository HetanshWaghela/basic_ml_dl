Purpose of activation functions is to introduce non-linearity

## Sigmoid 

$\sigma(x) = \frac{1}{1 + e^{-x}}$

Range: (0,1)

When you take derivative of this, the answer is 0.25(at x=0). So the max value becomes 0.25. This is the root cause of Vanishing gradients issue.

To solve this, we use

## ReLU (Rectified Liner Unit) 
$$ \mathrm{ReLU}(x) = \max(0, x)$$

Range: [0, infinity)

For positive inputs, the derivative is exactly 1. This preserves the gradient magnitude through the layers, solving the vanishing gradient problem.

Weakness: For negative inputs , the output and gradient are 0. This leads to "dying ReLU" problem , where a neuron effectively dies and never updates because gradient remains zero.

## Tanh (Hyperbolic Tangent)
$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$

Unlike sigmoid, this is zero centered , which helps gradients flow more efficiently during optimization.




