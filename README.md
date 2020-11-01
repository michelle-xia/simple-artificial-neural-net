# simple-artificial-neural-net
 The purpose of this project is to gain a deeper understanding of the inner workings of artificial neural networks. In order to accomplish this, I created a simple artificial neural net from scratch using a quadratic function on numbers 1-5 without any ML libraries.

# Data
I used a simple 5 x 1 matrix with numbers 1, 2, 3, 4, 5 as input, two 5 x 5 weight matrices containing randomized floats between 0 and 1, and a 5 x 1 matrix with 1, 4, 9, 16, 25 as the target.

# Method
1) I calculated two layers of output, z2 and z3 with matrix multiplication of input * weights. For the transformation function, I used sigmoid (1 / (1 + e^-x)) to get a2 and a3 (output). Sigmoid is not the best activation function, so I'm thinking about changing this to a ReLU.
2) Output was compared to the target 5 x 1 matrix and sum of squared error (SSE) was calculated. SSE drives the training process.
3) To backpropagate, I took the partial derivative of the SSE equation with respect to each of the weights, multiplied this derivative by a convergence factor, and subtracted it to update both sets of weights. 
4) New output is computed and SSE is updated with these new values as long as SSE is greater than the accepted error.
