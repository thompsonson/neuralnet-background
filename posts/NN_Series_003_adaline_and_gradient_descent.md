# [NN Series 3/n] Calculating the error before quantisation: Gradient Descent

Next I'm looking at the Adaline in python code. This post is a mixture of what I've learnt in my degree, Sebestien Raschka's book/code, and the 1960 paper that delivered the Adaline Neuron.

## Difference between the Perceptron and the Adaline

In the [first post](https://matt.thompson.gr/2025/02/12/nn-series-n-from-neurons.html) we looked at the Perceptron as a flow of inputs (x), multiplied by weights (w), then summed in the Aggregation Function and finally quantised in the Threshold Function.

![Computation Graph of a Perceptron](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/perceptron/perceptron.png)

The error for fitting/training came from the difference of the final output to the desired output. The python code to do that is quite simple (see [here](https://github.com/thompsonson/neuralnet-background/blob/main/perceptron.ipynb?short_path=a0eec92#L133) for context):

```python
update = self.eta * (target - self.predict(xi))
```

In the [second post](https://matt.thompson.gr/2025/02/24/nn-series-n-circuits-that.html) we looked at the different approach the Adaline takes, getting the error from an Activiation Function before the output is quantised with the Threshold Function.

![Computational Graph of Adaline](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/adaline/adaline_computional_graph.png)

The python code becomes slightly more complex as the derivative of the cost (error) with respect to (wrt) the weights needs to be calculated.

## Python code for an Adaline Neuron

Due to the additional complexity of the code in the Adaline I've modified Sebestien's code to be more explicit and link up to calculating the dervitive and the chain rule.

Implementation steps:
1. Initialize weights with small random values
2. For each epoch:
    - a. Calculate the net input (weighted sum via the aggregation function)
    - b. Apply activation function (in this case it is the identity function)
    - c. Calculate errors (difference between actual and predicted)
    - d. Calculate the derivative (gradient) of the cost function wrt the weights and bias
    - e. Update all weights and biases based on the derivatives multipled by the learning rate
    - f. Calculate and store cost for this epoch

Note: the deriveative of the cost function wrt the weights is calculated using the chain rule:

> ∂E/∂w_j = ∂E/∂φ * ∂φ/∂z * ∂z/∂w_j

where:

> ∂E/∂φ = -(y - φ) = (φ - y)  _ [The actual output (**aggregation_function**) minus the desired output (**y**)]
>
> ∂φ/∂z = 1   _________________ [The derivative of the activation (**activation_function**) wrt the net input (**aggregation_function**)]
>
> ∂z/∂w_j = x_j _______________ [The derivative of the net input (**aggregation_function**) wrt the jth input (**X_j**)]

Here we jump to using numpy in the code.

The important thing to remember is the Jacobian matrix is the first order derivative of the cost function.

Rather than looping though each input (also called a feature) and updating the weights, we can use matrix multiplication to update all weights at once.

The matrix multiplication is done by taking the dot product of the transpose of the input matrix (`X.T`) and the error vector (`y - self.activation_function(self.aggregation_function(X))`).

This reduces to this code: `X.T.dot(X.dot(w) - y) = X.T.dot(y - self.activation_function(self.aggregation_function(X)))`

and using numpy here's the actual code

```python
        rgen = np.random.RandomState(self.random_state)
        # Step 1: Initialize weights with small random values
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            # Step 2a: Calculate net input (weighted sum via aggregation_function)
            net_input = self.aggregation_function(X)

            # Step 2b: Apply activation function (in this case it is the identity function)
            output = self.activation_function(net_input)

            # Step 2c: Calculate prediction error
            error_vector = output - y

            # Step 2d: Calculate the derivative (gradient) of the cost function wrt the weights and bias
            # using the chain rule: ∂E/∂w = X.T.dot(output - y)
            derivative_cost_wrt_weights = X.T.dot(error_vector)
            derivative_cost_wrt_bias = error_vector.sum()

            # Step 2e: Apply standard gradient descent update: w = w - eta * gradient
            self.w_[1:] -= self.eta * derivative_cost_wrt_weights
            self.w_[0] -= self.eta * derivative_cost_wrt_bias

            # Step 2f: Calculate and store cost
            cost = (error_vector**2).sum() / 2.0  # Sum of squared errors / 2
            self.cost_.append(cost)
        return self
```

## Taking the new Neuron for a test drive

The full code is available [here](https://github.com/thompsonson/neuralnet-background/blob/main/adaline.ipynb), this includes the investigation into the affect of changing the learning rate on the optimisation and then the decision boundary.

After seeing that there were clear differences between learning rates like ` [0.001, 0.0006, 0.0005, 0.0002, 0.00009]` I grouped the training into one function to compare what occurred.

## Finding a learning rate that converges on the optimal weights

The below diagram clearly shows that a small learning rate (eta below 0.0002) is needed to achieve a stable model.

![Training with different learning rates](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/adaline/training.png)

What piqued my interest was the larger learning rates that started to descend but then overran the minima and did not converge. A learning rate of 0.0005 best shows this

## Close but no cigar

I ran an experiment on an Adaline with a learning rate (eta) of `0.0005` for six different epochs `[5, 7, 8, 40, 50, 100],`.

In these graphs we can see the error descend, the decision boundary improves (the first 3 graphs), however somewhere around epoch 25 the error starts to ascend and the decision boundary passes over the top of all of the data, clearly not providing any usable classification for this data.

![Decision Boundary for eta=0.0005](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/adaline/decision_boundary.png)

Cleary we have the left hand side with this learning rate of `0.0005` (and those greater than it).

![Gradient Descent vs Ascent](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/adaline/gradient_decent_and_ascent.png)

## Conclusion

So this post has been about two things:

1. Bridging the Maths of Gradient Descent, the design of the Adaline, and the Python code implementation of both of these concepts.
2. Understanding the impact of learning rates on the training

Next up I'm sharing code that will implement Feature Normalisation and show the benefits of that on training a model.





