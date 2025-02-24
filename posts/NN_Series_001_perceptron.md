# [NN Series 1/n] From Neurons to Neural Networks: The Perceptron

This post looks at the Percepton, from Frank Rosenblatt's original paper to a practical implementation classifying Iris flowers.

The Perceptron is the original Artificial Neuron and provided a way to train a model to classify linearly separable data sets.

The Perceptron itself had a short life, with the Adaline coming in 3 years later. However it's name lives on as neural networks have, Multilayer Perceptrons (MLPs). The naming shows the importance of this discovery. 

## Background to Artificial Neurons: The Perceptron

Frank Rosenblatt introduced the Perceptron, an artificial neuron, to the world outside of the US Navy in a 1958 Psychology Review article titled [The Perceptron: A probabilistic model for information storage and organisation in the Brain](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf).

In the article he discussed the second and third these questions: 

    1. How is information about the physical world sensed, or detected, by the biological system?
    2. In what form is information stored, or remembered?
    3. How does information contained in storage, or in memory, influence recognition and behavior?

### Information storage

There are a few key sections that really speak to me, here he highlights the information retained is connected to the input (it is stored as a preferred response) and concludes connections are important:

    Whatever information is retained must somehow be stored as a preference for a particular response; i.e., the information is contained in connections or associations rather than topographic representation.

### Probabilistic approach

He looked at the work of other researchers work, particularlly around symbolic logic and thought another model is needed, choosing probability theory: 

    Unfortunately, the language of symbolic logic and Boolean algebra is less well suited for such investigations. The need for a suitable language for the mathematical analysis of events in systems where only the gross organization can be characterized, and the precise structure is unknown, has led the author to formulate the current model in terms of probability theory rather than symbolic logic.

At the time, the preferred approach was to first create deterministic models of perception (perceiving and recognising stimuli) and then tweak these models to explain the workings of a more realistic nervous system. 

Rosenblatt decided to do it the opposite way, he wrote: 

    that a mere refinement or improvement of the principles already suggested can never account for biological intelligence; a difference in principle is clearly indicated.

### The work Rosenblatt built on

He was clear that previous work on the subject of learning lacked the rigor of similar work using boolean algebra and proved to be an obstacle. However he highlights the current position as one with the following assumptions (see the paper for the names and references associated to previous work):

    1. The physical connections of the nervous system which are involved in learning and recognition are not identical from one organism to another. At birth, the construction of the most important networks is largely random, subject to a minimum number of genetic constraints.

    2. The original system of connected cells is capable of a certain amount of plasticity; after a period of neural activity, the probability that a stimulus applied to one set of cells will cause a response in some other set is likely to change, due to some relatively long-lasting changes in the neurons themselves.

    3. Through exposure to a large sample of stimuli, those which are most "similar" (in some sense which must be defined in terms of the particular physical system) will tend to form pathways to the same sets of responding cells. Those which are markedly "dissimilar" will tend to develop connections to different sets of responding cells.

    4. The application of positive and/or negative reinforcement (or stimuli which serve this function) may facilitate or hinder whatever formation of connections is currently in progress.

    5. Similarity, in such a system, is represented at some level of the nervous system by a tendency of similar stimuli to activate the same sets of cells. Similarity is not a necessary attribute of particular formal or geometrical classes of stimuli, but depends on the physical organization of the perceiving system, an organization which evolves through interaction with a given environment. The structure of the system, as well as the ecology of the stimulus-environment, will affect, and will largely determine, the classes of "things" into which the perceptual world is divided.

### Adding Mathematical rigour

He goes into a lot of detail to set up the notation he uses to explain the connections from the _sensory units_ (S) through to _association cells_ (A) the _responses_ (R). Below is an example from the article that:

    has only two responses, but there is clearly no limit on the number that might be included.

![Figure2](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/perceptron/Neuron_and_venn_diagram.png)

This is clearly insightful

    If such a system is to be capable of learning, then it must be possible to modify the A-units or their connections in such a way that stimuli of one class will tend to evoke a stronger impulse in the Ri source-set than in the Ra source-set, stimuli of another (dissimilar) class will tend to evoke a stronger impulse in the Ra source-set than in the Ri source-set.

### Key conclusions

He goes into a lot of detail, which I do not cover here, instead I'd like to highlight two quotes from his conclusion:

    In an environment of random stimuli, a system consisting of randomly connected units, subject to the parametric constraints discussed above, can learn to associate specific responses to specific stimuli. 

and

    Verifiability. Previous quantitative learning theories, apparently without exception, have had one important characteristic in common: they have all been based on measurements of behavior, in specified situations, using these measurements (after theoretical manipulation) to predict behavior in other situations. Such a procedure, in the last analysis, amounts to a process of curve fitting and extrapolation, in the hope that the constants which describe one set of curves will hold good for other curves in other situations. 

The first shows that Rosenblatt saw the Perceptron as a tool for learning in any environment. 

The second shows how he was breaking from the current methodology of observation to an independent explanation of how learning occurs. 

Clearly this is not an exact model of how organsims learn but for better or worse, it was the neuron that has gone into the massive Data Centre sized artificial "brains" we have now.

## The Perceptron model 

### Computation Graph of a Perceptron

![Computation Graph of a Perceptron](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/perceptron/perceptron.png)

The character 𝚫 (Delta) represents a change or update in the weight w_k. This represents the adjustment made to the weights during the training process.

Where:
- x_k: The input feature corresponding to the weight w_k.
- w_k: The weight w_k of the perceptron.
- y is the known desired  output.
- ŷ (y_hat) is the predicted output. 
- 𝚫w_k: The change in the weight w_k.
- Ƞ (eta): The learning rate, which controls the size of the weight updates.
- error: The difference between the predicted value (ŷ) and the desired  value (y).

Components:
- Aggregate function: a summation of the outputs of x . w. 
- Threshold function: sets the predicted value (ŷ) to 1 or -1 depending on the output of the aggregation function.

### The Perceptron in code

```python
# code comes from Sebastian Raschka's Python Machine Learning book


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch

    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Training function that implements the perceptron learning rule.

        1. Initialize weights to zero
    
        For each epoch:

            2. For each sample:
                - Make prediction
                - Calculate error
                - Update weights if prediction is wrong
            3. Track number of errors per epoch
        
        4. return the instance of the Perceptron object

        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        ------------
        self : object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        The Aggregation Function - Calculates the weighted sum of inputs (dot product).
        Formula: z = w0 + w1x1 + w2x2 + ... + wnxn
        where w0 is the bias unit and w1...wn are the weights
        """
        # uses numpy's dot method to calculate the dot product of the inputs X and the weights w_
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        The Threshold function - Implements the unit step function (threshold).
        Returns:
        1 if net input >= 0
        -1 if net input < 0
        """
        # uses numpy's where method to return 1 if net_input is greater than or equal to 0, otherwise -1
        return np.where(self.net_input(X) >= 0.0, 1, -1)
```

## Practical application

### Data preparation

```python
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

df.tail()
```

Here's an example output:

| Index | Sepal Length | Sepal Width | Petal Length | Petal Width | Species |
|-------|--------------|-------------|--------------|-------------|----------|
| 145 | 6.7 | 3.0 | 5.2 | 2.3 | Iris-virginica |
| 146 | 6.3 | 2.5 | 5.0 | 1.9 | Iris-virginica |
| 147 | 6.5 | 3.0 | 5.2 | 2.0 | Iris-virginica |
| 148 | 6.2 | 3.4 | 5.4 | 2.3 | Iris-virginica |
| 149 | 5.9 | 3.0 | 5.1 | 1.8 | Iris-virginica |

and we can plot all of the data using MatLab pyplot

```python
# Extract the first hundred class labels (it is known that 50 are Iris-setosa and 50 are Iris-virginica)
y = df.iloc[0:100, 4].values
# update the class labels to -1 or 1
y = np.where(y == 'Iris-setosa', -1, 1)
# Extract the first hundred features in columns 0 and 1, representively representing sepal length and petal length 
X = df.iloc[0:100, [0, 2]].values

# now plot the data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
```
![Data to train the Perceptron](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/perceptron/output_data.png)

note: this data is clearly linearly separable

### Training the Perceptron to classify Iris flowers

```python
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
```

We can plot the training curve by epoch with the `errors_` attribute in the Perceptron.

```python
# plot the errors per epoch

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
```

![Trainin Curve](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/perceptron/training_curve.png)

Here we can see that the training converges on a solution after six epochs.

Seeing the convergence is positive, visualizing the decision boundary helps understand how the Perceptron separates the classes.

### Viewing the Decision Boundary

Here I deviate slightly from the original example. I am new to numpy so I needed to get an understanding of the methods `arange`, `meshgrid`, and `ravel`. I also misunderstood the reason the original function was using the full X and y data and calling the predict function on the Perceptron classifier.

I skip the work I did in this post, you can see that [in this Juypter notebook](https://github.com/thompsonson/neuralnet-background/blob/main/perceptron.ipynb).

Here is the update code

```python

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    Plot decision regions for a classifier in a 2D feature space.

    Parameters:
    X : array-like, shape = [n_samples, n_features]
        Feature matrix.
    y : array-like, shape = [n_samples]
        Target vector.
    classifier : object
        Trained classifier with a predict method.
    resolution : float, optional (default=0.02)
        Resolution of the mesh grid used to plot the decision surface.

    Returns:
    None

    This function visualizes the decision boundaries of a classifier by plotting
    the decision surface in a 2D feature space.
    """
    # setup marker generator and color map
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # calculate a grid of the features (feature_1 is the sepal length and feature_2 is the petal length)
    
    # firstly get the min and max of each feature
    feature_1_min, feature_1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    feature_2_min, feature_2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # Create a fine grid of points covering our feature space so we can visualize
    # how the perceptron classifies every possible combination of feature values
    feature_1_values, feature_2_values = np.meshgrid(np.arange(feature_1_min, feature_1_max, resolution),
                                                     np.arange(feature_2_min, feature_2_max, resolution))
    # now flatten the 2 dimensional arrays into a one dimension array for both features
    # the array is transposed so that each point in the grid is represented as a row
    feature_grid = np.array([feature_1_values.ravel(), feature_2_values.ravel()]).T
    
    # use the classifier to calculate the label at each point on the gird
    label_per_point_on_feature_grid = classifier.predict(feature_grid)
    # Reshape the predictions back to match our grid dimensions for plotting
    label_per_point_on_feature_grid = label_per_point_on_feature_grid.reshape(feature_1_values.shape)

    # plot the decision surface

    # Create a filled contour plot where different colors show the different predicted classes
    # alpha=0.4 makes the coloring semi-transparent so we can see the data points
    plt.contourf(feature_1_values, feature_2_values, label_per_point_on_feature_grid, alpha=0.4, cmap=cmap)
    # Set the plot limits to show the full decision boundary region
    plt.xlim(feature_1_values.min(), feature_1_values.max())
    plt.ylim(feature_2_values.min(), feature_2_values.max())
```
```python
def plot_class_samples(X, y, label_names):
    """
    Plot class samples in a 2D feature space.

    Parameters:
    X : array-like, shape = [n_samples, n_features]
        Feature matrix.
    y : array-like, shape = [n_samples]
        Target vector.
    label_names : list
        List of label names corresponding to the unique classes in y.

    Returns:
    None

    This function plots the class samples in a 2D feature space with a legend.
    """
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[cmap(idx)],
                    marker=markers[idx], label=label_names[idx])
    plt.legend()
```
```python
plot_decision_regions(X, y, classifier=ppn)
label_names = ['setosa', 'versicolor']
plot_class_samples(X, y, label_names)
# Label the axes with the feature names to show what we're plotting
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.show()
```

The output is a contour displaying the classification either side of the decision boundary

![Decision Boundary Plot](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/perceptron/decision_boundary.png)

## Summary

The Perceptron has the key concepts that make neural networks so useful: 

- weight initialization 
- learning through error correction
- binary classification
    
While limited to linearly separable problems, it provides building blocks for understanding more complex neural architectures. 

Rosenblatt's probabilistic approach and emphasis on learning from random stimuli remain relevant in modern deep learning, where neural networks learn general approximations from large, diverse datasets and decision boundaries span many dimensions.

