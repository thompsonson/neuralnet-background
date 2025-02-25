# [NN Series 2/n] Circuits that can be trained to match patterns: The Adaline

In the [previous post](https://matt.thompson.gr/2025/02/12/nn-series-n-from-neurons.html) we saw that Frank Rosenblatt introduced the Perceptron as an alternative to the approaches taken with Boolean Algebra and symbolic logic. His work has proved to be a cornerstone of the modern day artificial neuron however there is a significant addition from the Boolean Algebra approach, from work done by Bernard Widron and Macrian E. (Ted) Hoff in 1960 \- [ADAPTIVE SWITCHING CIRCUITS \- Bernard Widrow](https://www-isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf).

## Building on the shoulders of giants

The Adaline version of an artificial neuron comes from the work on Adaptive Switching Circuits and contains an additional function to the Perceptron; that of the Activation Function. What I find interesting is that the paper didn’t focus on this in the way I expected (i.e. to enable multi-layer networks) and working backwards I was confused about what actually occurred.

I have heard three different benefits from this paper, this is in the order I learned about them:

1. The Adaline introduces a **non-linear activation function**. This is very important as it enables a multi-layered network of neurons to be created. Without a non-linear activation function all of the layers could be reduced to a signal layer that acts as a function of the input.   
2. The Adaline algorithm illustrates the key concept of **defining and minimising cost functions**. This capability is fundamental for more advanced machine learning algorithms including logistic regression (classification models), support vector machines, and regression models.  
3. **An Adaptive Pattern Classifier**. I like the old style text so I include a screenshot of a key section here. The concept here is interesting and has me wondering if this is the root of the current wave of “Justism” for saying AI is *just a pattern matcher*. 

![Adaptive Pattern Classifier](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/adaline/Adaptive_Pattern_Matching.png)  


## Side notes on Justism: 

Justism is a term I have heard via the paper linked in the first bullet point and is a term used to describe the arguments that an AI cannot do X Y, or Z as it is "just something something".

- this paper [OSF Preprints | A rebuttal of two common deflationary stances against LLM cognition](https://osf.io/preprints/osf/y34ur_v2) covers two other Justism arguments for dismissing AI  
- [Here are links](https://matt.thompson.gr/2025/01/31/proof-llms-an-llm-learning.html) to Quanta articles and related papers on LLMs learning skills rather than being stochastic parrots

## So what is an Adaline Neuron?

The Adaptive Linear Neuron was introduced as a hardware unit and has this schematic. 

![Schematic of Adaline](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/adaline/adaline_schematic.png)  

And with this there was one-dimensional searching of the solution space.   

![one-dimensional searching of the solution space](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/adaline/solution_space.png)

The patterns being matched were basic representations of letters  

![Patterns for classification experiment](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/adaline/patterns.png)

With the project tracking the Mean Square Error (MSE) over the number of input patterns  

![Adaptive-element performance curve](https://raw.githubusercontent.com/thompsonson/neuralnet-background/refs/heads/main/images/adaline/mse_per_pattern.png)

## A quick look back and then forward

In looking at this paper, I found myself interested in the work/papers that support it. Two particular findings have popped up:

- Shannon’s work on [A Symbolic Analysis of Relay \- and Switching Circuits" \- Claude E. Shannon](https://www.cs.virginia.edu/~evans/greatworks/shannon38.pdf)    
- The connection back to George Boole’s work on linking Mathematics and Logic (which before was solely in the area of Philosophy).

Next I’m going through the Adaline as Python code. I’ve not added it here as the code, from Sebastien Raschka’s book Python Machine Learning, includes a little bit on Feature Normalisation. 

I found the Feature Normalisation quite significant and have dug into it a bit more than the book goes into.    

## conclusion

This post is purposefull short and too the point as I was confused between the circumsance for the Adaline and how it has become used. The confusion is also relevant with the naming conventions (Perceptron is used for Multilayer networks when a string argument can be made that it should be Multilayer Adaline). 

As such I've focused on getting clarity of what was proposed in the paper and how it benefits us today (trainable and non-linear).

Closing out why it's short: in writing and experiementing with the code for an Adaline I looked at a lot of different hyperparameters and Feature Normalisation. By splitting them I feel the key concepts of both can be emphased (remembered!!)
