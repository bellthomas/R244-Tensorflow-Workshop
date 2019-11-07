import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create data to fit.
samples = 100
xs = np.linspace(-5, 5, samples)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, samples)

# First, create TensorFlow placeholders for input data (xs) and
# output (ys) data. Placeholders are inputs to the computation graph.
# When we run the graph, we need to feed values for the placerholders into the graph.
X = tf.compat.v1.placeholder(tf.float64, name='X', shape=None)
Y = tf.compat.v1.placeholder(tf.float64, name='Y', shape=None)


# We will try minimzing the mean squared error between our predictions and the
# output. Our predictions will take the form X*W + b, where X is input data,
# W are ou weights, and b is a bias term:
# minimize ||(X*w + b) - y||^2
# To do so, you will need to create some variables for W and b. Variables
# need to be initialised; often a normal distribution is used for this.
w = tf.Variable(name='w', initial_value=np.random.normal(size=(1,)))
b = tf.Variable(name='b', initial_value=np.random.normal(size=(1,)))


# Next, you need to create a node in the graph combining the variables to predict
# the output: Y = X * w + b. Find the appropriate TensorFlow operations to do so.
predictions = tf.math.add(tf.math.multiply(X, w), b) 


# Finally, we need to define a loss that can be minimized using gradient descent:
# The loss should be the mean squared difference between predictions
# and outputs.
loss = tf.math.reduce_mean(tf.math.squared_difference(predictions, Y))
# loss = tf.reduce_sum(tf.pow(predictions - Y, 2))/(samples)


# Use gradient descent to optimize your variables
learning_rate = 0.001
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# We create a session to use the graph and initialize all variables
session = tf.compat.v1.Session()
session.run(tf.compat.v1.global_variables_initializer())

# Optimisation loop
epochs = 1000
previous_loss = 0.0
for epoch in range(epochs):
    for (inputs, outputs) in zip(xs, ys):
        session.run(optimizer, feed_dict={vX: inputs, Y: outputs }) 
      

    training_cost = session.run(loss, feed_dict={ X: xs, Y: ys }) 
    print('Training cost = {}'.format(training_cost))

    # Termination condition for the optimization loop
    if np.abs(previous_loss - training_cost) < 0.000001:
        break

    previous_loss = training_cost


predictions_y = session.run(predictions, feed_dict={ X: xs })
plt.plot(xs, predictions_y)
plt.show()

# For Tensorboard.
# writer = tf.summary.FileWriter('tmp', graph=tf.get_default_graph())
