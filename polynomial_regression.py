import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate training data.
samples = 100
xs = np.linspace(-3, 3, samples)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, samples)

# Create plot and show truth points.
fig, axis = plt.subplots()
axis.set_ylim([-3, 3])
axis.scatter(xs, ys)
fig.show()
plt.draw()

# (x,y) variables.
X = tf.compat.v1.placeholder(tf.float64, name='X', shape=None)
Y = tf.compat.v1.placeholder(tf.float64, name='Y', shape=None)

# Prediction function.
predictions = tf.Variable(name='b', initial_value=np.random.normal(size=(1,)))
for power in range(1, 5):
    weighting = tf.Variable(name='w_%d' % power, initial_value=np.random.normal(size=(1,)))
    predictions = tf.add(
        tf.multiply(tf.pow(X, power), weighting),
        predictions
    )

# Los/cost function.
cost = tf.reduce_sum(tf.pow(predictions - Y, 2)) / (samples - 1)

# Initialise optimiser.
learning_rate = 0.01
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

epochs = 1000
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    previous_loss = 0.0
    for epoch in range(epochs):
        for (inputs, outputs) in zip(xs, ys):
            session.run(optimizer, feed_dict={ X: inputs, Y: outputs })

        training_cost = session.run(cost, feed_dict={ X: xs, Y: ys })
        
        # Feedback every 100 iterations.
        if epoch % 100 == 0:
            print("Epoch " + str(epoch) + ": " + str(training_cost))

        # Break if minimum reached.
        if np.abs(previous_loss - training_cost) < 0.000001:
            break

        previous_loss = training_cost
    
    # Plot fitted curve.
    prediction = predictions.eval(feed_dict={ X: xs })
    axis.plot(xs, prediction)
    fig.show()
    plt.draw()

fig.show()
plt.show()