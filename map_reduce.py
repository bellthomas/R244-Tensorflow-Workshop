import numpy as np
import tensorflow as tf
import threading

# Set to False to kill all workers started with start_worker().
running = True

# Create worker, ready for computation.
def start_worker(index, hosts):
    task = int(index)
    port = 2222 + index
    cluster_spec = tf.train.ClusterSpec({"local": hosts})
    server = tf.distribute.Server(cluster_spec, job_name="local", task_index=task)
    while True:
        global running
        if not running: 
            break

# Create a cluster spec matching your server spec
num_workers = 6
hosts = ["localhost:{}".format(2222 + i) for i in range(0, num_workers)]
for worker in range(0, num_workers):
    threading.Thread(target=start_worker, args=(worker, hosts)).start()
    print("Task {} started...".format(worker))

# Input data specification.
data_size = 10_000_000
task_input = tf.compat.v1.placeholder(tf.float32, data_size, name="task_input")

# Partition equally amongst workers and distribute.
partition_size = (data_size // num_workers)
calculated_means = [0 for i in range(0, num_workers)]
for worker in range(0, num_workers):   
    with tf.device("/job:local/task:{}".format(worker)):
        start_index = worker * partition_size
        local_input = tf.slice(task_input, [start_index], [partition_size])
        calculated_means[worker] = tf.reduce_mean(local_input)

# Remote session specification.
with tf.compat.v1.Session("grpc://localhost:2222") as sess:
    # Sample data and compute the overall result by combining both results.
    data = np.random.random(data_size)
    mean = tf.math.divide(tf.math.reduce_sum(calculated_means), num_workers)

    # Run the session to compute the overall using your workers
    # and the input data. Output the result.
    result = sess.run(mean, feed_dict={ task_input: data })
    running = False
    print("\nCompleted: " + str(result))
    print("Servers shutdown.")
    