#Accuracy 94.56%
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
# creates nodes in a graph
# "construction phase"
# x1 = tf.constant(5)
# x2 = tf.constant(6)
#
# result = tf.multiply(x1,x2)
# print(result)

# with tf.Session() as sess:
#   with tf.device("/gpu:1"):
#     matrix1 = tf.constant([[3., 3.]])
#     matrix2 = tf.constant([[2.],[2.]])
#     product = tf.matmul(matrix1, matrix2)
#
# print(product)

#layer sizes
n_nodes_hl1 = 1024
n_nodes_hl2 = 1024
n_nodes_hl3 = 1024
n_classes = 10

batch_size = 100

#graph shapes
condensed_size = 784 #28*28 into single row vector
x = tf.placeholder('float', [None, condensed_size])
y = tf.placeholder('float')

#model
def neural_network_model(data):

    #layer size and propagation dictionaries
    # hidden_layer_sizes = [condensed_size, n_nodes_hl1, n_nodes_hl2, n_nodes_hl3, n_classes]
    # hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer = 0,0,0,0
    # hidden_layers = [data, hidden_1_layer, hidden_2_layer, hidden_3_layer, output_layer]
    # for _ in range(1, len(hidden_layer_sizes)):
    #     hidden_layers[_] = {'weights':tf.Variable(tf.random_normal([hidden_layer_sizes[_-1], hidden_layer_sizes[_]])),
    #                   'biases':tf.Variable(tf.random_normal([hidden_layer_sizes[_]]))}



    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([condensed_size, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    #
    # l1,l2,l3,output = 0,0,0,0
    # layers = [data,l1,l2,l3,output]
    # for _ in range(1, len(layers) - 1):
    #
    #     layers[_] = tf.add(tf.matmul(layers[_ - 1],hidden_layers[_]['weights']), hidden_layers[_]['biases'])
    #     layers[_] = tf.nn.relu(layers[_])

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)


    # output = tf.matmul(layers[len(layers) -1],hidden_layers[len(hidden_layers) -1]['weights']) + hidden_layers[len(hidden_layers) -1 ]['biases']

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


#training section:

def train_neural_network(x):
    prediction = neural_network_model(x) #use previously made model
    # define cost function as reduce mean of cross entropy (difference) between prediction and y label(given correct answer)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost) #optimization model (backpropagation) = randomized gradient descent

    hm_epochs = 3 #cycles
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #run initializer

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):  #run training
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Cycle', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            # track correct count
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x) #run program