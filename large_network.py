import tensorflow as tf
import numpy as np


class LargeNetwork:

    n_classes = 12
    lr = 3e-4
    n_epochs = 1
    n_batches = 3
    batch_size = 4
    n_map_features = 14
    n_player_features = 1

    def model(self, x1, x2):
        weights = {
            'W_conv1': tf.Variable(tf.random_normal([3, 3, self.n_map_features, 128])),
            'W_conv2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
            'W_conv3': tf.Variable(tf.random_normal([3, 3, 128, 64])),
            'W_conv4': tf.Variable(tf.random_normal([3, 3, 64, 64])),
            'W_fc1': tf.Variable(tf.random_normal([11 * 11 * (self.n_map_features + self.n_player_features), 2048])),
            'W_fc2': tf.Variable(tf.random_normal([2048, 512])),
            'W_fc3': tf.Variable(tf.random_normal([512, 128])),
            'out': tf.Variable(tf.random_normal([128, self.n_classes])),
        }

        biases = {
            'b_conv1': tf.Variable(tf.random_normal([128])),
            'b_conv2': tf.Variable(tf.random_normal([128])),
            'b_conv3': tf.Variable(tf.random_normal([64])),
            'b_conv4': tf.Variable(tf.random_normal([64])),
            'b_f1': tf.Variable(tf.random_normal([2048])),
            'b_fc2': tf.Variable(tf.random_normal([512])),
            'b_fc3': tf.Variable(tf.random_normal([128])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        def _conv2d(x, W, b, strides=1):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

        # Reshape input to a 4D tensor
        x1 = tf.reshape(x1, shape=[-1, 11, 11, self.n_map_features])
        x2 = tf.reshape(x2, shape=[-1, 11, 11, self.n_player_features])

        conv1 = _conv2d(x1, weights['W_conv1'], biases['b_conv1'])
        conv2 = _conv2d(conv1, weights['W_conv2'], biases['b_conv2'])
        conv3 = _conv2d(conv2, weights['W_conv3'], biases['b_conv3'])
        conv4 = _conv2d(conv3, weights['W_conv4'], biases['b_conv4'])
        concat = tf.concat([conv4, x2], 3)
        reshaped = tf.reshape(concat, [-1, 11 * 11 * (self.n_map_features + self.n_player_features)])
        fc1 = tf.add(tf.matmul(reshaped, weights['W_fc1']), biases['b_f1'])
        fc2 = tf.add(tf.matmul(fc1, weights['W_fc2']), biases['b_f2'])
        fc3 = tf.add(tf.matmul(fc2, weights['W_fc3']), biases['b_f3'])
        out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])

        return out

    def run_model(self):
        # tf Graph input
        x1 = tf.placeholder(tf.float32, [None, 11, 11, self.n_map_features])
        x2 = tf.placeholder(tf.float32, [None, 11, 11, self.n_player_features])
        y = tf.placeholder(tf.float32, [None, self.n_classes])

        # Model
        logits = self.model(x1, x2)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.n_epochs):
                for batch in range(self.n_batches):
                    batch_x1 = np.zeros((self.batch_size, 11, 11, self.n_map_features))
                    batch_x2 = np.zeros((self.batch_size, 11, 11, self.n_player_features))
                    batch_y = np.zeros(self.n_classes)
                    sess.run(optimizer, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y})

                    # Calculate batch loss and accuracy
                    loss = sess.run(cost, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y})
                    valid_acc = sess.run(accuracy, feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y})

                    print('Epoch {:>2}, Batch {:>3} - Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                        epoch + 1,
                        batch + 1,
                        loss,
                        valid_acc))

            # Calculate Test Accuracy


LargeNetwork().run_model()
