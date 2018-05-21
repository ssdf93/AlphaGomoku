# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5

@author: Xiang Zhong
"""

import numpy as np
import tensorflow as tf


class PolicyValueNet():
    def __init__(self, board_width, board_height, training = True, model_file=None):
        self.board_width = board_width
        self.board_height = board_height

        with tf.device('/cpu:0'):
            # Define the tensorflow neural network
            # 1. Input:
            self.input_states = tf.placeholder(
                    tf.float32, shape=[None, 10, board_height, board_width])
            self.input_states_reshaped = tf.transpose(self.input_states, [0,2,3,1])

            print(self.input_states.get_shape())
            print(self.input_states_reshaped.get_shape())
            print("OK")
            first_conv = tf.layers.conv2d(inputs=self.input_states_reshaped,
                                      filters=128, kernel_size=[3, 3],
                                      padding="same", activation=tf.nn.relu)

            self.resnet_output = first_conv
            # 2. Common Networks Layers
            for i in range(2):
                resnet_input = self.resnet_output

                conv1 = tf.layers.conv2d(inputs=resnet_input, filters=128,
                                              kernel_size=[3, 3], padding="same",
                                              activation=tf.identity)
                conv1 = tf.nn.relu(conv1)
                conv1 = tf.layers.batch_normalization(conv1, training = training)
                print(conv1.get_shape())
                conv2 = tf.layers.conv2d(inputs=conv1, filters=128,
                                              kernel_size=[3, 3], padding="same",
                                              activation=tf.identity)
                conv2 = tf.layers.batch_normalization(conv2, training = training)
                self.resnet_output = tf.nn.relu(resnet_input + conv2)
                print(self.resnet_output.get_shape())

            # 3-1 Action Networks
            self.action_conv = tf.layers.conv2d(inputs=self.resnet_output, filters=4,
                                                kernel_size=[1, 1], padding="same",
                                                activation=tf.nn.relu)
            # Flatten the tensor
            self.action_conv_flat = tf.reshape(
                    self.action_conv, [-1, 4 * board_height * board_width])
            # 3-2 Full connected layer, the output is the log probability of moves
            # on each slot on the board
            self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                             units=board_height * board_width,
                                             activation=tf.nn.log_softmax)
            # 4 Evaluation Networks
            self.evaluation_conv = tf.layers.conv2d(inputs=self.resnet_output, filters=2,
                                                    kernel_size=[1, 1],
                                                    padding="same",
                                                    activation=tf.nn.relu)
            self.evaluation_conv_flat = tf.reshape(
                    self.evaluation_conv, [-1, 2 * board_height * board_width])
            self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                                  units=64, activation=tf.nn.relu)
            # output the score of evaluation on current state
            self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                                  units=1, activation=tf.nn.tanh)

            # Define the Loss function
            # 1. Label: the array containing if the game wins or not for each state
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])
            # 2. Predictions: the array containing the evaluation score of each state
            # which is self.evaluation_fc2
            # 3-1. Value Loss function
            self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                           self.evaluation_fc2)
            # 3-2. Policy Loss function
            self.mcts_probs = tf.placeholder(
                    tf.float32, shape=[None, board_height * board_width])
            self.policy_loss = tf.negative(tf.reduce_mean(
                    tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
            # 3-3. L2 penalty (regularization)
            l2_penalty_beta = 1e-5
            vars = tf.trainable_variables()
            l2_penalty = l2_penalty_beta * tf.add_n(
                [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
            # 3-4 Add up to be the Loss function
            self.loss = self.value_loss + self.policy_loss + l2_penalty

            # Define the optimizer we use for training
            self.learning_rate = tf.placeholder(tf.float32)
            self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss)


        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth=True
        tf_config.allow_soft_placement=True

        # Make a session
        self.session = tf.Session(config=tf_config)

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.tight_availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 10, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
