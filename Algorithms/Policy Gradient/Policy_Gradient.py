"""
Reinforcement Learning using Policy Gradient with 3 FC layers
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


class PolicyGradient():
    def __init__(self, n_x, n_y, learning_rate=0.01, reward_decay=0.95, load_path=None, save_path=None):
        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay

        # path to save
        self.save_path = None
        if self.save_path is not None:
            self.save_path = save_path

        # episode data
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []

        # ...
        self.build_network()
        self.cost_history = []
        self.sess = tf.Session()

        # tensor-board -- log_dir = logs
        tf.summary.FileWriter("data/model/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # saver
        self.saver = tf.train.Saver()

        # restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)

    def store_transition(self, ob, a, r):
        """
        store play memory for training
        :param s: observations
        :param a: action taken
        :param r: reward after action
        :return:
        """
        self.episode_observations.append(ob)
        self.episode_rewards.append(r)
        # store actions as a list of arrays
        action = np.zeros(self.n_y)
        action[a] = 1
        self.episode_actions.append(action)

    def choose_action(self, observation):
        """
        choose an action based on observation
        :param observation: state function
        :return: index of action the agent choose
        """
        # reshape the observation to (num_features, 1)
        observation = observation[:, np.newaxis]

        # run forward propagation to get softmax probability
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict={self.X: observation})

        # select action randomly
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())

        return action

    def learn(self):
        # discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
            self.X: np.vstack(self.episode_observations).T,
            self.Y: np.vstack(np.array(self.episode_actions)).T,
            self.discounted_episode_rewards_norm: discounted_episode_rewards_norm
        })

        # reset the episode data
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []

        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            print("Model saved in file: %s" % save_path)

        return discounted_episode_rewards_norm

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        cumulative = 0                             # vt: reward
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative

        """######################## check if normalization works ########################"""
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    def build_network(self):
        # create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # Initialize parameters
        units_layer_1 = 10
        units_layer_2 = 10
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [units_layer_1, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [units_layer_2, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [self.n_y, units_layer_2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [self.n_y, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1, self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)
            A3 = tf.nn.softmax(Z3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(Z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')

        """######################## train some new loss function ########################"""
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)         # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        # plot cost - episode
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Episodes')
        plt.show()

        # plot reward - episode
        plt.plot(self.episode_rewards, self.cost_history)
        plt.ylabel('Reward')
        plt.xlabel('Training Episode')
        plt.show()





