import os
from platform import node
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, nodes=1, n_actions=1,
            name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.nodes=nodes
        self.n_actions=n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.inp = Input(shape=(2*self.nodes+self.n_actions,))
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        self.inp = tf.concat([state, action], axis=1)# shape (1,2,1)
        action_value = self.fc1(self.inp)
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, nodes=1,
            rng=10, name='actor',chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.nodes = nodes
        self.n_actions = n_actions
        self.rng = rng

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.inp = Input(shape=(2*self.nodes,))
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        self.inp = state
        prob = self.fc1(self.inp)
        prob = self.fc2(prob)

        mu = self.mu(prob)*(self.rng/2)+(self.rng/2)

        return mu