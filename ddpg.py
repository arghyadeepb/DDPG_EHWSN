import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import CriticNetwork, ActorNetwork

class Agent:
    def __init__(self, input_dims, nodes, lam, lamE, rng, alpha=0.001, beta=0.002, env=None,
            gamma=0.99, n_actions=2, max_size = 1000, tau=0.005,
            fc1=400, fc2=300, batch_size=64, noise=0.1, scale=False):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.nodes = nodes
        self.lam = lam
        self.lamE = lamE
        self.rng = rng
        self.max_action = self.rng*np.ones((1, self.nodes**2))
        self.min_action = np.zeros((1, self.nodes**2))
        if scale==False:
            self.name_str = '_Nodes_' + str(nodes) + '_Range_' + str(rng) + '_lam_' + str(list(lam)) + '_lam_' + str(lamE)
        else:
            self.name_str = '_Nodes_' + str(nodes) + '_Range_' + str(rng)

        self.actor = ActorNetwork(n_actions=n_actions, nodes=self.nodes, rng=self.rng,
                                    fc1_dims=fc1, fc2_dims=fc2, name='actor'+self.name_str)
        self.critic = CriticNetwork(n_actions=n_actions, nodes=self.nodes,
                                    fc1_dims=fc1, fc2_dims=fc2, name='critic'+self.name_str)
        self.target_actor = ActorNetwork(n_actions=n_actions, nodes=self.nodes, rng=self.rng,
                                    fc1_dims=fc1, fc2_dims=fc2, name='target_actor'+self.name_str)
        self.target_critic = CriticNetwork(n_actions=n_actions, nodes=self.nodes,
                                    fc1_dims=fc1, fc2_dims=fc2, name='target_critic'+self.name_str)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def save_models(self):
        print()
        print('...Saving Models...')
        print()
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
    
    def save_final_models(self):
        print()
        print('...Saving Models...')
        print()
        self.actor.save_weights(self.actor.checkpoint_file_final)
        self.critic.save_weights(self.critic.checkpoint_file_final)
        self.target_actor.save_weights(self.target_actor.checkpoint_file_final)
        self.target_critic.save_weights(self.target_critic.checkpoint_file_final)
    
    def load_models(self):
        print('...loading models...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
    
    def load_final_models(self):
        print('...loading models...')
        self.actor.load_weights(self.actor.checkpoint_file_final)
        self.critic.load_weights(self.critic.checkpoint_file_final)
        self.target_actor.load_weights(self.target_actor.checkpoint_file_final)
        self.target_critic.load_weights(self.target_critic.checkpoint_file_final)
    
    def choose_action(self, observation, noise, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
