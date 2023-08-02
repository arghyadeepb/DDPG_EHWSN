import gym
from gym import spaces
import numpy as np

class Env(gym.Env):
    def __init__(self, rng, nodes, lam, lamE):

        self.rng = rng
        self.nodes = nodes
        self.lam = lam
        self.lamE = lamE

        self.Q = np.zeros(self.nodes)
        self.E = np.zeros(self.nodes)
        self.state = np.zeros(2*self.nodes)

        self.state_dims = []
        self.action_dims = []
        for i in range(self.nodes):
            self.state_dims.append(spaces.Discrete(self.rng+1))
            self.state_dims.append(spaces.Discrete(self.rng+1))
            for i in range(self.nodes):
                self.action_dims.append(spaces.Discrete(self.rng+1))

        self.action_space = spaces.Box( np.zeros((self.nodes, self.nodes, self.rng)),
                            10*np.ones((self.nodes, self.nodes, self.rng)) )
        # spaces.Tuple(tuple(self.action_dims)) 
        self.state_space = spaces.Box( np.zeros((self.nodes, self.nodes, self.rng)),
                            10*np.ones((self.nodes, self.nodes, self.rng)) )
        # spaces.Tuple(tuple(self.state_dims))
    
    def trans_func(self,x):
        return np.log2(1+x)
    
    def cut(self,x):
        x = np.maximum(np.zeros(len(x)),np.minimum(x,self.rng*np.ones(len(x))))
        return x # Numpy Clip
    
    def next_state(self,action):
        X =[]
        Y =[]
        for i in range(self.nodes):
            X.append(np.random.poisson(self.lam[i]))
            Y.append(np.random.poisson(self.lamE[i]))
        X, Y = np.array(X), np.array(Y)
        for i in range(self.nodes):
            if np.sum(action[i])!=0:
                action[i] = (self.E[i]+Y[i])*(action[i]/np.sum(action[i]))
            else:
                pass
        E_used = np.sum(action, axis=0)
        E_spent = np.sum(action, axis=1)
        #E_transmit = np.diag(action)
        self.Q = self.Q - self.trans_func(E_used) + X
        self.E = self.E - E_spent + Y
        Q = self.cut(self.Q)###################not done sharing
        E = self.cut(self.E)
        marginQ = self.Q-Q
        marginE = self.E-E
        self.Q = Q
        self.E = E
        next = np.append(self.Q,self.E)
        LossQ = np.sum(marginQ[np.where(marginQ>0)])
        LossE = np.sum(marginE[np.where(marginE>0)])
        return next, LossQ, LossE

    def reward(self, next, action, lossQ, lossE):
        R = -np.sum(np.square(next[:self.nodes])) - 10*(lossQ**2)
        # R = -np.sum(next[:self.nodes])
        # R = -np.sum(np.square(next[:self.nodes]))
        return R
    
    def step(self, action):

        next, lossq, losse = self.next_state(action)
        reward = self.reward(next, action, lossq, losse)
        done = False
        info = {}

        return  next, reward, done, lossq
    
    def reset(self):
        state = np.random.rand(2*self.nodes)*self.rng
        #state = np.zeros(2*self.nodes)
        return state