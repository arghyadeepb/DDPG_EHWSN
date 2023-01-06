import numpy as np

class Env():
    def __init__(self, range, nodes, lam, lamE):

        self.range = range
        self.nodes = nodes
        self.lam = lam
        self.lamE = lamE

        self.Q = np.zeros(self.nodes)
        self.E = np.zeros(self.nodes)
        self.state = np.zeros(2*self.nodes)

        self.state_dims = []
        for i in range(self.nodes):
            self.state_dims.append(self.range+1)
            self.state_dims.append(self.range+1)
        
        self.action_dims = []
        for i in range(self.nodes):
            for i in range(self.nodes):
                self.action_dims.append(self.range+1)

        self.action_space = tuple(self.action_dims)
        #spaces.Box( np.zeros((self.nodes, self.nodes, self.range)),
        #  10*np.ones((self.nodes, self.nodes, self.range)) )
        self.state_space = tuple(self.state_dims)
    
    def trans_func(self,x):
        return np.log2(x)
    
    def cut(self,x):
        x = np.max(0,np.min(x,self.range))
        return x
    
    def next_state(self,action):
        E_used = np.sum(action, axis=0)
        E_transmit = np.diag(action)
        X =[]
        Y =[]
        for i in range(self.nodes):
            X.append(np.random.poisson(self.lam[i]))
            Y.append(np.random.poisson(self.lamE[i]))
        X, Y = np.array(X), np.array(Y)
        self.Q = self.Q - self.trans_func(E_used) + X
        self.E = self.E - E_transmit + Y
        Q = self.cut(self.Q)###################not done sharing
        E = self.cut(self.E)
        marginQ = self.Q-Q
        marginE = self.E-E
        self.Q = Q
        self.E = E
        next = np.append(self.Q,self.E)
        LossQ = np.sum(marginQ)
        LossE = np.sum(marginE)
        return next, LossQ, LossE

    def reward(self, next, action, lossQ, lossE):
        R = -(np.sum(np.square(next[:len(next)//2])) + 10*np.sum(np.square(lossQ)) + np.sum(lossE))
        return R
    
    def step(self, action):

        next, lossq, losse = self.next_state(action)
        reward = self.reward(next, action, lossq, losse)
        done = False
        info = {}

        return  next, reward, done, info
    
    def reset(self):
        state = np.zeros((self.nodes,self.nodes))
        return state