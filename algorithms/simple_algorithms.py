import numpy as np

class BaseAgent(object):
    
    def __init__(self, actions) -> None:
        self._actions = actions
        self._action = None
        self._Qs = self.initialize(actions)
        
    @staticmethod
    def initialize(actions):
        diction = {}
        for a in actions:
            diction[a] = 0
            
        return diction
    
    def decide(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    @property
    def actions(self):
        return self._actions
    
    @property
    def action(self):
        return self._action
    
    @property
    def estimations(self):
        return self._Qs
        
    

class epGreedyAgent(BaseAgent):
    
    def __init__(self, actions, ep) -> None:
        super().__init__(actions)
        self.ep = ep
        self._action = None # It stores the former action to update, and action is `None` at first round
        self._Ns = self.initialize(actions)
    
    def decide(self):
        if self._action is None:
            action = np.random.choice(self._actions)
            self._action = action
            return self._action
         
        r_ep = np.random.uniform(0,1)
        if r_ep <= self.ep:
            action = np.random.choice(self._actions)
            self._action = action
        else:
            action = max(self._Qs, key=lambda k: self._Qs[k])
            self._action = action
        return action
    
    def update(self, reward) -> None:
        self._Ns[self._action] += 1
        self._Qs[self._action] += (reward - self._Qs[self._action])/self._Ns[self._action]


if __name__ == "__main__":
    actions = list(range(10))
    ep = 0.1
    eg_agent = epGreedyAgent(actions, ep)
    print("actions: ", eg_agent.actions)
    action = eg_agent.decide()
    print("action: ", eg_agent.action)
    eg_agent.update(1)
    print(eg_agent.estimations)
         
        
            
        
    
    
        