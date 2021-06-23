import numpy as np
from math import ceil

class BaseAgent(object):
    
    def __init__(self, actions) -> None:
        self._actions = actions
        self._action = None
        self._Qs = self.initialize(actions)
        self.t = 1
            
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
    
    @staticmethod
    def initialize(actions):
        diction = {}
        for a in actions:
            diction[a] = 0
            
        return diction
        
    

class epGreedyAgent(BaseAgent):
    
    def __init__(self, actions, ep) -> None:
        """[summary]

        Args:
            actions ([type]): [description]
            ep ([type]): [description]
        """
        super().__init__(actions)
        self.ep = ep
        self._action = None # It stores the former action to update, and action is `None` at first round
        self._Ns = self.initialize(actions)
    
    def decide(self):
        if self._action is None:
            self._action = np.random.choice(self._actions)
            return self._action
         
        r_ep = np.random.uniform(0,1)
        ep = self.ep(self.t) if callable(self.ep) else self.ep
        if r_ep <= ep:
            self._action = np.random.choice(self._actions)
        else:
            self._action = max(self._Qs, key=lambda k: self._Qs[k])
        return self._action
    
    def update(self, reward) -> None:
        self._Ns[self._action] += 1
        self._Qs[self._action] += (reward - self._Qs[self._action])/self._Ns[self._action]
        self.t += 1
        
 
class UCBAgent(BaseAgent):
     
    def __init__(self, actions, alg_name, alpha=None) -> None:
        super().__init__(actions)  
        if alg_name == "UCB1":
            pass
        elif alg_name == "UCB2":
            if alpha is None or (alpha > 1) or (alpha < 0):
                raise ValueError
            self.alpha = alpha
        else:
            raise ValueError
        
        self.alg_name = alg_name
        self._action = None
        self._Ns = self.initialize(actions)
        self._Rs = self.initialize(actions)
        
        self.rest = 0
        
    def decide(self):
        # initial actions
        if self.t <= len(self._actions):
            # print("initial time: ", self.t)
            self._action = self._actions[self.t-1]
            return self._action
        
        if self.rest == 0:
            if self.t > len(self._actions)+1 and self.alg_name == "UCB2":
                # print("time: ",self.t)
                self._Rs[self._action] += 1
            self._action = self._find_maximal_UB()
            self._get_rest(self._action) 
            return self._action
        
        elif self.rest > 0: # Work only for UCB2, self.rest will always be 0 in UCB1
            self.rest -= 1
            return self._action
        
    def update(self, reward):
        self._Ns[self._action] += 1
        self._Qs[self._action] += (reward - self._Qs[self._action])/self._Ns[self._action]
        self.t += 1     
        
    def _find_maximal_UB(self):
        if self.alg_name == "UCB1":
            Us = self.compute_UB1(self._Qs, self._Ns, self.t)
        elif self.alg_name == "UCB2":
            Us = self.compute_UB2(self._Qs, self._Rs, self.alpha, self.t, self.tau_f)
        return max(Us, key=lambda k: Us[k])
        
    def _get_rest(self, j):
        if self.alg_name == "UCB1":
            self.rest = 0
        elif self.alg_name == "UCB2":
            self.rest = self.tau_f(self._Rs[j]+1, self.alpha) - self.tau_f(self._Rs[j], self.alpha) -1
          
    @staticmethod
    def compute_UB1(Qs, Ns, t):
        Us = {}
        for a in Qs.keys():
            Us[a] = Qs[a] + np.sqrt(2*np.log(t)/Ns[a])
        return Us

    @staticmethod
    def compute_UB2(Qs, Rs, alpha, t, tau_f):
        Us = {}
        for a in Rs.keys():
            Us[a] = Qs[a] + \
                np.sqrt(
                    (1+alpha)*np.log(np.e*t/tau_f(Rs[a], alpha))/(2*tau_f(Rs[a], alpha))
                )
        return Us
                
    @staticmethod
    def tau_f(r, alpha): return ceil((1+alpha)**r)
    
             
    
    
        
                  

if __name__ == "__main__":
    # actions = list(range(10))
    # def f(t): return 0.1*t**(0.01)
    # ep = 0.1
    # eg_agent = epGreedyAgent(actions, f)
    # print("actions: ", eg_agent.actions)
    # action = eg_agent.decide()
    # print("action: ", eg_agent.action)
    # eg_agent.update(1)
    # print(eg_agent.estimations)
    # eg_agent.decide()
    # eg_agent.update(0)
    # ======================
    actions = list(range(3))
    alpha = 0.8 
    ucb1_agent = UCBAgent(actions, "UCB1")
    ucb2_agent = UCBAgent(actions, "UCB2", alpha)
    rewards = np.random.randint(0, 2, size=10)
    for r in rewards:
        ucb2_agent.decide()
        ucb2_agent.update(r)
        print(ucb2_agent.rest)
        print(ucb2_agent._Rs)
        
         
        
            
        
    
    
        