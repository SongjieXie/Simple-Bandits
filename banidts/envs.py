import numpy as np

class BaseArm(object):
    
    def __init__(self, name) -> None:
        self._name = name
        
    @property 
    def name(self):
        return self._name 
    
    def reward(self) -> int:
        raise NotImplementedError
    
    
class BernArm(BaseArm):
    
    def __init__(self, name, p) -> None:
        super().__init__(name)
        self.p = p
        
    def reward(self) -> int:
        r = np.random.uniform(0,1)
        rew = 0
        if r <= self.p:
            rew = 1
            
        return rew
    


class BaseEnv(object):
    
    def __init__(self, arms_names) -> None:
        self._arms_names = arms_names
        self._optimal = None
        self._optimal_reward = None
        self._regrets = []
        
    def take(self) -> int:
        raise NotImplementedError
    
    @property 
    def arms_names(self):
        return self._arms_names
    
    @property
    def optimal(self):
        return self._optimal
    
    @property
    def optimal_reward(self):
        return self._optimal_reward
    
    @property
    def regrets(self):
        return self._regrets
    
    
class MutiBernArmEnv(BaseEnv):
    
    def __init__(self, arms_names, ps) -> None:
        super().__init__(arms_names)
        self.ps = ps
        self.arms = self.set_up_arms(self._arms_names, ps) 
        self._optimal = self._arms_names[
            np.argmax(ps)]
        self._optimal_reward = np.max(ps)
        
    def take(self, a) -> int:
        if a not in self._arms_names:
            raise ValueError
        r = self.arms[a].reward()
        self._regrets.append(
            self.optimal_reward-r
        )
        return r
        
    @staticmethod
    def set_up_arms(arms_names, ps) -> dict:
        arms = {}
        for i in range(len(arms_names)):
            name = arms_names[i]
            p = ps[i]
            arms[name] = BernArm(name, p)
        return arms
        
        
if __name__ == "__main__":
    arms_names = list(range(10))
    ps = np.random.uniform(0,1, size= 10)
    mba = MutiBernArmEnv(arms_names, ps)
    print("ps: ", ps)
    print(mba.arms_names)
    print(mba.optimal)
    print(mba.optimal_reward)
    for i in range(9):
        result = mba.take(i)
        print("reward: ", result)
    print(mba.regrets)
    