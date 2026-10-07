'''Matthew W Meza Chang. Two Step Task Environment.''' 

import numpy as np

'''
There will be 4 actions
    1. Initiate
    2. Left 
    3. Right 
    4. Outcome

There will be 5 states
    1. Choice
    2. A 
    3. B 
    4. A_Reward
    5. B_Reward 

Rewarded:
Choice --(Left)--> A --(Outcome)--> A_Reward --(Initiate)--> Choice 

Unrewarded:
Choice --(Left)--> A --(Outcome)--> Choice 
'''

# State and Action IDs
initiate = 0 
left = 1
right = 2
outcome = 3 

choice = 0
A = 1
B = 2
A_reward = 3 
B_reward = 4

high_A = 0
high_B = 1 

class Two_Step():
    
    def __init__(self):
        """
        Two Step constructor
        """
        self.p_common = 0.8                                      # Common transition probability is 0.8 
        self.window = 8                                          # We are determing the % correct based on last 8 trials

        self.n_actions = 4                                       # There are four possible actions
        self.n_states = 5                                        # There are five possible states

        self.reset()

    def reset(self):
        """
        Initialize the new session
        """
        self.high_state = np.random.choice([high_A, high_B])    # Current high state

        self.switch_criterion = False                           # Block switch criterion
        self.trials_until_switch = None                         # Trials until block switch
        self.trial_history = []                                 # Keep track whether trials correct

        self.trial_n = 0                                        # Counter for how many trials done
        self.state = choice                                     # Start at choice state 

        self.block_switch_flag = False                          # Flag ... 

        return self.state

    def transition(self, action):
        """
        Helper function
            - Input only allows Left and Right actions
            - Output is hidden state A or B 
        """
        dice = np.random.rand()

        if action == left:
            if dice < self.p_common:
                return 'A'
            else:
                return 'B'
        elif action == right:
            if dice < self.p_common:
                return 'B'
            else:
                return 'A'
        else:
            raise ValueError("Action must be left or right")
            
    def reward(self, state):
        """
        Helper function
            - Input is hidden states A or B
            - Output is whether rewarder or not 
        """
        if self.high_state == high_A:
            p_reward_A = 0.9
            p_reward_B = 0.1 
        elif self.high_state == high_B:
            p_reward_A = 0.1 
            p_reward_B = 0.9

        dice = np.random.rand()
        if state == 'A':
            return int(dice < p_reward_A)
        elif state == 'B':
            return int(dice < p_reward_B)
        else:
            raise ValueError("State must be A or B")

    def step(self, action):
        """
        Given an action return the state and reward. 
        """
        reward = 0

        # Choice state, only acceptable action is Right or Left 
        if self.state == choice: 
            if action not in (left, right):
                return self.state, reward

            second_step = self.transition(action)

            # Update our trial history 
            if self.high_state == high_A:
                is_correct = int(action == left)
            elif self.high_state == high_B:
                is_correct = int(action == right)
            self.trial_history.append(is_correct)

            if not self.switch_criterion:
                if len(self.trial_history) >= self.window:
                    curr_accuracy = np.mean(self.trial_history[-self.window:])

                    if curr_accuracy >= 0.7:
                        self.switch_criterion = True            # Criterion met the 70% accuracy 
                        self.trials_until_switch = np.random.randint(5,21)

            elif self.switch_criterion:
                self.trials_until_switch -= 1                   # Decrement

                if self.trials_until_switch == 0:
                    self.block_switch_flag = True               # Flag is raised, so ready to block switch

                    '''
                    if self.high_state == high_A:
                        self.high_state = high_B
                    elif self.high_state == high_B:
                        self.high_state = high_A

                    # Reset
                    self.switch_criterion = False
                    self.trials_until_switch = None
                    self.trial_history = []
                    '''

            if second_step == 'A':
                self.state = A
            elif second_step == 'B':
                self.state = B

        elif self.state == A:
            if action != outcome:
                return self.state, reward
            reward = self.reward('A')

            if reward == 1:
                self.state = A_reward
            else:
                self.state = choice 
                self.trial_n += 1

                if self.block_switch_flag:
                    if self.high_state == high_A:
                        self.high_state = high_B
                    elif self.high_state == high_B:
                        self.high_state = high_A

                    self.switch_criterion = False
                    self.trials_until_switch = None
                    self.trial_history = []
                    self.block_switch_flag = False 

        elif self.state == B:
            if action != outcome:
                return self.state, reward 
            reward = self.reward('B')

            if reward == 1:
                self.state = B_reward
            else:
                self.state = choice
                self.trial_n += 1

                if self.block_switch_flag:
                    if self.high_state == high_A:
                        self.high_state = high_B
                    elif self.high_state == high_B:
                        self.high_state = high_A
                
                    self.switch_criterion = False
                    self.trials_until_switch = None
                    self.trial_history = []
                    self.block_switch_flag = False 

        elif self.state in (A_reward, B_reward):
            if action != initiate:
                return self.state, reward
            self.state = choice 
            self.trial_n += 1

            if self.block_switch_flag:
                if self.high_state == high_A:
                    self.high_state = high_B
                elif self.high_state == high_B:
                    self.high_state = high_A

                # Reset
                self.switch_criterion = False 
                self.trials_until_switch = None
                self.trial_history = []
                self.block_switch_flag = False

        return self.state, reward