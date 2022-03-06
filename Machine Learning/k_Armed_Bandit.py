#from statistics import mean
#from pandas import pd
import numpy as np
import matplotlib.pyplot as plt

class Instance(object): #   Initialising test instance parameters class
    def __init__(self, k, m, sd): #  k number of arms, mean as m and standard deviation as d
        self.k = k #    K arms of bandit problem
        self.m = m #    Mean for Gaussian distribution
        self.sd = sd #    Standard deviation for Gaussian distribution
        self.a_value = np.zeros(k) #    Initialise all action values for k arms to 0
        self.max_a_value = 0 #  Stores maximum action value
        self.Reset() #  Resets instance at the end of a run

    def Reset(self): #  Reset instance at the end of a run
        self.a_value = np.random.normal(self.m, self.sd, self.k) #  Initialise distribution
        self.max_a_value = np.argmax(self.a_value) #    Store maximum action value at current position

class Bandit(object): # Models an arm to be pulled
    def __init__(self, k, e):
        self.k = k # K arms of bandit problem
        self.e = e # Epsilon value
        self.timestep = 0 # Keeps track of timestep
        self.pre_act = None #   Last action taken
        self.k_act = np.zeros(k) # K time step count
        self.rewards = np.zeros(k) #     Initialise sum of rewards from each play
        self.est_a_value = np.zeros(k) #     Initialise action value estimates

    def __str__(self): #    To be used for graph labelling
        if self.e == 0:
            return "Greedy-Approach"
        else:
            return "Epsilon-greedy = " + str(self.e)

    def play(self): #   # Picks an arm to play
        act = np.random.random() #    exploitation/exploration random variable generation
        if act < self.e:    # If generated values is below epsilon value, Explore
            arm = np.random.choice(len(self.est_a_value)) 
        else:   #    If generated values is above epsilon value, Exploit
            greedy_arm = np.argmax(self.est_a_value)
            a = np.where(self.est_a_value == greedy_arm)[0] # a stores the greedy arm identifier
            if len(a) == 0: # Choose greedy action
                arm = greedy_arm
            else: # However, if there are multiple equivalent outcomes, choose randomly
                arm = np.random.choice(a)

        self.pre_act = arm #    Saves the arm that was last pulled
        return arm

    def update(self, reward): # Calculates estimated action value 
        arm_idx = self.pre_act
        self.k_act[arm_idx] = self.k_act[arm_idx] + 1
        self.rewards[arm_idx] = self.rewards[arm_idx] + reward
        self.est_a_value[arm_idx] = self.rewards[arm_idx] / self.k_act[arm_idx] #  New action value
        self.timestep = self.timestep + 1
    
    def reset(self): #  Reset all variable values
        self.timestep = 0
        self.pre_act = None
        self.k_act[:] = 0
        self.rewards[:] = 0
        self.est_a_value[:] = 0

class Controller(object): # Controller that directs the actions of Bandits and Instances
    def __init__ (self, instance, bandits, play, iter):
        self.instance = instance # Initialises an instance
        self.bandit = bandits # Initialises a bandit
        self.play = play # Initialises number of plays
        self.iter = iter # Initialises number of iterations

    def run(self): #    Method to run bandits on an instantiated distribution environment
        results = np.zeros((self.play, len(self.bandit)))
        max_actions = np.zeros((self.play, len(self.bandit))) # Helps calculate count of optimal selections

        for iteration in range(self.iter): # Runs iter number of iterations
            self.instance.Reset() # Reset instance first
            for bandit in self.bandit:
                bandit.reset()  #   Clear bandits

            for a_play in range(self.play): #  Executes k Armed Bandit implementation for 'play' number of times
                count = 0

                #Carry out play on each bandit
                for k_bandit in self.bandit:
                    arm_pulled = k_bandit.play()
                    arm_reward = np.random.normal(self.instance.a_value[arm_pulled], scale=1) # Get reward from pulling lever
                    k_bandit.update(reward = arm_reward)
                    results[a_play, count] += arm_reward
                    if arm_pulled == self.instance.max_a_value: # Tracks action selected, if optimal, stores in max actions array
                        max_actions[a_play, count] = max_actions[a_play, count] + 1

                    count += 1

        avg_reward = results/self.iter # Store average reward
        opt_action = max_actions/self.iter #    Store number of optimal action choices

        return avg_reward, opt_action   #   Return average reward and optimal action carried out by bandits

# CLASS AND OBJECT CREATION, INSTANTIATION AND EXECUTION #
# THESE VARIABLES ARE ASSIGNED IN ACCORDANCE TO FIG 2.2 PARAMETERS
k = 10  #   k number of bandits
iter = 2000 #   Number of times to carry out plays
plays = 1000 #  Number of plays

instance = Instance(k,0,1) #   Instantiating an instance of k armed bandit implementation; (k arms, mean, standard deviation)
bandits = [Bandit(k, 0),Bandit(k,0.1),Bandit(k,0.01)] # Creating bandits with varying epsilon classes and k arms
controller = Controller(instance, bandits, plays, iter) #   controller object to execute behaviour

# Implementation Execution
print("KINDLY WAIT FOR A MINUTE, IMPLEMENTATION HAS BEGUN RUNNING-----------------")
print("Graphs will display results")
results, optimal_choice = controller.run() # Store the average results of bandits as well as their optmial choice selectiosn for graphing

#The graph depicting average rewards over plays
plt.figure(figsize=(5,5))
plt.title("Average Drawn Rewards on 10 Armed Bandit Implementation")
plt.plot(results)
plt.xlabel('Number of Plays')
plt.ylabel('Resulting Average Reward')
plt.legend(bandits, loc=7)
plt.show()

#The graph depicting exploitation % of optimal action chosen over plays
plt.figure(figsize=(5,5))
plt.title("Percentage of optimal selection on 10 Armed Bandit Implementation")
plt.plot(optimal_choice * 100)
plt.xlabel('Number of Plays')
plt.ylabel('Percentage Optimal Action')
plt.legend(bandits, loc=7)
plt.show()

print("EXECUTION HAS FINISHED RUNNING!\n")