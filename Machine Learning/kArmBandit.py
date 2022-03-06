#import modules used in implementation
import numpy as np 
import matplotlib.pyplot as plt 
import time

class armed_bandit:

    # This class takes as input, k number of arms
    # an epsilon value e and 'iter' no of iterations
    def __init__(self, k, e, iter):
        self.k = k #    K Arms
        self.e = e #    Epsilon-Greedy value
        self.iter = iter #  Number of plays
        self.step_count = 0 #   Time step count
        self.k_step_count = np.zeros(k) # K time step count
        self.m_reward = 0 #   Store mean reward
        self.rewards = np.zeros(iter) #     Store reward from each play
        self.k_Arm_reward = np.zeros(k) #   Store mean reward for k-Arm
        self.mean = np.random.normal(0, 1, k) # Draw mean from normal (Gaussian) distribution

    # Picks an arm to play
    def play(self):
        act = np.random.random() #    exploitation/exploration random variable generation
        if self.e == 0 and self.step_count == 0: #  Pick a random arm initially
            action = np.random.choice(self.k)
        elif act < self.e: #    If generated values is below epsilon value, Explore
            action = np.random.choice(self.k) #     Explore
        else: #    If generated values is above epsilon value, Exploit
            action = np.argmax(self.k_Arm_reward) #   Exploit

        #if action == np.argmax(self.k_Arm_reward):

        
        rewards = np.random.normal(self.mean[action], 1) #   Action reward from normal (Gaussian) distribution

        self.step_count = self.step_count + 1 #   Update time step count
        self.k_step_count[action] = self.k_step_count[action] + 1 #   Update k Arm step count

        self.m_reward = self.m_reward + (rewards - self.m_reward) / self.step_count #   Update mean reward

        self.k_Arm_reward[action] = self.k_Arm_reward[action] + (rewards - self.k_Arm_reward[action]) / self.k_step_count[action] # Update estimated action value for k Arm

    def run(self):
        for i in range(self.iter):
            self.play()
            self.rewards[i] = self.m_reward
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_step_count = np.zeros(k)
        self.m_reward = 0
        self.rewards = np.zeros(iter)
        self.k_Arm_reward = np.zeros(k)



k = 10
iters = 100

eps_0_rewards = np.zeros(iters)
eps_01_rewards = np.zeros(iters)
eps_1_rewards = np.zeros(iters)

eps_0_selection = np.zeros(k)
eps_01_selection = np.zeros(k)
eps_1_selection = np.zeros(k)

runs = 200
# Run experiments
for i in range(runs):
    # Initialize bandits
    eps_0 = armed_bandit(k, 0, iters)
    eps_01 = armed_bandit(k, 0.01, iters)
    eps_1 = armed_bandit(k, 0.1, iters)
    
    # Run experiments
    y = eps_0.run()
    #print(y)
    eps_01.run()
    eps_1.run()
    
    # Update long-term averages
    eps_0_rewards = eps_0_rewards + (
        eps_0.rewards - eps_0_rewards) / (i + 1)
    eps_01_rewards = eps_01_rewards + (
        eps_01.rewards - eps_01_rewards) / (i + 1)
    eps_1_rewards = eps_1_rewards + (
        eps_1.rewards - eps_1_rewards) / (i + 1)

    # Average actions per episode
    eps_0_selection = eps_0_selection + (
        eps_0.k_step_count - eps_0_selection) / (i + 1)
    eps_01_selection = eps_01_selection + (
        eps_01.k_step_count - eps_01_selection) / (i + 1)
    eps_1_selection = eps_1_selection + (
        eps_1.k_step_count - eps_1_selection) / (i + 1)

print(eps_01_rewards)
    
plt.figure(figsize=(12,8))
plt.plot(eps_0_rewards, label="$\epsilon=0$ (greedy)")
plt.plot(eps_01_rewards, label="$\epsilon=0.01$")
plt.plot(eps_1_rewards, label="$\epsilon=0.1$")
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average $\epsilon-greedy$ Rewards after " + str(runs) 
    + " Runs")
plt.show()

plt.figure(figsize=(12,8))
plt.plot(eps_0_selection, label="$\epsilon=0$ (greedy)")
plt.plot(eps_01_selection, label="$\epsilon=0.01$")
plt.plot(eps_1_selection, label="$\epsilon=0.1$")
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average $\epsilon-greedy$ Rewards after " + str(runs) 
    + " Runs")
plt.show()