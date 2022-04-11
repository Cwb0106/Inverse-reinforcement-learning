import gym
import numpy as np


"""""
This is a code about Maximum Entropy Reinforcement Learning. Please refer to the paper "Maximum Entropy Reinforcement Learning" for specific theories.
The concrete code reference for 
https://github.com/clam004/max_ent_irl/blob/main/mountaincar_discrete_maxIRL.ipynb

"""""

# Create the "MountainCar-v0" environment
env = gym.make('MountainCar-v0')
env.reset()

# Action space: the car has 3 movements in total: left, stationary and right
# Left = 0
# Stay = 1
# Right = 2

class MaxEntropyIRL():
    def __init__(self, expert_file, env_low, env_high, env_distance, gamma=0.99, theta_learning_rate=0.05, q_learning_rate=0.03, n_feature_bins=20, Max_episode=30000):
        self.n_feature_bins = n_feature_bins  # The position and velocity in the MountainCar state
                                              # S =[position,velocity] are artificially discretized into 20 states each
        self.Max_episode = Max_episode
        self.env_low = env_low
        self.env_high = env_high
        self.env_distance = env_distance
        self.n_states_all = 400                # a total of 20*20 states can be combined for position and speed
        self.n_actions = 3                     #  Left   stay    Right
        self.q_learning_rate = q_learning_rate
        self.gamma = gamma
        self.theta = -(np.random.uniform(size=(self.n_states_all,)))    # (400, 0)
        self.theta_learning_rate = theta_learning_rate

        self.expert_file = expert_file
        expert_demo = np.load(self.expert_file)
        self.demonstrations = np.zeros((len(expert_demo), len(expert_demo[0]), 3))
        for n in range(len(expert_demo)):
            for t in range(len(expert_demo[0])):
                state_idx = self.idx_state(expert_demo[n][t])
                self.demonstrations[n][t][0] = state_idx
                self.demonstrations[n][t][1] = expert_demo[n][t][2]


        self.q_table = np.zeros((self.n_states_all, self.n_actions))  #  (400, 3)

    def idx_state(self, state):
        """""
        
        """""
        position_index = int((state[0] - self.env_low[0]) / self.env_distance[0])
        velocity_index = int((state[1] - self.env_low[1]) / self.env_distance[1])
        state_index = position_index + velocity_index * self.n_feature_bins
        return state_index

    def update_q_table(self, state, action, reward, next_state):
        """""
        Update the Q table with the formula
        Q(s,a) <-- Q(s,a) + alpha * (r + gamma * max_a'(Q(s',a')) - Q(s,a))
        """""
        Q_1 = self.q_table[state][action]
        Q_2 = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.q_learning_rate * (Q_2 - Q_1)

    def get_reward(self, state_index):
        irl_rewards = self.theta[state_index]
        return irl_rewards

    def update_theta(self, expert, learner):
        """""
        According to the maximum entropy theory, the optimization problem can be formalized
        (1) max(-p*log(p))  s.t. 1) Experts expect = Learner trajectory expectation ∑p*f_learner = f_experts
        2) ∑p = 1 By the Lagrange multiplier method, it can be converted to 
        min_L = ∑p*log(p) - ∑lambda_1*(p*f_learner-f_experts) - lambda_0*(∑p-1))
        The gradient for this Loss function L is
        ∇L(𝜃) = f_expert - ∑trajectories P(trajectory|𝜃)*f_learner = f_expert - ∑states Ds*fs
        Where Ds is the expected state visitation frequencies
        𝜃  <--  𝜃 + theta_learning_rate * ∇L(𝜃)  
        """""
        gradient = expert - learner
        self.theta += self.theta_learning_rate * gradient
        self.theta = np.clip(self.theta, -float('inf'), 0)

    def get_expert_feature_expectations(self):
        """""
        Calculate the probability of occurrence of each state 
        s=[position,velocity] in all expert tracks
        """""
        feature_expectations = np.zeros(self.n_states_all)

        for demonstration in self.demonstrations:
            for state_index, _, _ in demonstration:
                feature_expectations[int(state_index)] += 1
        feature_expectations /= self.demonstrations.shape[0]
        return feature_expectations

    def train(self):

        expected_expert_feats = self.get_expert_feature_expectations()

        learner_feature_counts = np.zeros(self.n_states_all)     #(400,)

        episodes = []
        scores = []
        best_score = -200
        for episode in range(self.Max_episode):
            state = env.reset()
            score = 0
            if (episode >= 1000 and episode % 1000 == 0):
                print('MaxEnt Update')
                expected_learner_feats = learner_feature_counts / episode
                self.update_theta(expected_expert_feats, expected_learner_feats)
            while True:
                state_index = self.idx_state(state)

                action = np.argmax(self.q_table[state_index])

                next_state, reward, done, _ = env.step(action)
                irl_reward = self.get_reward(state_index)
                next_state_idx = self.idx_state(next_state)

                self.update_q_table(state_index, action, irl_reward, next_state_idx)

                learner_feature_counts[int(state_index)] += 1
                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break


            if episode % 1000 == 0:
                score_avg = np.mean(scores[-100:])
                print('{} episode score is {:.2f}'.format(episode, score_avg))
                if score_avg > -110:
                    print('solved!')
                    break

                if score_avg > best_score:
                    best_score = score_avg
                    print('saving new best')
                    np.save("Data/maxent_q_table", arr=self.q_table)


    def test(self):
        self.q_table = np.load('Expert_Data/maxent_q_table.npy')
        state = env.reset()
        env.render()
        for _ in range(200):
            state_indx = self.idx_state(state)
            action = np.argmax(self.q_table[state_indx])
            state, reward, done, _ = env.step(action)
            env.render()


if __name__=='__main__':
    env = gym.make('MountainCar-v0')
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / 20
    expert_file = 'Expert_Data/expert_demo.npy'
    agent = MaxEntropyIRL(env_low=env_low, env_high=env_high, env_distance=env_distance, expert_file=expert_file)

    agent.train()
    agent.test()



