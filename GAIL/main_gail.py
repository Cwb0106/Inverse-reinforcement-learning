import numpy as np
import torch
from gail import gail
from env import ArmEnv



def train():
    env_with_Dead = False
    env = ArmEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1
    expl_noise = 0.25
    MAX_EPISODES = 500
    MAX_EP_STEPS = 200
    pre_p = 10

    kwargs = {
        "env_with_Dead":env_with_Dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "hidden_size": 256,
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "batch_size":128,
    }

    model = gail(**kwargs)

    best_score = -float('inf')

    for ep in range(MAX_EPISODES):
        observation = env.reset()
        ep_r = 0.
        expl_noise *= 0.99
        for step in range(MAX_EP_STEPS):
            action = (model.select_action(observation) + np.random.normal(0, max_action * expl_noise, size = action_dim)
            ).clip(-max_action, max_action)
            observation_, reward, done = env.step(action)
            cal_r = model.get_reward(observation, action)
            model.memory.add(observation, action, cal_r, observation_, done)

            if ep>pre_p:
                model.discriminator_train()
                model.train()

            ep_r += reward
            observation = observation_

            if done or step == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (ep, '---' if not done else 'done', ep_r, step))
                if ep_r >= best_score:
                    model.save()
                    best_score = ep_r
                break


def eval():
    env_with_Dead = False
    env = ArmEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1

    kwargs = {
        "env_with_Dead":env_with_Dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "hidden_size": 256,
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "batch_size":128,
    }
    model = gail(**kwargs)
    model.load()
    observation = env.reset()
    while True:
        action = model.select_action(observation)
        observation_, reward, done = env.step(action)
        env.render()
        observation = observation_



if __name__ == '__main__':

    train()

    # eval()