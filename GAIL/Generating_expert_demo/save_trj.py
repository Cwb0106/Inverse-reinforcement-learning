import numpy as np
import torch

from TD3 import TD3
import pickle

from env import ArmEnv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    env_with_Dead = False
    env = ArmEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1
    expl_noise = 0.25

    kwargs = {
        "env_with_Dead":env_with_Dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "net_width": 200,
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "Q_batchsize":256,
    }

    model = TD3(**kwargs)


    for i in range(400):
        s = env.reset()
        ep_r = 0.
        expl_noise *= 0.99
        for j in range(200):
            # env.render()

            a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size = action_dim)
            ).clip(-max_action, max_action)
            s_, r, done = env.step(a)


            model.memory.add(s, a, r, s_, done)

            if i>120: model.train()


            ep_r += r


            s = s_
            if done or j == 200-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    model.save(400)

def eval():
    env_with_Dead = False
    env = ArmEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1
    expl_noise = 0.25
    kwargs = {
        "env_with_Dead":env_with_Dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "net_width": 200,
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "Q_batchsize":256,
    }
    model = TD3(**kwargs)
    model.load(400)

    episode_traj = []
    traj_pool = []
    for j in range(1):
        s = env.reset()
        for joi in range(300):
            a = model.select_action(s)
            s_,r,done = env.step(a)
            print('s',s.shape)
            print('a',a.shape)
            episode_traj.append([s,a])
            s = s_
        traj_pool.extend(episode_traj)
        del episode_traj[:]
        if j % 50 == 0:
            print('目前存了几回合',j)
    print(len(traj_pool))


    # file = open('save_trj.pkl', 'wb')
    # pickle.dump(traj_pool, file)
    # file.close()

if __name__ == '__main__':
    # main()
    eval()