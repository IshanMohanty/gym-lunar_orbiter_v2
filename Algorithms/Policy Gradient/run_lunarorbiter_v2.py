import gym
from gym import wrappers
from Policy_Gradient import PolicyGradient
import gym_lunar_orbiter_v2
import matplotlib.pyplot as plt
import numpy as np
import time
import os


env = gym.make('LunarOrbiter-v2')
env = env.unwrapped


# policy gradient has high variance, seed for reproducability
env.seed(1)


print('env.action_space', env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)


RENDER_ENV = True
EPISODES = 1000
rewards = []
RENDER_REWARD_MIN = 1000
TRAINING_LOG_DIR = 'data/training_log/'
TESTING_LOG_DIR = 'data/testing_log/'
RECORD_DIR = 'data/video/'
RECORD_FILENAME = 'training-video'


if __name__ == '__main__':
    # load checkpoint               what does this part mean and how to change it
    load_version = 8
    save_version = load_version + 1
    # load_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(load_version)
    # save_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(save_version)

    PG = PolicyGradient(n_x=env.observation_space.shape[0],
                        n_y=env.action_space.n,
                        learning_rate=0.005,
                        reward_decay=0.99,
                        # load_path=load_path,
                        # save_path=save_path
                        )

    # initialize episode reward log
    if not os.path.exists(TRAINING_LOG_DIR):
        os.makedirs(TRAINING_LOG_DIR)
    f = open(TRAINING_LOG_DIR + 'episode_reward_log.txt', 'w')
    f.write('EPISODE\tTIME_ELAPSED\tTOTAL_REWARD')
    f.write('\n')
    f.close()

    # training
    for episode in range(EPISODES):
        observation = env.reset()
        episode_reward = 0

        tic = time.time()

        while True:
            if RENDER_ENV:
                env.render()

            # 1. choose an action based on observation
            action = PG.choose_action(observation)

            # 2. take an action
            observation_, reward, done, info = env.step(action)

            # 3. store transition data for training
            PG.store_transition(observation, action, reward)

            toc = time.time()
            elapsed_time = toc - tic
            if elapsed_time > 3:
                done = True

            episode_reward_sum = sum(PG.episode_rewards)
            if episode_reward_sum < -200:
                done = True

            if done:
                episode_reward_sum = sum(PG.episode_rewards)
                rewards.append(episode_reward_sum)
                max_reward_so_far = np.max(rewards)

                # print training data in command line
                print("==========================================")
                print("Episode: ", episode)
                print("Seconds: ", elapsed_time)
                print("Reward: ", episode_reward_sum)
                print("Max reward so far: ", max_reward_so_far)

                # save data
                f = open(TRAINING_LOG_DIR + 'episode_reward_log.txt', 'a')
                f.write(str(episode) + '\t' + str(int(elapsed_time/0.001)) + '\t' + str(episode_reward_sum))
                f.write('\t')
                f.close()

                # 4. train nn
                discounted_episode_reward_norm = PG.learn()

                if max_reward_so_far > RENDER_REWARD_MIN:
                    RENDER_ENV = True

                break

            # 5. save new observation
            observation = observation_

        # start recording video
        # if episode % 50 == 0:
        #     env = wrappers.Monitor(env, RECORD_DIR + RECORD_FILENAME, force=True)


