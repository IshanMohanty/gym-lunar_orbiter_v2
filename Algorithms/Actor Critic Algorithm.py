# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:01:30 2019

@author: tchat
"""

#Import the various gym, keras, numpy and libraries we will require

import gym
import gym.spaces
import gym_lunar_orbiter_v2
import gym.wrappers
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import time

from collections import deque
from keras.layers import Flatten, Dense
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from multiprocessing import Pool, freeze_support

def build_model_critic(num_input_nodes, num_output_nodes, lr = 0.001, size = [256]):
	
	model = Sequential()
	model.add(Dense(size[0], input_shape = (8,), activation = 'relu'))
	for i in range(1,len(size)):
		model.add(Dense(size[i], activation = 'relu'))
	
	model.add(Dense(num_output_nodes, activation = 'linear')) 
	adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
	model.compile(loss = 'mse', optimizer = adam)
	#print('Critic Model Summary:')
	#model.summary()
	return model

def build_model_actor(num_input_nodes, num_output_nodes, lr = 0.001, size = [256]):
	
	model = Sequential()
	model.add(Dense(size[0], input_shape = (num_input_nodes,), activation = 'relu'))
	for i in range(1, len(size)):
		model.add(Dense(size[i], activation = 'relu'))
	
	model.add(Dense(num_output_nodes, activation = 'softmax')) 
	adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
	model.compile(loss = 'categorical_crossentropy', optimizer = adam)
	#print('Actor Model Summary:')
	#model.summary()
	return model

def decide_action(actor, state):

	flat_state = np.reshape(state, [1,8])
	action = np.random.choice(4, 1, p = actor.predict(flat_state)[0])[0]
	return(action)
    
def run_episode(env, actor, r = False):
    
    memory = [] 
    bestyet = float('-inf')      
    state = env.reset()
    episode_reward = 0
    cnt = 0 
    done = False
    while not done and cnt <350:
        cnt += 1
        if r:
            env.render()

        action = decide_action(actor, state)
        observation, reward, done, _ = env.step(action)  
        episode_reward += reward
        state_new = observation 
        memory.append((state, action, reward, state_new, done))
        state = state_new 

    return(memory, episode_reward)  
    
def train_models(actor, critic, memory, gamma):

	random.shuffle(memory)
	
	for i in range(len(memory)):

		state, action, reward, state_new, done = memory[i]			
		flat_state_new = np.reshape(state_new, [1,8])
		flat_state = np.reshape(state, [1,8])		
		target = np.zeros((1, 1))
		advantages = np.zeros((1, 4))
		value = critic.predict(flat_state)
		next_value = critic.predict(flat_state_new)

		if done:
			advantages[0][action] = reward - value
			target[0][0] = reward
		else:
			advantages[0][action] = reward + gamma * (next_value) - value
			target[0][0] = reward + gamma * next_value
		
		actor.fit(flat_state, advantages, epochs=1, verbose=0)
		critic.fit(flat_state, target, epochs=1, verbose=0)    

def play_game(iters, r = True):
    env = gym.make('LunarOrbiter-v2')
    totalrewardarray = []
    for i in range(iters):
    
        state = env.reset()
        totalreward = 0
        cnt = 0
        done = False
        while not done and cnt <450:

            cnt += 1
            if r:
                import PIL
                PIL.Image.fromarray(env.render(mode='rgb_array')).resize((320, 420))

            action = decide_action(actor, state)
            observation, reward, done, _ = env.step(action)  
            totalreward += reward
            state_new = observation 
            state = state_new            
        totalrewardarray.append(totalreward)

    return totalrewardarray
        
def run_train_plot(alr, clr, gamma, numepisodes):
    
    env = gym.make('LunarOrbiter-v2')  
    i = 0
    actor = build_model_actor(num_input_nodes = 8, num_output_nodes = 4, lr = alr, size = [64,64,64])
    critic = build_model_critic(num_input_nodes = 8, num_output_nodes = 1, lr= clr, size = [64,64,64])

    totrewardarray = [] #For storing the total reward from each episode
    best = float('-inf') #For storing the best rolling average reward
    episodes = len(totrewardarray) #Counting how many episodes have passed
    
    scores = []

    while episodes < numepisodes:   

        i+= 1
        memory, episode_reward = run_episode(env, actor, r = False)
        totrewardarray.append(episode_reward)
        episodes = len(totrewardarray)
        print(episode_reward) 
        scores.append(episode_reward)
    
        if episodes >= 100:
            score = np.average(totrewardarray[-100:-1])
            if score > best:
                best = score
                actor.save('actormodel.h5')
                critic.save('criticmodel.h5')
            if episodes%500==0:
                print('ALR:', alr, ' CLR:', clr, 'episode ', episodes, 'of',numepisodes, 'Average Reward (last 100 eps)= ', score)

        train_models(actor, critic, memory, gamma)

        avgarray = []
        cntarray = []

    for i in range(100,len(totrewardarray),10):
        avgarray.append(np.average(totrewardarray[i-100:i]))
        cntarray.append(i)

    
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    plt.figure()
    plt.plot(cntarray, avgarray, label = 'Best 100 ep av. reward = '+str(best))
        
    plt.title('Rolling Average (previous 100) vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.legend(loc='best')
    
    plt.show()   
    
    
#run_train_plot(5e-6, 5e-4, 0.999, 5000)

#Load the saved model at its best performance
actor=load_model('actormodel.h5')
critic=load_model('criticmodel.h5')


rewards = play_game(iters = 10, r = True)
plt.hist(rewards, 40, rwidth=0.8)
plt.title('Performance of trained models')
plt.xlabel('Episode reward')
plt.ylabel('Number of Occurrences')
plt.show()  
   