# gym-lunar_orbiter_v2
Lunar Orbiter V2 

1. Install gym: pip install gym

2. Download from github by:
   git clone https://github.com/IshanMohanty/gym-lunar_orbiter_v2
   cd gym_lunar_orbiter_v2/
   pip install -e .
   
3. If you already have the Folder directory structure with all files then:
   do not follow step 2, just follow step3.
   cd gym_lunar_orbiter_v2/
   pip install -e .
   
4. Test the custom environment by copying the below code onto a python file say test.py and running it test.py:

   import gym
   import gym.spaces
   import gym_lunar_orbiter_v2

    env = gym.make('LunarOrbiter-v2')
    env.reset()
    
    while True:
      env.render()
      action = env.action_space.sample()
      observation,reward,done,info =env.step(action)

      if PRINT_DEBUG_MSG:
        print("Action Taken  ",action)
        print("Observation   ",observation)
        print("Reward Gained ",reward)
        print("Info          ",info,end='\n\n')

      if done:
        print("Simulation done.")
        break
      env.close()

5. you should be able to see the environment visually.


Algorithms:

cd gym-lunar_orbiter_v2

1. DQN Algorithm:
   cd /Algorithms/DQN/
   python3 dqn.py
   
2. Actor-Critic Algorithm:
   cd /Algorithms/Actor-Critic/
   python3 actor_critic_algorithm.py
 
3. Policy Gradient Algorithm:
   cd /Algorithms/Policy Gradient
   python3 run_lunarorbiter_v2.py
   
Weights for DQN after training: weights.pth
Weights for actor critic after training: actormodel.h5, criticmodel.h5
Live Training for Policy Gradients, no weights saved.
                                             
