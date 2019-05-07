import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_record = pd.read_csv('episode_reward_log@April_30_15dian.txt', sep='\t')

a = training_record['EPISODE']
b = training_record['TOTAL_REWARD']
b = b.array; a = a.array

epi_array = np.zeros(shape=(60,))
for i in range(1, 61):
    epi_array[i-1] = i*10
    i += 1

avg_array = np.zeros(shape=(60,))
avg = 0; sum = 0
for i in range(0, 60):
    for j in range(0, 10):
        sum += b[10*i + j]
    avg = sum/10
    sum = 0
    avg_array[i] = avg
    avg = 0

fig = plt.figure()
plt.plot(epi_array, avg_array, '.-', c='blue')
plt.xlabel('Episode')
plt.ylabel('Learning Performance')
fig.savefig('episode_reward_log@April_30_21dian.jpeg', format='jpeg', dpi=300, bbox_inches='tight')
