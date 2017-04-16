#!/usr/bin/python
# Filename: ndpromp_demo.py

import ipromp
import numpy as np
import matplotlib.pyplot as plt

# close the current windows
plt.close()

# create a ProMP object
p = ipromp.NDProMP(num_joints=3)

# Generate and plot trajectory Data
x = np.arange(0,1.01,0.01)           # time points for trajectories
nrTraj=30                            # number of trajectoreis for training
sigmaNoise=0.02                      # noise on training trajectories
A1 = np.array([.5, .0, .0])    # the weight of different func 
A2 = np.array([.0, .5, .0])    # the weight of different func 
A3 = np.array([.0, .0, .5])    # the weight of different func 
X = np.vstack( (np.sin(5*x), x**3, x ))    # the basis func


# add demonstration
for traj in range(0, nrTraj):
    sample1 = np.dot(A1 + sigmaNoise * np.random.randn(1,3), X)[0]
    sample2 = np.dot(A2 + sigmaNoise * np.random.randn(1,3), X)[0]
    sample3 = np.dot(A3 + sigmaNoise * np.random.randn(1,3), X)[0]
    samples = np.array([sample1, sample2, sample3]).T
    p.add_demonstration(samples)
    label = 'training set' if traj==0 else ''
    plt.plot(x, samples, 'grey', label=label)

# add via point as observation
p.add_viapoint(0.33, np.array([0.5+0.008, 0.02+0.008, 0.18+0.008]))

# plot the trained model and generated traj
p.plot(x=p.x)

# show the plot
plt.legend()
plt.show()