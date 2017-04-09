#!/usr/bin/python
# Filename: simple.py

import ipromp
import numpy as np
import matplotlib.pyplot as plt

# create a ProMPs object
p = ipromp.ProMP()

# close the current windows
plt.close()

# Generate and plot trajectory Data
x = np.arange(0,1.01,0.01)           # time points for trajectories
nrTraj=30                            # number of trajectoreis for training
sigmaNoise=0.02                      # noise on training trajectories
A = np.array([.2, .2, .01, -.05])
X = np.vstack( (np.sin(5*x), x**2, x, np.ones((1,len(x))) ))

Y = np.zeros( (nrTraj,len(x)) )
for traj in range(0, nrTraj):
    sample = np.dot(A + sigmaNoise * np.random.randn(1,4), X)[0]
    label = 'training set' if traj==0 else ''
    plt.plot(x, sample, 'grey', label=label)
    p.add_demonstration(sample)
    
p.plot(x=p.x, legend='trained module                                                                   ', color='r')
    
p.set_goal(0.05)
#plt.plot(1.0, 0.05, 'o', label="point",color='red')

plt.plot(p.x, p.generate_trajectory(), 'g', label='generated traj',linewidth=3)

plt.legend()
plt.show()
