#!/usr/bin/python
# Filename: simple.py

import ipromp
import numpy as np
import matplotlib.pyplot as plt

# close the current windows
plt.close()

# create a ProMP object
p = ipromp.ProMP()

# Generate and plot trajectory Data
x = np.arange(0,1.01,0.01)           # time points for trajectories
nrTraj=30                            # number of trajectoreis for training
sigmaNoise=0.02                      # noise on training trajectories
A = np.array([.2, .2, .01, -.05])    # the weight of different func 
X = np.vstack( (np.sin(5*x), x**2, x, np.ones((1,len(x))) ))    # the basis func

# add demonstration
for traj in range(0, nrTraj):
    sample = np.dot(A + sigmaNoise * np.random.randn(1,4), X)[0]
    label = 'training set' if traj==0 else ''
    plt.plot(x, sample, 'grey', label=label)
    p.add_demonstration(sample)

# add via point as observation
p.set_start(-0.04+0.01)
p.add_viapoint(0.1, 0.055+0.01)
p.add_viapoint(0.2, 0.130+0.01)

# plot the trained model and via point
p.plot(x=p.x, color='r')

# plot the generated traj
plt.plot(p.x, p.generate_trajectory(), 'g', label='generated traj',linewidth=3)

plt.legend()
plt.show()
