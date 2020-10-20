###########################################################################
# March 2019, Orit Peleg, orit.peleg@colorado.edu
# Code for HW3 CSCI 4314/5314 Dynamic Models in Biology
###########################################################################

import numpy as np
import math
import matplotlib.pyplot as plt
import random

class flock():
    def flocking_python(self):
        N = 400 #No. of Boids
        frames = 100 #No. of frames
        limit = 100 #Axis Limits
        L  = limit*2
        P = 10 #Spread of initial position (gaussian) // Test: P = 50 (Initial: P = 10)
        V = 10 #Spread of initial velocity (gaussian)
        delta = 1 #Time Step
        c1 = 0.00001 #Attraction Scaling factor 0.01 (0.00001)
        c2 = 0.01 #Repulsion scaling factor 0.1 (0.01)
        c3 = 1 #Heading scaling factor 5, 0.5 (1)
        c4 = 0.01 #Randomness scaling factor (0.01)
        vlimit = 1 #Maximum velocity

        #Initialize
        p = P*np.random.randn(2,N) # Matrix: Row: x, y Column: Agent No. (Position, Velocity)
        v = V*np.random.randn(2,N)

        additionalAgents = 0

        # Obstacle: Agents: Static
        # additionalAgents = 16
        # p = P*np.random.randn(2, N + additionalAgents) # Matrix: Row: x, y Column: Agent No. (Position, Velocity)
        # v = V*np.random.randn(2, N + additionalAgents)
        #
        # p[:, N] = [0, -60]
        # v[:, N] = [0.2, 0.2]
        #
        # p[:, N + 1] = [0, 60]
        # v[:, N + 1] = [0.2, 0.2]
        #
        # p[:, N + 2] = [0, -55]
        # v[:, N + 2] = [0.2, 0.2]
        #
        # p[:, N + 3] = [0, 55]
        # v[:, N + 3] = [0.2, 0.2]
        #
        # p[:, N + 4] = [0, -50]
        # v[:, N + 4] = [0.2, 0.2]
        #
        # p[:, N + 5] = [0, 50]
        # v[:, N + 5] = [0.2, 0.2]
        #
        # p[:, N + 6] = [0, -45]
        # v[:, N + 6] = [0.2, 0.2]
        #
        # p[:, N + 7] = [0, 45]
        # v[:, N + 7] = [0.2, 0.2]
        #
        # p[:, N + 8] = [0, -40]
        # v[:, N + 8] = [0.2, 0.2]
        #
        # p[:, N + 9] = [0, 40]
        # v[:, N + 9] = [0.2, 0.2]
        #
        # p[:, N + 10] = [0, -30]
        # v[:, N + 10] = [0.2, 0.2]
        #
        # p[:, N + 11] = [0, 30]
        # v[:, N + 11] = [0.2, 0.2]
        #
        # p[:, N + 12] = [0, -20]
        # v[:, N + 12] = [0.2, 0.2]
        #
        # p[:, N + 13] = [0, 20]
        # v[:, N + 13] = [0.2, 0.2]
        #
        # p[:, N + 14] = [0, -10]
        # v[:, N + 14] = [0.2, 0.2]
        #
        # p[:, N + 15] = [0, 10]
        # v[:, N + 15] = [0.2, 0.2]

        #Initializing plot
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)


        for i in range(0, frames):
            v1 = np.zeros((2,N))
            v2 = np.zeros((2,N))

            #YOUR CODE HERE
            #Calculate Average Velocity v3

            # Calculate: v3 = Mean: [X, Y]
            v3 = np.array([[np.mean(v[0, :])*c3], [np.mean(v[1, :])*c3]])

            # print("V3 Shape: ", v3.shape)
            # v3 = np.array([[np.sum(v[0, :])*c3], [np.sum(v[1, :])*c3]])

            if (np.linalg.norm(v3) > vlimit): #limit maximum velocity
                v3 = v3*vlimit/np.linalg.norm(v3)

            for n in range(0, N): # Agents: n
                for m in range(0, N + additionalAgents):
                    if m!=n:
                        #YOUR CODE HERE
                        #Compute vector r from one agent to the next-

                        # Calculate: r = pm - pn
                        # Formula: r  = pm - pn
                        r = p[:, m] - p[:, n]

                        # Wrap Around
                        if r[0] > L/2:
                            r[0] = r[0]-L
                        elif r[0] < -L/2:
                            r[0] = r[0]+L

                        if r[1] > L/2:
                            r[1] = r[1]-L
                        elif r[1] < -L/2:
                            r[1] = r[1]+L

                        #YOUR CODE HERE

                        # Calculate: rmag = Square Root (r[0]^2 + r[1]^2)
                        #Compute distance between agents rmag-
                        rmag = math.sqrt((r[0])**2 + (r[1])**2)

                        # Calculate: v1 = v1 + (c1)(r)
                        #Compute attraction v1-
                        v1[:, n] = v1[:, n] + c1*r

                        # Calculate: v2 = v2 - ((c2)(r))/(rmag^2)
                        #Compute Repulsion [non-linear scaling] v2-
                        v2[:, n] = v2[:, n] - (c2*r)/(rmag**2)

                #YOUR CODE HERE

                # Calculate: v4 = (c4)(randomNumber)
                #Compute random velocity component v4-
                v4 = c4*np.random.randn(2, 1)

                #Update velocity-

                # Calculate: v = v1 + v2 + v3 + v4
                v[:, n] = v1[:, n] + v2[:, n] + v3[:, 0] + v4[:, 0]

            #YOUR CODE HERE

            #Update position
            # Calculate: p = p + (v)(delta)
            # p = p + v*delta
            p[:, N - 1] = p[:, N - 1] + v[:, N - 1]*delta

            #Periodic boundary
            tmp_p = p

            tmp_p[0, p[0,:]>L/2] = tmp_p[0,p[0,:]> (L/2)] - L
            tmp_p[1, p[1,:] > L/2] = tmp_p[1, p[1,:] > (L/2)] - L
            tmp_p[0, p[0,:] < -L/2]  = tmp_p[0, p[0,:] < (-L/2)] + L
            tmp_p[1, p[1,:] < -L/2]  = tmp_p[1, p[1,:] < (-L/2)] + L

            p = tmp_p
            # Can Also be written as:
            # p[p > limit] -= limit * 2
            # p[p < -limit] += limit * 2

            line1, = ax.plot(p[0, 0], p[1, 0])

            #update plot
            ax.clear()
            ax.quiver(p[0,:], p[1,:], v[0,:], v[1,:]) # For drawing velocity arrows
            plt.xlim(-limit, limit)
            plt.ylim(-limit, limit)
            line1.set_data(p[0,:], p[1,:])

            fig.canvas.draw()
            plt.show()

flock_py = flock()
flock_py.flocking_python()
