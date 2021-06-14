# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:37:15 2020

@author: fredr
"""
import numpy as np
#import num_int_adaptive as nia
from random import random
from scipy.stats import linregress
from scipy.integrate import quad

import matplotlib.pyplot as plt
from scipy.signal import argrelextrema 
from random import random


def x_deriv(x, y, z, sigma):
    return sigma*(y - x)

def y_deriv(y, z, x, rho):
    return (x*(rho-z)-y)

def z_deriv(z, x, y, beta):
    return (x*y-beta*z)

def update_values(x, y, z, sigma, rho, beta, delta, TOL):
#    x_new = x + nia.adap_q(x, x+delta, x_deriv, nia.simpsons, TOL, y, z, sigma)
#    y_new = y + nia.adap_q(y, y+delta, y_deriv, nia.simpsons, TOL, z, x, rho)
#    z_new = z + nia.adap_q(z, z+delta, z_deriv, nia.simpsons, TOL, x, y, beta)   
    x_new = x + quad(x_deriv, x, x+delta, (y,z,sigma), epsrel = 1e-3)[0]
    y_new = y + quad(y_deriv, y, y+delta, (z,x,rho), epsrel = 1e-3)[0]
    z_new = z + quad(z_deriv, z, z+delta, (x, y, beta), epsrel = 1e-3)[0]
    return x_new, y_new, z_new


def run_lorenz(x_0, y_0, z_0, t_f, delta, sigma, rho, beta, TOL):
    N = int(t_f/delta) - 1        #Number of iterations 
    X, Y, Z, T = [np.zeros(N+1), np.zeros(N+1), np.zeros(N+1), np.zeros(N+1)]
    X[0],Y[0], Z[0] = x_0, y_0, z_0 #Set initial conditions
    for i in range(N):
        X[i+1], Y[i+1], Z[i+1] = update_values(X[i], Y[i], Z[i], sigma, rho, beta, delta, TOL)
        T[i+1] = T[i]+delta
    return X, Y, Z, T


#Task 2.2c i
def point_after_time(t_start, delta, sigma, rho, beta, TOL): #Define x_0 as the point reached after "t_start"
    x, y, z, _ = run_lorenz(random()*1e-5, random()*1e-5, random()*1e-5, t_start, delta, sigma, rho, beta, TOL)
    return np.array([x[-1], y[-1], z[-1]])

#Task 2.2c ii
def point_in_neighbourhood(point_0, d): #Returns a random vector at a distance "d" from x_0
    theta = random()*2*np.pi #Defines two random angles.
    phi = random()*2*np.pi   
    return (point_0 + np.array([d*np.cos(phi)*np.sin(theta), d*np.sin(phi)*np.sin(theta), d*np.cos(theta)]))


def lorenz_seperation(t_f, delta, sigma, rho, beta, TOL):
    point_0 = point_after_time(50, delta, sigma, rho, beta, TOL) #Find start point point_0
    point_1 = point_in_neighbourhood(point_0, 1e-6)              #Find point at a distance 1e-6 point_0
    #Run analysis for point_0 and point_1:
    x_0, y_0, z_0, T = run_lorenz(point_0[0], point_0[1], point_0[2], t_f, delta, sigma, rho, beta, TOL)
    x_1, y_1, z_1, _ = run_lorenz(point_1[0], point_1[1], point_1[2], t_f, delta, sigma, rho, beta, TOL)
    #Make a matrix of seperations between x's, y's and z's at each point
    seperation_matrix = np.array([x_0 - x_1, y_0 - y_1, z_0 - z_1])   
    seperation_array = []
    for column in range(len(seperation_matrix[0])): #Calculate absolute seperation at each time_step. Notice that each column represents a new time step
        sep_at_one_point_in_time = np.sqrt((seperation_matrix[0][column])**2 + \
                                           (seperation_matrix[1][column])**2 + \
                                           (seperation_matrix[2][column])**2)
        seperation_array.append(np.log(sep_at_one_point_in_time))
    return seperation_array, T

def liapunov_constant(T_c, delta, sigma, rho, beta, TOL, N): #N = number of iterations to average over
    #This calculates the Liapunov constant for the Lorenz system averaged over N simulations. 
    liapunov_array = np.zeros(N)
    for n in range(N):
        sep, T = lorenz_seperation(T_c, delta, sigma, rho, beta, TOL)
        liapunov_array[n], _, _, _, _ = linregress(sep, T)
    return liapunov_array.mean(), sep, T

def generate_points(t_simulate, N, delta, sigma, rho, beta, TOL): #returns a Nx3 matrix with locations of the points
    particles = np.empty((0,3)) #A 2D-array with locations of all "particles", each defined 3 spatial coordinates
    particles = np.append(particles, np.array([point_after_time(50, delta, sigma, rho, beta, TOL)]), axis = 0) #Append a single point on the attractor to particles. Point found using point_after_time
    for i in range(N-1):
        new_particle = point_in_neighbourhood(particles[0], 1e-2) #Generate a new random point in the neighbourhood of first point
        x_1, y_1, z_1, _ = run_lorenz(new_particle[0], new_particle[1], new_particle[2], 0, t_simulate, delta, sigma, rho, beta, TOL) #Time evolve this new point
        particles = np.append(particles, np.array([[x_1[-1], y_1[-1], z_1[-1]]]), axis = 0) #Append the new point to the set of particles
    return particles 

def plot(array_1, array_2, label_x, label_y, style = "o-", c = "blue"):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.plot(array_1, array_2, style, color = c)
    return 1

def work_station():
    x_0, y_0, z_0 = [0.001]*3 #Set Initial values
    delta, TOL = 1e-3, 1e-4  #Set partition delta and quadrature error tolerance TOL
    sigma, rho, beta = 10, 28, 8
    t_limit = 40
    x,y,z,T = run_lorenz(x_0, y_0, z_0, t_limit, delta, sigma, rho, beta, TOL)
    plot()


print("Ran as ", __name__)
if __name__ == "__main__":
    work_station()
    return 0