import math

class Vector3D:
    def __init__(self, initial_x = 0.0, initial_y = 0.0, initial_z = 0.0):
        self.x = initial_x
        self.y = initial_y
        self.z = initial_z
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    # Operator overloading for adding two vecs  
    def __add__(self, v): 
        return Vector3D(self.x + v.x, self.y + v.y, self.z + v.z)
    
    def __sub__(self, v): 
        return Vector3D(self.x - v.x, self.y - v.y, self.z - v.z)
    
    def __mul__(self, scalar):
        return Vector3D(self.x*scalar, self.y*scalar, self.z*scalar)
    
    #printing overloaded
    def __str__(self): 
        return "x=" + str(self.x) + ", y=" + str(self.y) + ", z=" + str(self.z)


import math

class Particle:
    
    def __init__(self, initial_m = 1.0, diameter = 2.0, initial_position = Vector3D(0.0, 0.0, 0.0), initial_velocity = Vector3D(0.0, 0.0, 0.0)):
        self.m = initial_m
        self.d = diameter
        self.position = initial_position
        self.velocity = initial_velocity
        self.sigma = 1.0
        self.eps = 1.0
        
    def volume(self):
        self.volume = (4.0/3.0) * math.pi * ((self.d/2.0)**3)
    
    def update_position(self, dt):
         self.position = self.position + (self.velocity * dt)
            
    def get_force_on_block(self):
        #self.force = self.position * -1.0 * k
        #temp = Vector3D(0.0, 0.0, 0.0)
        #temp.x = 48 * self.eps * ( ((self.sigma)**12/(self.position.x **13)) - (0.5*(self.sigma)**6/(self.position.x **7)) )
        #self.force = temp
        # consider a fixed particle in 0, 0, 0
        rvec = self.position - Vector3D(0.0, 0.0, 0.0)
        r = rvec.magnitude()
        self.force = rvec * 48 * self.eps * (1/r**2) * ( ((self.sigma**12)/(r**12)) - (0.5*(self.sigma**6)/(r**6)) )

    def update_velocity(self, dt):
         self.velocity = self.velocity + (self.force * (dt / self.m))        

    def kinetic_energy(self):
         self.ke = 0.5 * self.m * (self.velocity.magnitude()**2)
            
    def get_energy_on_block(self):
        #self.pe = 0.5 * k * (self.position.x **2)
        # consider a fixed particle in 0, 0, 0
        r = (self.position - Vector3D(0.0, 0.0, 0.0)).magnitude()
        self.pe =  4 * self.eps * ( ((self.sigma**12)/(r**12)) - ((self.sigma**6)/(r**6)) )
        

import math
import time

def velocity_verlet(mass=None, initial_pos=2.0, time=100, deltaT=0.01):
    
    print("Modeling the block-spring system")
    print("Need a useful abstraction of the problem: a point particle")
    print("Make a Particle class")
    print("Set up initial conditions")

    
    sphere = Particle(initial_m = 1.0, diameter = 2.0, initial_position = Vector3D(0.0, 0.0, 0.0), initial_velocity = Vector3D(0.0, 0.0, 0.0))
    sphere_volume = sphere.volume()
    print("volume of a unit (radius = 1) sphere is {}".format(sphere_volume))
    
    # inputs
    if mass is None:
        print("enter mass of the block: ")
        time.sleep(0.1) # This sleep is not needed, just added to get input box below the print statements
        mass = float(input())
    
    block = Particle(initial_m = mass, diameter = 2.0, initial_position = Vector3D(initial_pos, 0.0, 0.0), initial_velocity = Vector3D(0.0, 0.0, 0.0));
    
    # we can compute the initial force on the block
    block.get_force_on_block()

    #Print the system
    print("mass of the block is {}".format(block.m))
    print("initial position of the block is {}".format(block.position.x))
    print("initial velocity of the block is {}".format(block.velocity.x))
    print("initial force on the block is {}".format(block.force.x))
     
    t = time
    dt=deltaT
    S = int(t // dt)

    #simulation Begins here
    simulated_result = open("data/dynamics_mass={}_x0={:.2f}_t={}_deltaT={}.out".format(mass,initial_pos,t, dt), "w")
    block.get_energy_on_block()
    block.kinetic_energy()
    simulated_result.write("{0:.6f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}  {5:.6f} {6:.6f} \n".format(0*dt, block.position.x, block.velocity.x, block.ke, block.pe, (block.ke + block.pe), block.force.x))
     
    for i in range(S):
        block.update_velocity(dt/2.0) #update velocity half timestep
        block.update_position(dt) #update position full timestep
        block.get_force_on_block()
        block.update_velocity(dt/2.0)
        #filing the time, position of the block
        block.kinetic_energy()
        block.get_energy_on_block()
        simulated_result.write("{0:.6f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}  {5:.6f} {6:.6f} \n".format((i+1)*dt, block.position.x, block.velocity.x, block.ke, block.pe, (block.ke + block.pe), block.force.x))
    
    simulated_result.close() 
    print("Simulation is over.")


import itertools
import multiprocessing
import os, time
from multiprocessing.pool import ThreadPool
import os, time
import numpy as np
import glob
import pickle
from sklearn.utils import shuffle
import random
import threading

# Helper function to unpack args
def merge_names_unpack(args):
    print("Proccess id: ", os.getpid())
    return velocity_verlet(*args)

if __name__ == "__main__":
    #Generate processes equal to the number of cores
    pool_size  = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(pool_size)
    #Generate values for each parameter
    mass_ = range(1,11,1)
    initial_pos = np.arange(1,3.05,0.05)
    time=[100] 
    deltaT=[0.001]

    #Generate a list of tuples where each tuple is a combination of parameters.
    #The list will contain all possible combinations of parameters.
    # Sequance is important: mass=mass_, k=k_, noise_freq=f_
    paramlist = list(itertools.product(mass_, initial_pos, time, deltaT))
    #Distribute the parameter sets evenly across the cores
    pool_output = pool.map(merge_names_unpack, paramlist)
    print(pool_output)