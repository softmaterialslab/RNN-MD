from multiprocessing import Pool
import os, time



import math
import numpy as np

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
    
    def __mul__(self, scalar):
        return Vector3D(self.x*scalar, self.y*scalar, self.z*scalar)
    
    #printing overloaded
    def __str__(self): 
        return "x=" + str(self.x) + ", y=" + str(self.y) + ", z=" + str(self.z)
    
        
import math

class Particle:
    
    def __init__(self, initial_m = 1.0, diameter = 2.0, initial_position = Vector3D(0.0, 0.0, 0.0), initial_velocity = Vector3D(0.0, 0.0, 0.0), noise_freq=None):
        self.m = initial_m
        self.d = diameter
        self.position = initial_position
        self.velocity = initial_velocity
        self.noise_freq = noise_freq
        self.current_noise = 0.0
        # This is gaussian noise
        # 0 is the mean of the normal distribution you are choosing from
        # 1 is the standard deviation of the normal distribution
        # 1000 is the number of elements you get in array noise
        self.noise = np.random.normal(0,1,1000)
        
    def volume(self):
        self.volume = (4.0/3.0) * math.pi * ((self.d/2.0)**3)
    
    def update_position(self, dt):
        self.position = self.position + (self.velocity * dt)
                  
    def get_force_on_block(self, k, i):
        if self.noise_freq is not None and (i%self.noise_freq == 0) and i!=0:
            self.current_noise = self.noise[np.random.randint(len(self.noise), size=1)[0]]
            
            
        self.force = self.position * -1.0 *  k
        self.force.x +=  k*self.current_noise
        
    def update_velocity(self, dt):
        self.velocity = self.velocity + (self.force * (dt / self.m))
        

    def kinetic_energy(self):
        self.ke = 0.5 * self.m * (self.velocity.magnitude()**2)
            
    def get_energy_on_block(self, k):
        self.pe = 0.5 * k * ((self.position.x - self.current_noise) **2)
        #self.pe = 0.5 * k * (self.position.x **2)


import math
import time

def velocity_verlet(mass=None, k=None, x0=None, noise_freq=None):
    
    start = time.time()
    
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
    if k is None:
        print("enter spring constant: ")
        time.sleep(0.1) # This sleep is not needed, just added to get input box below the print statements
        k = float(input())

    if x0 is None:
        intial_pos = -1.0
    else:
        intial_pos = x0

    block = Particle(initial_m = mass, diameter = 2.0, initial_position = Vector3D(intial_pos, 0.0, 0.0), initial_velocity = Vector3D(0.0, 0.0, 0.0), noise_freq=noise_freq)
    
    # we can compute the initial force on the block
    block.get_force_on_block(k, 0)

    #Print the system
    print("mass of the block is {}".format(block.m))
    print("spring constant is {}".format(k))
    print("initial position of the block is {}".format(block.position.x))
    print("initial velocity of the block is {}".format(block.velocity.x))
    print("initial force on the block is {}".format(block.force.x))
     
    #we are interested in simulating the dynamics of this block against the spring force 
    #exact solution is available
    #x(t) = - cos(sqrt(k/m)*t)
    #discretize the exact solution and file it

    t = 1000 #time
    #N = 10000 #number of data points for which exact solution is discretized
    
    '''
    exact_solution = []
    omega = math.sqrt(k/mass)
    A = abs(intial_pos)
    for i in range(N):
        exact_solution.append(-A*math.cos(omega*i*t/N))
        #exact_solution.write("{0:.3f}  {1:.3f}\n".format(i*t/N, -math.cos(i*t/N)))
    '''
    #exact_solution.close()  
    #use the computer to simulate the dynamics of the particle
    #need integrators that evolve the trajectory using the equation of motion
    #one timestep dt at a time

    #simulation Begins here
    
    simulated_result = open("data/dynamics_mass={}_k={}_x={}_f={}_t={}.out".format(mass,k,intial_pos,noise_freq,t), "w")

    S = 100000;
    dt = t/S;

    for i in range(S):
        block.update_velocity(dt/2.0) #update velocity half timestep
        block.update_position(dt) #update position full timestep
        block.get_force_on_block(k, i)
        block.update_velocity(dt/2.0)
        #filing the time, position of the block
        block.kinetic_energy()
        block.get_energy_on_block(k)
        # time, extact_dynamics, MD_dynamics, velocity, ke, pe, (ke + pe)
        simulated_result.write("{0:.3f}  {1:.3f}  {2:.3f}  {3:.3f}  {4:.3f}  {5:.3f} {6:.3f} \n".format(i*dt, block.position.x, block.velocity.x, block.ke, block.pe, (block.ke + block.pe), block.current_noise))
    
    simulated_result.close() 
    print("Simulation is over.")
    end = time.time()
    print("Time: "+str(end - start) + " for MD simulation ")


import itertools
import multiprocessing

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
    k_ = range(1,11,1)
    x_ = range(-10,0,1)
    #f_values = [1, 10, 100, 1000, 10000]
    f_values = [2000000]
    
    #Generate a list of tuples where each tuple is a combination of parameters.
    #The list will contain all possible combinations of parameters.
    # Sequance is important: mass=mass_, k=k_, noise_freq=f_
    paramlist = list(itertools.product(mass_,k_,x_,f_values))
    #Distribute the parameter sets evenly across the cores
    pool_output = pool.map(merge_names_unpack, paramlist)
    print(pool_output)