// This is the particle class
// It provides features to the particle and also lists how their positions and velocities are updated

#ifndef _PARTICLE_H
#define _PARTICLE_H

#include "vector3d.h"

class PARTICLE 
{
  public:

  // members
  int id;		// id of the particle
  double diameter;	// diameter of the particle
  double m; 		// mass of the particle
  VECTOR3D posvec;	// position vector of the particle
  VECTOR3D velvec;	// velocity vector of the particle
  VECTOR3D forvec;	// force vector on the particle
  double pe;		// potential energy
  long double ke;	// kinetic energy
  double energy;	// energy
  double lx, ly, lz;	// box lengths-- useful information
  
  // member functions
  
  // make a particle
  PARTICLE(int initial_id = 0, double initial_diameter = 0, double initial_mass = 0, VECTOR3D initial_position = VECTOR3D(0,0,0), double initial_lx = 0, double initial_ly = 0, double initial_lz = 0)
  {
    id = initial_id;
    m = initial_mass;
    posvec = initial_position;
    lx = initial_lx;
    ly = initial_ly;
    lz = initial_lz;
  }
  
  // the next two functions are central to the velocity-Verlet algorithm
  // update position of the particle
  void update_position(double dt)		// dt is taken from outside; it is the time-step that the user supplies (based on steps)
  {
    posvec = ( posvec + (velvec * dt) );	// position updated to a full time-step
    // periodic boundary
    if (posvec.x > lx/2.0)
      posvec.x = posvec.x - lx;
    if (posvec.x < -lx/2.0)
      posvec.x = posvec.x + lx;
    if (posvec.y > ly/2.0)
      posvec.y = posvec.y - ly;
    if (posvec.y < -ly/2.0)
      posvec.y = posvec.y + ly;
    if (posvec.z > lz/2.0)
      posvec.z = posvec.z - lz;
    if (posvec.z < -lz/2.0)
      posvec.z = posvec.z + lz;
    return;
  }
  
  // update velocity of the particle
  void update_velocity(double dt)	
  {
    velvec = ( velvec + ( (forvec) * ( 0.5 * dt / m ) ) );	// notice the half time-step
    return;
  }
  
  // calculate kinetic energy of a particle
  void kinetic_energy()				
  {
    ke = 0.5 * m * velvec.Magnitude() * velvec.Magnitude();	// note that Magnitude function is a member of the VECTOR3D class and gets you the magnitude of a 3-D vector.
    return;
  }
};

#endif