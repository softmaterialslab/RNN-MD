// This file contains the routines 

#include "functions.h"

// overload out
ostream& operator<<(ostream& os, VECTOR3D vec)
{
  os << vec.x << setw(15) << vec.y << setw(15) << vec.z;
  return os;
}

// make movie
void make_movie(int num, vector<PARTICLE>& atom, double bx, double by, double 
bz, ofstream& outdump)
{
  outdump << "ITEM: TIMESTEP" << endl;
  outdump << num - 1 << endl;
  outdump << "ITEM: NUMBER OF ATOMS" << endl;
  outdump << atom.size() << endl;
  outdump << "ITEM: BOX BOUNDS" << endl;
  outdump << -0.5*bx << "\t" << 0.5*bx << endl;
  outdump << -0.5*by << "\t" << 0.5*by << endl;
  outdump << -0.5*bz << "\t" << 0.5*bz << endl;
  outdump << "ITEM: ATOMS index type x y z v" << endl;
  for (unsigned int i = 0; i < atom.size(); i++)
  {
    outdump << i+1 << "   " << "1" << "   " << atom[i].posvec.x << "   " << atom[i].posvec.y << "   " << atom[i].posvec.z << "   " << (atom[i].velvec.Magnitude()) << endl;
  }
  return;
}

// initialize velocities of particles to start simulation
void initialize_particle_velocities(vector<PARTICLE>& ljatom)
{
  
  // initialize velocities
  for (unsigned int i = 0; i < ljatom.size(); i++) 
  { 
    ljatom[i].velvec = VECTOR3D(0,0,0);	
  }
  
  // average velocity should be 0; as there is no net flow of the system in any particular direction; we do this next
  VECTOR3D average_velocity_vector = VECTOR3D(0,0,0);
  for (unsigned int i = 0; i < ljatom.size(); i++) 
    average_velocity_vector = average_velocity_vector + ljatom[i].velvec;
  average_velocity_vector = average_velocity_vector*(1.0/ljatom.size());
  
  // subtract this computed average_velocity_vector from the velocity of each particle to ensure that the total average after this operation is 0
  for (unsigned int i = 0; i < ljatom.size(); i++) 
    ljatom[i].velvec = ljatom[i].velvec - average_velocity_vector;
  
  return;
}

// display progress
void ProgressBar (double fraction_completed)
{
    int val = (int) (fraction_completed * 100);
    int lpad = (int) (fraction_completed * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% |%.*s%*s|", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}