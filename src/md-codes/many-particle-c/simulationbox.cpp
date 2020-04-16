// This file contains member functions for interface class

#include "simulationbox.h"
#include "functions.h"

void SIMULATIONBOX::put_ljatoms_inside(vector<PARTICLE>& ljatom, int number_ljatom, double ljatom_diameter, double ljatom_mass, double density)
{
  // initialize the positions of the particles on a lattice (not randomly)
  
  //ofstream out_initialize_position("ip.lammpstrj", ios::out);
  cout << "# of atoms created in the box... " << "  ";

  //This is the diameter of a particle
  double a = 1;
  cout << "lattice spacing (based on crystal of closest packing) is " << a << endl;
  
  unsigned int num_atoms_linear = int(ceil(pow((double(number_ljatom)),1.0/3.0)));
  cout << "in each direction, about " << num_atoms_linear << " atoms " << endl;

  //double lx_start = -lx/2, ly_start = -ly/2, lz_start = -lz/2;
  //(num_atoms_linear-1): -1 is because worst case: fully packed d~1.0 atom center is at the edge of the box
  double lx_start = -a*(num_atoms_linear-1)/2,  ly_start = -a*(num_atoms_linear-1)/2, lz_start = -a*(num_atoms_linear-1)/2;

  for (unsigned int i = 0; i < num_atoms_linear; i++)
  {
    for (unsigned int j = 0; j < num_atoms_linear; j++)
    {
      for (unsigned int k = 0; k < num_atoms_linear; k++)
      {
	if (int(ljatom.size()) < number_ljatom)	// stop if atoms created are equal to the requested number
	{
	  double x = lx_start + i*a;
	  double y = ly_start + j*a;
	  double z = lz_start + k*a;
	  if (z >= lz/2.0 - a/2.0 || y >= ly/2.0 - a/2.0 || x >= lx/2.0 - a/2.0)
	    continue;
	  VECTOR3D position = VECTOR3D(x,y,z);
	  PARTICLE fresh_atom = PARTICLE(int(ljatom.size())+1,ljatom_diameter,ljatom_mass,position,lx,ly,lz);
	  ljatom.push_back(fresh_atom);
	  // make a movie as you go to see how you are doing
	  // not the most efficient movie making as I take all the atoms created into the movie funciton and not just the new atom
	  //make_movie(ljatom.size(),ljatom,lx,ly,lz,out_initialize_position);
	}
      }
    }
  }
  
  // make a snapshot movie of initial positions of the particles
  //ofstream list_ljatoms("final_ip.lammpstrj", ios::out);
  //make_movie(ljatom.size(),ljatom,lx,ly,lz,list_ljatoms);
  //list_ljatoms.close();
  
  return;
} 
