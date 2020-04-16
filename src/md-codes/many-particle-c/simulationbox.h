// This is a header file for the SIMULATIONBOX class.  

#ifndef _SIMULATIONBOX_H
#define _SIMULATIONBOX_H

using namespace std;

#include<vector>
#include<iostream>
#include<fstream>
#include<iomanip>
#include "vector3d.h"
#include "particle.h"
//OPENMP
#include <omp.h>

class SIMULATIONBOX
{
  public:

  VECTOR3D posvec;		// origin point of the simulation box
  double lx;			// length of the box in x direction
  double ly;			// length of the box in y direction
  double lz; 			// length of the box in z direction
  
  // put particles inside the box
  void put_ljatoms_inside(vector<PARTICLE>&, int, double, double, double);
  
  SIMULATIONBOX(VECTOR3D m_posvec = VECTOR3D(0,0,0), double m_lx = 10.0, double m_ly = 10.0, double m_lz = 10.0)
  {
    posvec = m_posvec;
    lx = m_lx;
    ly = m_ly;
    lz = m_lz;
  }
};

#endif

