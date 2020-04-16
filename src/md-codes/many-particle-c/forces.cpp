// This file contains the routine that computes the LJ force on the particle

using namespace std;

#include <vector>
#include "particle.h"
#include "simulationbox.h"

void update_forces(vector<PARTICLE>& ljatom, SIMULATIONBOX& box, double dcut) 
{
  unsigned int i,j;
  double elj = 1.0;
  VECTOR3D r_vec, flj;
  double d, d2, d6, d12, dcut2;
  d = 1; // recall that we are working in reduced units where the unit of length is the diameter of the particle
  //r2 = (r_vec.Magnitude()) * (r_vec.Magnitude());
  d2 = d * d;
  d6 = d2 * d2 * d2;
  d12 = d6 * d6;
  dcut2 = dcut*dcut;

#pragma omp parallel for schedule(dynamic) default(shared) private(i, j, flj, r_vec)
  for (i = 0; i < ljatom.size(); i++)
  {
    flj = VECTOR3D(0,0,0);
    for (j = 0; j < ljatom.size(); j++)
    {
      if (j == i) continue;
      r_vec = ljatom[i].posvec - ljatom[j].posvec;
      
      // the next 6 lines take into account the periodic nature of the boundaries of our simulation box
      if (r_vec.x>box.lx/2) r_vec.x -= box.lx;
      else if (r_vec.x<-box.lx/2) r_vec.x += box.lx;
      if (r_vec.y>box.ly/2) r_vec.y -= box.ly;
      else if (r_vec.y<-box.ly/2) r_vec.y += box.ly;
      if (r_vec.z>box.lz/2) r_vec.z -= box.lz;
      else if (r_vec.z<-box.lz/2) r_vec.z += box.lz;

      double r2, r6, r12;

      r2 = r_vec.Magnitude_sqared();

      if (r2 < dcut2*d2)
      {
	    r6 = r2 * r2 * r2;
	    r12 = r6 * r6;

	    flj = flj + ( r_vec * ( 48 * elj * (  (d12 / r12)  - 0.5 *  (d6 / r6) ) * ( 1 / r2 ) ) );
      }
    }

      ljatom[i].forvec =  flj;
  }


  return; 
}

