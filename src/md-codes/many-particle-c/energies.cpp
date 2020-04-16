// This function has the potential energy evaluations 
// For now, we have LJ interaction potential between particles

using namespace std;
#include <vector>
#include "particle.h"
#include "simulationbox.h"

double update_energies(vector<PARTICLE>& ljatom, SIMULATIONBOX& box, double dcut)
{
  // potential energy
  vector<double> lj_atom_atom;
  
  // what you need to compute energy
  VECTOR3D r_vec;
  double r, r2, r6, d, d2, d6;
  
  // recall that we are working with a truncated-shifted LJ potential
  // while the shift does not matter in the force calculation (why?), it matters here, hence the energy_shift calculation below
  // compute the energy shift term  
  double elj = 1.0;
  double dcut2 = dcut * dcut;
  double dcut6 = dcut2 * dcut2 * dcut2;
  double dcut12 = dcut6 * dcut6;
  double energy_shift = 4*elj*(1/dcut12 - 1/dcut6);
  d = 1;
  d2 = d * d;
  d6 = d2 * d2 * d2;
  
  // energy is computed as pair-wise sums
  for (unsigned int i = 0; i < ljatom.size(); i++)
  {
    double uljpair = 0.0;
    for (unsigned int j = 0; j < ljatom.size(); j++)
    {
      if (j == i) continue;
      
      r_vec = ljatom[i].posvec - ljatom[j].posvec;
      
      if (r_vec.x > box.lx/2) r_vec.x -= box.lx;
      if (r_vec.x < -box.lx/2) r_vec.x += box.lx;
      if (r_vec.y > box.ly/2) r_vec.y -= box.ly;
      if (r_vec.y < -box.ly/2) r_vec.y += box.ly; 
      if (r_vec.z > box.lz/2) r_vec.z -= box.lz;
      if (r_vec.z < -box.lz/2) r_vec.z += box.lz;
      
      r2 = r_vec.Magnitude_sqared();
      	// note: reduced units, all particles have the same diameter
      if (r2 < dcut2 * d2)
      {
	    r6 = r2 * r2 * r2;

	    uljpair = uljpair +  4 * elj * (d6 / r6) * ( ( d6 / r6 ) - 1 ) - energy_shift;
      }

    }
    lj_atom_atom.push_back(uljpair);
  }
  
  double total_lj_atom_atom = 0;
  for (unsigned int i = 0; i < ljatom.size(); i++)
    total_lj_atom_atom += lj_atom_atom[i];
  total_lj_atom_atom = 0.5 * total_lj_atom_atom;
  // factor of half for double counting -- does that make sense?
  
  return total_lj_atom_atom;
}
