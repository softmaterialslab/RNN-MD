// This is main.
// This is MD simulation of particles interacting with a LJ potential in a periodic box 

using namespace std;

#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<vector>
#include "simulationbox.h"
#include "particle.h"
#include "functions.h"
#include <iostream>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <boost/program_options.hpp>


using namespace std;
using namespace boost::program_options;

void update_forces(vector<PARTICLE>&, SIMULATIONBOX&, double);
double update_energies(vector<PARTICLE>&, SIMULATIONBOX&, double);

int main(int argc, char* argv[])

{
  // we begin with some fundamental numbers/physical constants
  const double pi = 3.141593;	// Pi
  const double kB = 1.38e-23;	// Joules per Kelvin
  const double mol = 6.0e23;	// Avogadro number
  
  // reduced units
  double unitlength = 0.3405e-9; // m; unit of length is these many meters (diameter of a typical atom)
  double unitenergy = 119.8 * kB;// Joules; unit of energy is these many Joules (typical interaction strength)
  double unitmass = 0.03994 / mol; // kg; unit of mass (mass of a typical atom)
  double unittime = sqrt(unitmass * unitlength * unitlength / unitenergy); // Unit of time
  double unittemperature = unitenergy/kB;			// unit of temperature

  // essential parameters needed for particles in a box simulation
  /**********************/
  double ljatom_density=0.8442 ;	// this is taken in as mass density (in reduced units)
  int number_ljatom=8;		// total number of particles in your system
  double ljatom_diameter;	
  double ljatom_mass;
  double bx, by, bz;		// box edge lengths
  double temperature;
  double dcut;			// cutoff distance for potential in reduced units
  /**********************/

    options_description desc("Usage:\n Manyparticle <options>");
    desc.add_options()
            ("help,h", "print usage message")
            ("density,d", value<double>(&ljatom_density)->default_value(0.8442),
             "enter density (in LJ reduced units: recall unit of length is diameter of your lj particle); tested for rho = 0.8442 ")                // enter in nanometers
            ("number_ljatom,n", value<int>(&number_ljatom)->default_value(108),
             "enter total number of ljatoms (108 is a typical value; tested for this value) ");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

  cout << "\nProgram starts\n";
  cout << "units of length, energy, mass are given near the top of the main function" << endl;
  cout << "unit of length is " << unitlength << " meters" << endl;
  cout << "unit of mass is " << unitmass << " kilograms" << endl;
  cout << "unit of energy is " << unitenergy << " Joules" << endl;
  cout << "unit of time is " << unittime << " seconds" << endl;
  cout << "enter density (in LJ reduced units: recall unit of length is diameter of your lj particle); tested for rho = 0.8442  :" << ljatom_density <<endl;
  //cin >> ljatom_density;
  cout << "enter total number of ljatoms (108 is a typical value; tested for this value) :" << number_ljatom << endl;
  //cin >> number_ljatom;
  dcut = 2.5;
  
  ljatom_diameter = 1.0;	// in reduced units
  ljatom_mass = 1.0;
  
  double edge_length = pow(number_ljatom/ljatom_density,1.0/3.0);
  cout << "edge length calculated to be " << edge_length << endl;

#pragma omp parallel default(shared)
  {
    if (omp_get_thread_num() == 0) {
     printf("The app comes with OpenMP parallelization)\n");
     printf("Number of OpenMP threads %d\n", omp_get_num_threads());
     printf("Make sure that number of ions is greater than %d\n", omp_get_num_threads());
   }
  }

  // Different parts of the system
  vector<PARTICLE> ljatom;		// all particles in the system
  
  bx = edge_length; 
  by = edge_length;
  bz = edge_length;
  SIMULATIONBOX simulation_box = SIMULATIONBOX(VECTOR3D(0,0,0),bx,by,bz);		// place where the action happens, periodic boundaries
  simulation_box.put_ljatoms_inside(ljatom, number_ljatom, ljatom_diameter, ljatom_mass, ljatom_density);				
  
  // output to screen the parameters of the problem
  cout << "Box dimensions x | y | z " << setw(15) << simulation_box.lx << setw(15) << simulation_box.ly << setw(15) << simulation_box.lz << endl;
  cout << "Number of LJ atoms inside the box " << ljatom.size() << endl;
  cout << "LJ atom diameter (in SI units; meter) " << ljatom_diameter*unitlength << endl;
  cout << "LJ atom mass (in SI units; kg) " << ljatom_mass*unitmass << endl;
  cout << "cut off distance (in reduced units) " << dcut << endl;
  
  initialize_particle_velocities(ljatom);
  
  update_forces(ljatom, simulation_box, dcut);	// expensive step
  
  double delta_t, totaltime;
  totaltime = 200;
  int steps;		// number of time discretizations (slices)
  steps = 200000;

  delta_t = totaltime/steps;	// the code deterimes the time-step delta_t. choose steps carefully, make sure you have a fine discretization
  cout << "timestep (reduced units) " << delta_t << endl;
  cout << "timestep (in SI units; second) " << delta_t*unittime << endl;
  cout << "total simulation time (in real units) " << totaltime*unittime << endl;

  char filename_movie[200];
  sprintf(filename_movie, "propagation_rho=%f_N=%d.lammpstrj", ljatom_density, number_ljatom);

  ofstream list_propagation(filename_movie, ios::out); // create a file to store data; first line displays the labels
  
  // filing the data with naming within the code. -- sometimes could be handy -- uncomment next three lines and comment the 4th if you want that.
  char filename[200];
  sprintf(filename, "energy_rho=%f_N=%d.out", ljatom_density, number_ljatom);
  ofstream output_energy(filename, ios::out);
  //ofstream output_energy("energy.out", ios::out);
  
  double totalpe = update_energies(ljatom, simulation_box, dcut);

  double totalke = 0.0;
  for (unsigned int i = 0; i < ljatom.size(); i++)
  {
    ljatom[i].kinetic_energy();
    totalke += ljatom[i].ke;
  }
  output_energy << 0 << "  " << totalke/ljatom.size() << "  " << totalpe/ljatom.size() << "  " << (totalke+totalpe)/ljatom.size() << endl;
  cout << "initial total energy per lj particle is " << (totalke+totalpe)/ljatom.size() << endl;
  
  int movie_frequency = 100;
  
  double average_pe = 0.0;
  double average_ke = 0.0;
  int data_collect_frequency = 1000;
  int samples = 0;
  
  int hit_eqm = 3000; // this is your choice of where you think the system hit equilibrium

  make_movie(0,ljatom,simulation_box.lx,simulation_box.ly,simulation_box.lz,list_propagation);

  // Molecular Dynamics
  cout << "progress..." << endl;
  for (int num = 1; num <= steps; num++)
  {
    // velocity-Verlet
    for (unsigned int i = 0; i < ljatom.size(); i++)
      ljatom[i].update_velocity(delta_t);// update velocity half timestep
    for (unsigned int i = 0; i < ljatom.size(); i++)
      ljatom[i].update_position(delta_t);// update position full timestep
      
    update_forces(ljatom, simulation_box, dcut);	// expensive step
    double totalpe;
    if (num%data_collect_frequency == 0)
      totalpe = update_energies(ljatom, simulation_box, dcut);
    
    for (unsigned int i = 0; i < ljatom.size(); i++)
      ljatom[i].update_velocity(delta_t);// update velocity half timestep
     
    double totalke = 0.0;
    for (unsigned int i = 0; i < ljatom.size(); i++)
    {
      ljatom[i].kinetic_energy();
      totalke += ljatom[i].ke;
    }
      
    // calling a movie function to get a movie of the simulation
    if (num%movie_frequency == 0)
      make_movie(num,ljatom,simulation_box.lx,simulation_box.ly,simulation_box.lz,list_propagation);

      // outputting the energy to make sure simulation can be trusted
    if (num%data_collect_frequency == 0)
        output_energy << num << "  " << totalke/ljatom.size() << "  " << totalpe/ljatom.size() << "  " << (totalke+totalpe)/ljatom.size() << endl;
    
    if (num > hit_eqm && num%data_collect_frequency == 0)
    {
      average_pe = average_pe + totalpe;
      average_ke = average_ke + totalke;
      samples++;
    }
    double fraction_completed = ((num)/(double)steps);
    ProgressBar(fraction_completed);
  }
  
  cout << endl;
  cout << "average pe per particle is " << (average_pe/samples)/ljatom.size() << endl;
  cout << "average ke per particle is " << (average_ke/samples)/ljatom.size() << endl;
  cout << "average T per particle is " << 2*(average_ke/samples)/(3*number_ljatom) << endl;
  
  return 0;
} 
// End of main