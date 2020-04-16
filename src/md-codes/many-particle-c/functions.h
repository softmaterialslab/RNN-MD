#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

#include<iostream>
#include<iomanip>
#include<fstream>
#include<vector>
#include<cmath>
#include "vector3d.h"
#include "particle.h"

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

using namespace std;

// general functions
// -----------------

// overloaded << to print 3d vectors
ostream& operator<<(ostream&, VECTOR3D);

// make movie
void make_movie(int, vector<PARTICLE>&, double, double, double, ofstream&);

// initialize particle velocities
void initialize_particle_velocities(vector<PARTICLE>&);

// display progress of the simulation (code from the internet)
void ProgressBar(double);

#endif
