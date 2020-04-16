# This is a makefile.
# This makes a parallel simulation for different dielectric problem
# Use option -p in CC for profiling with gprof

BIN = ../bin

# nanoHUB flags. 
nanoHUBCC = mpicxx -O3 -g -Wall -fopenmp -std=c++11 -Wunused-variable -Wunknown-pragmas
nanoHUBLFLAG = -lgsl -lgslcblas -lm -L${BOOST_LIBDIR} -lboost_program_options -lboost_mpi -lboost_serialization
nanoHUBCFLAG = -I${BOOST_INCDIR}
nanoHUBOFLAG = -o
# BigRed2 flags. 
BigRed2CC = CC -O3 -g -Wall -fopenmp -std=c++11 -Wunused-variable -Wunknown-pragmas
BigRed2LFLAG = -lgsl -lgslcblas -lm -lboost_program_options -lboost_mpi -lboost_serialization
BigRed2CFLAG = -c
BigRed2OFLAG = -o
# General purpose flags.
CC = mpicxx -O3 -g -Wall -fopenmp -std=c++11 -Wunused-variable -Wunknown-pragmas
LFLAG = -lgsl -lgslcblas -lm -L${BOOST_LIBDIR} -lboost_program_options -lboost_mpi -lboost_serialization
CFLAG = -c
OFLAG = -o

PROG = simulate_many_particle_dynamics
OBJ = main.o simulationbox.o functions.o forces.o energies.o

all: $(PROG)

nanoHUB-install: create-dirs
	@echo "compiling the np_electrostatics code on Nanohub"
	. /etc/environ.sh; use -e -r boost-1.62.0-mpich2-1.3-gnu-4.7.2; make CCF=nanoHUB all
	cp -f $(PROG) $(BIN)
			
install: create-dirs
	@echo "compiling the np_electrostatics code in local computer"
	make CCF=LOCAL all
	cp -f $(PROG) $(BIN)
	
cluster-install: create-dirs
	@echo "compiling the np_electrostatics code on BIGRED 2"
	module swap PrgEnv-cray PrgEnv-gnu && module load boost/1.65.0 && module load gsl; make CCF=BigRed2 all
	cp -f $(PROG) $(BIN)	

create-dirs:
	@echo "Checking and creating needed sub-directories in the $(BIN) directory"
	if ! [ -d $(BIN) ]; then mkdir $(BIN); fi
	if ! [ -d $(BIN)/outfiles ]; then mkdir $(BIN)/outfiles; fi
	@echo "Directory creation is over."

$(PROG) : $(OBJ)
ifeq ($(CCF),BigRed2)	
	$(BigRed2CC) $(BigRed2OFLAG) $(PROG) $(OBJ) $(BigRed2LFLAG)
%.o : %.cpp
	$(BigRed2CC) -c $(BigRed2CFLAG) $< -o $@
else ifeq ($(CCF),nanoHUB)
	$(nanoHUBCC) $(nanoHUBOFLAG) $(PROG) $(OBJ) $(nanoHUBLFLAG)
%.o : %.cpp
	$(nanoHUBCC) -c $(nanoHUBCFLAG) $< -o $@
else
	$(CC) $(OFLAG) $(PROG) $(OBJ) $(LFLAG)
%.o : %.cpp
	$(CC) -c $(CFLAG) $< -o $@	
endif

clean:
	rm -f *.o
	rm -f $(PROG)

dataclean:
	rm -f $(BIN)/outfiles/*.dat $(BIN)/outfiles/*.xyz  $(BIN)/outfiles/*.lammpstrj  $(BIN)/datafiles/*.dat verifiles/*.dat $(BIN)/computedfiles/*.dat
	rm -f $(BIN)/*.log
	rm -f $(BIN)/*.pbs

distclean: clean
	rm -f $(BIN)/$(PROG)