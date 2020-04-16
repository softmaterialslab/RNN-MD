cmake_minimum_required(VERSION 3.9)
project(many_particle)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        energies.cpp
        forces.cpp
        functions.cpp
        functions.h
        main.cpp
        particle.h
        simulationbox.cpp
        simulationbox.h
        vector3d.h
        )

add_executable(many_particle ${SOURCE_FILES})
find_package(GSL REQUIRED)
target_link_libraries(many_particle GSL::gsl GSL::gslcblas)
find_package(Boost COMPONENTS system program_options REQUIRED)
target_link_libraries(many_particle ${Boost_LIBRARIES})
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



