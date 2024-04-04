cmake_minimum_required(VERSION 3.14)
project(classifim_gen_libs)

# Enable C++23 or later
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefix is removed since we are already in that directory:
# classifim_gen/cpp_src/fil24_hamiltonian.cpp
set(SOURCE_FILES
  fil24_hamiltonian.cpp
  fil24_hamiltonian.h
  ising_mcmc.cpp
  ising_mcmc.h
)

set(WRAPPER_FILES
  fil24_hamiltonian_wrapper.h
  fil24_hamiltonian_wrapper.cpp
  ising_mcmc_wrapper.h
  ising_mcmc_wrapper.cpp
)

# Add a shared library target for classifim_gen.so
add_library(classifim_gen SHARED ${SOURCE_FILES} ${WRAPPER_FILES})
target_include_directories(classifim_gen PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(classifim_gen PUBLIC xtensor)
set_target_properties(classifim_gen PROPERTIES OUTPUT_NAME "classifim_gen")
# Or is it classifim_gen.dll?
if(WIN32)
  set_target_properties(classifim_gen PROPERTIES SUFFIX ".dll")
else()
  # Enable warnings:
  target_compile_options(classifim_gen PRIVATE -Wall -Wextra -pedantic)
endif()
