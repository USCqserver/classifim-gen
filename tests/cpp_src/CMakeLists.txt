cmake_minimum_required(VERSION 3.14)
project(test_classifim_bench)

# Enable C++23 or later
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Catch2 3 REQUIRED)
set(SOURCE_FILES
    ../../src/classifim_gen/cpp_src/fil24_hamiltonian.cpp
    ../../src/classifim_gen/cpp_src/fil24_hamiltonian.h
    ../../src/classifim_gen/cpp_src/ising_mcmc.cpp
    ../../src/classifim_gen/cpp_src/ising_mcmc.h
)

set(TEST_FILES
    test_fil24_hamiltonian.cpp
    test_ising_mcmc.cpp
)

add_executable(tests ${TEST_FILES} ${SOURCE_FILES})

# Link the Catch2::Catch2WithMain target to the tests executable
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
