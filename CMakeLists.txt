cmake_minimum_required(VERSION 3.14)
project(classifim_bench)

# Example usages:
# 1. Run classifim_bench tests (requires Catch2v3):
#   mkdir build && cd build && cmake -DBUILD_TESTS=ON .. && cmake --build . \
#     && test_classifim_bench/cpp_src/tests
# 2. Build classifim_bench shared library:
#   mkdir build && cd build \
#     && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Debug .. && cmake --build .
# 3. Release build classifim_bench shared library:
#   mkdir build && cd build \
#     && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release .. \
#     && cmake --build .
# 4. Cross-platform release build classifim_bench shared library on Linux targeting Windows:
#   mkdir build && cd build \
#     && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/x86_64-w64-mingw32.cmake .. \
#     && cmake --build .
# 5. You may want to copy that library after compilation:
#   # assuming you are in build directory and ran #2 or #3:
#   cp classifim_bench/cpp_src/libclassifim_bench.so ../classifim_bench/lib/
#   # for #4:
#   cp classifim_bench/cpp_src/libclassifim_bench.dll* ../classifim_bench/lib/
# 6. Cleanup
#   cd .. # if you are in build directory
#   rm -r build

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# You may want to copy that library after compilation:
# cp libfil24_hamiltonian.so ../classifim_bench/lib/

if(BUILD_SHARED_LIBS)
  add_subdirectory(src/classifim_gen/cpp_src)
endif()

if(BUILD_TESTS)
  add_subdirectory(test_classifim_gen/cpp_src)
endif()
