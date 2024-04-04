# Scikit-build requires CMake 3.15
cmake_minimum_required(VERSION 3.15)
project(classifim_gen)

# Example usages:
# 1. Run classifim_gen tests (requires Catch2v3):
#   mkdir build && cd build && cmake -DBUILD_TESTS=ON .. && cmake --build . \
#     && tests/cpp_src/tests
# 2. Build classifim_gen shared library:
#   mkdir build && cd build \
#     && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Debug .. && cmake --build .
# 3. Release build classifim_gen shared library:
#   mkdir build && cd build \
#     && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release .. \
#     && cmake --build .
# 4. Cross-platform release build classifim_gen shared library on Linux targeting Windows:
#   mkdir build && cd build \
#     && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/x86_64-w64-mingw32.cmake .. \
#     && cmake --build .
# 5. You may want to copy that library after compilation:
#   # assuming you are in build directory and ran #2 or #3:
#   cp src/classifim_gen/cpp_src/libclassifim_gen.so ../src/classifim_gen/lib/
#   # for #4:
#   cp src/classifim_gen/cpp_src/libclassifim_gen.dll* ../src/classifim_gen/lib/
# 6. Cleanup
#   cd .. # if you are in build directory
#   rm -r build

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

if(BUILD_SHARED_LIBS)
  add_subdirectory(src/classifim_gen/cpp_src)
endif()

if(BUILD_TESTS)
  add_subdirectory(tests/cpp_src)
endif()
