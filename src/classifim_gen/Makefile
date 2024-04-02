CXX = g++
CXXFLAGS = -std=c++17 -O3 -fPIC -Iinclude
LDFLAGS = -shared
LIB_NAME = libclassifim_bench.so

SRC_DIR = src
LIB_DIR = lib
INCLUDE_DIR = include
TEST_SRC_DIR = ../test_classifim_bench/cpp_src
PYBIND_DIR = pybind

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

TEST_SOURCES = $(wildcard $(TEST_SRC_DIR)/*.cpp)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)

# https://stackoverflow.com/a/29800774/301644
# Ensure that all variables are defined
# This decreases the likelihood of bugs like accidentally running
# 'rm -rf /*' due to undefined variables.
# Make sure all dangerous targets depend on no-make-warnings
MAKEFLAGS += --warn-undefined-variables
ifndef MAKECMDGOALS
MAKECMDGOALS = all clean print_args
endif

# Fails if the Makefile contains any warnings.
# Run this Makefile with the same goals, but with the -n flag.
# Grep for warnings, and fail if any are found.
no-make-warnings:
	! make -n $(MAKECMDGOALS) 2>&1 >/dev/null | grep warning

all: $(LIB_DIR)/$(LIB_NAME)

$(LIB_DIR)/$(LIB_NAME): $(OBJECTS)
	mkdir -p $(LIB_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run_tests: $(TEST_OBJECTS) $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: test clean print_args no-make-warnings

test: run_tests
	./run_tests

clean: no-make-warnings
	rm $(SRC_DIR)/*.o
	rm $(TEST_SRC_DIR)/*.o
	rm run_tests

