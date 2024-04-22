#include "../../src/classifim_gen/cpp_src/ising_mcmc.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

TEST_CASE("IsingNNNMCMC::step_lines_vertical") {
  classifim_gen::IsingNNNMCMC mcmc(1, 20, 20, -1.0, 0.0, 0.0, 0.0, 0.0, 3.0);
  std::vector<std::uint64_t> test_state(20, 0xfffffULL);
  mcmc.set_state(test_state);
  REQUIRE(mcmc.get_beta_energy() == -1200.0);
  mcmc.step_lines_vertical();
  REQUIRE(mcmc.get_beta_energy() == -1200.0);
}
