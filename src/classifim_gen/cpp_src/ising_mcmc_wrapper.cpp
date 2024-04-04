#include "ising_mcmc_wrapper.h"

#include "ising_mcmc.h"
#include <cassert>
#include <stdexcept>

void ising_mcmc2D_base_get_state(const classifim_bench::IsingMCMC2DBase *mcmc,
                                 std::uint64_t *state, std::size_t size) {
  mcmc->get_state(std::span<std::uint64_t>{state, size});
}

void ising_mcmc2D_base_produce_shifted_state(classifim_bench::IsingMCMC2D *mcmc,
                                             std::uint64_t *state,
                                             std::size_t size) {
  mcmc->produce_shifted_state(std::span<std::uint64_t>{state, size});
}

void ising_mcmc2D_step_combined_ef(classifim_bench::IsingMCMC2D *mcmc,
                                   int n_steps, int n_energies,
                                   std::int32_t *energies, bool flip) {
  if (0 > n_energies) {
    throw std::invalid_argument("n_energies must be non-negative.");
  }
  if (n_energies > n_steps) {
    throw std::invalid_argument("n_energies must be at most n_steps.");
  }
  // 0 <= n_energies <= n_steps
  int energy_idx = 0;
  int record_counter = 0;
  for (int i = 0; i < n_steps; ++i) {
    mcmc->step();
    record_counter += n_energies; // We add n_energies * n_steps in total here.
    if (record_counter >= n_steps) {
      energies[energy_idx] = static_cast<std::int32_t>(mcmc->get_energy0());
      ++energy_idx;
      record_counter -= n_steps; // This has to run exactly n_energies times.
    }
  }
  assert(record_counter == 0);
  if (flip) {
    mcmc->step_flip();
  }
}
