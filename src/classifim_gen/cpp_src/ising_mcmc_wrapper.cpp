#include "ising_mcmc_wrapper.h"

#include "ising_mcmc.h"
#include <cassert>
#include <stdexcept>

void ising_mcmc2D_base_get_state(const classifim_gen::IsingMCMC2DBase *mcmc,
                                 std::uint64_t *state, std::size_t size) {
  mcmc->get_state(std::span<std::uint64_t>{state, size});
}

void ising_mcmc2D_base_produce_shifted_state(
    classifim_gen::IsingMCMC2DBase *mcmc, std::uint64_t *state,
    std::size_t size) {
  mcmc->produce_shifted_state(std::span<std::uint64_t>{state, size});
}

void ising_mcmc2D_step_combined_ef(classifim_gen::IsingMCMC2D *mcmc,
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

void ising_nnn_mcmc_step_combined_2of(classifim_gen::IsingNNNMCMC *mcmc,
                                      int n_steps, int n_obs_samples,
                                      std::int32_t *observables, bool flip) {
  if (0 > n_obs_samples) {
    throw std::invalid_argument("n_obs_samples must be non-negative.");
  }
  if (n_obs_samples > n_steps) {
    throw std::invalid_argument("n_obs_samples must be at most n_steps.");
  }
  // 0 <= n_obs_samples <= n_steps
  int obs_idx = 0;
  int record_counter = 0;
  for (int i = 0; i < n_steps; ++i) {
    mcmc->step();
    // We add n_obs_samples * n_steps in total here:
    record_counter += n_obs_samples;
    if (record_counter >= n_steps) {
      observables[obs_idx++] = static_cast<std::int32_t>(mcmc->get_total_nn());
      observables[obs_idx++] = static_cast<std::int32_t>(mcmc->get_total_nnn());
      record_counter -= n_steps; // This has to run exactly n_obs_samples times.
    }
  }
  assert(record_counter == 0);
  if (flip) {
    mcmc->step_flip();
  }
}
