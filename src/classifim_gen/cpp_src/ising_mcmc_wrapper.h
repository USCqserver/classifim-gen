#ifndef INCLUDED_ISING_MCMC_WRAPPER
#define INCLUDED_ISING_MCMC_WRAPPER
#include "ising_mcmc.h"

#ifndef __cplusplus
#error "This header requires C++"
#endif

extern "C" {

classifim_bench::IsingMCMC2D *create_ising_mcmc2D(std::uint64_t seed, int width,
                                                 int height, double beta,
                                                 double h) {
  return new classifim_bench::IsingMCMC2D(seed, width, height, beta, h);
}

void delete_ising_mcmc2D(classifim_bench::IsingMCMC2D *mcmc) { delete mcmc; }

void ising_mcmc2D_get_state(const classifim_bench::IsingMCMC2D *mcmc,
                            std::uint64_t *state, std::size_t size);

void ising_mcmc2D_produce_shifted_state(classifim_bench::IsingMCMC2D *mcmc,
                                        std::uint64_t *state,
                                        std::size_t size);

void ising_mcmc2D_reset(classifim_bench::IsingMCMC2D *mcmc) { mcmc->reset(); }

void ising_mcmc2D_adjust_parameters(classifim_bench::IsingMCMC2D *mcmc,
                                    double beta, double h) {
  mcmc->adjust_parameters(beta, h);
}

void ising_mcmc2D_step(classifim_bench::IsingMCMC2D *mcmc, int n_steps) {
  mcmc->step(n_steps);
}

void ising_mcmc2D_step_flip(classifim_bench::IsingMCMC2D *mcmc) {
  mcmc->step_flip();
}

int ising_mcmc2D_get_width(const classifim_bench::IsingMCMC2D *mcmc) {
  return mcmc->get_width();
}

int ising_mcmc2D_get_height(const classifim_bench::IsingMCMC2D *mcmc) {
  return mcmc->get_height();
}

int ising_mcmc2D_get_beta(const classifim_bench::IsingMCMC2D *mcmc) {
  return mcmc->get_beta();
}

double ising_mcmc2D_get_h(const classifim_bench::IsingMCMC2D *mcmc) {
  return mcmc->get_h();
}

double ising_mcmc2D_get_magnetization(
    const classifim_bench::IsingMCMC2D *mcmc) {
  return mcmc->get_magnetization();
}

int ising_mcmc2D_get_energy0(const classifim_bench::IsingMCMC2D *mcmc) {
  return mcmc->get_energy0();
}

// Step + recording energy + flip
void ising_mcmc2D_step_combined_ef(
    classifim_bench::IsingMCMC2D *mcmc, int n_steps, int n_energies,
    std::int32_t *energies, bool flip);

} // extern "C"
#endif // INCLUDED_ISING_MCMC_WRAPPER
