#ifndef INCLUDED_ISING_MCMC_WRAPPER
#define INCLUDED_ISING_MCMC_WRAPPER
#include "ising_mcmc.h"

#ifndef __cplusplus
#error "This header requires C++"
#endif

extern "C" {
// IsingMCMC2DBase:
void delete_ising_mcmc2D_base(classifim_bench::IsingMCMC2DBase *mcmc) {
  delete mcmc;
}

void ising_mcmc2D_base_get_state(const classifim_bench::IsingMCMC2DBase *mcmc,
                                 std::uint64_t *state, std::size_t size);

void ising_mcmc2D_base_produce_shifted_state(
    classifim_bench::IsingMCMC2DBase *mcmc, std::uint64_t *state,
    std::size_t size);

void ising_mcmc2D_base_reset(classifim_bench::IsingMCMC2DBase *mcmc) {
  mcmc->reset();
}

void ising_mcmc2D_base_step(classifim_bench::IsingMCMC2DBase *mcmc,
                            int n_steps) {
  mcmc->step(n_steps);
}

int ising_mcmc2D_base_get_width(const classifim_bench::IsingMCMC2DBase *mcmc) {
  return mcmc->get_width();
}

int ising_mcmc2D_base_get_height(const classifim_bench::IsingMCMC2DBase *mcmc) {
  return mcmc->get_height();
}

double ising_mcmc2D_base_get_magnetization(
    const classifim_bench::IsingMCMC2DBase *mcmc) {
  return mcmc->get_magnetization();
}

int ising_mcmc2D_base_get_energy0(
    const classifim_bench::IsingMCMC2DBase *mcmc) {
  return mcmc->get_energy0();
}

// IsingMCMC2D:
classifim_bench::IsingMCMC2D *create_ising_mcmc2D(std::uint64_t seed, int width,
                                                  int height, double beta,
                                                  double h) {
  return new classifim_bench::IsingMCMC2D(seed, width, height, beta, h);
}

void ising_mcmc2D_adjust_parameters(classifim_bench::IsingMCMC2D *mcmc,
                                    double beta, double h) {
  mcmc->adjust_parameters(beta, h);
}

void ising_mcmc2D_step_flip(classifim_bench::IsingMCMC2D *mcmc) {
  mcmc->step_flip();
}

int ising_mcmc2D_get_beta(const classifim_bench::IsingMCMC2D *mcmc) {
  return mcmc->get_beta();
}

double ising_mcmc2D_get_h(const classifim_bench::IsingMCMC2D *mcmc) {
  return mcmc->get_h();
}

// Step + recording energy + flip
void ising_mcmc2D_step_combined_ef(classifim_bench::IsingMCMC2D *mcmc,
                                   int n_steps, int n_energies,
                                   std::int32_t *energies, bool flip);

// IsingNNNMCMC:
classifim_bench::IsingNNNMCMC *
create_ising_nnn_mcmc(std::uint64_t seed, int width, int height, double beta,
                      double jh, double jv, double jp, double jm, double h) {
  return new classifim_bench::IsingNNNMCMC(seed, width, height, beta, jh, jv,
                                           jp, jm, h);
}

void ising_nnn_mcmc_adjust_parameters(classifim_bench::IsingNNNMCMC *mcmc,
                                      double beta, double jh, double jv,
                                      double jp, double jm, double h) {
  mcmc->adjust_parameters(beta, jh, jv, jp, jm, h);
}

void ising_nnn_mcmc_step_flip(classifim_bench::IsingNNNMCMC *mcmc) {
  mcmc->step_flip();
}

int ising_nnn_mcmc_get_beta(const classifim_bench::IsingNNNMCMC *mcmc) {
  return mcmc->get_beta();
}

double ising_nnn_mcmc_get_h(const classifim_bench::IsingNNNMCMC *mcmc) {
  return mcmc->get_h();
}

} // extern "C"
#endif // INCLUDED_ISING_MCMC_WRAPPER
