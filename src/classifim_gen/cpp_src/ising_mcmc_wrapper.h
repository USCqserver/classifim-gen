#ifndef INCLUDED_ISING_MCMC_WRAPPER
#define INCLUDED_ISING_MCMC_WRAPPER
#include "ising_mcmc.h"

#ifndef __cplusplus
#error "This header requires C++"
#endif

extern "C" {
// IsingMCMC2DBase:
void delete_ising_mcmc2D_base(classifim_gen::IsingMCMC2DBase *mcmc) {
  delete mcmc;
}

void ising_mcmc2D_base_get_state(const classifim_gen::IsingMCMC2DBase *mcmc,
                                 std::uint64_t *state, std::size_t size);

void ising_mcmc2D_base_set_state(classifim_gen::IsingMCMC2DBase *mcmc,
                                 const std::uint64_t *state, std::size_t size);

void ising_mcmc2D_base_produce_shifted_state(
    classifim_gen::IsingMCMC2DBase *mcmc, std::uint64_t *state,
    std::size_t size);

void ising_mcmc2D_base_reset(classifim_gen::IsingMCMC2DBase *mcmc) {
  mcmc->reset();
}

void ising_mcmc2D_base_step(classifim_gen::IsingMCMC2DBase *mcmc, int n_steps) {
  mcmc->step(n_steps);
}

int ising_mcmc2D_base_get_width(const classifim_gen::IsingMCMC2DBase *mcmc) {
  return mcmc->get_width();
}

int ising_mcmc2D_base_get_height(const classifim_gen::IsingMCMC2DBase *mcmc) {
  return mcmc->get_height();
}

double ising_mcmc2D_base_get_magnetization(
    const classifim_gen::IsingMCMC2DBase *mcmc) {
  return mcmc->get_magnetization();
}

int ising_mcmc2D_base_get_total_nn(const classifim_gen::IsingMCMC2DBase *mcmc) {
  return mcmc->get_total_nn();
}

int ising_mcmc2D_base_get_total_nnn(
    const classifim_gen::IsingMCMC2DBase *mcmc) {
  return mcmc->get_total_nnn();
}

int ising_mcmc2D_base_get_energy0(const classifim_gen::IsingMCMC2DBase *mcmc) {
  return mcmc->get_energy0();
}

int ising_mcmc2D_base_step_parallel_tempering(
    classifim_gen::IsingMCMC2DBase *a, classifim_gen::IsingMCMC2DBase *b) {
  return a->step_parallel_tempering(*b);
}

double
ising_mcmc2D_base_get_beta_energy(const classifim_gen::IsingMCMC2DBase *mcmc) {
  return mcmc->get_beta_energy();
}

// IsingMCMC2D:
classifim_gen::IsingMCMC2D *create_ising_mcmc2D(std::uint64_t seed, int width,
                                                int height, double beta,
                                                double h) {
  return new classifim_gen::IsingMCMC2D(seed, width, height, beta, h);
}

void ising_mcmc2D_adjust_parameters(classifim_gen::IsingMCMC2D *mcmc,
                                    double beta, double h) {
  mcmc->adjust_parameters(beta, h);
}

void ising_mcmc2D_step_flip(classifim_gen::IsingMCMC2D *mcmc) {
  mcmc->step_flip();
}

double ising_mcmc2D_get_beta(const classifim_gen::IsingMCMC2D *mcmc) {
  return mcmc->get_beta();
}

double ising_mcmc2D_get_h(const classifim_gen::IsingMCMC2D *mcmc) {
  return mcmc->get_h();
}

// Step + recording energy + flip
void ising_mcmc2D_step_combined_ef(classifim_gen::IsingMCMC2D *mcmc,
                                   int n_steps, int n_energies,
                                   std::int32_t *energies, bool flip);

// IsingNNNMCMC:
classifim_gen::IsingNNNMCMC *
create_ising_nnn_mcmc(std::uint64_t seed, int width, int height, double beta,
                      double jh, double jv, double jp, double jm, double h) {
  return new classifim_gen::IsingNNNMCMC(seed, width, height, beta, jh, jv, jp,
                                         jm, h);
}

void ising_nnn_mcmc_adjust_parameters(classifim_gen::IsingNNNMCMC *mcmc,
                                      double beta, double jh, double jv,
                                      double jp, double jm, double h) {
  mcmc->adjust_parameters(beta, jh, jv, jp, jm, h);
}

void ising_nnn_mcmc_step_flip(classifim_gen::IsingNNNMCMC *mcmc) {
  mcmc->step_flip();
}

double ising_nnn_mcmc_get_beta(const classifim_gen::IsingNNNMCMC *mcmc) {
  return mcmc->get_beta();
}

double ising_nnn_mcmc_get_jh(const classifim_gen::IsingNNNMCMC *mcmc) {
  return mcmc->get_jh();
}

double ising_nnn_mcmc_get_jv(const classifim_gen::IsingNNNMCMC *mcmc) {
  return mcmc->get_jv();
}

double ising_nnn_mcmc_get_jp(const classifim_gen::IsingNNNMCMC *mcmc) {
  return mcmc->get_jp();
}

double ising_nnn_mcmc_get_jm(const classifim_gen::IsingNNNMCMC *mcmc) {
  return mcmc->get_jm();
}

double ising_nnn_mcmc_get_h(const classifim_gen::IsingNNNMCMC *mcmc) {
  return mcmc->get_h();
}

// Step + recording 2 observables + flip
// observables buffer must be of size 2 * n_obs_samples
void ising_nnn_mcmc_step_combined_2of(classifim_gen::IsingNNNMCMC *mcmc,
                                      int n_steps, int n_obs_samples,
                                      std::int32_t *observables, bool flip);

void ising_nnn_mcmc_step_spins(classifim_gen::IsingNNNMCMC *mcmc) {
  mcmc->step_spins();
}

void ising_nnn_mcmc_step_lines(classifim_gen::IsingNNNMCMC *mcmc) {
  mcmc->step_lines();
}

void ising_nnn_mcmc_step_lines_horizontal(classifim_gen::IsingNNNMCMC *mcmc) {
  mcmc->step_lines_horizontal();
}

void ising_nnn_mcmc_step_lines_vertical(classifim_gen::IsingNNNMCMC *mcmc) {
  mcmc->step_lines_vertical();
}

} // extern "C"
#endif // INCLUDED_ISING_MCMC_WRAPPER
