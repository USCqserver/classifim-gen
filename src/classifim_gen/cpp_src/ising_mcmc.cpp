#include "ising_mcmc.h"

#include <cassert>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace classifim_bench {
namespace {
// Helper functions
void _basic_step_flip(int width, int height, double beta_h,
                      std::mt19937_64 &rng, std::uint64_t *state) {
  // Decide whether to flip the entire lattice or not.
  double p_accept;
  if (beta_h == 0) {
    p_accept = 0.5;
  } else {
    // Calculate total magnetization
    int total_magnetization = 0;
    for (int row_i = 1; row_i <= height; ++row_i) {
      total_magnetization +=
          __builtin_popcountll(state[row_i] & ((1ull << width) - 1));
    }

    // Calculate energy change due to flipping
    double delta_energy =
        2 * beta_h * (2 * total_magnetization - width * height);

    // Calculate probability of flipping
    double p_ratio = exp(-delta_energy);
    // The choice of acceptance probability function $f$
    // has to satisfy $f(r) / f(1/r) = r$ and $f(r) \leq 1.0$.
    // * Metropolis: $f(r) = min(1, r)$.
    // * Barker: $f(r) = r / (1 + r)$.
    p_accept = p_ratio / (1 + p_ratio);
  }

  // Generate a random number between 0 and 1
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double rnd = distribution(rng);

  // Flip the lattice if the random number is less than the flip probability
  if (rnd < p_accept) {
    std::uint64_t mask2 = (1ull << (width + 2)) - 1;
    for (int row_i = height + 1; row_i >= 0; --row_i) {
      state[row_i] = ~state[row_i] & mask2;
    }
  }
}
} // namespace

// class IsingMCMC2DBase:
IsingMCMC2DBase::IsingMCMC2DBase(std::uint64_t seed, int width, int height)
    : m_width(width), m_height(height), m_rng(seed),
      m_mask((1ull << (width + 1)) - 2) {
  // Validate the width and height
  if (width < 2 || width > 62) {
    throw std::invalid_argument("width must be between 2 and 62 (inclusive).");
  }
  if (width % 2 != 0) {
    throw std::invalid_argument(
        "Currently only even integers are supported in _update_row.");
  }

  if (height < 2) {
    throw std::invalid_argument("height must be a positive integer >= 2.");
  }

  m_state.resize(m_height + 2);
  reset();
}

void IsingMCMC2DBase::get_state(std::span<std::uint64_t> state) const {
  if (state.size() != static_cast<std::size_t>(m_height)) {
    throw std::invalid_argument("state must have length equal to height.");
  }

  for (int i = 0; i < m_height; ++i) {
    state[i] = (m_state[i + 1] & m_mask) >> 1;
  }
}

void IsingMCMC2DBase::produce_shifted_state(std::span<std::uint64_t> state) {
  if (state.size() != static_cast<std::size_t>(m_height)) {
    throw std::invalid_argument("state must have length equal to height.");
  }

  // Generate random offsets for rows and columns
  std::uniform_int_distribution<int> dist_height(0, m_height - 1);
  std::uniform_int_distribution<int> dist_width(0, m_width - 1);
  int row_offset = dist_height(m_rng);
  int col_offset = dist_width(m_rng);
  int col_offset2 = m_width - col_offset;
  std::uint64_t mask = (1ull << m_width) - 1ull;

  for (int i = 0; i < m_height; ++i) {
    int in_rowi = (i + row_offset) % m_height;
    std::uint64_t row = m_state[in_rowi] & mask;
    row = ((row >> col_offset) | (row << col_offset2)) & mask;

    state[i] = row;
  }
}

void IsingMCMC2DBase::reset() {
  // Initialize the 2D lattice to a random state.
  // Bit value 0 means Z = +1, bit value 1 means Z = -1.
  std::uniform_int_distribution<std::uint64_t> dist(0, (1ull << m_width) - 1);
  for (int i = 1; i <= m_height; ++i) {
    m_state[i] = dist(m_rng) << 1;
  }
  _fix_boundary();
}

// class IsingMCMC2D:
void IsingMCMC2D::_precompute_thresholds() {
  constexpr std::uint64_t m_rng_max = std::mt19937_64::max();
  // m_rng.max(); TODO:9: figure out why we can't refer to type of m_rng here:
  // error: constexpr variable 'm_rng_max' must be initialized by a constant
  //   expression
  // note: implicit use of 'this' pointer is only allowed within the
  //   evaluation of a call to a 'constexpr' member function
  // decltype(m_rng)::max(); (?)

  // Precompute the thesholds for the MCMC acceptance.
  // m_threshold[2 * j + b] / 2^64 represents the probability
  // of accepting a spin flip from state b to state 1 - b.
  for (int j = 0; j <= 4; ++j) {
    for (int b = 0; b <= 1; ++b) {
      constexpr double J = 1.0;
      // Energy E = J * E_J + h * E_h
      // We had j frustrated couplings (energy +J)
      // and (4 - j) non-frustrated couplings (energy -J).
      // E_J_before = 2 * j - 4
      // E_J_after = 4 - 2 * j
      // Bit b corresponds to z = 1 - 2 * b
      // E_h_before = -z = 2 * b - 1
      // E_h_after = z = 1 - 2 * b
      // delta_E = E_after - E_before = J * (8 - 4 * j) + h * (2 - 4 * b)
      double delta_e = J * (8 - 4 * j) + m_h * (2 - 4 * b);
      double p_accept = std::exp(-m_beta * delta_e);
      if (p_accept >= 1.0) {
        m_thresholds[2 * j + b] = m_rng_max;
      } else {
        m_thresholds[2 * j + b] = static_cast<std::uint64_t>(
            p_accept * static_cast<double>(m_rng_max));
      }
    }
  }
}

void IsingMCMC2D::adjust_parameters(double beta, double h) {
  m_beta = beta;
  m_h = h;
  _precompute_thresholds();
}

void IsingMCMC2D::step() {
  // Perform one step of the MCMC simulation.
  // The algorithm is the Metropolis-Hastings algorithm.
  _update_row(1);
  m_state[m_height + 1] = m_state[1];
  for (int row_i = 2; row_i <= m_height; ++row_i) {
    _update_row(row_i);
  }
  m_state[0] = m_state[m_height];
}

void IsingMCMC2D::step_flip() {
  _basic_step_flip(m_width, m_height, m_beta * m_h, m_rng, m_state.data());
}

// class IsingNNNMCMC:
void IsingNNNMCMC::_precompute_thresholds() {
  constexpr std::uint64_t m_rng_max = decltype(m_rng)::max();
  double jh = m_beta * m_jh;
  double jv = m_beta * m_jv;
  double jp = m_beta * m_jp;
  double jm = m_beta * m_jm;
  double h = m_beta * m_h;
  // Consider a single coupling between za = (-1)**xa and zb = (-1)**xb and the
  // coupling strength J. Its energy is
  // -J * za * zb = J * (1 - 2 * (xa^xb)).
  // Magnetic field can be considered as a coupling
  // with a fixed spin zb = 1 = (-1)**0.
  // Energy before the flip:
  // E0 = \sum_{dir} J_{dir} * (1 - 2 * n_{dir}),
  // where n_{dir} is the number of frustrated couplings.
  // Energy after the flip is E1 = -E0.
  // delta_E = E1 - E0 = -2 * E0 = \sum_{dir} J_{dir} * (4 * n_{dir} - 2).
  for (int n_jh = 0; n_jh <= 2; ++n_jh) {
    double delta_e0 = jh * (4 * n_jh - 2);
    int idx0 = n_jh << 7;
    for (int n_jv = 0; n_jv <= 2; ++n_jv) {
      double delta_e1 = delta_e0 + jv * (4 * n_jv - 2);
      int idx1 = idx0 + (n_jv << 5);
      for (int n_jp = 0; n_jp <= 2; ++n_jp) {
        double delta_e2 = delta_e1 + jp * (4 * n_jp - 2);
        int idx2 = idx1 + (n_jp << 3);
        for (int n_jm = 0; n_jm <= 2; ++n_jm) {
          double delta_e3 = delta_e2 + jm * (4 * n_jm - 2);
          int idx3 = idx2 + (n_jm << 1);
          for (int n_h = 0; n_h <= 1; ++n_h) {
            double delta_e = delta_e3 + h * (2 * n_h - 1);
            int idx = idx3 + n_h;
            double p_accept = std::exp(-delta_e);
            if (p_accept >= 1.0) {
              m_thresholds[idx] = m_rng_max;
            } else {
              m_thresholds[idx] = static_cast<std::uint64_t>(
                  p_accept * static_cast<double>(m_rng_max));
            }
          }
        }
      }
    }
  }
}

void IsingNNNMCMC::adjust_parameters(double beta, double jh, double jv,
                                     double jp, double jm, double h) {
  m_beta = beta;
  m_jh = jh;
  m_jv = jv;
  m_jp = jp;
  m_jm = jm;
  m_h = h;
  _precompute_thresholds();
}

void IsingNNNMCMC::step() {
  // Perform one step of the MCMC simulation.
  // The algorithm is the Metropolis-Hastings algorithm.
  _update_row(1);
  m_state[m_height + 1] = m_state[1];
  for (int row_i = 2; row_i <= m_height; ++row_i) {
    _update_row(row_i);
  }
  m_state[0] = m_state[m_height];
}

void IsingNNNMCMC::step_flip() {
  _basic_step_flip(m_width, m_height, m_beta * m_h, m_rng, m_state.data());
}

} // namespace classifim_bench
