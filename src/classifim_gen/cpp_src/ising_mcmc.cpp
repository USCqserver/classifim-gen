#include "ising_mcmc.h"

#include <cassert>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace classifim_gen {
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

/**
 * Update a single bit of 1D chain.
 *
 * For PBC the chain's Hamiltonian is
 * H = -sum_{j=1}^{width} couplings_1d[j-1] Z[j] Z[j-1]
 *     - sum_{j=1}^{width} biases_1d[j-1] Z[j].
 * Here Z[j] = 1 - 2 * ((flips_1d >> j) & 1).
 * PBC are Z[0] = Z[width], Z[width + 1] = Z[1].
 *
 * Other boundary conditions also work if `j` is not on the boundary.
 * This function flips a single bit Z[j]: it does not restore the boundary
 * conditions (this should be done in the caller).
 *
 * @param couplings_1d Couplings of the chain (length = width + 1).
 * @param biases_1d Biases of the chain (length = width).
 * @param j Index of the bit to update. 1 <= j <= width.
 * @param rng Random number generator.
 * @param flips_1d Current state of the chain.
 **/
void _couplings_1d_flip_one(const std::vector<double> &couplings_1d,
                            const std::vector<double> &biases_1d, int j,
                            std::mt19937_64 &rng, std::uint64_t &flips_1d) {
  int cur_bits = static_cast<int>(flips_1d >> (j - 1));
  int z0 = 1 - 2 * (cur_bits & 1);
  int z1 = 1 - (cur_bits & 2); // bit to flip
  int z2 = 1 - ((cur_bits >> 1) & 2);
  double delta_e =
      -2 * z1 *
      (z0 * couplings_1d[j - 1] + biases_1d[j - 1] + z2 * couplings_1d[j]);
  // Use Baker's acceptance probability:
  double p_accept = 1.0 / (1.0 + std::exp(-delta_e));
  std::bernoulli_distribution dist_accept(p_accept);
  flips_1d ^= static_cast<std::uint64_t>(dist_accept(rng)) << j;
}

/**
 * Compute ceil(log2(x)) for x > 0.
 *
 **/
int log2ub(std::uint64_t x) {
  int res;
  std::uint64_t xmax = 1;
  for (res = 0; res < 64 && x > xmax; ++res) {
    xmax <<= 1;
  }
  return res;
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

void IsingMCMC2DBase::set_state(std::span<const std::uint64_t> state) {
  if (state.size() != static_cast<std::size_t>(m_height)) {
    throw std::invalid_argument("state must have length equal to height.");
  }

  for (int i = 0; i < m_height; ++i) {
    m_state[i + 1] = (state[i] << 1) & m_mask;
  }
  _fix_boundary();
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
double IsingMCMC2D::_get_beta_energy_of(
    const std::vector<std::uint64_t> &state) const {
  return m_beta * (-m_h * _get_int_magnetization_of(state) +
                   1.0 * _get_energy0_of(state));
}

void IsingMCMC2D::_precompute_thresholds() {
  constexpr std::uint64_t m_rng_max = decltype(m_rng)::max();

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

double IsingNNNMCMC::_get_beta_energy_of(
    const std::vector<std::uint64_t> &state) const {
  int n_jh = 0, n_jv = 0, n_jp = 0, n_jm = 0, n_h = 0;
  for (int row_i = 1; row_i <= m_height; ++row_i) {
    std::uint64_t row0 = state[row_i - 1];
    std::uint64_t row1 = state[row_i];
    n_jh += __builtin_popcountll((row1 ^ row1 >> 1) & m_mask);
    n_jv += __builtin_popcountll((row1 ^ row0) & m_mask);
    // jp is -- direction, i.e. bit 0 of row0 should xor with bit 1 of row1:
    n_jp += __builtin_popcountll((row1 ^ row0 << 1) & m_mask);
    n_jm += __builtin_popcountll((row1 ^ row0 >> 1) & m_mask);
    n_h += __builtin_popcountll(row1 & m_mask);
  }
  int area = m_height * m_width;
  // -Z_i Z_j = 2 * xor(x_i, x_j) - 1
  // The number of couplings of each type is `area` (one coupling per site),
  // hence for each coupling type
  // -sum(Z_i Z_j) = 2 * n_{dir} - area
  return m_beta * (m_jh * (2 * n_jh - area) + m_jv * (2 * n_jv - area) +
                   m_jp * (2 * n_jp - area) + m_jm * (2 * n_jm - area) +
                   m_h * (2 * n_h - area));
}

IsingNNNMCMC::IsingNNNMCMC(std::uint64_t seed, int width, int height,
                           double beta, double jh, double jv, double jp,
                           double jm, double h)
    : IsingMCMC2DBase(seed, width, height), m_beta(beta), m_jh(jh), m_jv(jv),
      m_jp(jp), m_jm(jm), m_h(h) {
  if (width % 9 == 1) {
    throw std::invalid_argument(
        "Cases with width % 9 == 1 are not supported in _update_row. Sorry.");
  }
  _precompute_thresholds();
}

void IsingNNNMCMC::_precompute_thresholds() {
  constexpr std::uint64_t m_rng_max = decltype(m_rng)::max();
  double jh = m_beta * m_jh;
  double jv = m_beta * m_jv;
  double jp = m_beta * m_jp;
  double jm = m_beta * m_jm;
  double h = m_beta * m_h;
  // Consider a single coupling between za = (-1)**xa and zb = (-1)**xb and the
  // coupling strength J. Its energy is
  // -J * za * zb = J * (2 * (xa^xb) - 1).
  // Magnetic field can be considered as a coupling
  // with a fixed spin zb = 1 = (-1)**0.
  // Energy before the flip:
  // E0 = \sum_{dir} J_{dir} * (2 * n_{dir} - n_{dir,max}),
  // where n_{dir} is the number of frustrated couplings.
  // Energy after the flip is E1 = -E0.
  // delta_E = E0 - E1 = 2 * E0 =
  //   \sum_{dir} J_{dir} * (4 * n_{dir} - 2 * n_{dir,max}).
  for (int n_jh = 0; n_jh <= 2; ++n_jh) {
    double delta_e0 = jh * (4 * n_jh - 4);
    int idx0 = n_jh << 7;
    for (int n_jv = 0; n_jv <= 2; ++n_jv) {
      double delta_e1 = delta_e0 + jv * (4 * n_jv - 4);
      int idx1 = idx0 + (n_jv << 5);
      for (int n_jp = 0; n_jp <= 2; ++n_jp) {
        double delta_e2 = delta_e1 + jp * (4 * n_jp - 4);
        int idx2 = idx1 + (n_jp << 3);
        for (int n_jm = 0; n_jm <= 2; ++n_jm) {
          double delta_e3 = delta_e2 + jm * (4 * n_jm - 4);
          int idx3 = idx2 + (n_jm << 1);
          for (int n_h = 0; n_h <= 1; ++n_h) {
            // delta_e is beta * (E_cur - E_proposed)
            double delta_e = delta_e3 + h * (4 * n_h - 2);
            int idx = idx3 + n_h;
            if (delta_e >= 0) {
              m_thresholds[idx] = m_rng_max;
            } else {
              double p_accept = std::exp(delta_e);
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

void IsingNNNMCMC::step_spins() {
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

void IsingNNNMCMC::step_lines_horizontal() {
  _step_line_horizontal(1);
  m_state[m_height + 1] = m_state[1];
  for (int rowi = 2; rowi <= m_height; ++rowi) {
    _step_line_horizontal(rowi);
  }
  m_state[0] = m_state[m_height];
}
void IsingNNNMCMC::step_lines_vertical() {
  // In C++ vector<double> is initalized with 0.0 values by default:
  std::vector<double> couplings_1d(static_cast<std::size_t>(m_width + 1));
  std::vector<double> biases_1d(static_cast<std::size_t>(m_width));
  const int n_sum_bits = log2ub(m_height + 1);
  const int n_sums = 64 / n_sum_bits;
  std::uint64_t sum_item_mask = (1ULL << n_sum_bits) - 1ULL;
  std::uint64_t sum_mask =
      ((1ULL << (n_sum_bits * n_sums)) - 1ULL) / sum_item_mask;
  for (int shift = 0; shift < n_sum_bits; ++shift) {
    std::uint64_t sum_jh = 0;
    std::uint64_t sum_jp = 0;
    std::uint64_t sum_jm = 0;
    std::uint64_t sum_h = 0;
    for (int rowi = 1; rowi <= m_height; ++rowi) {
      std::uint64_t row0 = m_state[rowi - 1] >> shift;
      std::uint64_t row1 = m_state[rowi] >> shift;
      std::uint64_t row2 = m_state[rowi + 1] >> shift;
      std::uint64_t row = row1 >> 1;
      sum_jh += (row1 ^ row) & sum_mask;
      sum_jp += (row0 ^ row) & sum_mask;
      sum_jm += (row2 ^ row) & sum_mask;
      sum_h += row & sum_mask;
    }
    for (int j = shift; j < m_width; j += n_sum_bits) {
      int sum_shift = j - shift;
      // Non-frustrated couplings are positive: e.g. if the grid are
      // all 0 (Z = 1), and all couplings m_jh, m_jp, m_jm, and m_h are
      // positive, and m_beta > 0, then
      // couplings_1d[j] > 0 and biases_1d[j] > 0.
      couplings_1d[j] =
          m_beta * m_jh *
              (m_height -
               2 * static_cast<int>((sum_jh >> sum_shift) & sum_item_mask)) +
          m_beta * m_jp *
              (m_height -
               2 * static_cast<int>((sum_jp >> sum_shift) & sum_item_mask)) +
          m_beta * m_jm *
              (m_height -
               2 * static_cast<int>((sum_jm >> sum_shift) & sum_item_mask));
      biases_1d[j] = m_beta * m_h *
                     (m_height - 2 * static_cast<int>((sum_h >> sum_shift) &
                                                      sum_item_mask));
    }
  }
  couplings_1d[m_width] = couplings_1d[0];
  std::uint64_t flips_1d = 0;
  // Now we have 1D chain with
  // H = -sum_j couplings_1d[j] Z[j] Z[j-1] - sum_j biases_1d[j] Z[j];
  // Here Z[j] = 1 - 2 * ((flips_1d >> (j + 1)) & 1)
  _couplings_1d_flip_one(couplings_1d, biases_1d, 1, m_rng, flips_1d);
  flips_1d ^= flips_1d << m_width;
  for (int j = 2; j <= m_width; ++j) {
    _couplings_1d_flip_one(couplings_1d, biases_1d, j, m_rng, flips_1d);
  }
  flips_1d ^= (flips_1d >> m_width) & 1;
  for (int rowi = m_height + 1; rowi >= 0; --rowi) {
    m_state[rowi] ^= flips_1d;
  }
}

} // namespace classifim_gen
