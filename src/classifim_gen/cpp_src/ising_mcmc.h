#ifndef INCLUDED_ISING_MCMC
#define INCLUDED_ISING_MCMC

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace classifim_bench {
class IsingMCMC2D {
public:
  /**
   * Constructor to initialize the 2D Ising model for MCMC simulation.
   *
   * @param seed: The seed for random number generation, used in the MCMC steps.
   * @param width: The width of the 2D lattice, between 2 and 62 (inclusive).
   * @param height: The height of the 2D lattice, a positive integer >= 2.
   * @param beta: The inverse temperature parameter of the Ising model.
   * @param h: The external magnetic field.
   */
  IsingMCMC2D(std::uint64_t seed, int width, int height, double beta, double h);

  /**
   * Fetches the current state of the Ising model lattice.
   *
   * The state is stored in a 1D array (std::span), where each std::uint64_t
   * represents a single row of 2D lattice sites.
   *
   * @param state: A span to fill with the lattice state of length `height`.
   */
  void get_state(std::span<std::uint64_t> state) const;

  /**
   * Produces a shifted version of the current state of the Ising model lattice.
   *
   * Shifts are random offsets in both directions; the randomness is derived
   * from the same rng as in the MCMC steps, hence the method is not const:
   * consecutive calls to this method will produce states shifted by different
   * offsets.
   *
   * The state is stored in a 1D array (std::span), where each std::uint64_t
   * represents a single row of 2D lattice sites.
   *
   * @param state: A span to fill with the shifted lattice state of length
   * `height`.
   */
  void produce_shifted_state(std::span<std::uint64_t> state);

  /**
   * Resets the Ising model to a random initial state.
   */
  void reset();

  /**
   * Adjusts the parameters of the Ising model without changing the state.
   *
   * @param beta: The new inverse temperature parameter.
   * @param h: The new external magnetic field.
   */
  void adjust_parameters(double beta, double h);

  /**
   * Executes one step of the MCMC simulation.
   */
  void step();

  /**
   * Executes `n_steps` steps of the MCMC simulation.
   * @param n_steps: The number of steps to execute.
   */
  void step(int n_steps) {
    for (int i = 0; i < n_steps; ++i) {
      step();
    }
  }

  /**
   * Executes an alternative step: flipping (or not) the whole lattice.
   */
  void step_flip();

  /**
   * Executes an alternative step: rotating the
   * lattice by random offsets in both directions.
   */
  // void step_rotate();

  int get_width() const { return m_width; }
  int get_height() const { return m_height; }
  double get_beta() const { return m_beta; }
  double get_h() const { return m_h; }
  int get_int_magnetization() const{
    int res = 0;
    for (int i = 1; i <= m_height; ++i) {
      std::uint64_t row = m_state[i] & m_mask;
      // Add popcount of row
      res += __builtin_popcountll(row);
    }
    // This is bit -> Z conversion: 0 -> +1, 1 -> -1:
    int area = m_width * m_height;
    return area - 2 * res;
  }
  // This computes the scaled magnetization:
  double get_magnetization() const {
    int res = get_int_magnetization();
    int area = m_width * m_height;
    return static_cast<double>(res) / area;
  }
  /**
   * Computes the energy of the current state of the Ising model assuming h=0.
   *
   * Total energy can be computed as `energy0 - h * int_magnetization`.
   *
   * @return The energy.
   */
  int get_energy0() const {
    int frustration = 0;
    for (int row_i = 1; row_i <= m_height; ++row_i) {
      std::uint64_t row = m_state[row_i];
      frustration += __builtin_popcountll((row ^ (row >> 1)) & m_mask);
      frustration += __builtin_popcountll((row ^ m_state[row_i - 1]) & m_mask);
    }
    // The total number of couplings is 2 * width * height.
    // Frustrated coupling has energy +1, non-frustrated coupling has energy -1.
    return 2 * frustration - 2 * m_width * m_height;
  }

private:
  // Data:
  const int m_width, m_height;
  double m_beta, m_h;
  // (m_state[i + 1] & m_mask) >> 1 represents the i-th row of the lattice.
  // The first and last rows of m_state are copies of the last and first rows
  // of the lattice, respectively, to implement periodic boundary conditions.
  // Similarly, the first and last bits of each m_state[j]
  // are copies of the last and
  // first bits, respectively.
  std::vector<std::uint64_t> m_state; // Length is m_height + 2
  std::mt19937_64 m_rng;
  std::uint64_t m_mask; // represents inner m_width bits of std::uint64_t
  std::uint64_t m_thresholds[16]; // Precomputed thresholds for MCMC acceptance

  // Helper functions:
  void _precompute_thresholds();
  void _fix_boundary() {
    // Fix the (periodic) boundary conditions.
    for (int i = 1; i <= m_height; ++i) {
      std::uint64_t &state = m_state[i];
      state &= m_mask;
      state |= ((state & 2) << m_width) | ((state >> m_width) & 1);
    }
    m_state[0] = m_state[m_height];
    m_state[m_height + 1] = m_state[1];
  }
  void _update_row(int row_i) {
    std::uint64_t row_up = m_state[row_i - 1];
    std::uint64_t row = m_state[row_i];
    std::uint64_t row_down = m_state[row_i + 1];
    for (int shift = 1; shift < 5; ++shift) {
      constexpr std::uint64_t int4_mask = 0x1111111111111111ULL;
      std::uint64_t cur_row = (row >> shift) & int4_mask;
      std::uint64_t cur_right = (row >> (shift - 1)) & int4_mask;
      std::uint64_t cur_left = (row >> (shift + 1)) & int4_mask;
      std::uint64_t cur_up = (row_up >> shift) & int4_mask;
      std::uint64_t cur_down = (row_down >> shift) & int4_mask;
      // In each int4 block of frustration the bits are organized as follows:
      // block & 1: the current bit of the row.
      // (block >> 1) & 7: total frustration.
      std::uint64_t frustration = (cur_row ^ cur_right) + (cur_row ^ cur_left) +
                                  (cur_row ^ cur_up) + (cur_row ^ cur_down);
      frustration = (frustration << 1) | cur_row;
      // col_i = 4 * i + shift - 1 < m_width, so
      int imax = (m_width - shift + 4) / 4;
      for (int i = 0; i < imax; ++i) {
        // col_i + 1 --- index of the current bit we are processing.
        int col_i_adj = 4 * i + shift;
        std::size_t cur_frustration = frustration & 0xf;
        frustration >>= 4;
        std::uint64_t rnd = m_rng();
        std::uint64_t threshold = m_thresholds[cur_frustration];
        row ^= (static_cast<std::uint64_t>(rnd < threshold) << col_i_adj);
      }
      row &= m_mask;
      row |= ((row & 2) << m_width) | ((row >> m_width) & 1);
    }
    m_state[row_i] = row;
  }
};
} // namespace classifim_bench
#endif // INCLUDED_ISING_MCMC
