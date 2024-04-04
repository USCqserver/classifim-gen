#ifndef INCLUDED_FIL24_HAMILTONIAN
#define INCLUDED_FIL24_HAMILTONIAN

#include <cstdint>
#include <span>
#include <vector>

namespace classifim_gen {
// nsites is the number of sites in the lattice.
// There are 2 qubits per lattice site.
// 1 <= nsites <= 15.
class Fil24Orbits {
public:
  const int nsites;

public:
  Fil24Orbits(int nsites_, bool do_init = true);
  void init();

  const std::vector<std::uint32_t> &get_z_to_vi() const { return z_to_vi; }
  const std::vector<std::uint32_t> &get_vi_to_z() const { return vi_to_z; }
  const std::vector<std::uint8_t> &get_orbit_sizes() const {
    return orbit_sizes;
  }
  bool is_initialized() const { return !vi_to_z.empty(); }

private:
  // Orbits and obs derived from main data. Computed in init().
  std::vector<std::uint32_t> z_to_vi;
  std::vector<std::uint32_t> vi_to_z;
  std::vector<std::uint8_t> orbit_sizes;
};

struct CsrMatrix {
public:
  // Last element of row_ptrs should be col_idxs.size().
  std::vector<int> row_ptrs;
  std::vector<int> col_idxs;
  std::vector<double> data;
  int get_nrows() const { return row_ptrs.size() - 1; }
  std::span<int> col_idxs_for_row(int row) {
    return {col_idxs.data() + row_ptrs[row],
            col_idxs.data() + row_ptrs[row + 1]};
  }
  std::span<const int> col_idxs_for_row(int row) const {
    return {col_idxs.data() + row_ptrs[row],
            col_idxs.data() + row_ptrs[row + 1]};
  }
  std::span<double> data_for_row(int row) {
    return {data.data() + row_ptrs[row], data.data() + row_ptrs[row + 1]};
  }
  std::span<const double> data_for_row(int row) const {
    return {data.data() + row_ptrs[row], data.data() + row_ptrs[row + 1]};
  }
};

class Fil1DFamily {
public:
  Fil24Orbits &orbits; // Not owned
  const std::vector<int> edge_dirs;

public:
  // Caller retains the ownership of orbits_
  Fil1DFamily(Fil24Orbits &orbits_, std::vector<int> edge_dirs_)
      : orbits{orbits_}, edge_dirs{std::move(edge_dirs_)} {}
  void init();
  int get_nsites() const { return orbits.nsites; }
  const CsrMatrix &get_op_x() const { return op_x; }
  const std::vector<double> &get_op_kterms() const { return op_kterms; }
  const std::vector<double> &get_op_uterms() const { return op_uterms; }
  bool is_initialized() const { return !op_x.row_ptrs.empty(); }

private:
  CsrMatrix op_x;
  std::vector<double> op_kterms;
  std::vector<double> op_uterms;

private:
  void init_op_x();
  void init_op_zs();
};

// Reverse bit pairs in a 32-bit integer.
// For example
// 0b10110011000000000000000000000000u -> 0b00000000000000000000000011001101u.
inline std::uint32_t reverse_bit_pairs(std::uint32_t z) {
  z = ((z & 0x33333333u) << 2) | ((z & 0xCCCCCCCCu) >> 2);
  z = ((z & 0x0F0F0F0Fu) << 4) | ((z & 0xF0F0F0F0u) >> 4);
  z = ((z & 0x00FF00FFu) << 8) | ((z & 0xFF00FF00u) >> 8);
  z = ((z & 0x0000FFFFu) << 16) | ((z & 0xFFFF0000u) >> 16);
  return z;
}

inline std::uint32_t reverse_bit_pairs(std::uint32_t z, const int nsites) {
  return reverse_bit_pairs(z) >> (32 - 2 * nsites);
}

inline std::uint32_t rotate_bit_pairs_right(const std::uint32_t z,
                                            const int nsites) {
  return (z >> 2u) | ((z & 3u) << (2 * (nsites - 1)));
}

inline std::uint32_t rotate_bit_pairs_right(const std::uint32_t z,
                                            const int nsites,
                                            const int rotate_by) {
  const int rshift_bits = 2 * rotate_by;
  const int lshift_bits = 2 * (nsites - rotate_by);
  const std::uint32_t mask = (1u << rshift_bits) - 1u;
  return (z >> rshift_bits) | ((z & mask) << lshift_bits);
}
} // namespace classifim_gen
#endif // INCLUDED_FIL24_HAMILTONIAN
