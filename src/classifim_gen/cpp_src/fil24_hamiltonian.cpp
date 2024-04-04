#include "fil24_hamiltonian.h"

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace classifim_gen {
namespace {

// process_translations and process_transforms are utility functions for
// initialization of z_to_vi and orbit_sizes: information about the orbits
// of a dihedral group D(nsites) with 2 * nsites elements on bitstrings of
// length 2 * nsites.
//
// Assumption: before process_transforms for a given vi is called,
//   z_to_vi[z] == std::numeric_limits<std::uint32_t>::max()
//   for all z in orbit # vi and orbit_sizes[vi] = 0.
// Result of calling process_transforms:
//   z_to_vi[z] == vi for all z in orbit # vi and
//   orbit_sizes[vi] == size of orbit # vi (i.e. the number of distinct such z).
//   That gives 1 <= orbit_sizes[vi] <= 2 * nsites.
void process_translations(std::uint32_t z, std::uint32_t vi, const int nsites,
                          std::vector<std::uint32_t> &z_to_vi,
                          std::vector<std::uint8_t> &orbit_sizes) {
  for (int i = 0; i < nsites; ++i) {
    if (z_to_vi[z] == std::numeric_limits<std::uint32_t>::max()) {
      z_to_vi[z] = vi;
      ++orbit_sizes[vi];
    }
    z = rotate_bit_pairs_right(z, nsites);
  }
}

void process_transforms(std::uint32_t z, const std::uint32_t vi,
                        const int nsites, std::vector<std::uint32_t> &z_to_vi,
                        std::vector<std::uint8_t> &orbit_sizes) {
  process_translations(z, vi, nsites, z_to_vi, orbit_sizes);
  const std::uint32_t z_flipped = reverse_bit_pairs(z, nsites);
  process_translations(z_flipped, vi, nsites, z_to_vi, orbit_sizes);
}
} // namespace

Fil24Orbits::Fil24Orbits(int nsites_, bool do_init)
    : nsites{nsites_}, z_to_vi(std::size_t(1) << (2 * nsites),
                               std::numeric_limits<std::uint32_t>::max()) {
  assert(nsites >= 1);
  assert(nsites <= 15);
  if (do_init) {
    init();
  }
}

void Fil24Orbits::init() {
  // The total number of orbits is
  // A032275(nsites) = T(nsites, 4) where T is given by A051137.
  // See https://oeis.org/A032275 and https://oeis.org/A051137.
  std::size_t num_vi_upper_bound =
      (1u << (2 * nsites - 1)) / nsites + 3u * (1u << (nsites - 1));
  vi_to_z.reserve(num_vi_upper_bound);
  orbit_sizes.reserve(num_vi_upper_bound);
  std::uint32_t z_max = std::uint32_t(1) << (2 * nsites);
  std::uint32_t cur_vi = 0;
  for (std::uint32_t z = 0; z < z_max; ++z) {
    if (z_to_vi[z] != std::numeric_limits<std::uint32_t>::max()) {
      continue;
    }
    vi_to_z.push_back(z);
    orbit_sizes.push_back(0);
    process_transforms(z, cur_vi, nsites, z_to_vi, orbit_sizes);
    ++cur_vi;
  }
}

void Fil1DFamily::init() {
  if (!orbits.is_initialized()) {
    orbits.init();
  }
  init_op_x();
  init_op_zs();
}

void Fil1DFamily::init_op_x() {
  assert(orbits.is_initialized());
  assert(op_x.row_ptrs.empty());
  int nsites = orbits.nsites;
  auto &z_to_vi = orbits.get_z_to_vi();
  auto &vi_to_z = orbits.get_vi_to_z();
  auto &orbit_sizes = orbits.get_orbit_sizes();
  int max_vi = vi_to_z.size();
  std::map<std::uint32_t, int> vi_to_counts;
  // No vi_to_counts.reserve(2 * nsites); here
  // since std::map does not support reserve.
  for (int vi_from = 0; vi_from < max_vi; ++vi_from) {
    std::uint32_t z_from = vi_to_z[vi_from];
    for (int j = 0; j < 2 * nsites; ++j) {
      std::uint32_t z_to = z_from ^ (1u << j);
      vi_to_counts[z_to_vi[z_to]] += 1;
    }
    int count_from = orbit_sizes[vi_from];
    // Add a row to op_x:
    op_x.row_ptrs.push_back(op_x.data.size());
    for (auto &vi_to_count : vi_to_counts) {
      int vi_to = vi_to_count.first;
      int count_to = orbit_sizes[vi_to];
      // Count of X terms with the same vi_from and vi_to
      int x_count = vi_to_count.second;
      double x_value = -std::sqrt(double(count_from) / count_to) * x_count;
      op_x.data.push_back(x_value);
      op_x.col_idxs.push_back(vi_to);
    }
    vi_to_counts.clear();
  }
  op_x.row_ptrs.push_back(op_x.data.size());
}

void Fil1DFamily::init_op_zs() {
  assert(orbits.is_initialized());
  assert(op_kterms.empty());
  assert(op_uterms.empty());
  auto &vi_to_z = orbits.get_vi_to_z();
  int num_vis = vi_to_z.size();
  int nsites = get_nsites();
  // TODO:9: we count every edge twice here and
  // (other than edge_dirs) the Hamiltonian family
  // is hardcoded. Make a more flexible version
  // of this class with arbitrary stoquastic interactions
  // symmetric with respect to translations and reflections.
  // For now the fix is to use 1.0 instead of 2.0 below.
  double horizontal_factor = 1.0 / edge_dirs.size();
  op_kterms.reserve(num_vis);
  op_uterms.reserve(num_vis);
  for (int vi = 0; vi < num_vis; ++vi) {
    std::uint32_t z = vi_to_z[vi];
    std::uint32_t zb = z & 0x55555555u;
    std::uint32_t zt = (z >> 1) & 0x55555555u;
    // \sum_{j} Z_{T{j}}:
    double term_zt = nsites - 2 * std::popcount(zt);
    // \sum_{j} Z_{B{j}}:
    double term_zb = nsites - 2 * std::popcount(zb);
    // \sum_{j} Z_{B{j}} Z_{T{j}}:
    double term_zbzt = nsites - 2 * std::popcount(zb ^ zt);
    // \sum_{(j,k) in NN} ((Z_{T{j}} Z_{T{k}} - Z_{B{j}} Z_{B{k}})
    //   * 2 / horizontal_degree).
    double term_horizontal = 0;
    for (int delta : edge_dirs) {
      std::uint32_t zt_shifted = rotate_bit_pairs_right(zt, nsites, delta);
      std::uint32_t zb_shifted = rotate_bit_pairs_right(zb, nsites, delta);
      term_horizontal -= 2 * std::popcount(zt ^ zt_shifted);
      term_horizontal += 2 * std::popcount(zb ^ zb_shifted);
    }
    term_horizontal *= horizontal_factor;
    // Signs: -ZT + ZT * ZT - ZT * ZB - ZB * ZB:
    op_kterms.push_back(term_horizontal - term_zt - term_zbzt);
    // Signs: +ZB
    op_uterms.push_back(term_zb / 2.0);
  }
}
} // namespace classifim_gen
