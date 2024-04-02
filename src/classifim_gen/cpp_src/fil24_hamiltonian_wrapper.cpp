#include "fil24_hamiltonian_wrapper.h"

#include <map>
#include <stdexcept>

namespace {
classifim_bench::Fil24Orbits *get_orbits(int nsites) {
  static std::map<int, classifim_bench::Fil24Orbits> orbits_map;
  auto it = orbits_map.find(nsites);
  if (it == orbits_map.end()) {
    it = orbits_map
             .emplace(std::piecewise_construct, std::forward_as_tuple(nsites),
                      std::forward_as_tuple(nsites, true))
             .first;
  }
  return &it->second;
}
} // namespace

classifim_bench::Fil1DFamily *create_fil1d_family(int nsites, int *edge_dirs,
                                                 int num_edge_dirs) {
  classifim_bench::Fil24Orbits *orbits = get_orbits(nsites);
  classifim_bench::Fil1DFamily *res = new classifim_bench::Fil1DFamily(
      *orbits, std::vector<int>(edge_dirs, edge_dirs + num_edge_dirs));
  try {
    res->init();
    return res;
  } catch (...) {
    delete res;
    throw;
  }
}

void delete_fil1d_family(classifim_bench::Fil1DFamily *fil1d_family) {
  delete fil1d_family;
}

const double *
fil1d_family_get_op_kterms(classifim_bench::Fil1DFamily *fil1d_family,
                           int *size) {
  auto &op_kterms = fil1d_family->get_op_kterms();
  *size = op_kterms.size();
  return op_kterms.data();
}

const double *
fil1d_family_get_op_uterms(classifim_bench::Fil1DFamily *fil1d_family,
                           int *size) {
  auto &op_uterms = fil1d_family->get_op_uterms();
  *size = op_uterms.size();
  return op_uterms.data();
}

void fil1d_family_get_op_x(classifim_bench::Fil1DFamily *fil1d_family,
                           const int **row_ptrs, int *nrows,
                           const int **col_idxs, const double **data,
                           int *nnz) {
  auto &op_x = fil1d_family->get_op_x();
  if (op_x.row_ptrs.back() != op_x.col_idxs.size()) {
    throw std::runtime_error("row_ptrs.back() != col_idxs.size()");
  }
  if (op_x.row_ptrs.back() != op_x.data.size()) {
    throw std::runtime_error("row_ptrs.back() != data.size()");
  }
  *row_ptrs = op_x.row_ptrs.data();
  *nrows = op_x.row_ptrs.size() - 1;
  *col_idxs = op_x.col_idxs.data();
  *data = op_x.data.data();
  *nnz = op_x.data.size();
}

const std::uint32_t *
fil1d_family_get_z_to_vi(const classifim_bench::Fil1DFamily *fil1d_family,
                         int *size) {
  const std::vector<std::uint32_t> &z_to_vi =
      fil1d_family->orbits.get_z_to_vi();
  *size = z_to_vi.size();
  return z_to_vi.data();
}

const std::uint32_t *
fil1d_family_get_vi_to_z(const classifim_bench::Fil1DFamily *fil1d_family,
                         int *size) {
  const std::vector<std::uint32_t> &vi_to_z =
      fil1d_family->orbits.get_vi_to_z();
  *size = vi_to_z.size();
  return vi_to_z.data();
}

const std::uint8_t *
fil1d_family_get_orbit_sizes(const classifim_bench::Fil1DFamily *fil1d_family,
                             int *size) {
  const std::vector<std::uint8_t> &orbit_sizes =
      fil1d_family->orbits.get_orbit_sizes();
  *size = orbit_sizes.size();
  return orbit_sizes.data();
}

// Implement more functions here
