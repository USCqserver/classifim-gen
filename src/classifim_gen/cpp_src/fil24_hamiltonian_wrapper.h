#ifndef INCLUDED_FIL24_HAMILTONIAN_WRAPPER
#define INCLUDED_FIL24_HAMILTONIAN_WRAPPER
#include "fil24_hamiltonian.h"

#ifndef __cplusplus
#error "This header requires C++"
#endif

extern "C" {

classifim_bench::Fil1DFamily *create_fil1d_family(int nsites, int *edge_dirs,
                                                 int num_edge_dirs);
void delete_fil1d_family(classifim_bench::Fil1DFamily *fil1d_family);
const double *
fil1d_family_get_op_kterms(classifim_bench::Fil1DFamily *fil1d_family,
                           int *size);
const double *
fil1d_family_get_op_uterms(classifim_bench::Fil1DFamily *fil1d_family,
                           int *size);
void fil1d_family_get_op_x(classifim_bench::Fil1DFamily *fil1d_family,
                           const int **row_ptrs, int *nrows,
                           const int **col_idxs, const double **data, int *nnz);
const std::uint32_t *
fil1d_family_get_z_to_vi(const classifim_bench::Fil1DFamily *fil1d_family,
                         int *size);
const std::uint32_t *
fil1d_family_get_vi_to_z(const classifim_bench::Fil1DFamily *fil1d_family,
                         int *size);
const std::uint8_t *
fil1d_family_get_orbit_sizes(const classifim_bench::Fil1DFamily *fil1d_family,
                             int *size);

} // extern "C"
#endif // INCLUDED_FIL24_HAMILTONIAN_WRAPPER
