#include "../../src/classifim_gen/cpp_src/fil24_hamiltonian.h"

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

TEST_CASE("nsites=6", "[Fil24Orbits]") {
  auto orbits6 = classifim_gen::Fil24Orbits(6);
  REQUIRE(orbits6.is_initialized());
  auto &z_to_vi = orbits6.get_z_to_vi();
  auto &vi_to_z = orbits6.get_vi_to_z();
  auto &orbit_sizes = orbits6.get_orbit_sizes();
  SECTION("Verify orbits") {
    REQUIRE(z_to_vi.size() == 4096);
    REQUIRE(vi_to_z.size() == 430);
    REQUIRE(orbit_sizes[z_to_vi.at(0)] == 1);
    REQUIRE(orbit_sizes[z_to_vi.at(0b000000000011)] == 6);
    REQUIRE(orbit_sizes[z_to_vi.at(0b000000000110)] == 12);
    for (std::uint32_t z = 0; z < 4096; ++z) {
      std::uint32_t vi = z_to_vi.at(z);
      std::uint32_t z1 = vi_to_z.at(vi);
      std::uint32_t vi1 = z_to_vi.at(z1);
      REQUIRE(vi == vi1);
    }
  }
  SECTION("Fil1DFamily") {
    // For base lattice with 6 sites we have 6 * 2 = 12 qubits.
    auto family12 = classifim_gen::Fil1DFamily(orbits6, {1, 2, 4, 5});
    REQUIRE(!family12.is_initialized());
    family12.init();
    REQUIRE(family12.is_initialized());
    REQUIRE(family12.get_nsites() == 6);
    {
      auto &op_x = family12.get_op_x();
      REQUIRE(op_x.get_nrows() == 430);
      REQUIRE(op_x.col_idxs.size() == op_x.data.size());
      // Z_{T_0} = -1. All other Z_{*_*} are +1:
      std::uint32_t z = 0b100000000000;
      std::uint32_t vi = z_to_vi.at(z);
      std::span<const int> z_col_idxs = op_x.col_idxs_for_row(vi);
      std::span<const double> z_data = op_x.data_for_row(vi);
      // Possible z_to types which can be obtained from z by flipping a single
      // bit (up to symmetry), format:
      // * z_to_type, x_count, vi_to_orbit_size
      // * [00] * 6 ,       1, 1
      // * [00] * 5 + [11], 1, 6
      // * [00] * (4-j) + [01] + [00] * j + [10], 2, 12 (j = 0, 1)
      // * [00] * (4-j) + [10] + [00] * j + [10], 2, 6 (j = 0, 1)
      // * [00] * 2 + [01] + [00] * 2 + [10], 1, 6
      // * [00] * 2 + [10] + [00] * 2 + [10], 1, 3
      REQUIRE(z_col_idxs.size() == 8);
      REQUIRE(z_data.size() == 8);
      REQUIRE(z_col_idxs[0] == z_to_vi.at(0b000000000000));
      REQUIRE(z_data[0] == -std::sqrt(6.0 / 1.0) * 1.0);
      REQUIRE(z_col_idxs[2] == z_to_vi.at(0b000000000110));
      REQUIRE(z_data[2] == -std::sqrt(6.0 / 12.0) * 2.0);
    }

    {
      auto &op_kterms = family12.get_op_kterms();
      auto &op_uterms = family12.get_op_uterms();
      REQUIRE(op_kterms.size() == 430);
      REQUIRE(op_uterms.size() == 430);
      std::uint32_t z = 0b100011000000;
      std::uint32_t vi = z_to_vi.at(z);
      double k_term_expected = -6 + 2 * 2   // -ZT
                               + 6 - 1 * 6  // ZT * ZT
                               - 6 + 2 * 1  // -ZT * ZB
                               - 6 + 1 * 4; // ZB * ZB
      REQUIRE(op_kterms[vi] == k_term_expected);
      double u_term_expected = (6 - 2 * 1) / 2.0; // -ZB
      REQUIRE(op_uterms[vi] == u_term_expected);
    }
  }
}

TEST_CASE("initialization", "[Fil24Orbits]") {
  auto orbits3 = classifim_gen::Fil24Orbits(3, false);
  REQUIRE_FALSE(orbits3.is_initialized());
  orbits3.init();
  REQUIRE(orbits3.is_initialized());
  auto &vi_to_z = orbits3.get_vi_to_z();
  REQUIRE(vi_to_z.size() == 20);
}

TEST_CASE("nsites=12 benchmark", "[!benchmark]") {
  BENCHMARK("orbits12") {
    auto orbits12 = classifim_gen::Fil24Orbits(12);
    return orbits12.get_vi_to_z().size();
  };
}

TEST_CASE("nsites=6", "[ReverseBitPairs]") {
  const std::uint32_t z = 0b000110111111;
  REQUIRE(classifim_gen::reverse_bit_pairs(z, 6) == 0b111111100100);
}

TEST_CASE("nsites=6, rotate_by=1", "[RotateBitPairsRight]") {
  const std::uint32_t z = 0b000110111111;
  REQUIRE(classifim_gen::rotate_bit_pairs_right(z, 6) == 0b110001101111);
}

TEST_CASE("nsites=6, rotate_by=5", "[RotateBitPairsRight]") {
  const std::uint32_t z = 0b000110111111;
  REQUIRE(classifim_gen::rotate_bit_pairs_right(z, 6, 5) == 0b011011111100);
}
