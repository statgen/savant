/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "grm.hpp"

#include <savvy/reader.hpp>
#include <omp.hpp>

#include <iostream>
#include <getopt.h>

bool load_geno_matrix_dense(savvy::reader& geno_file, std::vector<std::vector<float>>& geno)
{
  std::vector<float> geno_vec;
  savvy::variant var;
  geno.resize(geno_file.samples().size());
  while (geno_file >> var) // && geno.size() < 5000)
  {
    if (var.alts().size() != 1) continue;
    if (!var.get_format("GT", geno_vec))
      return std::cerr << "Error: variant mssing GT field\n", false; // TODO: allow skipping as an option

    std::size_t an = geno_vec.size();
    std::size_t ac = 0;
    for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
    {
      if (std::isnan(*it)) // missing
      {
        --an;
      }
      else
      {
        ac += *it;
      }
    }

    if (ac == 0 || ac == an) continue; // skipping monomorphic //TODO: log first occurence

    float af = static_cast<float>(ac) / an;

    //if (af >= 0.05) continue;

    assert(af > 0.f && af < 1.f);

    savvy::stride_reduce(geno_vec, geno_vec.size() / geno_file.samples().size());


    std::size_t stride = geno_vec.size() / geno_file.samples().size();
    for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
    {
      if (std::isnan(*it))
        *it = 0.; // mean impute missing
      else
        *it = (*it - 2.f * af) / std::sqrt(2.f * af * (1.f - af));
    }

    assert(geno.size() == geno_vec.size());
    for (std::size_t i = 0; i < geno.size(); ++i)
      geno[i].emplace_back(geno_vec[i]);
  }

  if (geno_file.bad())
    return std::cerr << "Error: read failure\n", false;

  return true;
}

int grm_main(int argc, char** argv)
{
  assert(argc > 1);
  savvy::reader geno_file(argv[1]);
  std::vector<std::vector<float>> geno_matrix;
  if (!load_geno_matrix_dense(geno_file, geno_matrix))
    return EXIT_FAILURE;

  std::size_t non_zero_cnt = 0;

  std::ofstream output_file("/dev/stdout", std::ios::binary);
  output_file << "ID1\tID2\tKinship\n";

  omp::internal::thread_pool2 tpool(1);
  std::size_t n_samples = geno_file.samples().size();
  for (std::size_t i = 0; i < n_samples; ++i)
  {
    if (i % 1000 == 0)
      std::cerr << "Processing " << i << "th row ..." << std::endl;
//    for (std::size_t j = 0; j < i; ++j)
//    {
//      if (j > 0)
//        std::cout.put('\t');
//      std::cout.put('0');
//    }

    assert(geno_matrix.size() == n_samples);
    std::vector<float> aggs(n_samples);
    omp::parallel_for_exp(omp::static_schedule(), geno_matrix.begin() + i, geno_matrix.end(),[i, &geno_matrix, &aggs](std::vector<float>& j_vec, const omp::iteration_context& ctx)
    {
      assert(geno_matrix[i].size() == j_vec.size());
      float agg = 0.f;
      for (std::size_t k = 0; k < geno_matrix[i].size(); ++k)
      {
        agg += geno_matrix[i][k] * j_vec[k];
      }

      aggs[i + ctx.index] = agg;

//      std::size_t j = ctx.index;
//      if (std::abs(agg / geno_matrix[i].size()) > 0.05)
//        std::cout << i << "\t" << j << "\t" << (agg / geno_matrix[i].size()) << "\n";

//      if (j > 0)
//        std::cout.put('\t');
//      float coef = agg / geno_matrix.size();
//      std::cout << (std::abs(coef) < 0.05 ? 0.f : coef);
    }, tpool);

    for (std::size_t j = i; j < n_samples; ++j)
    {
      float val = aggs[j] / geno_matrix[i].size();
      if (std::abs(val) > 0.05)
      {
        output_file << geno_file.samples()[i] << "\t" << geno_file.samples()[j] << "\t" << val << "\n";
        ++non_zero_cnt;
      }
    }

    //std::cout << std::endl;
  }

  std::cerr << "Sparsity: " << (double(non_zero_cnt) / double(n_samples * n_samples)) * 100. << "%" << std::endl;

  return EXIT_SUCCESS;
}