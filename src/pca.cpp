#include "pca.hpp"

#include <savvy/reader.hpp>
#include <xtensor/xarray.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <fstream>

template <typename T>
auto compute_eigen_dense(const std::vector<std::vector<T>>& geno_matrix, std::size_t num_pcs = 10, std::size_t n_simulations = 128)
{
  using namespace xt;
  using namespace xt::linalg;

  assert(geno_matrix.size());

  xarray<double> Q = random::rand<double>({geno_matrix[0].size(), num_pcs});
  xarray<double> R;
  std::tie(Q, R) = qr(Q);
  xarray<double> Q_prev = Q;

  for (std::size_t i = 0; i < n_simulations; ++i)
  {
    xarray<double> S = xt::zeros<double>(Q.shape());

    for (std::size_t j = 0; j < geno_matrix.size(); ++j)
    {
      auto x = adapt(geno_matrix[j], {std::size_t(1), geno_matrix[j].size()});
      auto x_t = adapt(geno_matrix[j], {geno_matrix[j].size(), std::size_t(1)});
      S += dot(x_t, dot(x, Q));
      //std::cerr << S << std::endl;
    }

    std::tie(Q, R) = qr(S);
    auto delta = Q - Q_prev;
    auto err = sum(delta * delta);
    if (i % 10 == 0 || i + 1 == n_simulations)
      std::cerr << "SSE at iteration " << i << ": " << err << std::endl;
    Q_prev = Q;
  }

  return std::make_tuple(eval(diagonal(R)), Q);
}

int pca_main(int argc, char** argv)
{
  if (argc < 3)
    return std::cerr << "Error: missing argument (path to results file)\n", EXIT_FAILURE;

  std::string geno_file_path = argv[2];
  double min_af = 0.01;
  double max_af = 1.;

  savvy::reader geno_file(geno_file_path);
  if (!geno_file)
    return std::cerr << "Error: could not open genotype file ("<< geno_file_path << ")\n", EXIT_FAILURE;

  std::vector<std::vector<double>> geno_matrix(1);
  savvy::variant var;
  while (geno_file >> var)
  {
    if (!var.get_format("GT", geno_matrix.back()))
      continue;

    double ac = std::accumulate(geno_matrix.back().begin(), geno_matrix.back().end(), 0.);
    double af = ac / geno_matrix.back().size();
    if (af < min_af || af > max_af)
      continue;

    savvy::stride_reduce(geno_matrix.back(), geno_matrix.back().size() / geno_file.samples().size());

    geno_matrix.emplace_back();
    if (geno_matrix.size() == 101)
      break;
  }

  geno_matrix.pop_back();

  if (geno_file.bad())
    return EXIT_FAILURE;

  xt::xarray<double> eigvals, eigvecs;
  std::tie(eigvals, eigvecs) = compute_eigen_dense(geno_matrix);
  std::cerr << "evals: " << eigvals << std::endl;
  //std::cerr << "evecs: " << eigvecs << std::endl;

  std::ofstream eigvec_file("/dev/stdout", std::ios::binary);
  for (std::size_t i = 1; i <= eigvecs.shape(1); ++i)
    eigvec_file << (i > 1 ? "\tpc" : "pc") << i;
  eigvec_file << std::endl;

  for (std::size_t i = 0; i < eigvecs.shape(0); ++i)
  {
    eigvec_file << eigvecs(i, 0);
    for (std::size_t j = 1; j < eigvecs.shape(1); ++j)
    {
      eigvec_file << "\t" << eigvecs(i, j);
    }
    eigvec_file << "\n";
  }



  return EXIT_SUCCESS;
}