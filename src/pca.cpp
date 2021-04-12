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
auto compute_cov_mat_eigen(const T& X, std::size_t num_pcs = 10, std::size_t max_iterations = 128, double tolerance = 1e-8)
{
  using namespace xt;
  using namespace xt::linalg;

  xarray<double> Q = random::rand<double>({X.shape(1), num_pcs});
  xarray<double> R;
//  std::cerr << "Q: " << Q << std::endl;
//  std::cerr << "R: " << R << std::endl;
  std::tie(Q, R) = qr(Q);
  xarray<double> Q_prev;

//  std::cerr << "Q: " << Q << std::endl;
//  std::cerr << "R: " << R << std::endl;

  for (std::size_t i = 0; i < max_iterations; ++i)
  {
    Q_prev = Q;
    //auto start = std::chrono::steady_clock::now();
    //auto Z = xt::eval(dot(X, Q));
    //std::cerr << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() << "us" <<  std::endl;
    std::tie(Q, R) = qr(dot(X, Q));

    auto delta = Q - Q_prev;
    double err = sum(delta * delta)();
    if (i % 10 == 0 || i + 1 == max_iterations || err < tolerance)
    {
      std::cerr << "SSE after iteration " << (i + 1) << ": " << err << std::endl;
      if (err < tolerance)
        break;
    }
  }

  //std::cerr << "Q: " << Q << std::endl;
  //std::cerr << "R: " << R << std::endl;

  return std::make_tuple(xt::eval(xt::diagonal(R)), Q);
}

template <typename MaT>
auto nipals_dense(const MaT& xgeno, std::size_t num_pcs = 10, std::size_t max_iterations = 128, double tolerance = 1e-8)
{
  using namespace xt;
  using namespace xt::linalg;

  assert(xgeno.size());

  xarray<double> Q = random::rand<double>({xgeno.shape(1), num_pcs});
  xarray<double> R;
  std::tie(Q, R) = qr(Q);
  xarray<double> Q_prev;

  for (std::size_t i = 0; i < max_iterations; ++i)
  {
    Q_prev = Q;

    //auto T = dot(xgeno, Q);
    //auto S = dot(transpose(xgeno), T);

    std::tie(Q, R) = qr(dot(transpose(xgeno), dot(xgeno, Q)));
    auto delta = Q - Q_prev;
    auto err = sum(delta * delta)();
    if (i % 10 == 0 || i + 1 == max_iterations || err < tolerance)
    {
      std::cerr << "SSE after iteration " << (i + 1) << ": " << err << std::endl;
      if (err < tolerance)
        break;
    }
  }

  return std::make_tuple(eval(diagonal(R)), Q);
}

template <typename T>
auto compute_eigen_dense(const std::vector<std::vector<T>>& geno_matrix, std::size_t num_pcs = 10, std::size_t max_iterations = 128, double tolerance = 1e-8)
{
  using namespace xt;
  using namespace xt::linalg;

  assert(geno_matrix.size());

  xarray<double> Q = random::rand<double>({geno_matrix[0].size(), num_pcs});
  xarray<double> R;
  std::tie(Q, R) = qr(Q);
  xarray<double> Q_prev;

  for (std::size_t i = 0; i < max_iterations; ++i)
  {
    Q_prev = Q;
    xarray<double> S = xt::zeros<double>(Q.shape());

    for (std::size_t j = 0; j < geno_matrix.size(); ++j)
    {
      auto x = adapt(geno_matrix[j], {std::size_t(1), geno_matrix[j].size()});
      auto x_t = adapt(geno_matrix[j], {geno_matrix[j].size(), std::size_t(1)});
      S += dot(x_t, dot(x, Q));
    }
    std::tie(Q, R) = qr(S);
    auto delta = Q - Q_prev;
    auto err = sum(delta * delta)();
    if (i % 10 == 0 || i + 1 == max_iterations || err < tolerance)
    {
      std::cerr << "SSE after iteration " << (i + 1) << ": " << err << std::endl;
      if (err < tolerance)
        break;
    }
  }

  return std::make_tuple(eval(diagonal(R)), Q);
}

template <typename T>
auto compute_eigen_sparse(const std::vector<savvy::compressed_vector<T>>& geno_matrix, std::size_t num_pcs = 10, std::size_t max_iterations = 128, double tolerance = 1e-8)
{
  using namespace xt;
  using namespace xt::linalg;

  assert(geno_matrix.size());

  xarray<double> Q = random::rand<double>({geno_matrix[0].size(), num_pcs});
  xarray<double> R;
  std::tie(Q, R) = qr(Q);
  xarray<double> Q_prev;

  for (std::size_t i = 0; i < max_iterations; ++i)
  {
    Q_prev = Q;

    xarray<double> S = xt::zeros<double>(Q.shape());

    for (std::size_t j = 0; j < geno_matrix.size(); ++j)
    {
      //auto x = adapt(geno_matrix[j], {std::size_t(1), geno_matrix[j].size()});
      std::vector<double> tmp(Q.shape(1));
      for (std::size_t k = 0; k < Q.shape(1); ++k)
      {
        for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
          tmp[k] += *gt * col(Q, k)[gt.offset()];
      }

      //auto x_t = adapt(geno_matrix[j], {geno_matrix[j].size(), std::size_t(1)});

      for (std::size_t k = 0; k < S.shape(1); ++k)
      {
        for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
        {
          S(gt.offset(), k) += *gt * tmp[k];
        }
      }

      //S += dot(x_t, dot(x, Q));
      //std::cerr << S << std::endl;
    }

    std::tie(Q, R) = qr(S);
    auto delta = Q - Q_prev;
    double err = sum(delta * delta)();
    if (i % 10 == 0 || i + 1 == max_iterations || err < tolerance)
    {
      std::cerr << "SSE after iteration " << (i + 1) << ": " << err << std::endl;
      if (err < tolerance)
        break;
    }
  }

  return std::make_tuple(eval(diagonal(R)), Q);
}

bool load_geno_matrix(savvy::reader& geno_file, xt::xtensor<double, 2>& xgeno)
{
  std::vector<savvy::compressed_vector<std::int8_t>> geno_matrix(1);
  savvy::variant var;
  while (geno_file >> var)
  {
    if (!var.get_format("GT", geno_matrix.back()))
      return std::cerr << "Error: variant mssing GT field\n", false;

//    if (std::isnan(af) || af < min_af || af > max_af)
//      continue;

    geno_matrix.emplace_back();
//    if (geno_matrix.size() == 101)
//      break;
  }

  geno_matrix.pop_back();

  if (geno_file.bad())
    return std::cerr << "Error: read failure\n", false;

  xgeno.resize({geno_matrix.size(), geno_file.samples().size()});
  xgeno.fill(0.);

  for (std::size_t i = 0; i < geno_matrix.size(); ++i)
  {
    std::size_t an = geno_matrix[i].size();
    double ac{};
    for (auto jt = geno_matrix[i].begin(); jt != geno_matrix[i].end(); ++jt)
    {
      if (*jt < 0) // missing
        --an;
      else
        ac += *jt;
    }

    double af = ac / an;

    std::size_t stride = geno_matrix[i].size() / geno_file.samples().size();
    if (an != geno_matrix[i].size())
    {
      for (auto jt = geno_matrix[i].begin(); jt != geno_matrix[i].end(); ++jt)
      {
        if (*jt < 0)
          xgeno(i, jt.offset() / stride) += af; // mean impute missing
        else
          xgeno(i, jt.offset() / stride) += *jt;
      }
    }
    else
    {
      for (auto jt = geno_matrix[i].begin(); jt != geno_matrix[i].end(); ++jt)
        xgeno(i, jt.offset() / stride) += *jt;
    }

    for (std::size_t j = 0; j < geno_file.samples().size(); ++j)
      xgeno(i, j) = (xgeno(i, j) - 2. * af) / std::sqrt(2. * af * (1. - af)); // row normalize
  }

  return true;
}

int pca_main(int argc, char** argv)
{
  if (argc < 3)
    return std::cerr << "Error: missing argument (path to results file)\n", EXIT_FAILURE;

  std::string geno_file_path = argv[2];

  savvy::reader geno_file(geno_file_path);
  if (!geno_file)
    return std::cerr << "Error: could not open genotype file ("<< geno_file_path << ")\n", EXIT_FAILURE;

  xt::xtensor<double, 2> xgeno;
  if (!load_geno_matrix(geno_file, xgeno))
    return std::cerr << "Error: failed loading geno matrix from file\n", EXIT_FAILURE;

  auto nipals_complexity = [](std::size_t m, std::size_t n) { return std::size_t(2) * m * n; };
  auto cov_complexity = [](std::size_t m, std::size_t n) { return n * n; };

  std::size_t m = xgeno.shape(0), n = xgeno.shape(1);
  std::cerr << "Loaded " << m << " variants from " << n << " samples" << std::endl;
  xt::xarray<double> eigvals, eigvecs;
  std::tie(eigvals, eigvecs) = nipals_complexity(m, n) < cov_complexity(m, n) ?
    nipals_dense(xgeno, 10)
    :
    compute_cov_mat_eigen(xt::linalg::dot(xt::transpose(xgeno), xgeno), 10);
  std::cerr << "evals: " << eigvals << std::endl;
  //std::cerr << "evecs: " << eigvecs << std::endl;

  std::ofstream eigvec_file("/dev/stdout", std::ios::binary);
  eigvec_file << "sample_id";
  for (std::size_t i = 1; i <= eigvecs.shape(1); ++i)
    eigvec_file << "\tpc" << i;
  eigvec_file << std::endl;

  assert(eigvecs.shape(0) == geno_file.samples().size());
  for (std::size_t i = 0; i < eigvecs.shape(0); ++i)
  {
    eigvec_file << geno_file.samples()[i];
    for (std::size_t j = 0; j < eigvecs.shape(1); ++j)
    {
      eigvec_file << "\t" << eigvecs(i, j);
    }
    eigvec_file << "\n";
  }



  return EXIT_SUCCESS;
}