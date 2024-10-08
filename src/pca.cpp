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
#include <getopt.h>

class pca_prog_args
{
private:
  std::vector<option> long_options_;
  std::string input_path_;
  std::string output_path_ = "/dev/stdout";
  double tolerance_ = 1e-8;
  std::size_t max_iterations_ = 128;
  std::int8_t use_cov_mat_ = -1;
  std::size_t num_pcs_ = 10;
  bool is_sparse_ = false;
  bool help_ = false;
public:
  pca_prog_args() :
    long_options_(
      {
        {"cov-mat", required_argument, 0, 'c'},
        {"help", no_argument, 0, 'h'},
        {"iterations", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"pcs", required_argument, 0, 'p'},
        {"sparse", no_argument, 0, 's'},
        {"tolerance", required_argument, 0, 'e'},
        {0, 0, 0, 0}
      })
  {
  }

  const std::string& input_path() const { return input_path_; }
  const std::string& output_path() const { return output_path_; }

  double tolerance() const { return tolerance_; }
  std::size_t max_iterations() const { return max_iterations_; }
  std::size_t num_pcs() const { return num_pcs_; }
  std::int8_t use_cov_mat() const { return use_cov_mat_; }
  bool is_sparse() const { return is_sparse_; }
  bool help_is_set() const { return help_; }

  void print_usage(std::ostream& os)
  {
    os << "Usage: savant pca [opts ...] <in.sav> \n";
    os << "\n";
    os << " -c, --cov-mat     Explicitly set whether to compute covariance matrix (0 or 1; default: auto)\n";
    os << " -e, --tolerance   Tolerance used for convergence (default: 1e-8)\n";
    os << " -h, --help        Print usage\n";
    os << " -i, --iterations  Maximum number of iterations (default: 128)\n";
    os << " -o, --output      Output path (default: /dev/stdout)\n";
    os << " -p, --pcs         Number of PCs to generate (default: 10)\n";
    //os << " -t, --threads     Number of threads to use (default: 1)\n";

    os << std::flush;
  }

  bool parse(int argc, char** argv)
  {
    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "c:e:hi:o:p:s", long_options_.data(), &long_index )) != -1)
    {
      char copt = char(opt & 0xFF);
      switch (copt)
      {
      case 'c':
        use_cov_mat_ = std::atoi(optarg ? optarg : "");
        break;
      case 'e':
        tolerance_ = std::atof(optarg ? optarg : "");
        break;
      case 'h':
        help_ = true;
        return true;
      case 'i':
        max_iterations_ = std::atoi(optarg ? optarg : "");
        break;
      case 'o':
        output_path_ = optarg ? optarg : "";
        break;
      case 'p':
        num_pcs_ = std::atoi(optarg ? optarg : "");
        break;
      case 's':
        is_sparse_ = true;
        break;
      default:
        return false;
      }
    }

    int remaining_arg_count = argc - optind;

    if (remaining_arg_count == 1)
    {
      input_path_ = argv[optind];
    }
    else if (remaining_arg_count < 1)
    {
      std::cerr << "Too few arguments\n";
      return false;
    }
    else
    {
      std::cerr << "Too many arguments\n";
      return false;
    }

    return true;
  }
};

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
      std::cerr << "SSE after iteration " << i << ": " << err << std::endl;
      if (err < tolerance)
        break;
    }
  }

  //std::cerr << "Q: " << Q << std::endl;
  //std::cerr << "R: " << R << std::endl;

  return std::make_tuple(xt::eval(xt::diagonal(R)), Q);
}
#if 0
// https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <boost/numeric/ublas/matrix_sparse.hpp>

template <typename MaT>
auto nipals_sparse_ublas(const MaT& xgeno, std::size_t num_pcs = 10, std::size_t max_iterations = 128, double tolerance = 1e-8)
{
  boost::numeric::ublas::compressed_matrix<double> G(0, 0);
  boost::numeric::ublas::matrix<double> Q;

  Q = boost::numeric::ublas::prod(G, Q);

}

template <typename MaT>
auto nipals_sparse_eigen3(const MaT& xgeno, std::size_t num_pcs = 10, std::size_t max_iterations = 128, double tolerance = 1e-8)
{
  Eigen::SparseMatrix<double> G(xgeno.shape(0), xgeno.shape(1)); // default is column major

  std::vector<Eigen::Triplet<double>> triplets;
  for (std::size_t i = 0; i < xgeno.shape(0); ++i)
  {
    for (std::size_t j = 0; j < xgeno.shape(1); ++j)
    {
      if (xgeno(i, j))
        triplets.emplace_back(i, j, xgeno(i, j));
    }
  }
  //triplets.emplace_back(16, 745, 1.7);
  G.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::MatrixXd Q = Eigen::MatrixXd::Random(G.outerSize(), num_pcs);
  Eigen::MatrixXd R;
  Eigen::FullPivHouseholderQR<Eigen::MatrixXd> fullPivHouseholderQR(Q.rows(), Q.cols());
  fullPivHouseholderQR.compute(Q);
  Q = fullPivHouseholderQR.matrixQ();
  R = fullPivHouseholderQR.matrixQR().template  triangularView<Eigen::Upper>();
  Eigen::MatrixXd Q_prev;

//  xarray<double> Q = random::rand<double>({xgeno.shape(1), num_pcs});
//  xarray<double> R;
//  std::tie(Q, R) = qr(Q);
//  xarray<double> Q_prev;

  for (std::size_t i = 0; i < max_iterations; ++i)
  {
    Q_prev = Q;

    //auto T = dot(xgeno, Q);
    //auto S = dot(transpose(xgeno), T);
    auto P = (G.transpose() * (G * Q)).eval();
    fullPivHouseholderQR.compute(P);
    Q = fullPivHouseholderQR.matrixQ();
    R = fullPivHouseholderQR.matrixQR().template  triangularView<Eigen::Upper>();
    auto delta = Q - Q_prev;
    auto err = (delta.cwiseProduct(delta)).sum();
    if (i % 10 == 0 || i + 1 == max_iterations || err < tolerance)
    {
      std::cerr << "SSE after iteration " << i << ": " << err << std::endl;
      if (err < tolerance)
        break;
    }
  }

}
#endif

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
    //std::cerr << eval(dot(transpose(xgeno), dot(xgeno, Q))) << std::endl;
    std::tie(Q, R) = qr(dot(transpose(xgeno), dot(xgeno, Q)));
    auto delta = Q - Q_prev;
    auto err = sum(delta * delta)();
    if (i % 10 == 0 || i + 1 == max_iterations || err < tolerance)
    {
      std::cerr << "SSE after iteration " << i << ": " << err << std::endl;
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
      std::cerr << "SSE after iteration " << i << ": " << err << std::endl;
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

    //auto start = std::chrono::steady_clock::now();
    std::vector<double> tmp(Q.shape(1));
    for (std::size_t j = 0; j < geno_matrix.size(); ++j)
    {
      //auto x = adapt(geno_matrix[j], {std::size_t(1), geno_matrix[j].size()});
      std::fill(tmp.begin(), tmp.end(), 0.);
//      for (std::size_t k = 0; k < Q.shape(1); ++k)
//      {
//        for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
//          tmp[k] += *gt * col(Q, k)[gt.offset()];
//      }

      for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
      {
        double* p = &Q(gt.offset(), 0);
        for (std::size_t k = 0; k < Q.shape(1); ++k)
        {
          tmp[k] += *gt * p[k];
        }
      }

      //auto x_t = adapt(geno_matrix[j], {geno_matrix[j].size(), std::size_t(1)});

//      for (std::size_t k = 0; k < S.shape(1); ++k)
//      {
//        for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
//        {
//          S(gt.offset(), k) += *gt * tmp[k];
//        }
//      }

      for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
      {
        double* p = &S(gt.offset(), 0);
        for (std::size_t k = 0; k < S.shape(1); ++k)
          p[k] += static_cast<double>(*gt) * tmp[k];
      }

      //S += dot(x_t, dot(x, Q));
      //std::cerr << S << std::endl;
    }
    //std::cerr << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() << "s" <<  std::endl;

    //std::cerr << S << std::endl;
    std::tie(Q, R) = qr(S);
    auto delta = Q - Q_prev;
    double err = sum(delta * delta)();
    if (i % 10 == 0 || i + 1 == max_iterations || err < tolerance)
    {
      std::cerr << "SSE after iteration " << i << ": " << err << std::endl;
      if (err < tolerance)
        break;
    }
  }

  return std::make_tuple(eval(diagonal(R)), Q);
}

template <typename ScalarT, typename CVecT>
auto compute_eigen_sparse_centered4(const std::vector<savvy::compressed_vector<ScalarT>>& geno_matrix, const CVecT& standardization_vec, std::size_t num_pcs = 10, std::size_t max_iterations = 128, double tolerance = 1e-8)
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

    //auto start = std::chrono::steady_clock::now();
    // ========== //
    // center right product
    xt::xtensor<double, 2> T = xt::xtensor<double, 2>::from_shape({geno_matrix.size(), Q.shape(1)});
    xt::xtensor<double, 1> col_sum_Q = -xt::sum(Q, {0});
    for (std::size_t j = 0; j < T.shape(0); ++j)
      xt::row(T, j) = col_sum_Q * standardization_vec[j];
    // ========== //

    //std::vector<double> tmp(Q.shape(1));
    for (std::size_t j = 0; j < geno_matrix.size(); ++j)
    {
      double* tmp = &T(j, 0);

      for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
      {
        double* p = &Q(gt.offset(), 0);
        for (std::size_t k = 0; k < Q.shape(1); ++k)
        {
          tmp[k] += *gt * p[k];
        }
      }

      for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
      {
        double* p = &S(gt.offset(), 0);
        for (std::size_t k = 0; k < S.shape(1); ++k)
          p[k] += static_cast<double>(*gt) * tmp[k];
      }

      //std::cerr << S << std::endl;
    }

    // ========== //
    // center left product
    xt::xtensor<double, 1> temp2 = xt::linalg::dot(standardization_vec, T);
    S -= temp2;
    // ========== //

    //std::cerr << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() << "s" <<  std::endl;

    //std::cerr << S << std::endl;
    std::tie(Q, R) = qr(S);
    auto delta = Q - Q_prev;
    double err = sum(delta * delta)();
    if (i % 10 == 0 || i + 1 == max_iterations || err < tolerance)
    {
      std::cerr << "SSE after iteration " << i << ": " << err << std::endl;
      if (err < tolerance)
        break;
    }
  }

  return std::make_tuple(eval(diagonal(R)), Q);
}

template <typename ScalarT, typename CVecT>
auto compute_eigen_sparse_centered5(const std::vector<savvy::compressed_vector<ScalarT>>& geno_matrix, const CVecT& standardization_vec, std::size_t num_pcs = 10, std::size_t max_iterations = 128, double tolerance = 1e-8)
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

    //auto start = std::chrono::steady_clock::now();
    // ========== //
    // center right product
    //xt::xtensor<double, 2> T = xt::xtensor<double, 2>::from_shape({geno_matrix.size(), Q.shape(1)});
    xt::xtensor<double, 1> col_sum_Q = -xt::sum(Q, {0});
//    for (std::size_t j = 0; j < T.shape(0); ++j)
//      xt::row(T, j) = col_sum_Q * standardization_vec[j];
    // ========== //

    std::vector<double> tmp(Q.shape(1));
    xt::xtensor<double, 1> left_center_accum = xt::zeros<double>({Q.shape(1)});
    for (std::size_t j = 0; j < geno_matrix.size(); ++j)
    {
      // ========== //
      // center right product
      double std_coef = standardization_vec[j];
      for (std::size_t k = 0; k < Q.shape(1); ++k)
        tmp[k] = col_sum_Q[k] * std_coef;
      // ========== //

      for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
      {
        double* p = &Q(gt.offset(), 0);
        for (std::size_t k = 0; k < Q.shape(1); ++k)
        {
          tmp[k] += *gt * p[k];
        }
      }

      // ========== //
      // accumulate center for left product
      for (std::size_t k = 0; k < Q.shape(1); ++k)
        left_center_accum[k] += tmp[k] * std_coef;
      // ========== //

      for (auto gt = geno_matrix[j].begin(); gt != geno_matrix[j].end(); ++gt)
      {
        double* p = &S(gt.offset(), 0);
        for (std::size_t k = 0; k < S.shape(1); ++k)
          p[k] += static_cast<double>(*gt) * tmp[k];
      }

      //std::cerr << S << std::endl;
    }

    // ========== //
    // center left product
    //xt::xtensor<double, 1> temp2 = xt::linalg::dot(standardization_vec, T);
    //S -= temp2;
    S -= left_center_accum;
    // ========== //

    //std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << "ms" <<  std::endl;

    //std::cerr << S << std::endl;
    std::tie(Q, R) = qr(S);
    auto delta = Q - Q_prev;
    double err = sum(delta * delta)();
    if (i % 10 == 0 || i + 1 == max_iterations || err < tolerance)
    {
      std::cerr << "SSE after iteration " << i << ": " << err << std::endl;
      if (err < tolerance)
        break;
    }
  }

  return std::make_tuple(eval(diagonal(R)), Q);
}

bool load_geno_matrix(savvy::reader& geno_file, std::vector<savvy::compressed_vector<double>>& geno, std::vector<double>& centering_vec)
{
  geno.resize(0);
  savvy::variant var;
  savvy::compressed_vector<std::int8_t> geno_vec;
  savvy::compressed_vector<double> bi_geno_vec;
  while (geno_file >> var)
  {
    if (var.alts().size() < 1) continue;
    if (!var.get_format("GT", geno_vec))
      return std::cerr << "Error: variant mssing GT field\n", false; // TODO: allow skipping as an option

    for (std::size_t allele_idx = 1; allele_idx <= var.alts().size(); ++allele_idx)
    {
      std::size_t an = geno_vec.size();
      std::size_t ac = 0;
      for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
      {
        if (*it < 0) // missing
          --an;
        else
          ac += int(*it == allele_idx);
      }

      bi_geno_vec.clear();
      bi_geno_vec.resize(geno_vec.size());

      double af = static_cast<double>(ac) / an;
      double denom = std::sqrt(2. * af * (1. - af));
      for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
      {
        if (*it < 0)
          bi_geno_vec[it.offset()] = af / denom; // mean impute missing
        else if (*it == allele_idx)
          bi_geno_vec[it.offset()] = 1. / denom;
      }

      savvy::stride_reduce(bi_geno_vec, bi_geno_vec.size() / geno_file.samples().size());

      geno.emplace_back(bi_geno_vec.value_data(), bi_geno_vec.value_data() + bi_geno_vec.non_zero_size(), bi_geno_vec.index_data(), bi_geno_vec.size()); // copy operators not yet overloaded.
      centering_vec.emplace_back(2. * af / denom);
    }
  }

  if (geno_file.bad())
    return std::cerr << "Error: read failure\n", false;

  return true;
}

bool load_geno_matrix(savvy::reader& geno_file, std::size_t& n_variants, xt::xtensor<double, 2>& xgeno, bool center)
{
  xgeno.resize({n_variants, geno_file.samples().size()});
  xgeno.fill(0.);

  n_variants = 0;
  savvy::variant var;
  savvy::compressed_vector<std::int8_t> geno_vec;
  while (geno_file >> var)
  {
    if (var.alts().size() < 1) continue;
    if (!var.get_format("GT", geno_vec))
      return std::cerr << "Error: variant mssing GT field\n", false; // TODO: allow skipping as an option

    const std::size_t allele_idx = 1;

    std::size_t an = geno_vec.size();
    std::size_t ac = 0;
    for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
    {
      if (*it < 0) // missing
      {
        --an;
      }
      else
      {
        // Only first allele is considered when using xtensor matrix. Since record counts from s1r index are unaware of multiallelics.
        // Could potentially do a first pass wih SAV files (as we do with BCF/VCF) to compute true variant count in order to support other alleles.
        ac += int(*it == allele_idx);
      }
    }

    if (ac == 0 || ac == an) continue; // skipping monomorphic //TODO: log first occurence

    double af = static_cast<double>(ac) / an;

    assert(af > 0. && af < 1.);


    std::size_t stride = geno_vec.size() / geno_file.samples().size();
    for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
    {
      if (*it < 0)
        xgeno(n_variants, it.offset() / stride) += af; // mean impute missing
      else
        xgeno(n_variants, it.offset() / stride) += double(*it == allele_idx);
    }

    if (center)
    {
      double mean = 2. * af;
      double denom = std::sqrt(2. * af * (1. - af));
      for (std::size_t i = 0; i < xgeno.shape(1); ++i)
      {
        xgeno(n_variants, i) = (xgeno(n_variants, i) - mean) / denom;
        assert(!std::isnan(xgeno(n_variants, i)));
      }
    }

    ++n_variants;
  }

  if (geno_file.bad())
    return std::cerr << "Error: read failure\n", false;

  return true;
}

int pca_main(int argc, char** argv)
{
#ifdef NDEBUG
  std::cerr << "release build" << std::endl;
#else
  std::cerr << "debug build" << std::endl;
#endif
  pca_prog_args args;
  if (!args.parse(argc, argv))
  {
    args.print_usage(std::cerr);
    return EXIT_FAILURE;
  }

  if (args.help_is_set())
  {
    args.print_usage(std::cout);
    return EXIT_SUCCESS;
  }

  savvy::reader geno_file(args.input_path());
  if (!geno_file)
    return std::cerr << "Error: could not open genotype file ("<< args.input_path() << ")\n", EXIT_FAILURE;

  xt::xarray<double> eigvals, eigvecs;
  if (args.is_sparse())
  {
    std::vector<savvy::compressed_vector<double>> geno;
    std::vector<double> centering_vec;
    if (!load_geno_matrix(geno_file, geno, centering_vec) || geno.empty())
      return std::cerr << "Error: failed loading geno matrix from file\n", EXIT_FAILURE;

    std::size_t m = geno.size(), n = geno[0].size();
    std::cerr << "Loaded " << m << " sparse variants from " << n << " samples" << std::endl;

    bool center = true;
    if (center)
      std::tie(eigvals, eigvecs) = compute_eigen_sparse_centered5(geno, xt::adapt(centering_vec), args.num_pcs(), args.max_iterations(), args.tolerance());
    else
      std::tie(eigvals, eigvecs) = compute_eigen_sparse(geno, args.num_pcs(), args.max_iterations(), args.tolerance());
  }
  else
  {
    std::size_t n_variants = 0;
    for (const auto& s : savvy::s1r::stat_index(args.input_path()))
    {
      n_variants += s.record_count; // assuming biallelic
    }

    if (n_variants == 0)
    {
      std::cerr << "Reading through file to get variant count (this step is not necessary when using SAV file format)" << std::endl;
      savvy::variant var;
      while (geno_file >> var)
        ++n_variants;

      std::cerr << "Will allocate memory for " << n_variants << " variants" << std::endl;
      geno_file = savvy::reader(args.input_path());
      if (!geno_file)
        return std::cerr << "Error: could not reopen genotype file ("<< args.input_path() << ")\n", EXIT_FAILURE;
    }

    xt::xtensor<double, 2> xgeno;
    if (!load_geno_matrix(geno_file, n_variants, xgeno, true))
      return std::cerr << "Error: failed loading geno matrix from file\n", EXIT_FAILURE;

    auto nipals_complexity = [](std::size_t m, std::size_t n) { return std::size_t(2) * m * n; };
    auto cov_complexity = [](std::size_t m, std::size_t n) { return n * n; };

    // load_geno_matrix() updates n_variants
    std::size_t m = n_variants /*xgeno.shape(0)*/, n = xgeno.shape(1);
    std::cerr << "Loaded " << m << " variants from " << n << " samples" << std::endl;

    bool use_cov_mat = args.use_cov_mat() < 0 ? nipals_complexity(m, n) > cov_complexity(m, n) : (bool)args.use_cov_mat();

    if (use_cov_mat)
      std::cerr << "Using covariance matrix" << std::endl;
    else
      std::cerr << "Not using covariance matrix" << std::endl;

    std::tie(eigvals, eigvecs) = use_cov_mat ?
      compute_cov_mat_eigen(xt::linalg::dot(xt::transpose(xgeno), xgeno), args.num_pcs(), args.max_iterations(), args.tolerance())
      :
      nipals_dense(xgeno, args.num_pcs(), args.max_iterations(), args.tolerance());
  }

  std::cerr << "evals: " << eigvals << std::endl;
  //std::cerr << "evecs: " << eigvecs << std::endl;

  std::ofstream eigvec_file(args.output_path(), std::ios::binary);
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