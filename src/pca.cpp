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
    while ((opt = getopt_long(argc, argv, "c:e:hi:o:p:", long_options_.data(), &long_index )) != -1)
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

bool load_geno_matrix(savvy::reader& geno_file, xt::xtensor<double, 2>& xgeno, bool center)
{
  std::vector<savvy::compressed_vector<std::int8_t>> geno_matrix(1);
  savvy::variant var;
  while (geno_file >> var)
  {
    if (var.alts().size() != 1) continue; // SNPs only. //TODO: Support indels
    if (!var.get_format("GT", geno_matrix.back()))
      return std::cerr << "Error: variant mssing GT field\n", false; // TODO: allow skipping as an option

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

    double mean = center ? 2. * af : 0.;
    for (std::size_t j = 0; j < geno_file.samples().size(); ++j)
    {
      xgeno(i, j) = (xgeno(i, j) - mean) / std::sqrt(2. * af * (1. - af)); // row normalize
      assert(!std::isnan(xgeno(i, j)));
    }
  }

  return true;
}

int pca_main(int argc, char** argv)
{
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

  xt::xtensor<double, 2> xgeno;
  if (!load_geno_matrix(geno_file, xgeno, true))
    return std::cerr << "Error: failed loading geno matrix from file\n", EXIT_FAILURE;

  auto nipals_complexity = [](std::size_t m, std::size_t n) { return std::size_t(2) * m * n; };
  auto cov_complexity = [](std::size_t m, std::size_t n) { return n * n; };

  std::size_t m = xgeno.shape(0), n = xgeno.shape(1);
  std::cerr << "Loaded " << m << " variants from " << n << " samples" << std::endl;

  bool use_cov_mat = args.use_cov_mat() < 0 ? nipals_complexity(m, n) > cov_complexity(m, n) : (bool)args.use_cov_mat();

  if (use_cov_mat)
    std::cerr << "Using covariance matrix" << std::endl;
  else
    std::cerr << "Not using covariance matrix" << std::endl;

  xt::xarray<double> eigvals, eigvecs;
  std::tie(eigvals, eigvecs) = use_cov_mat ?
    compute_cov_mat_eigen(xt::linalg::dot(xt::transpose(xgeno), xgeno), args.num_pcs(), args.max_iterations(), args.tolerance())
    :
    nipals_dense(xgeno, args.num_pcs(), args.max_iterations(), args.tolerance());
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