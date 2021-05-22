/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "utility.hpp"
#include "fit.hpp"

#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/Dense>

#include <boost/math/distributions/chi_squared.hpp>

#include <iostream>
#include <iomanip>

class mixed_effects_model
{
public:
  typedef Eigen::Map<Eigen::VectorXd> res_t;
  typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> cov_t;
  typedef double scalar_type;

  struct stats_t
  {
    scalar_type pvalue;
    scalar_type beta;
    scalar_type se;
    scalar_type chi_square;
    //scalar_type r2;
  };
private:
  Eigen::VectorXd residuals_;
  double gamma_;
public:
  /*
  P = Vi - ViX (XtViX)i XtVi
  -0.5 ( log|V| + log|XtViX| + yt Py )
  Py = Vi y - ViX (XtViX) XtVi y
  */
  double reml_chol2(const Eigen::SparseMatrix<double>& K, const Eigen::VectorXd& pheno, const Eigen::MatrixXd& covar, double Vg, double Ve)
  {
    std::size_t n = K.cols();
    std::size_t c = covar.cols();

    Eigen::SparseMatrix<double> I(n , n);
    I.setIdentity();

    Eigen::SparseMatrix<double> V(n, n);
    V = Vg * K + Ve * I;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> V_ldlt;
    V_ldlt.compute(V);
    assert(V_ldlt.info() == Eigen::Success);

    Eigen::MatrixXd ViX = V_ldlt.solve(covar); // n*c

    Eigen::MatrixXd XtViX = covar.transpose() * ViX; // c*c

    Eigen::MatrixXd Viy = V_ldlt.solve(pheno); // n*1

    //Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(XtViX);
    //double log_det_XtViX = cod.logAbsDeterminant();
    //XtViX = cod.pseudoInverse();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtViX);
    assert(qr.isInvertible());


    Eigen::VectorXd Py = Viy - ViX * (qr.inverse() * (covar.transpose() * Viy));

    Eigen::VectorXd d = V_ldlt.vectorD();
    double d_sum = d.array().sum();
    double d_min = d.minCoeff();
    //assert(d_min >= 0.);
    double d_max = d.maxCoeff();
    double d_size = d.array().size();
    double d_expected = d(0) * d_size;
//  std::cerr << "V_ldlt.determinant(): " << V_ldlt.determinant() << std::endl;
//  std::cerr << "d.array().sum(): " << d.array().sum() << std::endl;
//  std::cerr << "log(d.array().sum()): " << std::log(d.array().sum()) << std::endl;
//  std::cerr << "d.array().log().sum(): " << d.array().log().sum() << std::endl;

    double log_det_V = d.array().abs().log().sum();
    double log_det_XtViX = qr.logAbsDeterminant();
    double ytPy = pheno.dot(Py);

    double log_likelihood = -0.5 * (log_det_V + log_det_XtViX + ytPy);
    return log_likelihood;
  }

  mixed_effects_model(const res_t& y_orig, const cov_t& x_orig, const Eigen::SparseMatrix<double>& K, const std::vector<savvy::compressed_vector<float>>& grammar_genotypes)
  {
    res_t y = y_orig;
    std::size_t n = y.size();

    double pheno_mean = y.array().mean();
    double pheno_std_dev = std::sqrt((y.array() - pheno_mean).square().sum()/(y.size()-1));
    y.array() -= pheno_mean;
    y /= std::sqrt((y.array() - pheno_mean).square().sum()/y.size());
    //std::cerr << "pheno: " << pheno << std::endl;


    // ====================================== //
    // Grid search
    double Vp = y.array().square().sum() / (n - 1);

    std::size_t n_partitions = 64;
    std::size_t recursion_depth = 8;

    double lower = 0.;
    double upper = Vp;

    //std::vector<double> partitions(n_partitions);
    std::cerr << "log_likelihood\tVg\tVe" << std::endl;
    double max_log_likelihood = std::numeric_limits<double>::lowest();
    double optimal_Vg = std::numeric_limits<double>::lowest();
    for (std::size_t i = 0; i < recursion_depth; ++i)
    {
      double step_size = (upper - lower) / n_partitions;
      assert(step_size > 0.); // TODO: handle.
      for (std::size_t j = 0; j < n_partitions; ++j)
      {
        double Vg = lower + step_size * j + step_size / 2.;
        double Ve = Vp - Vg;
        double log_likelihood = reml_chol2(K, y, x_orig, Vg, Ve);
        std::cerr << max_log_likelihood << "\t" << Vg << "\t" << Ve << std::endl;
        if (std::isfinite(log_likelihood) && log_likelihood > max_log_likelihood)
        {
          max_log_likelihood = log_likelihood;
          optimal_Vg = Vg;
        }
      }
      std::cerr << "---------------------" << std::endl;

//      Vg = optimal_Vg;
//      Ve = Vp - Vg;
//
//      assert(Ve >= 0.);

      lower = std::max(0., optimal_Vg - step_size / 2.);
      upper = std::min(Vp, optimal_Vg + step_size / 2.);
    }

    assert(max_log_likelihood > std::numeric_limits<decltype(max_log_likelihood)>::lowest());
    double Vg = optimal_Vg;
    double Ve = Vp - Vg;
    std::cerr << max_log_likelihood << "\t" << Vg << "\t" << Ve << std::endl;
    // ====================================== //

    // ====================================== //
    // grammar-gamma
    Eigen::SparseMatrix<double> I(K.cols() , K.cols());
    I.setIdentity();

    Eigen::SparseMatrix<double> V(K.cols(), K.cols());
    V = Vg * K + Ve * I;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> V_ldlt;
    V_ldlt.compute(V);
    residuals_ = V_ldlt.solve(y);

    gamma_ = 0.f;
    Eigen::VectorXd dense_geno(residuals_.size());
    for (const auto& geno_vec : grammar_genotypes)
    {
      dense_geno.setZero();
      assert(geno_vec.size() == dense_geno.size());
      for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
        dense_geno[it.offset()] = *it;

      Eigen::VectorXd Vi_x = V_ldlt.solve(dense_geno);
      float numerator = 0.f;
      float denominator = 0.f;
      for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
      {
        numerator += Vi_x[it.offset()] * *it;
        denominator += *it * *it;
      }

      gamma_ += numerator / denominator;
    }

    gamma_ /= grammar_genotypes.size();
    std::cerr << "gamma: " << gamma_ << std::endl;
    // ====================================== //
  }

  std::size_t sample_size() const { return residuals_.size(); }

  template <typename T>
  static T square(T v)
  {
    return v * v;
  }

  template <typename GenoT>
  stats_t test_single(const std::vector<GenoT>& x, scalar_type s_x) const
  {
    stats_t ret;
    return ret;
  }

  template <typename GenoT>
  stats_t test_single(const savvy::compressed_vector<GenoT>& x, scalar_type s_x) const
  {
    stats_t ret;

    scalar_type numerator{};
    scalar_type denominator{};
    for (auto it = x.begin(); it != x.end(); ++it)
    {
      numerator += residuals_[it.offset()] * *it;
      denominator += *it * *it;
    }

    scalar_type beta = (numerator / denominator) / gamma_;
    scalar_type beta_se = std::sqrt(1. / (denominator * gamma_));

    scalar_type chi_square = square(beta / beta_se);

    boost::math::chi_squared_distribution<scalar_type> dist(1.f);
    ret.pvalue = boost::math::cdf(boost::math::complement(dist, chi_square));
    ret.beta = beta;
    ret.se = beta_se;
    ret.chi_square = chi_square;

    return ret;
  }

  static bool load_kinship(const std::string& kinship_file_path, Eigen::SparseMatrix<double>& kinship_matrix, const std::vector<std::string>& sample_intersection)
  {
    std::ifstream kinship_file(kinship_file_path, std::ios::binary);
    if (!kinship_file)
      return std::cerr << "Error: could not open kinship file\n", false;

    std::unordered_map<std::string, std::size_t> sample_map;
    sample_map.reserve(sample_intersection.size());

    for (std::size_t i = 0; i < sample_intersection.size(); ++i)
      sample_map.emplace(sample_intersection[i], i);

    std::size_t id1_idx = std::size_t(-1);
    std::size_t id2_idx = std::size_t(-1);
    std::size_t kin_idx = std::size_t(-1);

    std::string line;
    if (!std::getline(kinship_file, line))
      return std::cerr << "Error: kinship file empty\n", false;


    auto header_names = utility::split_string_to_vector(line.c_str(), '\t');
    if (header_names.empty())
      return std::cerr << "Error: empty header\n", false;

    if (header_names[0].size() && header_names[0][0] == '#')
      header_names[0].erase(header_names[0].begin());

    std::vector<std::size_t> mask(header_names.size());

    for (std::size_t i = 0; i < header_names.size(); ++i)
    {
      if (header_names[i] == "ID1")
      {
        id1_idx = i;
      }
      else if (header_names[i] == "ID2")
      {
        id2_idx = i;
      }
      else if (header_names[i] == "Kinship")
      {
        kin_idx = i;
      }
    }

    if (id1_idx == std::size_t(-1))
      return std::cerr << "Error: missing ID1 column\n", false; // TODO: better error message
    if (id2_idx == std::size_t(-1))
      return std::cerr << "Error: missing ID2 column\n", false; // TODO: better error message
    if (kin_idx == std::size_t(-1))
      return std::cerr << "Error: missing Kinship column\n", false; // TODO: better error message

    std::size_t cnt = 0;
    float min = 1e28f;
    float max = 0.f;
    while (std::getline(kinship_file, line))
    {
      auto str_fields = utility::split_string_to_vector(line.c_str(), '\t');

      if (str_fields.size() <= std::max(id1_idx, std::max(id2_idx, kin_idx)))
        return std::cerr << "Error: not enough columns in kinship file\n", false;

      auto id1_it = sample_map.find(str_fields[id1_idx]);
      auto id2_it = sample_map.find(str_fields[id2_idx]);
      if (id1_it == sample_map.end() || id2_it == sample_map.end())
        continue;

      float coef = 2. * std::atof(str_fields[kin_idx].c_str());
      max = std::max(max, coef);
      min = std::min(min, coef);
      assert(id1_it->second != id2_it->second);
      kinship_matrix.coeffRef(id1_it->second, id2_it->second) = coef;
      //kinship_matrix.coeffRef(id2_it->second, id1_it->second) = coef;
      ++cnt;
    }

    std::cerr << "min: " << min << std::endl;
    std::cerr << "max: " << max << std::endl;
    std::cerr << "cnt: " << cnt << std::endl;

    Eigen::SparseMatrix<double> I(kinship_matrix.rows(), kinship_matrix.cols());
    I.setIdentity();

    kinship_matrix += Eigen::SparseMatrix<double>(kinship_matrix.transpose()); // TODO: maybe use selfAdjointView downstream to reduce memory usage.
    kinship_matrix += I;

    return true;
  }

  static bool load_grammar_variants(savvy::reader& geno_file, const std::string& fmt_field, const Eigen::Map<Eigen::VectorXd>& pheno, std::vector<savvy::compressed_vector<float>>& grammar_genotypes)
  {
    grammar_genotypes.clear();
    grammar_genotypes.reserve(1000);

    std::size_t i = 0;
    savvy::variant var;
    while (i < 1000 && geno_file >> var)
    {
      grammar_genotypes.emplace_back();
      var.get_format(fmt_field, grammar_genotypes.back());

      if (grammar_genotypes.back().non_zero_size() == 0)
      {
        grammar_genotypes.pop_back();
        continue;
      }

      savvy::stride_reduce(grammar_genotypes.back(), grammar_genotypes.back().size() / pheno.size());
      ++i;
    }
    return true;
  }
};

static std::ostream& operator<<(std::ostream& os, const mixed_effects_model& v)
{
  os << "pvalue\tbeta\tse\tchi_square"; //\tr2";
  return os;
}

static std::ostream& operator<<(std::ostream& os, const typename mixed_effects_model::stats_t& v)
{
  os << v.pvalue
     << "\t" << v.beta
     << "\t" << v.se
     << "\t" << v.chi_square;
     //<< "\t" << v.r2;
  return os;
}