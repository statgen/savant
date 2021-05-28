/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "utility.hpp"

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
  Eigen::VectorXd residuals_gamma_;
  double residuals_sum_ = 0.;
  double gamma_ = 0.;
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
//    Eigen::VectorXd Py_test = Viy - (ViX * qr.inverse()) * (covar.transpose() * Viy);
//
//    std::cerr << Py.sum() << " " << Py_test.sum() << std::endl;
//    std::cerr << Py.mean() << " " << Py_test.mean() << std::endl;

    Eigen::VectorXd d = V_ldlt.vectorD();
    double d_sum = d.array().sum();
    double d_min = d.minCoeff();
    //assert(d_min >= 0.);
    double d_max = d.maxCoeff();
    double d_size = d.array().size();
    double d_expected = d(0) * d_size;
    if (d_min <= 0.)
    {
      auto a = 0;
    }
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

  mixed_effects_model(const res_t& y_orig, const cov_t& x_orig, const Eigen::SparseMatrix<double>& K, const std::vector<savvy::compressed_vector<float>>& grammar_genotypes, const std::vector<std::string>& sample_intersection)
  {
    Eigen::MatrixXd ones(x_orig.rows(), 1);
    ones.setOnes();

    Eigen::MatrixXd X(x_orig.rows(), x_orig.cols() + 1);
    X << ones, x_orig;
    //std::cerr << X << std::endl;

    res_t y = y_orig;
    std::size_t n = y.size();

//    for (std::size_t i = 0; i < sample_intersection.size(); ++i)
//      std::cerr << sample_intersection[i] << "\t" << y[i] << "\n";
//    std::cerr << "----------------------------------------" << std::endl;

    {
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X.transpose() * X);
      auto pbetas = (qr.inverse() * X.transpose()) * y;

      y = y - X * pbetas;
      X = ones;
//      for (std::size_t i = 0; i < sample_intersection.size(); ++i)
//        std::cerr << sample_intersection[i] << "\t" << y[i] << "\n";
//      std::cerr.flush();
    }

    double pheno_mean = y.array().mean();
    double pheno_std_dev = std::sqrt((y.array() - pheno_mean).square().sum()/(y.size()-1));
    y.array() -= pheno_mean;
    //y /= std::sqrt((y.array() - pheno_mean).square().sum()/y.size());
    //std::cerr << "pheno: " << pheno << std::endl;

    // ====================================== //
    // Grid search
    double Vp = y.array().square().sum() / (n - 1);

#if 1
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
        double log_likelihood = reml_chol2(K, y, X, Vg, Ve);
        std::cerr << log_likelihood << "\t" << Vg << "\t" << Ve << std::endl;
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
#else
    double Vg = 0.00904266;
    double Ve = Vp - Vg;
#endif
    // ====================================== //
    // grammar-gamma
    Eigen::SparseMatrix<double> I(K.cols() , K.cols());
    I.setIdentity();

    Eigen::SparseMatrix<double> V(K.cols(), K.cols());
    V = Vg * K + Ve * I;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> V_ldlt;
    V_ldlt.compute(V);
    residuals_ = V_ldlt.solve(y);
    residuals_sum_ = residuals_.sum();
#if 1
    gamma_ = 0.f;
    Eigen::VectorXd dense_geno(residuals_.size());
    for (const auto& geno_vec : grammar_genotypes)
    {
      dense_geno.setZero();
      assert(geno_vec.size() == dense_geno.size());
      for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
      {
        assert(!std::isnan(*it));
        dense_geno[it.offset()] = *it;
      }

      Eigen::VectorXd Vi_x = V_ldlt.solve(dense_geno);
      float numerator = 0.f;
      float denominator = 0.f;
      for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
      {
        if (!std::isnan(*it))
        {
          numerator += Vi_x[it.offset()] * *it;
          denominator += *it * *it;
        }
      }

      gamma_ += numerator / denominator;
    }

    gamma_ /= grammar_genotypes.size();
#else
    gamma_ = 13.784334;
#endif
    std::cerr << "gamma: " << gamma_ << std::endl;
    // ====================================== //

    residuals_gamma_ = residuals_ / gamma_;
    assert(residuals_gamma_.size() == sample_intersection.size());
//    for (std::size_t i = 0; i < sample_intersection.size(); ++i)
//      std::cerr << sample_intersection[i] << "\t" << residuals_gamma_[i] << "\n";
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
    throw std::runtime_error("not implemented");
    stats_t ret;
    return ret;
  }

  template <typename GenoT>
  stats_t test_single(const savvy::compressed_vector<GenoT>& x, scalar_type s_x, std::size_t ploidy = 2) const
  {
    stats_t ret;

#if 0
    scalar_type numerator2{};
    scalar_type denominator2{};
    Eigen::VectorXd xvec(x.size());
    xvec.setZero();
    for (auto it = x.begin(); it != x.end(); ++it)
      xvec[it.offset()] = *it;
    double mean = s_x / x.size();
    double sd = mean * (1. - (mean/ploidy));
    double scale = 1. / std::sqrt(sd);
    double real_sd = std::sqrt((xvec.array() - mean).square().sum()/(xvec.size()-0));
    double real_scale = 1. / real_sd;

    xvec.array() -= mean;
    xvec.array() *= scale;
    for (std::size_t i = 0; i < xvec.size(); ++i)
    {
      numerator2 += residuals_[i] * xvec[i];
      denominator2 += xvec[i] * xvec[i];
    }

#else
    scalar_type numerator{};
    scalar_type denominator{};
    bool center = true;
    if (center)
    {
      double mean = s_x / x.size();
      double sd = std::sqrt(mean * (1. - (mean/ploidy)));
      double scale = 1. / sd;
      numerator = -mean * scale * residuals_sum_;
      denominator = -2 * mean * scale * s_x * scale + square(mean * scale) * x.size(); // (x - m)(x - m) == (xx - 2mx + mm)
      for (auto it = x.begin(); it != x.end(); ++it) // ((x - m)/d)((x - m)/d) == (x/d - m/d)(x/d - m/d) == ((x/d)(x/d) - 2(m/d)(x/d) + (m/d)(m/d))
      {
        assert(!std::isnan(*it));
        float gt = *it * scale;
        numerator += residuals_[it.offset()] * gt;
        denominator += gt * gt;
      }
    }
    else
    {
      for (auto it = x.begin(); it != x.end(); ++it)
      {
        assert(!std::isnan(*it));
        numerator += residuals_[it.offset()] * *it;
        denominator += *it * *it;
      }
    }
#endif
    scalar_type beta = (numerator / denominator) / gamma_;
    scalar_type beta_se = std::sqrt(1. / (denominator * gamma_));

    scalar_type chi_square = square(beta / beta_se);

    scalar_type chi_square_2 = square(numerator) / (denominator * gamma_);

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
      kinship_matrix.insert(id1_it->second, id2_it->second) = coef;
      //kinship_matrix.coeffRef(id2_it->second, id1_it->second) = coef;
      ++cnt;
    }

    kinship_matrix.makeCompressed();

    std::cerr << "min: " << min << std::endl;
    std::cerr << "max: " << max << std::endl;
    std::cerr << "cnt: " << cnt << std::endl;

    Eigen::SparseMatrix<double> I(kinship_matrix.rows(), kinship_matrix.cols());
    I.setIdentity();

    kinship_matrix += Eigen::SparseMatrix<double>(kinship_matrix.transpose()); // TODO: maybe use selfAdjointView downstream to reduce memory usage.
    kinship_matrix += I;

//    std::ofstream sp_kin_out("../test-data/fastGWA_sp_grm_missing_excluded.grm.sp");
//
//    for (int k=0; k<kinship_matrix.outerSize(); ++k)
//    {
//      for (Eigen::SparseMatrix<double>::InnerIterator it(kinship_matrix,k); it; ++it)
//      {
//        it.value();
//        it.row();   // row index
//        it.col();   // col index (here it is equal to k)
//        it.index(); // inner index, here it is equal to it.row()
//
//        sp_kin_out << it.col() << "\t" << it.row() << "\t" << it.value() << "\n";
//      }
//    }
//
//    std::ofstream sp_id_out("../test-data/fastGWA_sp_grm_missing_excluded.grm.id");
//    for (auto it = sample_intersection.begin(); it != sample_intersection.end(); ++it)
//      sp_id_out << *it << "\t" << *it << "\n";


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
      auto& cur_genos = grammar_genotypes.emplace_back();
      var.get_format(fmt_field, cur_genos);

      std::size_t an = cur_genos.size();
      std::size_t ac = 0;
      for (auto it = cur_genos.begin(); it != cur_genos.end(); ++it)
      {
        if (std::isnan(*it))
          --an;
        else
          ++ac;
      }

      if (ac == 0)
      {
        grammar_genotypes.pop_back();
        continue;
      }

      float af = static_cast<float>(ac) / static_cast<float>(an);

      if (an != cur_genos.size())
      {
        for (auto it = cur_genos.begin(); it != cur_genos.end(); ++it)
        {
          if (std::isnan(*it))
            *it = af;
        }
      }

      savvy::stride_reduce(cur_genos, cur_genos.size() / pheno.size());
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