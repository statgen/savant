/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SAVANT_LINEAR_MODEL_HPP
#define SAVANT_LINEAR_MODEL_HPP

#include "inv_norm.hpp"

#include <savvy/compressed_vector.hpp>

#include <xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <boost/math/distributions.hpp>

#include <iostream>

class linear_model
{
public:
  typedef double scalar_type;
  typedef xt::xtensor<scalar_type, 1> res_t;
  typedef xt::xtensor<scalar_type, 2> cov_t;
protected:
  res_t residuals_;
  scalar_type s_y_;
  scalar_type s_yy_;
public:
  struct stats_t
  {
    scalar_type pvalue = scalar_type();
    scalar_type beta = scalar_type();
    scalar_type se = scalar_type();
    scalar_type t = scalar_type();
    scalar_type r2 = scalar_type();

    static std::string header_column_names() { return "pvalue\tbeta\tse\ttstat\tr2"; }
    stats_t() {}
  };

  template <typename T>
  class variable_stats
  {
  private:
    T sum_ = T(0);
    T sum_squared_ = T(0);
  public:
    T sum() const { return sum_; }
    T sum_squared() const { return sum_squared_; }

    variable_stats() = default;

    variable_stats(T s, T ss) : sum_(s), sum_squared_(ss)
    {
    }

    template <typename VT>
    variable_stats(const std::vector<VT>& vec)
    {
      sum_ = std::accumulate(vec.begin(),  vec.end(), T());
      sum_squared_ = std::inner_product(vec.begin(),  vec.end(), vec.begin(), T());
    }

    template <typename VT>
    variable_stats(const xt::xtensor<VT, 1>& vec)
    {
      sum_ = std::accumulate(vec.begin(),  vec.end(), T());
      sum_squared_ = std::inner_product(vec.begin(),  vec.end(), vec.begin(), T());
    }

    template <typename VT>
    variable_stats(const savvy::compressed_vector<VT>& vec)
    {
      bool center = false;
      sum_ = std::accumulate(vec.begin(),  vec.end(), T());
      sum_squared_ = std::inner_product(vec.begin(),  vec.end(), vec.begin(), T());
      if (center)
      {
        T mean = sum_ / vec.size();
        sum_squared_ += (T(-2.) * mean * sum_) + (mean * mean * vec.size()); // (x - m)(x - m) == (xx - 2mx + mm)
        sum_ = T(0);
      }
    }
  };

  /*
    ## calculate the weights for each observation
    v <- exp(reml$linear.predictors) / (1 + exp(reml$linear.predictors))^2
    vx <- v * reml$x
    v2 <- t(vx) %*% reml$x
    iv2 <- solve(v2)
    iv2c <- chol(iv2) ## make t(iv2c) %*% iv2c = iv2
    x.res <- pheno - reml$fitted  # n * 1 matrix
    */


  linear_model(const res_t& y, const cov_t& x_orig, bool invnorm)
  {
    using namespace xt;
    using namespace xt::linalg;

    cov_t x = concatenate(xtuple(xt::ones<scalar_type>({y.size(), std::size_t(1)}), x_orig), 1);
    auto pbetas = dot(dot(pinv(dot(transpose(x), x)), transpose(x)), y);

    residuals_ = y - dot(x, pbetas);
    if (invnorm)
      inverse_normalize(residuals_);
    s_y_ = sum(residuals_)();
    s_yy_ = dot(residuals_, residuals_)();
  }

  virtual ~linear_model() {}

  std::size_t sample_size() const { return residuals_.size(); }

  template <typename GenoT>
  stats_t test_single(const std::vector<GenoT>& x, scalar_type s_x) const
  {
    const res_t& y = residuals_;

    assert(y.size() == x.size());
    const std::size_t n = x.size();
    //scalar_type s_x{}; //     = std::accumulate(x.begin(), x.end(), scalar_type());
    scalar_type s_xx{}; //    = std::inner_product(x.begin(), x.end(), x.begin(), scalar_type());
    scalar_type s_xy{}; //    = std::inner_product(x.begin(), x.end(), y.begin(), scalar_type());

    for (std::size_t i = 0; i < n; ++i)
    {
      //s_x += x[i];
      s_xx += x[i] * x[i];
      s_xy += x[i] * y[i];
    }

    const scalar_type m = (n * s_xy - s_x * s_y_) / (n * s_xx - s_x * s_x);
    const scalar_type b = (s_y_ - m * s_x) / n;
    auto fx             = [m,b](scalar_type x) { return m * x + b; };
    const scalar_type x_mean  = s_x / n;

//    scalar_type se_line{};
//    scalar_type se_x_mean{};
//    for (std::size_t i = 0; i < n; ++i)
//    {
//      se_line += square(y[i] - fx(x[i]));
//      se_x_mean += square(x[i] - x_mean);
//    }

    const scalar_type dof     = n - 2;
    //const scalar_type std_err = std::sqrt(se_line / dof) / std::sqrt(se_x_mean);
    const scalar_type std_err = std::sqrt((n * s_yy_ - s_y_ * s_y_ - m * m * (n * s_xx - s_x * s_x)) / ((n-2) * (n * s_xx - s_x * s_x)));
    scalar_type t = m / std_err;
    scalar_type r = (n * s_xy - s_x * s_y_) / std::sqrt((n * s_xx - s_x * s_x) * (n * s_yy_ - s_y_ * s_y_ ));


    boost::math::students_t_distribution<scalar_type> dist(dof);

    stats_t ret;
    ret.pvalue = boost::math::cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    ret.beta = m;
    ret.se = std_err;
    ret.t = t;
    ret.r2 = r * r; //1 - se_line / se_y_mean;

    return ret;
  }

  template <typename GenoT>
  stats_t test_single(const savvy::compressed_vector<GenoT>& x, scalar_type s_x) const
  {
    const res_t& y = residuals_;
    assert(y.size() == x.size());
    const std::size_t n = x.size();
    //scalar_type s_x{}; //     = std::accumulate(x.begin(), x.end(), scalar_type());
    scalar_type s_xx{}; //    = std::inner_product(x.begin(), x.end(), x.begin(), scalar_type());
    scalar_type s_xy{}; //    = x.dot(y, scalar_type());

    const auto x_beg = x.begin();
    const auto x_end = x.end();
    for (auto it = x_beg; it != x_end; ++it)
    {
      //s_x += *it;
      s_xx += (*it) * (*it);
      s_xy += (*it) * y[it.offset()];
    }

    //const float s_y     = std::accumulate(y.begin(), y.end(), 0.0f);
    const scalar_type m       = (n * s_xy - s_x * s_y_) / (n * s_xx - s_x * s_x);
    const scalar_type x_mean  = s_x / n;

//    scalar_type se_x_mean{};
//    for (auto it = x.begin(); it != x.end(); ++it)
//    {
//      se_x_mean += square(*it - x_mean);
//    }
//    se_x_mean += (square(0.0f - x_mean) * scalar_type(n - x.non_zero_size()));

    //scalar_type se2 = 1./(n*(n-2)) * (n*s_yy_ - s_y_*s_y_ - square(m)*(n*s_xx - square(s_x)));
    scalar_type r = (n * s_xy - s_x * s_y_) / std::sqrt((n * s_xx - s_x * s_x) * (n * s_yy_ - s_y_ * s_y_ ));

    const scalar_type dof = n - 2;
    //const scalar_type std_err_old = std::sqrt(se2) / std::sqrt(se_x_mean);
    const scalar_type std_err = std::sqrt((n * s_yy_ - s_y_ * s_y_ - m * m * (n * s_xx - s_x * s_x)) / ((n-2) * (n * s_xx - s_x * s_x)));
    scalar_type t = m / std_err;

    boost::math::students_t_distribution<scalar_type> dist(dof);

    stats_t ret;
    ret.pvalue = boost::math::cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    ret.beta = m;
    ret.se = std_err;
    ret.t = t;
    ret.r2 = r * r;

    return ret;
  }

  static stats_t ols(std::size_t n, scalar_type s_xy, const variable_stats<scalar_type>& x_stats, const variable_stats<scalar_type>& y_stats, std::size_t dof)
  {
    scalar_type s_x = x_stats.sum();
    scalar_type s_xx = x_stats.sum_squared();
    scalar_type s_y = y_stats.sum();
    scalar_type s_yy = y_stats.sum_squared();

    const scalar_type m       = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);

    scalar_type r = (n * s_xy - s_x * s_y) / std::sqrt((n * s_xx - s_x * s_x) * (n * s_yy - s_y * s_y));

    const scalar_type std_err = std::sqrt((n * s_yy - s_y * s_y - m * m * (n * s_xx - s_x * s_x)) / ((n-2) * (n * s_xx - s_x * s_x)));
    scalar_type t = m / std_err;

    boost::math::students_t_distribution<scalar_type> dist(dof);

    stats_t ret;
    ret.pvalue = boost::math::cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    ret.beta = m;
    ret.se = std_err;
    ret.t = t;
    ret.r2 = r * r;

    return ret;
  }

  template <typename GenoT>
  static stats_t ols(const std::vector<GenoT>& x, const res_t& y, scalar_type s_x, scalar_type s_y, scalar_type s_yy, std::size_t dof)
  {
    assert(y.size() == x.size());
    const std::size_t n = x.size();
    //scalar_type s_x{}; //     = std::accumulate(x.begin(), x.end(), scalar_type());
    scalar_type s_xx{}; //    = std::inner_product(x.begin(), x.end(), x.begin(), scalar_type());
    scalar_type s_xy{}; //    = std::inner_product(x.begin(), x.end(), y.begin(), scalar_type());

    for (std::size_t i = 0; i < n; ++i)
    {
      //s_x += x[i];
      s_xx += x[i] * x[i];
      s_xy += x[i] * y[i];
    }

    const scalar_type m = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
    const scalar_type b = (s_y - m * s_x) / n;
    auto fx             = [m,b](scalar_type x) { return m * x + b; };
    const scalar_type x_mean  = s_x / n;

    //    scalar_type se_line{};
    //    scalar_type se_x_mean{};
    //    for (std::size_t i = 0; i < n; ++i)
    //    {
    //      se_line += square(y[i] - fx(x[i]));
    //      se_x_mean += square(x[i] - x_mean);
    //    }

    //const scalar_type dof     = n - 2;
    //const scalar_type std_err = std::sqrt(se_line / dof) / std::sqrt(se_x_mean);
    const scalar_type std_err = std::sqrt((n * s_yy - s_y * s_y - m * m * (n * s_xx - s_x * s_x)) / ((n-2) * (n * s_xx - s_x * s_x)));
    scalar_type t = m / std_err;
    scalar_type r = (n * s_xy - s_x * s_y) / std::sqrt((n * s_xx - s_x * s_x) * (n * s_yy - s_y * s_y));


    boost::math::students_t_distribution<scalar_type> dist(dof);

    stats_t ret;
    ret.pvalue = boost::math::cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    ret.beta = m;
    ret.se = std_err;
    ret.t = t;
    ret.r2 = r * r; //1 - se_line / se_y_mean;

    return ret;
  }

  template <typename GenoT>
  static stats_t ols(const savvy::compressed_vector<GenoT>& x, const res_t& y, scalar_type s_x, scalar_type s_y, scalar_type s_yy, std::size_t dof)
  {
    //const res_t& y = residuals_;
    assert(y.size() == x.size());
    const std::size_t n = x.size();
    const scalar_type x_mean = s_x / n;
    // -2.0 * mean * scale * s_x * scale + square(mean * scale) * x.size();
    scalar_type s_xx{};
    scalar_type s_xy{};

    const auto x_beg = x.begin();
    const auto x_end = x.end();
    for (auto it = x_beg; it != x_end; ++it)
    {
      s_xx += (*it) * (*it);
      s_xy += (*it) * y[it.offset()];
    }

    //const float s_y     = std::accumulate(y.begin(), y.end(), 0.0f);
    const scalar_type m       = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);

    //    scalar_type se_x_mean{};
    //    for (auto it = x.begin(); it != x.end(); ++it)
    //    {
    //      se_x_mean += square(*it - x_mean);
    //    }
    //    se_x_mean += (square(0.0f - x_mean) * scalar_type(n - x.non_zero_size()));

    //scalar_type se2 = 1./(n*(n-2)) * (n*s_yy_ - s_y_*s_y_ - square(m)*(n*s_xx - square(s_x)));
    scalar_type r = (n * s_xy - s_x * s_y) / std::sqrt((n * s_xx - s_x * s_x) * (n * s_yy - s_y * s_y));

    //const scalar_type dof = n - 2;
    //const scalar_type std_err_old = std::sqrt(se2) / std::sqrt(se_x_mean);
    const scalar_type std_err = std::sqrt((n * s_yy - s_y * s_y - m * m * (n * s_xx - s_x * s_x)) / ((n-2) * (n * s_xx - s_x * s_x)));
    scalar_type t = m / std_err;

    boost::math::students_t_distribution<scalar_type> dist(dof);

    stats_t ret;
    ret.pvalue = boost::math::cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    ret.beta = m;
    ret.se = std_err;
    ret.t = t;
    ret.r2 = r * r;

    return ret;
  }

  template <typename GenoT>
  static stats_t ols(const savvy::compressed_vector<GenoT>& x, const res_t& y, const variable_stats<scalar_type>& x_stats, const variable_stats<scalar_type>& y_stats, std::size_t dof)
  {
    scalar_type s_x = x_stats.sum();
    scalar_type s_xx = x_stats.sum_squared();
    scalar_type s_y = y_stats.sum();
    scalar_type s_yy = y_stats.sum_squared();

    assert(y.size() == x.size());
    const std::size_t n = x.size();
    //const scalar_type x_mean = s_x / n;

    scalar_type s_xy{};
    for (auto it = x.begin(); it != x.end(); ++it)
      s_xy += (*it) * y[it.offset()];

    //const float s_y     = std::accumulate(y.begin(), y.end(), 0.0f);
    const scalar_type m       = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);

    //    scalar_type se_x_mean{};
    //    for (auto it = x.begin(); it != x.end(); ++it)
    //    {
    //      se_x_mean += square(*it - x_mean);
    //    }
    //    se_x_mean += (square(0.0f - x_mean) * scalar_type(n - x.non_zero_size()));

    //scalar_type se2 = 1./(n*(n-2)) * (n*s_yy_ - s_y_*s_y_ - square(m)*(n*s_xx - square(s_x)));
    scalar_type r = (n * s_xy - s_x * s_y) / std::sqrt((n * s_xx - s_x * s_x) * (n * s_yy - s_y * s_y));

    //const scalar_type dof = n - 2;
    //const scalar_type std_err_old = std::sqrt(se2) / std::sqrt(se_x_mean);
    const scalar_type std_err = std::sqrt((n * s_yy - s_y * s_y - m * m * (n * s_xx - s_x * s_x)) / ((n-2) * (n * s_xx - s_x * s_x)));
    scalar_type t = m / std_err;

    boost::math::students_t_distribution<scalar_type> dist(dof);

    stats_t ret;
    ret.pvalue = boost::math::cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    ret.beta = m;
    ret.se = std_err;
    ret.t = t;
    ret.r2 = r * r;

    return ret;
  }

  template <typename GenoT>
  static stats_t ols(const savvy::compressed_vector<GenoT>& x, const std::vector<scalar_type>& y, const variable_stats<scalar_type>& x_stats, const variable_stats<scalar_type>& y_stats, std::size_t dof)
  {
    scalar_type s_x = x_stats.sum();
    scalar_type s_xx = x_stats.sum_squared();
    scalar_type s_y = y_stats.sum();
    scalar_type s_yy = y_stats.sum_squared();

    assert(y.size() == x.size());
    const std::size_t n = x.size();
    //const scalar_type x_mean = s_x / n;

    scalar_type s_xy{};
    for (auto it = x.begin(); it != x.end(); ++it)
      s_xy += (*it) * y[it.offset()];

    //const float s_y     = std::accumulate(y.begin(), y.end(), 0.0f);
    const scalar_type m       = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);

    //    scalar_type se_x_mean{};
    //    for (auto it = x.begin(); it != x.end(); ++it)
    //    {
    //      se_x_mean += square(*it - x_mean);
    //    }
    //    se_x_mean += (square(0.0f - x_mean) * scalar_type(n - x.non_zero_size()));

    //scalar_type se2 = 1./(n*(n-2)) * (n*s_yy_ - s_y_*s_y_ - square(m)*(n*s_xx - square(s_x)));
    scalar_type r = (n * s_xy - s_x * s_y) / std::sqrt((n * s_xx - s_x * s_x) * (n * s_yy - s_y * s_y));

    //const scalar_type dof = n - 2;
    //const scalar_type std_err_old = std::sqrt(se2) / std::sqrt(se_x_mean);
    const scalar_type std_err = std::sqrt((n * s_yy - s_y * s_y - m * m * (n * s_xx - s_x * s_x)) / ((n-2) * (n * s_xx - s_x * s_x)));
    scalar_type t = m / std_err;

    boost::math::students_t_distribution<scalar_type> dist(dof);

    stats_t ret;
    ret.pvalue = boost::math::cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    ret.beta = m;
    ret.se = std_err;
    ret.t = t;
    ret.r2 = r * r;

    return ret;
  }

  template <typename GenoT>
  static stats_t ols(const std::vector<GenoT>& x, const res_t& y, const variable_stats<scalar_type>& x_stats, const variable_stats<scalar_type>& y_stats, std::size_t dof)
  {
    assert(y.size() == x.size());
    const std::size_t n = x.size();
    scalar_type s_x = x_stats.sum();
    scalar_type s_xx = x_stats.sum_squared();
    scalar_type s_y = y_stats.sum();
    scalar_type s_yy = y_stats.sum_squared();
    scalar_type s_xy{}; //    = std::inner_product(x.begin(), x.end(), y.begin(), scalar_type());

    for (std::size_t i = 0; i < n; ++i)
      s_xy += x[i] * y[i];

    const scalar_type m = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
    const scalar_type b = (s_y - m * s_x) / n;
    //auto fx             = [m,b](scalar_type x) { return m * x + b; };
    //const scalar_type x_mean  = s_x / n;

    //    scalar_type se_line{};
    //    scalar_type se_x_mean{};
    //    for (std::size_t i = 0; i < n; ++i)
    //    {
    //      se_line += square(y[i] - fx(x[i]));
    //      se_x_mean += square(x[i] - x_mean);
    //    }

    //const scalar_type dof     = n - 2;
    //const scalar_type std_err = std::sqrt(se_line / dof) / std::sqrt(se_x_mean);
    const scalar_type std_err = std::sqrt((n * s_yy - s_y * s_y - m * m * (n * s_xx - s_x * s_x)) / ((n-2) * (n * s_xx - s_x * s_x)));
    scalar_type t = m / std_err;
    scalar_type r = (n * s_xy - s_x * s_y) / std::sqrt((n * s_xx - s_x * s_x) * (n * s_yy - s_y * s_y));


    boost::math::students_t_distribution<scalar_type> dist(dof);

    stats_t ret;
    ret.pvalue = boost::math::cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    ret.beta = m;
    ret.se = std_err;
    ret.t = t;
    ret.r2 = r * r; //1 - se_line / se_y_mean;

    return ret;
  }

  template <typename GenoT>
  static stats_t ols(const std::vector<GenoT>& x, const std::vector<scalar_type>& y, const variable_stats<scalar_type>& x_stats, const variable_stats<scalar_type>& y_stats, std::size_t dof)
  {
    assert(y.size() == x.size());
    const std::size_t n = x.size();
    scalar_type s_x = x_stats.sum();
    scalar_type s_xx = x_stats.sum_squared();
    scalar_type s_y = y_stats.sum();
    scalar_type s_yy = y_stats.sum_squared();
    scalar_type s_xy{}; //    = std::inner_product(x.begin(), x.end(), y.begin(), scalar_type());

    for (std::size_t i = 0; i < n; ++i)
      s_xy += x[i] * y[i];

    const scalar_type m = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
    const scalar_type b = (s_y - m * s_x) / n;
    //auto fx             = [m,b](scalar_type x) { return m * x + b; };
    //const scalar_type x_mean  = s_x / n;

    //    scalar_type se_line{};
    //    scalar_type se_x_mean{};
    //    for (std::size_t i = 0; i < n; ++i)
    //    {
    //      se_line += square(y[i] - fx(x[i]));
    //      se_x_mean += square(x[i] - x_mean);
    //    }

    //const scalar_type dof     = n - 2;
    //const scalar_type std_err = std::sqrt(se_line / dof) / std::sqrt(se_x_mean);
    const scalar_type std_err = std::sqrt((n * s_yy - s_y * s_y - m * m * (n * s_xx - s_x * s_x)) / ((n-2) * (n * s_xx - s_x * s_x)));
    scalar_type t = m / std_err;
    scalar_type r = (n * s_xy - s_x * s_y) / std::sqrt((n * s_xx - s_x * s_x) * (n * s_yy - s_y * s_y));


    boost::math::students_t_distribution<scalar_type> dist(dof);

    stats_t ret;
    ret.pvalue = boost::math::cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    ret.beta = m;
    ret.se = std_err;
    ret.t = t;
    ret.r2 = r * r; //1 - se_line / se_y_mean;

    return ret;
  }
private:
  template <typename T>
  static T square(const T& v) { return v * v; }
};


static std::ostream& operator<<(std::ostream& os, const linear_model& v)
{
  os << "pvalue\tbeta\tse\ttstat\tr2";
  return os;
}

static std::ostream& operator<<(std::ostream& os, const typename linear_model::stats_t& v)
{
  os << v.pvalue
    << "\t" << v.beta
    << "\t" << v.se
    << "\t" << v.t
    << "\t" << v.r2;
  return os;
}


#endif //SAVANT_LINEAR_MODEL_HPP
