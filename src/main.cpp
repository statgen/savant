/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "single.hpp"
#include "plot.hpp"
#include "pca.hpp"
#include "grm.hpp"
#include "tcdf.hpp"
#include "logistic_score_model.hpp"
#include "linear_model.hpp"

#include <savvy/reader.hpp>

#include <iostream>

//#ifndef __cpp_lib_as_const
//#define __cpp_lib_as_const
//#endif
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Sparse>

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <chrono>
#include <getopt.h>
#include <tuple>
#include <complex>

std::ofstream debug_log;

int test_xtensor()
{
  using namespace xt;
  using namespace xt::linalg;

  xt::xarray<double> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  xt::xtensor<double, 1> y = {0., 1., 1., 0., 0.};
  xt::xtensor<double, 2> x = {
    {0.05, 0.12},
    {0.18, 0.22},
    {0.31, 0.35},
    {0.42, 0.38},
    {0.5, 0.49}};

  std::cerr << y.shape()[0] << std::endl;
  std::cerr << xt::transpose(y).shape()[0] << std::endl;
  std::cerr << x << std::endl;
  x = xt::concatenate(xt::xtuple(xt::ones<double>({y.size(),std::size_t(1)}), x), 1);
  std::cerr << x << std::endl;
  std::cerr << xt::adapt(y.shape()) << std::endl;
  std::cerr << y << std::endl;
  std::cerr << xt::adapt(xt::col(x, 0).shape()) << std::endl;
  std::cerr << xt::col(x, 0) << std::endl;


  std::cerr << xt::linalg::dot(xt::transpose(x), x) << std::endl;


  //B' = (X!X)−1X!y
  //b = inv(X.T.dot(X)).dot(X.T).dot(y)
  // b = (x'x)-1 X
  xt::xarray<double> betas = dot(dot(inv(dot(transpose(x), x)), transpose(x)), y);
  xt::xarray<double> pbetas = dot(dot(pinv(dot(transpose(x), x)), transpose(x)), y);
  std::cerr << betas << std::endl;
  std::cerr << pbetas << std::endl;
  xt::xarray<double> residuals = y - dot(x, betas);
  std::cerr << residuals << std::endl;
  double sos_res = 0.;
  for (auto e : residuals)
    sos_res += e*e;

  double psos_res = 0.;
  for (auto e : (y - dot(x, pbetas)))
    psos_res += e*e;

  {
    //auto [solution, sos_residuals, rank, s] = lstsq(x, y); // xt::transpose(y));
    xarray<double> solution, sos_residuals, rank, s;
    std::tie(solution, sos_residuals, rank, s) = lstsq(x, y);
    double sos_res2 = sos_residuals[0];

    std::cerr << solution << std::endl;
    std::cerr << sos_residuals << std::endl;
    std::cerr << rank << std::endl;
    std::cerr << s << std::endl;
  }
  return 0;
}

typedef double scalar_type;
//typedef xt::xtensor<scalar_type, 1> residuals_type;
typedef xt::xarray<scalar_type> residuals_type;

template <typename T, typename T2>
residuals_type compute_residuals(const T& y, const T2& x_orig)
{
  using namespace xt;
  using namespace xt::linalg;
  T2 x = concatenate(xtuple(xt::ones<scalar_type>({y.size(), std::size_t(1)}), x_orig), 1);
//  auto a = dot(transpose(x), x);
//  std::cerr << a << std::endl;
//  auto b = pinv(a);
//  std::cerr << "END A ------------------------------" << std::endl;
//  std::cerr << b << std::endl;
//  std::cerr << "END B ------------------------------" << std::endl;
//  auto c = dot(b, transpose(x));
//  std::cerr << c << std::endl;
  auto pbetas = dot(dot(pinv(dot(transpose(x), x)), transpose(x)), y);
  std::cerr << pbetas << std::endl;
  residuals_type residuals = y - dot(x, pbetas);
  std::cerr << "sum(y): " << sum(y) << std::endl;
  return residuals;
}

template <typename T, typename T2>
residuals_type compute_residuals_logit(const T& y, const T2& x_orig)
{
  using namespace xt;
  using namespace xt::linalg;
  const scalar_type tolerance = 0.00001;

  // ==== Fit Model ==== //
//  xarray<double> x = {53,57,58,63,66,67,67,67,68,69,70,70,70,70,72,73,75,75,76,76,78,79,81};
//  xtensor<double, 1> y = {1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0};
//  assert(y.size() == x.size());
  T2 x = xt::concatenate(xt::xtuple(xt::ones<double>({y.size(), std::size_t(1)}), x_orig), 1);
  xarray<double> x_transpose_copy; // We use this later for the result of (X^t)W. W is too large (n x n), so we populate the result without generating W.

  xtensor<double, 1> beta = dot(dot(pinv(dot(transpose(x), x)), transpose(x)), y);

  std::size_t n_iter = 8;
  for (std::size_t i = 0; i < n_iter; ++i)
  {
//    xarray<double> z = dot(x, beta);
//    std::cerr << z << std::endl;
//    xarray<double> p = 1. / (1. + xt::exp(-z));

    xarray<double> p = 1. / (1. + xt::exp(-dot(x, beta)));


    xarray<double> F = dot(transpose(x), y - p);

    x_transpose_copy = transpose(x);
    for (std::size_t i = 0; i < p.size(); ++i)
    {
      xt::col(x_transpose_copy, i) *= p(i) * (1. - p(i));
    }
    xtensor<double, 2> I = dot(x_transpose_copy, x);
    beta = beta + dot(pinv(I), F);
    std::cerr << beta << std::endl;
  }
  // =================== //



  auto xw = dot(x, beta);
  T y_hat = 1. / (1. + xt::exp(-xw));
  residuals_type residuals = y - y_hat;
  scalar_type se = xt::mean(residuals * residuals)();

  std::cerr << "sum(y): " << sum(y) << std::endl;
  return residuals;
}

template <typename T>
T square(T v)
{
  return v * v;
}
#if 0
auto linreg_ttest_old(const std::vector<float>& y, const std::vector<float>& x)
{
  const std::size_t n = x.size();
  const float s_x     = std::accumulate(x.begin(), x.end(), 0.0f);
  const float s_y     = std::accumulate(y.begin(), y.end(), 0.0f);
  const float s_xx    = std::inner_product(x.begin(), x.end(), x.begin(), 0.0f);
  const float s_xy    = std::inner_product(x.begin(), x.end(), y.begin(), 0.0f);
  const float m       = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
  const float b       = (s_y - m * s_x) / n;
  auto fx             = [m,b](float x) { return m * x + b; };
  float se_line       = 0.0f; for (std::size_t i = 0; i < n; ++i) se_line += square(y[i] - fx(x[i]));
  const float x_mean  = s_x / n;
  float se_x_mean     = 0.0f; for (std::size_t i = 0; i < n; ++i) se_x_mean += square(x[i] - x_mean);
  const float dof     = n - 2;
  const float std_err = std::sqrt(se_line / dof) / std::sqrt(se_x_mean);
  float t = m / std_err;
  //boost::math::students_t_distribution<float> dist(dof);
  float pval =  tcdf(t, n - 1); //cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
  return std::make_tuple(m, std_err, t, pval); // slope, std error, t statistic, p value
}
#endif
auto linreg_ttest(const std::vector<scalar_type>& y, const std::vector<scalar_type>& x, const scalar_type s_y)
{
  assert(y.size() == x.size());
  const std::size_t n = x.size();
  scalar_type s_x{}; //     = std::accumulate(x.begin(), x.end(), 0.0f);
  //scalar_type s_y{}; //     = std::accumulate(y.begin(), y.end(), 0.0f);
  scalar_type s_xx{}; //    = std::inner_product(x.begin(), x.end(), x.begin(), 0.0f);
  scalar_type s_xy{}; //    = std::inner_product(x.begin(), x.end(), y.begin(), 0.0f);

  for (std::size_t i = 0; i < n; ++i)
  {
    s_x += x[i];
    //s_y += y[i];
    s_xx += x[i] * x[i];
    s_xy += x[i] * y[i];
  }

  const scalar_type m       = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
  const scalar_type b       = (s_y - m * s_x) / n;
  auto fx             = [m,b](scalar_type x) { return m * x + b; };
  const scalar_type x_mean  = s_x / n;

  double se_line{};
  double se_x_mean{};
  for (std::size_t i = 0; i < n; ++i)
  {
    se_line += square(y[i] - fx(x[i]));
    se_x_mean += square(x[i] - x_mean);
  }

  const scalar_type dof     = n - 2;
  const scalar_type std_err = std::sqrt(se_line / dof) / std::sqrt(se_x_mean);
  scalar_type t = m / std_err;
  //boost::math::students_t_distribution<scalar-type> dist(dof);
  scalar_type pval =  tcdf(t, dof); //cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;

  return std::make_tuple(m, std_err, t, pval); // slope, std error, t statistic, p value
}
#if 0
auto sp_lin_reg_old(const std::vector<float>& y, const savvy::compressed_vector<float>& x)
{
  const std::size_t n = x.size();
  const float s_x     = std::accumulate(x.begin(), x.end(), 0.0f);
  const float s_y     = std::accumulate(y.begin(), y.end(), 0.0f);
  const float s_xx    = std::inner_product(x.begin(), x.end(), x.begin(), 0.0f);
  float s_xy    = 0.0f; for (auto it = x.begin(); it != x.end(); ++it) s_xy += (*it * y[x.index_data()[it - x.begin()]]);
  const float m       = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
  const float b       = (s_y - m * s_x) / n;
  auto fx             = [m,b](float x) { return m * x + b; };
  float se_line       = 0.0f; for (std::size_t i = 0; i < n; ++i) se_line += square(y[i] - fx(x[i]));
  const float x_mean  = s_x / n;
  float se_x_mean     = 0.0f; for (std::size_t i = 0; i < n; ++i) se_x_mean += square(x[i] - x_mean);
  const float dof     = n - 2;
  const float std_err = std::sqrt(se_line / dof) / std::sqrt(se_x_mean);
  float t = m / std_err;
  //std::students_t_distribution<float> dist(dof);
  float pval = tcdf(t, n - 1); //cdf(complement(dist, std::fabs(t))) * 2;
  /*
  beta = ((c+1)*sxy-sx*sy)/((c+1)*sxx-sx*sx);
  varE = 1/(c+1.)/(c-1.)*((c+1)*syy-sy*sy-beta*beta*((c+1)*sxx-sx*sx));
  sebeta = sqrt((c+1)*varE/((c+1)*sxx-sx*sx));
  r = ((c+1)*sxy-sx*sy)/sqrt(((c+1)*sxx-sx*sx)*((c+1)*syy-sy*sy));
  t = r * sqrt((c-1)/(1-r*r+pEmmaxHelper::ZEPS));
  pval = pEmmaxHelper::tcdf(t, c-1);
  */
  return std::make_tuple(m, std_err, t, pval); // slope, std error, t statistic, p value
}
#endif

auto linreg_ttest(const std::vector<scalar_type>& y, const savvy::compressed_vector<scalar_type>& x, const scalar_type& s_y, const scalar_type& s_yy)
{
  assert(y.size() == x.size());
  const std::size_t n = x.size();
  scalar_type s_x{}; //     = std::accumulate(x.begin(), x.end(), 0.0f);
  scalar_type s_xx{}; //    = std::inner_product(x.begin(), x.end(), x.begin(), 0.0f);
  scalar_type s_xy{}; //    = std::inner_product(x.begin(), x.end(), y.begin(), 0.0f);

  const auto x_beg = x.begin();
  const auto x_end = x.end();
//  const scalar_type* x_values = x.value_data();
//  const std::size_t* x_indices = x.index_data();
  for (auto it = x_beg; it != x_end; ++it)
  {
    s_x += *it;
    s_xx += (*it) * (*it);
    s_xy += (*it) * y[it.offset()];
  }

  //const float s_y     = std::accumulate(y.begin(), y.end(), 0.0f);
  const scalar_type m       = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
  const scalar_type x_mean  = s_x / n;

  float se_x_mean{};

  if (false)
  {
    const scalar_type b       = (s_y - m * s_x) / n;
    auto fx             = [m,b](scalar_type x) { return m * x + b; };
    const scalar_type f_of_zero = fx(0.0f);

    float se_line{};
    std::size_t i = 0;
    for (auto it = x.begin(); it != x.end(); ++i)
    {
      if (i == it.offset())
      {
        se_line += square(y[i] - fx(*it));
        se_x_mean += square(*it - x_mean);
        ++it;
      }
      else
      {
        se_line += square(y[i] - f_of_zero);
      }
    }

    for ( ; i < n; ++i)
    {
      se_line += square(y[i] - f_of_zero);
    }

    se_x_mean += (square(0.0f - x_mean) * scalar_type(n - x.non_zero_size()));

    const scalar_type dof = n - 2;
    const scalar_type std_err = std::sqrt(se_line / dof) / std::sqrt(se_x_mean);
    scalar_type t = m / std_err;
    //std::students_t_distribution<float> dist(dof);
    scalar_type pval = tcdf(t, dof); //cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    /*
    beta = ((c+1)*sxy-sx*sy)/((c+1)*sxx-sx*sx);
    varE = 1/(c+1.)/(c-1.)*((c+1)*syy-sy*sy-beta*beta*((c+1)*sxx-sx*sx));
    sebeta = sqrt((c+1)*varE/((c+1)*sxx-sx*sx));
    r = ((c+1)*sxy-sx*sy)/sqrt(((c+1)*sxx-sx*sx)*((c+1)*syy-sy*sy));
    t = r * sqrt((c-1)/(1-r*r+pEmmaxHelper::ZEPS));
    pval = pEmmaxHelper::tcdf(t, c-1);
    */

    return std::make_tuple(m, std_err, t, pval); // slope, std error, t statistic, p value
  }
  else
  {
    for (auto it = x.begin(); it != x.end(); ++it)
    {
      se_x_mean += square(*it - x_mean);
    }

    se_x_mean += (square(0.0f - x_mean) * scalar_type(n - x.non_zero_size()));
    double se2 = 1./(n*(n-2)) * (n*s_yy - s_y*s_y - square(m)*(n*s_xx - square(s_x)));

    const scalar_type dof = n - 2;
    const scalar_type std_err = std::sqrt(se2) / std::sqrt(se_x_mean);
    scalar_type t = m / std_err;
    //std::students_t_distribution<float> dist(dof);
    scalar_type pval = tcdf(t, dof); //cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;

    return std::make_tuple(m, std_err, t, pval); // slope, std error, t statistic, p value
  }
}

//void lin_reg(const residuals_type& y, const std::vector<scalar_type>& x)
//{
//}
//
//void lin_reg(const residuals_type& y, const savvy::compressed_vector<scalar_type>& x)
//{
//}



void slope_test()
{
  using namespace xt;
  using namespace xt::linalg;

  xarray<double> x = {1., 2., 3.};
  x.reshape({3,1});
  xtensor<double, 1> y = {3.1, 2.9, 3.2};
  //y.reshape({3,1});

  auto pbetas = dot(dot(pinv(dot(transpose(x), x)), transpose(x)), y);
  std::cerr << pbetas << std::endl;

  xtensor<double, 2> dmat = xt::concatenate(xtuple(xt::ones<double>({3, 1}), x), 1);
  std::cerr << dmat << std::endl;
  xarray<double> i = pinv(dot(transpose(dmat), dmat));
  auto pbetas2 = dot(dot(i, transpose(dmat)), y);
  std::cerr << pbetas2 << std::endl;

  //-----------------------//
  const std::size_t n = x.size();
  const double s_x     = std::accumulate(x.begin(), x.end(), 0.0);
  const double s_y     = std::accumulate(y.begin(), y.end(), 0.0);
  const double s_xx    = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
  const double s_xy    = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
  const double m       = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);

  const double b       = (s_y - m * s_x) / n;
  std::cerr << b << " , " << m << std::endl;
  auto fx              = [m,b](double x) { return m * x + b; };
  double se_line       = 0.0f; for (std::size_t i = 0; i < n; ++i) se_line += ::square(y[i] - fx(x[i]));
  const double x_mean  = s_x / n;
  double se_x_mean     = 0.0f; for (std::size_t i = 0; i < n; ++i) se_x_mean += ::square(x[i] - x_mean);
  const double dof     = n - 2;
  const double std_err = std::sqrt(se_line / dof) / std::sqrt(se_x_mean);
  float t = m / std_err;

}

void test()
{
  // Example from https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
  std::vector<double> x = {1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83};
//  std::vector<double> y = {52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46};
  std::vector<double> y = {52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 29.93, 61.29, 163.11, 164.47, 66.28, 68.10, 69.92, 72.19, 74.46};

  std::size_t n = x.size();
  double s_x = std::accumulate(x.begin(), x.end(), 0.);
  double s_y = std::accumulate(y.begin(), y.end(), 0.);
  double s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.);
  double s_yy = std::inner_product(y.begin(), y.end(), y.begin(), 0.);
  double s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.);
  std::cerr << s_x << " " << s_y << std::endl;
  std::cerr << s_xx << " " << s_yy << std::endl;
  std::cerr << s_xy << std::endl;

  double beta = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
  double alpha = (1./n) * s_y - beta * (1./n) * s_x;

  //se2 = se_line / dof
  double se2 = 1./(n*(n-2)) * (n*s_yy - s_y*s_y - beta*beta*(n*s_xx - s_x*s_x));
  double sbeta2 = n*se2 / (n*s_xx - s_x*s_x);
  double salpha2 = sbeta2*(1./n)*s_xx;

  double r = (n*s_xy - s_x*s_y) / std::sqrt((n*s_xx - s_x * s_x) * (n*s_yy - s_y * s_y));

  double m, std_err, t, pval;
  std::tie(m, std_err, t, pval) = linreg_ttest(y, x, s_y);

  return;
}

void challenger_test()
{
  using namespace xt;
  using namespace xt::linalg;

  xarray<double> x = {53,57,58,63,66,67,67,67,68,69,70,70,70,70,72,73,75,75,76,76,78,79,81};
  xtensor<double, 1> y = {1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0};
  assert(y.size() == x.size());
  x.reshape({y.size(), 1});
  x = xt::concatenate(xt::xtuple(xt::ones<double>({y.size(), std::size_t(1)}), x), 1);
  xarray<double> x_transpose_copy; // We use this later for the result of (X^t)W. W is too large (n x n), so we populate the result without generating W.

  xtensor<double, 1> beta = {2.90476190, -0.03738095}; // TODO: use linear model to produce initial betas
  //xtensor<double, 1> beta = dot(dot(pinv(dot(transpose(x), x)), transpose(x)), y);

  std::size_t n_iter = 8;
  for (std::size_t i = 0; i < n_iter; ++i)
  {
    //std::cerr << xt::transpose(beta) << std::endl;
    //std::cerr << x << std::endl;

    xarray<double> z = dot(x, beta);
    std::cerr << z << std::endl;
    xarray<double> p = 1. / (1. + xt::exp(-z));
    //std::cerr << p << std::endl;

    xarray<double> F = dot(transpose(x), y - p);
    //std::cerr << F << std::endl;
    x_transpose_copy = transpose(x);
    for (std::size_t i = 0; i < p.size(); ++i)
    {
      xt::col(x_transpose_copy, i) *= p(i) * (1. - p(i));
    }
    //std::cerr << transpose(x) << std::endl;
    //std::cerr << x_transpose_copy << std::endl;

    //xtensor<double, 2> I = -dot(x_transpose_copy, x);
    xtensor<double, 2> I = dot(x_transpose_copy, x);
    //std::cerr << I << std::endl;
    //std::cerr << pinv(I) << std::endl;

    beta = beta + dot(pinv(I), F);
    std::cerr << beta << std::endl;
  }

  auto a = 0;
}

template <typename T>
auto simultaneous_power_iteration(const T& X, std::size_t num_pcs = 0, std::size_t n_simulations = 8)
{
  using namespace xt;
  using namespace xt::linalg;

  if (num_pcs == 0)
    num_pcs = X.shape(1);

  xarray<double> Q = random::rand<double>({X.shape(1), num_pcs});
  xarray<double> R;
//  std::cerr << "Q: " << Q << std::endl;
//  std::cerr << "R: " << R << std::endl;
  std::tie(Q, R) = qr(Q);
  auto Q_prev = Q;

//  std::cerr << "Q: " << Q << std::endl;
//  std::cerr << "R: " << R << std::endl;

  for (std::size_t i = 0; i < n_simulations; ++i)
  {
    auto Z = dot(X, Q);
    std::tie(Q, R) = qr(Z);

    auto delta = Q - Q_prev;
    auto err = sum(delta * delta);
    std::cerr << "err: " << err << std::endl;
    Q_prev = Q;
  }

  std::cerr << "Q: " << Q << std::endl;
  std::cerr << "R: " << R << std::endl;

  return std::make_tuple(xt::eval(xt::diagonal(R)), Q);
}

template <typename T>
auto simultaneous_power_iteration_matrix_free(const T& X, std::size_t num_pcs = 0, std::size_t n_simulations = 8)
{
  using namespace xt;
  using namespace xt::linalg;

  if (num_pcs == 0)
    num_pcs = X.shape(1);

  xarray<double> Q = random::rand<double>({X.shape(1), num_pcs});
  xarray<double> R;
//  std::cerr << "Q: " << Q << std::endl;
//  std::cerr << "R: " << R << std::endl;
  std::tie(Q, R) = qr(Q);
  xarray<double> Q_prev = Q;
//  std::cerr << "Q: " << Q << std::endl;
//  std::cerr << "R: " << R << std::endl;
  //xt::xarray<double> r = xt::random::rand<double>({X.shape(1)});
  //r = r / xt::linalg::norm(r);
  //std::cerr << r << std::endl;

  for (std::size_t i = 0; i < n_simulations; ++i)
  {
    xarray<double> S = xt::zeros<double>(Q.shape());
    //std::cerr << S << std::endl;

    for (std::size_t j = 0; j < X.shape(0); ++j)
    {
      auto x = xt::row(X, j);
      //std::cerr << "x: " << x << std::endl;
      //std::cerr << "transpose(x): " << transpose(x) << std::endl;
      //std::cerr << "xt::reshape_view(x, {x.size(), 1}): " << xt::reshape_view(x, {x.size(), std::size_t(1)}) << std::endl;
      //std::cerr << "dot(x, Q): " << dot(x, Q) << std::endl;
      //std::cerr << "xt::reshape_view(dot(x, Q), {std::size_t(1), x.size()}): " << xt::reshape_view(dot(x, Q), {std::size_t(1), x.size()}) << std::endl;
      //std::cerr << "dot(transpose(x), dot(x, Q)): " << dot(xt::reshape_view(x, {x.size(), std::size_t(1)}), xt::reshape_view(dot(x, Q), {std::size_t(1), x.size()})) << std::endl;
      S += dot(xt::reshape_view(x, {x.size(), std::size_t(1)}), xt::reshape_view(dot(x, Q), {std::size_t(1), x.size()}));
      //std::cerr << s << std::endl;
    }
    //std::cerr << S << std::endl;

    //eigenvalue = xt::linalg::dot(xt::transpose(r), s)();
    //std::cerr << eigenvalue << std::endl;
    std::tie(Q, R) = qr(S);

    //auto err = xt::linalg::norm(eigenvalue * r - s);
    auto delta = Q - Q_prev;
    auto err = sum(delta * delta);
    std::cerr << "err: " << err << std::endl;
    //std::cerr << "err: " << err << std::endl;

    Q_prev = Q;
    //std::cerr << "r: " << r << std::endl;
  }

  std::cerr << "R: " << R << std::endl;
  std::cerr << "diag(R): " << diagonal(R) << std::endl;
  std::cerr << "Q: " << Q << std::endl;

  return std::make_tuple(eval(diagonal(R)), Q);
}

template <typename T>
auto power_iteration(const T& X, std::size_t n_simulations = 8)
{
  double eigval;
  xt::xarray<double> b_k = xt::ones<double>({X.shape(1)}) / std::sqrt(X.shape(1));
  //xt::xarray<double> b_k = xt::random::rand<double>({X.shape(1)});
  //std::cerr << b_k << std::endl;

  for (std::size_t i = 0; i < n_simulations; ++i)
  {
    // calculate the matrix-by-vector product Ab
    auto b_k1 = xt::linalg::dot(X, b_k);
    //std::cerr << "b_k1: " << b_k1 << std::endl;

    // calculate the norm
    auto b_k1_norm = xt::linalg::norm(b_k1);
    //std::cerr << "b_k1_norm: " << b_k1_norm << std::endl;

    // re normalize the vector
    b_k = b_k1 / b_k1_norm;
    //std::cerr << b_k << std::endl;

    auto tmp = xt::linalg::dot(b_k, xt::linalg::dot(X, b_k));
    //std::cerr << "eigval: " << tmp << std::endl;
    eigval = tmp();
  }

  return std::make_tuple(eigval, b_k);
}

template <typename T>
auto power_iteration_matrix_free(const T& X, std::size_t n_simulations = 8)
{
  double eigenvalue = 0.;
  xt::xarray<double> r = xt::ones<double>({X.shape(1)}) / std::sqrt(X.shape(1));
  //xt::xarray<double> r = xt::random::rand<double>({X.shape(1)});
  //r = r / xt::linalg::norm(r);
  //std::cerr << r << std::endl;

  for (std::size_t i = 0; i < n_simulations; ++i)
  {
    xt::xtensor<double, 1> s = xt::zeros<double>(r.shape());
    //std::cerr << s << std::endl;

    for (std::size_t j = 0; j < X.shape(0); ++j)
    {
      auto x = xt::row(X, j);
      s += xt::linalg::dot(x, r) * x;
      //std::cerr << s << std::endl;
    }

    eigenvalue = xt::linalg::dot(xt::transpose(r), s)();
    //std::cerr << eigenvalue << std::endl;

    auto err = xt::linalg::norm(eigenvalue * r - s);
    //std::cerr << "err: " << err << std::endl;

    r = s / xt::linalg::norm(s);
    //std::cerr << "r: " << r << std::endl;
  }

  return std::make_tuple(eigenvalue, r);
}

template <typename T>
auto nipals(const T& X, std::size_t n_simulations = 8)
{
  using namespace xt;
  using namespace xt::linalg;

  double eigenvalue = 0.;
  xt::xarray<double> t = xt::ones<double>({X.shape(0)}) / std::sqrt(X.shape(0));
  xt::xarray<double> r = xt::ones<double>({X.shape(1)}) / std::sqrt(X.shape(1));

  for (std::size_t i = 0; i < n_simulations; ++i)
  {
    t = dot(X, r);
    auto s = dot(transpose(X), t);
    //std::cerr << "s: " << s << std::endl;

    eigenvalue = xt::linalg::dot(xt::transpose(r), s)();
    //std::cerr << eigenvalue << std::endl;

    auto err = xt::linalg::norm(eigenvalue * r - s);
    //std::cerr << "err: " << err << std::endl;

    r = s / xt::linalg::norm(s);
    //std::cerr << "r: " << r << std::endl;
  }

  return std::make_tuple(eigenvalue, r);
}

template <typename MatT>
auto simultaneous_nipals(const MatT& X, std::size_t num_pcs = 0, std::size_t n_simulations = 8)
{
  using namespace xt;
  using namespace xt::linalg;

  if (num_pcs == 0)
    num_pcs = X.shape(1);

  xarray<double> Q = random::rand<double>({X.shape(1), num_pcs});
  xarray<double> R;
//  std::cerr << "Q: " << Q << std::endl;
//  std::cerr << "R: " << R << std::endl;
  std::tie(Q, R) = qr(Q);
  xarray<double> Q_prev = Q;
  xarray<double> T = xarray<double>::from_shape({X.shape(0), num_pcs});
//  std::cerr << "Q: " << Q << std::endl;
//  std::cerr << "R: " << R << std::endl;

  for (std::size_t i = 0; i < n_simulations; ++i)
  {
    T = dot(X, Q);
    auto S = dot(transpose(X), T);
    //std::cerr << S << std::endl;


    //eigenvalue = xt::linalg::dot(xt::transpose(r), s)();
    //std::cerr << eigenvalue << std::endl;
    std::tie(Q, R) = qr(S);

    //auto err = xt::linalg::norm(eigenvalue * r - s);
    auto delta = Q - Q_prev;
    auto err = sum(delta * delta);
    std::cerr << "err: " << err << std::endl;
    //std::cerr << "err: " << err << std::endl;

    Q_prev = Q;
    //std::cerr << "r: " << r << std::endl;
  }

  std::cerr << "R: " << R << std::endl;
  std::cerr << "diag(R): " << diagonal(R) << std::endl;
  std::cerr << "Q: " << Q << std::endl;

  return std::make_tuple(eval(diagonal(R)), Q);
}


int pca_test()
{
  using namespace xt;
  using namespace xt::linalg;

  xt::xtensor<double, 2> X = {{ 3.7,  -7.833333,     -9.5, -11.383333},
                              { 0.5,  19.166667,    -19.5,  11.916667},
                              {-1.4,  50.166667,     12.5,  -1.583333},
                              {-0.7, -53.833333,    -17.5, -13.083333},
                              {-0.5,  32.166667,     23.5,   8.016667},
                              {-1.6, -39.833333,     10.5,   6.116}};



  xt::xarray<std::complex<double>> cplx_evals, cplx_evecs;
  std::tie(cplx_evals, cplx_evecs) = xt::linalg::eig(xt::linalg::dot(xt::transpose(X), X));
  std::cerr << cplx_evals << std::endl;
  std::cerr << cplx_evecs << std::endl;

  xt::xarray<double> evals, evecs;
  xt::xarray<double> r = xt::random::rand<double>({X.shape(1)});

  auto xcov = xt::linalg::dot(xt::transpose(X), X);

  auto discard = simultaneous_power_iteration(xcov);
  std::cerr << "ev: " << std::get<0>(discard) << std::endl;
  auto discard2 = simultaneous_power_iteration_matrix_free(X);
  std::cerr << "ev: " << std::get<0>(discard2) << std::endl;
  auto discard3 = simultaneous_nipals(X);
  std::cerr << "ev: " << std::get<0>(discard3) << std::endl;

  auto P = xt::xtensor<double, 2>::from_shape({X.shape(1), 0});
  double eval;
  xt::xarray<double> evec;

  xt::xtensor<double, 2> A0 = X;
  for (std::size_t i = 0; i < A0.shape(1); ++i)
  {
    std::tie(eval, evec) = nipals(A0);
    std::cerr << "eval: " << eval << std::endl;
    evec.reshape({evec.size(), 1});
    P = xt::concatenate(xt::xtuple(P, evec), 1);

//    auto score_vec = xt::xtensor<double, 1>::from_shape({A.shape(0)});
//    for (std::size_t j = 0; j < A.shape(0); ++j)
//    {
//      score_vec[j] = dot(xt::row(A, j), evec)();
//    }

    for (std::size_t j = 0; j < A0.shape(0); ++j)
    {
      xt::row(A0, j) -= row(dot(xt::row(A0, j), evec)() * transpose(evec), 0);
    }
  }
  std::cerr << P << std::endl;

  P = xt::xtensor<double, 2>::from_shape({X.shape(1), 0});
  for (std::size_t i = 0; i < X.shape(1); ++i)
  {
    std::tie(eval, evec) = power_iteration(xcov);
    evec.reshape({evec.size(), 1});
    P = xt::concatenate(xt::xtuple(P, evec), 1);
    xcov -= eval * xt::linalg::dot(evec, xt::transpose(evec));
  }
  std::cerr << P << std::endl;

  P = xt::xtensor<double, 2>::from_shape({X.shape(1), 0});
  xt::xtensor<double, 2> A = X;
  for (std::size_t i = 0; i < A.shape(1); ++i)
  {
    std::tie(eval, evec) = power_iteration_matrix_free(A);
    evec.reshape({evec.size(), 1});
    P = xt::concatenate(xt::xtuple(P, evec), 1);

//    auto score_vec = xt::xtensor<double, 1>::from_shape({A.shape(0)});
//    for (std::size_t j = 0; j < A.shape(0); ++j)
//    {
//      score_vec[j] = dot(xt::row(A, j), evec)();
//    }

    for (std::size_t j = 0; j < A.shape(0); ++j)
    {
      xt::row(A, j) -= row(dot(xt::row(A, j), evec)() * transpose(evec), 0);
    }
  }
  std::cerr << P << std::endl;
  return 0;
}

int chol_test()
{
  using namespace xt;
  using namespace xt::linalg;

  {
    Eigen::MatrixXd FOO(3, 3);
    FOO <<
      0.0, 0.7, 9.0,
      0.7, 0.0, 1.0,
      9.0, 1.0, 0.0;
    std::cerr << "FOO L\n" << Eigen::MatrixXd(FOO.llt().matrixL()) << std::endl;
    std::cerr << "FOO L\n" << Eigen::MatrixXd(FOO.ldlt().matrixL()) << std::endl;
    std::cerr << "FOO D\n" << Eigen::VectorXd(FOO.ldlt().vectorD()).transpose() << std::endl;



    xtensor<double, 2> A = {
      {1.3, 0.7, 0.0},
      {0.7, 1.0, 0.0},
      {0.0, 0.0, 1.5},
    };

    auto L = eval(cholesky(A));
    std::cerr << "cholesky(A): " << L << std::endl;
    double p = 1.;
    for (std::size_t i = 0; i < L.shape(0); ++i) { p *= (L(i, i) * L(i, i)); }
    std::cerr << "p: " << p << std::endl;
    std::cerr << "log(p): " << std::log(p) << std::endl;

    double lp = 0.;
    for (std::size_t i = 0; i < L.shape(0); ++i) { lp += std::log(L(i, i) * L(i, i)); }
    std::cerr << "lp: " << lp << std::endl;
    std::cerr << "det(A): " << det(A) << std::endl;

    Eigen::MatrixXd A_eig(3, 3);
    A_eig << 1.3, 0.7, 0.0,
             0.7, 1.0, 0.0,
             0.0, 0.0, 1.5;

    std::cerr << "L eig:\n" << Eigen::MatrixXd(A_eig.llt().matrixL()) << std::endl;
    std::cerr << "L.square().log().sum():\n" << Eigen::MatrixXd(A_eig.llt().matrixL()).diagonal().array().square().log().sum() << std::endl;
    std::cerr << "LD eig:\n" << Eigen::MatrixXd(A_eig.ldlt().matrixL()) << std::endl;
    std::cerr << "D eig:\n" << Eigen::VectorXd(A_eig.ldlt().vectorD()).transpose() << std::endl;
    std::cerr << "D square:\n" << Eigen::MatrixXd(A_eig.llt().matrixL()).diagonal().array().square().transpose() << std::endl;
    std::cerr << "LD.log().sum():\n" << A_eig.ldlt().vectorD().array().log().sum() << std::endl;
  }


  xtensor<double, 2> X = {
    {0., 0., 0., 2., 1., 0.},
    {0., 0., 2., 0., 0., 0.},
    {0., 1., 0., 0., 0., 0.},
    {0., 2., 0., 0., 0., 0.},
    {0., 0., 1., 0., 0., 1.},
    {1., 0., 0., 0., 0., 0.},
    {0., 1., 0., 1., 0., 0.},
    {1., 0., 0., 0., 0., 1.}
  };

  //X = (X - xt::mean(X, 0)) / xt::stddev(X, 0);

  {
    for (std::size_t i = 0; i < X.shape(0); ++i)
    {
      auto r = row(X, i);
      double af = sum(r)() / (r.size() * 2);
      r = (r - 2. * af) / std::sqrt(2. * af * (1. - af));
    }

    std::cerr << "X: " << X << std::endl;

    xtensor<double, 2> A = dot(transpose(X), X);

    std::cerr << "A: " << A << std::endl;

    std::cerr << "A/M: " << (dot(transpose(X), X) / X.shape(0)) << std::endl;

//    xtensor<double, 2> D_inv = xt::zeros<double>(A.shape());
//    for (std::size_t i = 0; i < A.shape(0); ++i)
//      D_inv(i, i) = 1. / std::sqrt(A(i, i));
//
//    A = dot(dot(D_inv, A), D_inv);
//    std::cerr << "A_corr: " << A << std::endl;

    auto L = eval(cholesky(A));
    std::cerr << "L: " << L << std::endl;
    xtensor<double, 1> b = {0.1, 2.1, 0.12, 1.75, 0.41, 0.11};

    auto x = eval(solve_cholesky(L, b));
    std::cerr << "solve_cholesky(A, b): " << x << std::endl;
    std::cerr << "solve(A, b): " << eval(solve(A, b)) << std::endl;
    std::cerr << "lstsq(A, b): " << eval(dot(dot(pinv(dot(transpose(A), A)), transpose(A)), b)) << std::endl;
  }

  {
    Eigen::MatrixXd X2(X.shape(0), X.shape(1));
    for (std::size_t i = 0; i < X.shape(0); ++i)
    {
      for (std::size_t j = 0; j < X.shape(1); ++j)
      {
        X2(i, j) = X(i, j);
      }
    }

    Eigen::MatrixXd A = X2.transpose() * X2;
    Eigen::MatrixXd b (A.rows(), 1);
    b << 0.1, 2.1, 0.12, 1.75, 0.41, 0.11;
    Eigen::LDLT<Eigen::MatrixXd> solverA;
    solverA.compute(A);
    std::cerr << "Eigen L: " << Eigen::MatrixXd(solverA.matrixL()) << std::endl;
    std::cerr << "Eigen D: " << solverA.vectorD() << std::endl;
    auto x = solverA.solve(b);
    std::cerr << "Eigen x: " << x.transpose() << std::endl;
  }

  return 0;
}

class prog_args
{
private:
  std::string sub_command_;
  bool help_ = false;
  bool version_ = false;
public:
  prog_args()
  {
  }

  const std::string& sub_command() const { return sub_command_; }
  bool help_is_set() const { return help_; }
  bool version_is_set() const { return version_; }

  void print_usage(std::ostream& os)
  {
    //os << "----------------------------------------------\n";
    os << "Usage: savant <sub-command> [args ...]\n";
    os << "or: savant [opts ...]\n";
    os << "\n";
    os << "Sub-commands:\n";
    os << " single:    Single variant association analysis\n";
    //os << " group:     Group-wise association analysis\n";
    os << " pca:       Generates principal components\n";
    //os << " prune:     LD pruning\n";
    os << " plot:      Generates manhattan or QQ plot\n";
    os << "\n";
    os << "Options:\n";
    os << " -h, --help     Print usage\n";
    os << " -v, --version  Print version\n";
    //os << "----------------------------------------------\n";
    os << std::flush;
  }

  bool parse(int& argc, char**& argv)
  {
    if (argc > 1)
    {
      std::string str_opt_arg(argv[1]);

      if (str_opt_arg == "-h" || str_opt_arg == "--help")
        help_ = true;
      else if (str_opt_arg == "-v" || str_opt_arg == "--version")
        version_ = true;
      else if (str_opt_arg.size() && str_opt_arg.front() != '-')
      {
        sub_command_ = str_opt_arg;
        --argc;
        ++argv;
      }
    }

    if (!help_is_set() && !version_is_set() && sub_command_.empty())
    {
      std::cerr << "Missing sub-command\n";
      return false;
    }

    return true;
  }
};

int main(int argc, char** argv)
{
  //return mm_main(argc, argv);
  //return chol_test();
  //return pca_test();
  //challenger_test();
  //test();
  //slope_test();
  //return test_xtensor();

  prog_args args;
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

  if (args.version_is_set())
  {
    std::cout << "savant v" << SAVANT_VERSION << std::endl;
    return EXIT_SUCCESS;
  }

  if (args.sub_command() == "single")
    return single_main(argc, argv);
  else if (args.sub_command() == "plot")
    return plot_main(argc, argv);
  else if (args.sub_command() == "pca")
    return pca_main(argc, argv);
  else if (args.sub_command() == "grm")
    return grm_main(argc--, argv++);

  std::cerr << "Invalid sub-command (" << args.sub_command() << ")" << std::endl;
  args.print_usage(std::cerr);
  return EXIT_FAILURE;
}
