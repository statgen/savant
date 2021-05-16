/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "linear_model.hpp"

class ridge_regression
{
public:
  template <typename VecT, typename MatT>
  void fit(const VecT& y, const MatT& X, double lambda = 1.)
  {
    using namespace xt;
    using namespace xt::linalg;
    xtensor<double, 2> I = xt::diag(xt::ones<double>({X.shape(1)}));
    I(0,0) = 0.; // Don't penalize intercept
    xtensor<double, 1> beta = dot(pinv(dot(transpose(X), X) + lambda * I), dot(transpose(X), y));

    auto pred = dot(X, beta);
    xtensor<double, 1> resi = y - pred;
    std::cerr << "beta: " << beta << std::endl;
    std::cerr << "pred: " << pred << std::endl;
    std::cerr << "resi: " << resi << std::endl;
    std::cerr << "mse: " << dot(transpose(resi), resi) / y.shape(0) << std::endl;
  }

  template <typename VecT, typename MatT>
  void fit_gd(const VecT& y, const MatT& X, std::size_t max_epochs, double learning_rate, double tolerance, double lambda = 1.)
  {
    using namespace xt;
    using namespace xt::linalg;

    std::size_t n = X.shape(0);

    xtensor<double, 1> beta = xt::random::randn<double>({X.shape(1)}, 0., std::sqrt(1. / n));
    beta(0) = xt::mean(y)();
    std::cerr << "gd initial beta: " << beta << std::endl;
    for (std::size_t e = 0; e < max_epochs; ++e)
    {
      auto pred = dot(X, beta);
      auto resi = y - pred;

      xtensor<double, 1> reg = ((2./1.) * lambda) * beta;
      reg[0] = 0.; // Don't penalize intercept
      auto grad = -(2./n) * dot(transpose(X), resi) + reg;
      std::size_t not_finite = 0, nan_cnt = 0;
      for (std::size_t i = 0; i < grad.shape(0); ++i)
      {
        if (!std::isfinite(grad[i]))
          ++not_finite;
        if (std::isnan(nan_cnt))
          ++nan_cnt;
      }

      if (not_finite > 0)
      {
        std::size_t finite = grad.shape(0) - not_finite;
        auto foo = 0;
      }

      //beta = beta - learning_rate * grad;
      auto delta = learning_rate * grad;
      beta -= delta;
      if (e % 100 == 0)
      {
        std::cerr << "gd beta after iter " << e << ": " << beta << std::endl;
        std::cerr << "mse: " << dot(transpose(resi), resi) / n << std::endl;
      }
      if (false) //xt::all(xt::abs(delta) <= tolerance))
        break;
    }

    auto resi = (y - dot(X, beta));
    std::cerr << "beta gd: " << beta << std::endl;
    std::cerr << "pred gd: " << dot(X, beta) << std::endl;
    std::cerr << "resi gd: " << resi << std::endl;
    std::cerr << "mse: " << dot(transpose(resi), resi) / n << std::endl;
  }
};

class whole_genome_model : public linear_model
{
public:
  whole_genome_model(const res_t& y, const cov_t& x_orig, const xt::xtensor<double, 2>& geno_matrix) : linear_model(y, x_orig)
  {
    ridge_regression reg;
//    reg.fit(xt::view(residuals_, xt::range(0, 2504)), geno_matrix, 0.0);
//    reg.fit(xt::view(residuals_, xt::range(0, 2504)), geno_matrix, 1.0);
    reg.fit_gd(xt::view(residuals_, xt::range(0, 2504)), geno_matrix, 10000, 0.0001, 1e-5, 1.0);
  }
};