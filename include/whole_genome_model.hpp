/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "linear_model.hpp"

class ridge_regression
{
private:
  xt::xtensor<float, 1> beta_;
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
  void fit_gd(const VecT& y, const MatT& X, std::size_t max_epochs, typename VecT::value_type learning_rate, typename VecT::value_type tolerance, typename VecT::value_type lambda = 1.)
  {
    using namespace xt;
    using namespace xt::linalg;

    std::size_t n = X.shape(0);

    beta_ = xt::random::randn<typename VecT::value_type>({X.shape(1)}, 0., std::sqrt(1. / n));
    beta_(0) = xt::mean(y)();
    std::cerr << "gd initial beta: " << beta_ << std::endl;
    decltype(beta_) beta_prev = beta_;
    double prev_mse = std::numeric_limits<double>::max();
    for (std::size_t e = 0; e < max_epochs; ++e)
    {
      auto pred = dot(X, beta_);
      auto resi = y - pred;

      xtensor<typename VecT::value_type, 1> penalty = ((2.f) * lambda) * beta_;
      penalty[0] = 0.f; // Don't penalize intercept
      auto grad = (-2.f/n) * dot(transpose(X), resi) + penalty;
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
      beta_ -= delta;
      if (e % 100 == 0)
      {
        double mse = (dot(transpose(resi), resi) / n)();
        if (e > 0 && ((mse - prev_mse) > 1e-10 || !std::isfinite(mse)))
        {
          beta_ = beta_prev;
          learning_rate = learning_rate / 10.;
          std::cerr << "Resetting betas and decreasing learning rate to " << learning_rate << std::endl;
        }
        else
        {
          beta_prev = beta_;
          learning_rate += learning_rate * 0.01;
          prev_mse = mse;
          std::cerr << "Increasing learning rate to " << learning_rate << std::endl;
        }
        std::cerr << "gd beta after iter " << e << ": " << beta_ << std::endl;
        std::cerr << "mse: " << dot(transpose(resi), resi) / n << std::endl;
      }
      if (learning_rate < 1e-16) //xt::all(xt::abs(delta) <= tolerance))
      {
        std::cerr << "Stopping early" << std::endl;
        break;
      }
    }

    auto resi = (y - dot(X, beta_));
    std::cerr << "beta gd: " << beta_ << std::endl;
    std::cerr << "pred gd: " << dot(X, beta_) << std::endl;
    std::cerr << "resi gd: " << resi << std::endl;
    std::cerr << "mse: " << dot(transpose(resi), resi) / n << std::endl;
  }

  template <typename MatT>
  xt::xtensor<float, 1> predict(const MatT& X)
  {
    return xt::linalg::dot(X, beta_);
  }
};

class whole_genome_model : public linear_model
{
public:
  whole_genome_model(const res_t& y, const cov_t& x_orig, const xt::xtensor<float, 2>& geno_matrix, std::size_t max_epochs, double learning_rate, double tolerance, double lambda, bool invnorm) : linear_model(y, x_orig, invnorm)
  {
    ridge_regression reg;
//    reg.fit(xt::view(residuals_, xt::range(0, 2504)), geno_matrix, 0.0);
//    reg.fit(xt::view(residuals_, xt::range(0, 2504)), geno_matrix, 1.0);
    xt::xtensor<float, 1> y_copy = residuals_; //xt::view(residuals_, xt::range(0, 2504));
    reg.fit_gd(xt::view(y_copy, xt::range(0, 2504))/**/, geno_matrix, max_epochs, learning_rate, tolerance, lambda); // 10000, 0.001, 1e-5, 1.0);
    residuals_ = reg.predict(geno_matrix) - y_copy;
    s_y_ = sum(residuals_)();
    s_yy_ = xt::linalg::dot(residuals_, residuals_)();
  }
};