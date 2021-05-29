/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "fit.hpp"


#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <iostream>

class ridge_regression
{
public:
  template <typename VecT, typename MatT>
  void fit(const VecT& y, const MatT& X, double lambda = 1.)
  {
    using namespace xt;
    using namespace xt::linalg;
    auto I = xt::diag(xt::ones<double>({X.shape(1)}));
    xtensor<double, 1> beta = dot(pinv(dot(transpose(X), X) + lambda * I), dot(transpose(X), y));

    auto pred = dot(X, beta);
    xtensor<double, 1> resi = y - pred;
    std::cerr << "beta: " << beta << std::endl;
    std::cerr << "pred: " << pred << std::endl;
    std::cerr << "resi: " << resi << std::endl;
  }

  template <typename VecT, typename MatT>
  void fit_gd(const VecT& y, const MatT& X, std::size_t max_epochs, double learning_rate, double tolerance, double lambda = 1.)
  {
    using namespace xt;
    using namespace xt::linalg;

    xtensor<double, 1> beta = xt::random::rand<double>({X.shape(1)});
    for (std::size_t e = 0; e < max_epochs; ++e)
    {
      auto pred = dot(X, beta);
      auto resi = y - pred;

      auto grad = -2. * dot(transpose(X), resi) + 2. * lambda * beta;
      //beta = beta - learning_rate * grad;
      auto delta = learning_rate * grad;
      beta -= delta;
      if (e % 20 == 0)
        std::cerr << "gd beta at iter " << e << ": " << beta << std::endl;
      if (xt::all(xt::abs(delta) <= tolerance))
        break;
    }

    std::cerr << "beta gd: " << beta << std::endl;
    std::cerr << "pred gd: " << dot(X, beta) << std::endl;
    std::cerr << "resi gd: " << (y - dot(X, beta)) << std::endl;
  }
};

int fit_main(int argc, char** argv)
{

  xt::xtensor<double, 1> y = {0.2, 0.8, 0.9, 0.85};
  xt::xtensor<double, 2> X = {
    {1., 0.1, 0.3, 0.2, 0.002, 0.15},
    {1., 0.7, 0.7, 0.5, 0.77, 0.59},
    {1., 0.5, 0.72, 0.6, 0.71, 0.7},
    {1., 0.9, 0.71, 0.66, 0.68, 0.8}
  };

  ridge_regression reg;
  reg.fit(y, X, 1.5);
  reg.fit_gd(y, X, 1000, 0.01, 1e-6,  1.5);



  return EXIT_SUCCESS;
}