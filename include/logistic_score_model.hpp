/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SAVANT_LOGISTIC_SCORE_MODEL_HPP
#define SAVANT_LOGISTIC_SCORE_MODEL_HPP

#include "debug_log.hpp"

#include <savvy/compressed_vector.hpp>

#include <xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <boost/math/distributions.hpp>

#include <iostream>

class logistic_score_model
{
public:
  typedef double scalar_type;
  typedef xt::xtensor<scalar_type, 1> res_t;
  typedef xt::xtensor<scalar_type, 2> cov_t;
private:
  res_t v_;
  res_t residuals_;
  cov_t vx_dot_iv2ct_;
public:
  struct stats_t
  {
    double pvalue;
    double score;
  };

  logistic_score_model(const res_t& y, const cov_t& x_orig)
  {
    using namespace xt;
    using namespace xt::linalg;
    const scalar_type tolerance = 0.00001;

    // ==== Fit Logistic Model ==== //
    //xarray<double> x_orig = {53,57,58,63,66,67,67,67,68,69,70,70,70,70,72,73,75,75,76,76,78,79,81};
    //xtensor<double, 1> y = {1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0};
    //auto x = x_orig;
    //x.reshape({y.size(), 1});
//  assert(y.size() == x.size());
    auto x = xt::concatenate(xt::xtuple(xt::ones<double>({y.size(), std::size_t(1)}), x_orig), 1);
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
      //std::cerr << beta << std::endl;
    }
    // =================== //

    // =================== //
    /*
    ## calculate the weights for each observation
    v <- exp(reml$linear.predictors) / (1 + exp(reml$linear.predictors))^2
    vx <- v * reml$x
    v2 <- t(vx) %*% reml$x
    iv2 <- solve(v2)
    iv2c <- chol(iv2) ## make t(iv2c) %*% iv2c = iv2
    x.res <- pheno - reml$fitted  # n * 1 matrix
    */

    // v <- exp(reml$linear.predictors) / (1 + exp(reml$linear.predictors))^2
    auto xw = dot(x, beta);
    v_ = xt::exp(xw) / xt::square(1. + xt::exp(xw));
    debug_log << "shape: " << xt::adapt(v_.shape()) << std::endl;
    debug_log << "shape: " << xt::adapt(xw.shape()) << std::endl;
    debug_log << "shape: " << xt::adapt(x.shape()) << std::endl;
    // vx <- v * reml$x
    xarray<double> vx = x;
    debug_log << "shape: " << xt::adapt(x.shape()) << std::endl;
    for (std::size_t i = 0; i < vx.shape(1); ++i)
    {
      xt::col(vx, i) *= v_;
    }

    debug_log << "-------------------------------------" << std::endl;
    debug_log << xt::adapt(v_.shape()) << std::endl;
    debug_log << vx << std::endl;
    debug_log << x << std::endl;
    // v2 <- t(vx) %*% reml$x
    auto v2 = dot(transpose(vx), x);
    debug_log << "v2: " << v2 << std::endl;

    // iv2 <- solve(v2)
    auto iv2 = pinv(v2);
    debug_log << "iv2: " << iv2 << std::endl;

    // iv2c <- chol(iv2) ## make t(iv2c) %*% iv2c = iv2
    auto iv2c = transpose(cholesky(iv2)); // cholesky() returns lower triangle while R chol() returns upper
    debug_log << "iv2c: " << iv2c << std::endl;

    // x.res <- pheno - reml$fitted  # n * 1 matrix
    auto y_hat = 1. / (1. + xt::exp(-xw));
    residuals_ = y - y_hat;

    // tcrossprod(vx,iv2c) = vx %*% t(iv2c)
    vx_dot_iv2ct_ = dot(vx, transpose(iv2c));
    debug_log << "res: " << residuals_ << std::endl;
    debug_log << "vx_do_iv2ct: " << vx_dot_iv2ct_ << std::endl;
    // =================== //

    debug_log.flush();

//    T y_hat = 1. / (1. + xt::exp(-xw));
//    residuals_type residuals = y - y_hat;
//    scalar_type se = xt::mean(residuals * residuals)();
//
//    std::cerr << "sum(y): " << sum(y) << std::endl;
//    return residuals;
  }

  std::size_t sample_size() const { return residuals_.size(); }

  template <typename GenoVecT>
  stats_t test_single(const GenoVecT& geno_vec) const
  {
    using namespace xt;
    using namespace xt::linalg;
    /*
    ## Multiply weights for the genotypes
    U <- tcrossprod(t(x.res),genos)  # U is m * 1 matrix
    Vl <- tcrossprod(v,genos*genos)
    Vr <- genos %*% tcrossprod(vx,iv2c)
    V.s <- sqrt(Vl - rowSums(Vr*Vr) + 1e-10) ## to avoid negative variance estimate
    #print(min(Vl-rowSums(Vr*Vr)))

    ## compute pvalues and scores and return them
    T <- U/V.s
    return(list(p=pnorm(abs(T),lower.tail=F)*2,add=cbind(matrix(T,length(T),1)),cname=cname))
    */

    //T a = {1,2,3};
    //std::cerr << "dot(a, a): " << dot(a, a) << std::endl;

    //T geno_vec = {0,0,0,1,2,0,1,1,2,1,0,0,0,0,0,1,1,1,0,0,0,1,1};

    // U <- tcrossprod(t(x.res),genos)  # U is m * 1 matrix
    auto U = dot(residuals_, geno_vec);
    //std::cerr << "U: " << U << std::endl;

    // Vl <- tcrossprod(v,genos*genos)
    auto Vl = dot(v_, xt::square(geno_vec));
    //std::cerr << "Vl: " << Vl << std::endl;

    // Vr <- genos %*% tcrossprod(vx,iv2c)
    auto Vr = dot(geno_vec, vx_dot_iv2ct_);
    //std::cerr << "Vr: " << Vr << std::endl;

    // V.s <- sqrt(Vl - rowSums(Vr*Vr) + 1e-10) ## to avoid negative variance estimate
    auto Vs = xt::sqrt(Vl - xt::sum(Vr * Vr) + 1e-10);
    //std::cerr << "Vs: " << Vs << std::endl;

    auto t_stat = U / Vs;
    //std::cerr << "T: " << t_stat << std::endl;
    //std::cerr << "T(): " << t_stat() << std::endl;

    stats_t ret;
    ret.score = t_stat();

//    boost::math::students_t_distribution<double> dist(dof);
//    float pval =  cdf(complement(dist, std::fabs(std::isnan(t) ? 0 : t))) * 2;
    boost::math::normal_distribution<double> dist;
    ret.pvalue =  cdf(complement(dist, std::fabs(std::isnan(t_stat()) ? 0 : t_stat()))) * 2;
    //std::cerr << "p-value: " << ret.pvalue << std::endl;
    return ret;
  }

  stats_t test_single(const std::vector<double>& dense_geno) const
  {
    return test_single(xt::adapt(dense_geno));
  }

  stats_t test_single(const savvy::compressed_vector<double>& geno_vec) const
  {
    stats_t ret;
    return ret;
  }
};

static std::ostream& operator<<(std::ostream& os, const logistic_score_model& v)
{
  os << "pvalue\tscore";
  return os;
}


static std::ostream& operator<<(std::ostream& os, const logistic_score_model::stats_t& v)
{
  os << v.pvalue << "\t" << v.score;
  return os;
}


#endif //SAVANT_LOGISTIC_SCORE_MODEL_HPP
