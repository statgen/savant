#include <vector>
#include <functional>
#include <cstdint>
#include <algorithm>
#include <type_traits>

/*
 * Adapted from https://github.com/hyunminkang/invNorm
 */

double stdnormal_cdf(double u);
double stdnormal_inv(double p);

//
template <typename VecT>
typename std::enable_if<std::is_floating_point<typename VecT::value_type>::value, void>::type
inverse_normalize(VecT& vec, std::vector<std::reference_wrapper<typename VecT::value_type>>& rank_vec)
{
  typedef typename VecT::value_type T;
  std::size_t n = vec.size();
  rank_vec.assign(vec.begin(), vec.end());

  std::sort(rank_vec.begin(), rank_vec.end(), [](const T& l, const T& r) { return l < r; });

  std::size_t left = 0;
  T prev_val = rank_vec[0];
  for(std::size_t i=1; i < rank_vec.size(); ++i)
  {
    if ( prev_val == rank_vec[i] )
    {
      // do not resolve ties yet
    }
    else
    {
      // tie can be resolved from left ... i-1
      T v(stdnormal_inv((left + i)/2./n));
      for (std::size_t j = left; j < i; ++j)
        rank_vec[j].get() = v;
      prev_val = rank_vec[i];
      left = i;
    }
  }
}

template <typename VecT>
void inverse_normalize(VecT& vec)
{
  std::vector<std::reference_wrapper<typename VecT::value_type>> tmp;
  inverse_normalize(vec, tmp);
}