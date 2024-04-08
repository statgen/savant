/*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "qtl.hpp"
#include "utility.hpp"
#include "assoc.hpp"
#include "debug_log.hpp"
#include "bed_file.hpp"
#include "inv_norm.hpp"
#include "linear_model.hpp"

#include <cstdlib>

class qtl_prog_args : public assoc_prog_args
{
private:
  std::int64_t window_size_ = 48; //1000000;
public:
  qtl_prog_args() : assoc_prog_args("qtl", {})
  {

  }

  std::int64_t window_size() const { return window_size_; }
};


bool parse_covariates_file(const assoc_prog_args& args, const std::vector<std::string>& ids, xt::xtensor<scalar_type, 2>& dest)
{
  std::unordered_map<std::string, std::size_t> id_map;
  id_map.reserve(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i)
    id_map[ids[i]] = i;
  std::size_t match_count = 0, line_count = 0;

  if (args.cov_columns().empty())
    return std::cerr << "Error: must pass separate covariates file\n", false;
  std::string path = args.cov_columns()[0];
  std::ifstream pheno_file(path, std::ios::binary);

  std::string line;
  if (!std::getline(pheno_file, line))
    return std::cerr << "Error: empty covariates file\n", false;

  auto str_fields = utility::split_string_to_vector(line.c_str(), '\t');
  if (str_fields.empty())
    return std::cerr << "Error: first line in covariates file is empty\n", false;

  dest = xt::xtensor<scalar_type, 2>::from_shape({ids.size(), str_fields.size() - 1});

  char* p = nullptr;
  while (std::getline(pheno_file, line))
  {
    str_fields = utility::split_string_to_vector(line.c_str(), '\t');
    if (str_fields.empty())
      return std::cerr << "Error: empty line in covariates file\n", false;

    auto row_idx_it = id_map.find(str_fields[0]);
    if (row_idx_it != id_map.end())
    {
      ++match_count;
      for (std::size_t i = 1; i < str_fields.size(); ++i)
      {
        scalar_type v = std::strtod(str_fields[i].c_str(), &p);
        if (p == str_fields[i].c_str() && !str_fields[i].empty() && str_fields[i][0] != '.' && std::tolower(str_fields[i][0]) != 'n')
          return std::cerr << "Error: encountered non-numeric covariate\n", false;
        else if (p != str_fields[i].c_str())
          dest(row_idx_it->second, i) = v;
        else
          return std::cerr << "Error: missing covariates not supported\n", false;
      }
    }
  }

  if (match_count != ids.size())
    return std::cerr << "Error: missing covariates for " << (ids.size() - match_count) << " samples\n", false;
  return true;
}

class residualizer
{
public:
  typedef double scalar_type;
  typedef xt::xtensor<scalar_type, 1> res_t;
  typedef xt::xtensor<scalar_type, 2> cov_t;
private:
  cov_t x_;
  cov_t m_;
public:
  residualizer() {}
  residualizer(const cov_t& x_orig)
  {
    using namespace xt;
    using namespace xt::linalg;

    x_ = x_orig;

    m_ = xt::eval(dot(pinv(dot(transpose(x_), x_)), transpose(x_)));
  }

  template <typename T>
  res_t operator()(const xt::xtensor<T, 1>& v, bool invnorm = false) const
  {
    using namespace xt;
    using namespace xt::linalg;
    //cov_t x = concatenate(xtuple(xt::ones<scalar_type>({y.size(), std::size_t(1)}), x_orig), 1);
    auto pbetas = dot(m_, v);
    res_t residuals = v - dot(x_, pbetas);
    if (invnorm)
      inverse_normalize(residuals);
    return residuals;
  }

  template <typename T>
  std::vector<scalar_type> operator()(const std::vector<T>& v_std, bool invnorm = false) const
  {
    using namespace xt;
    using namespace xt::linalg;
    //cov_t x = concatenate(xtuple(xt::ones<scalar_type>({y.size(), std::size_t(1)}), x_orig), 1);
    auto v = xt::adapt(v_std, {v_std.size()});
    auto pbetas = dot(m_, v);
    res_t residuals = v - dot(x_, pbetas);
    if (invnorm)
      inverse_normalize(residuals);
    return std::vector<scalar_type> (residuals.begin(), residuals.end());
  }

};

bool process_cis_batch(const std::vector<bed_file::record>& phenos,  std::vector<residualizer>& residualizers, std::vector<std::vector<std::size_t>> subset_non_missing_map, /* std::deque<std::vector<std::int8_t>>& genos,*/ savvy::reader& geno_file, std::ostream& output_file, const qtl_prog_args& args)
{
  if (phenos.empty())
    return false;
  auto window_size = args.window_size();
  if (window_size >=0)
  {
    std::int64_t window_start = std::max(1ll, phenos.front().beg() + 1 - window_size);
    std::int64_t window_end = window_start;
    for (auto it = phenos.begin(); it != phenos.end(); ++it)
    {
      window_end = std::max(window_end, it->end() + window_size);
    }

    geno_file.reset_bounds(savvy::genomic_region(phenos.front().chrom(), window_start, window_end), savvy::bounding_point::any);
  }

  std::vector<scalar_type> s_y(phenos.size());
  std::vector<scalar_type> s_yy(phenos.size());
  std::vector<std::vector<scalar_type>> pheno_resids(phenos.size());
  for (std::size_t i = 0; i < phenos.size(); ++i)
  {
    pheno_resids[i] = residualizers[i](phenos[i].data(), args.invnorm());
    s_y[i] = std::accumulate(pheno_resids[i].begin(),  pheno_resids[i].end(), scalar_type());
    s_yy[i] = std::inner_product(pheno_resids[i].begin(),  pheno_resids[i].end(), pheno_resids[i].begin(), scalar_type());
  }

  std::vector<std::int8_t> geno;
  std::vector<scalar_type> geno_sub;
  savvy::variant var;
  while (geno_file.read(var))
  {
    var.get_format("GT", geno);
    for (std::size_t alt_idx = 1; alt_idx <= var.alts().size(); ++alt_idx)
    {
      std::int64_t var_end = std::max<std::int64_t>(var.pos(), var.pos() + var.alts()[alt_idx-1].size() - 1);
      for (std::size_t pheno_idx = 0; pheno_idx < pheno_resids.size(); ++pheno_idx)
      {
        if (window_size < 0 || (var.pos() > (phenos[pheno_idx].beg() - window_size) && var_end <= (phenos[pheno_idx].end() + window_size)))
        {
          std::int64_t an = geno.size();
          std::size_t ploidy = an / subset_non_missing_map[pheno_idx].size();
          float ac = 0.f;
          geno_sub.resize(0);
          geno_sub.resize(phenos[pheno_idx].data().size());
          for (std::size_t i = 0; i < geno.size(); ++i)
          {
            if (subset_non_missing_map[pheno_idx][i/ploidy] < geno_sub.size())
            {
              if (geno[i] == alt_idx)
              {
                geno_sub[subset_non_missing_map[pheno_idx][i / ploidy]] += 1;
                ac += 1.f;
              }
            }
            else
            {
              --an;
            }
          }

          float af = ac / an;
          float mac = (ac > (an/2.f) ? an - ac : ac);
          float maf = (af > 0.5f ? 1.f - af : af);

          if (an == 0) continue;
          if (mac < args.min_mac()) continue;
          if (maf < args.min_maf()) continue;

          geno_sub = residualizers[pheno_idx](geno_sub);

          linear_model::stats_t stats = linear_model::ols(geno_sub, xt::adapt(pheno_resids[pheno_idx], {pheno_resids[pheno_idx].size()}), std::accumulate(geno_sub.begin(), geno_sub.end(), scalar_type()), s_y[pheno_idx], s_yy[pheno_idx]);

          output_file << var.chromosome()
                      << "\t" << var.position()
                      << "\t" << var.ref()
                      << "\t" << (var.alts().empty() ? "." : var.alts()[0])
                      << "\t" << var.id()
                      << "\t" << maf
                      << "\t" << mac
                      << "\t" << phenos[pheno_idx].pheno_id()
                      << "\t" << phenos[pheno_idx].chrom()
                      << "\t" << phenos[pheno_idx].beg()
                      << "\t" << phenos[pheno_idx].end()
                      << "\t" << stats << "\n";

          // geno_sub
          // geno_resid =
          // pheno_it ~ geno_resid
          // results[pheno_idx].emplace_back(var_id, beta, beta_se, pval, dof, n_samples)

        }
      }
    }
  }

  return true;
}

int qtl_main(int argc, char** argv)
{
  qtl_prog_args args;
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

  if (args.debug_log_path().size())
    debug_log.open(args.debug_log_path());

  bed_file bed(args.pheno_path());

  if (bed.sample_ids().empty())
    return std::cerr << "Error: bed file contains no samples\n", EXIT_FAILURE;

  savvy::reader geno_file(args.geno_path());

  // vvv Create sample mapping TODO: put in function
  std::vector<std::string> sample_intersection = geno_file.subset_samples({bed.sample_ids().begin(), bed.sample_ids().end()});
  std::unordered_map<std::string, std::size_t> geno_id_map;
  geno_id_map.reserve(sample_intersection.size());
  for (std::size_t i = 0; i < sample_intersection.size(); ++i)
    geno_id_map[sample_intersection[i]] = i;

  std::size_t excluded_value = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> pheno_to_geno_map(bed.sample_ids().size(), excluded_value);
  for (std::size_t i = 0; i < bed.sample_ids().size(); ++i)
  {
    auto res = geno_id_map.find(bed.sample_ids()[i]);
    if (res != geno_id_map.end())
      pheno_to_geno_map[i] = res->second;
  }
  // ^^^ Create sample mapping

  xt::xtensor<scalar_type, 2> cov_mat;
  if (!parse_covariates_file(args, sample_intersection, cov_mat))
    return std::cerr << "Error: failed parsing covariates file\n", EXIT_FAILURE;

  shrinkwrap::bgzf::ostream output_file(args.output_path());
  output_file << "geno_chrom\tgeno_pos\tref\talt\tvariant_id\tmaf\tmac\tpheno_id\tpheno_chrom\tpheno_beg\tpheno_end\t" << linear_model::stats_t::header_column_names() << std::endl;

  std::size_t batch_size = 10;
  std::vector<bed_file::record> phenos;
  //std::vector<std::vector<std::int8_t>> genos;
  while (bed.read(phenos, pheno_to_geno_map, batch_size))
  {
    if (phenos.empty())
      break;

    std::vector<std::vector<std::size_t>> subset_non_missing_map(phenos.size(), std::vector<std::size_t>(phenos.front().data().size()));
    std::vector<std::size_t> keep_samples(cov_mat.shape()[1]);
    std::vector<residualizer> residualizers(phenos.size());
    for (std::size_t i = 0; i < phenos.size(); ++i)
    {
      keep_samples.resize(0);
      if (phenos[i].data().size() != subset_non_missing_map[i].size())
        return std::cerr << "Error: size mismatch at " << __FILE_NAME__ << ":" << __LINE__ << " (this should not happen)\n", false;

      //pheno_sub[i].resize(phenos[i].data().size());
      subset_non_missing_map[i] = phenos[i].remove_missing();
      for (std::size_t j = 0; j < subset_non_missing_map[i].size(); ++j)
      {
        if (subset_non_missing_map[i][j] <= j)
          keep_samples.push_back(j);
      }
      residualizers[i] = residualizer(xt::view(cov_mat, xt::keep(keep_samples), xt::all()));
    }

    if (!process_cis_batch(phenos, residualizers, subset_non_missing_map, /* genos,*/ geno_file, output_file, args))
      return std::cerr << "Error: processing batch failed\n", EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}