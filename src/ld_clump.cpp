/*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <savvy/reader.hpp>
#include <savvy/compressed_vector.hpp>
#include <shrinkwrap/gz.hpp>

#include <cmath>
#include <cstdlib>
#include <list>
#include <numeric>

#include "getopt_wrapper.hpp"
#include "summary_stats_input.hpp"

class clump_prog_args : public getopt_wrapper
{
private:
  std::string geno_path_;
  std::string results_path_;
  std::string output_path_ = "/dev/stdout";
  //std::unique_ptr<savvy::genomic_region> region_;
  double pval_threshold_ = 2.;
  double r2_threshold_ = 0.2;
  std::uint32_t seed_ = 7;
  bool write_all_ = false;
  bool help_ = false;
  bool version_ = false;
public:
  clump_prog_args() :
    getopt_wrapper("Usage: savant-clump [opts ...] <geno_file> <results_file>", {
      {"help", "", 'h', "Print usage"},
      {"output", "<file>", 'o', "Output path (default: /dev/stdout)"},
      {"max-pvalue", "<real>", 'p', "Max p-value to clump"},
      {"r2-threshold", "<real>", 's', "R-squared threshold for clumping (default: 0.2)"},
      {"write-all", "", 'a', "Write all records to output instead of only the most significant association from each LD group"},
      {"version", "", 'v', "Print version"}})
  {
  }

  virtual ~clump_prog_args() {}
  const std::string& geno_path() const { return geno_path_; }
  const std::string& output_path() const { return output_path_; }
  const std::string& results_path() const { return results_path_; }
  double pval_threshold() const { return pval_threshold_; }
  double r2_threshold() const { return r2_threshold_; }
  std::uint32_t seed() const { return seed_; }
  bool help_is_set() const { return help_; }
  bool version_is_set() const { return version_; }
  bool write_all() const { return write_all_; }

  bool parse(int argc, char** argv)
  {
    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, short_opt_string_.c_str(), long_options_.data(), &long_index )) != -1)
    {
      char copt = char(opt & 0xFF);
      switch (copt)
      {
      case 'h':
        help_ = true;
        return true;
      case 'o':
        output_path_ = optarg ? optarg : "";
        break;
      case 'p':
        pval_threshold_ = std::atof(optarg ? optarg : "");
        break;
      case 's':
        r2_threshold_ = std::atof(optarg ? optarg : "");
        if (r2_threshold_ <= 0. || r2_threshold_ > 1.)
          return std::cerr << "Error: --r2-threshold must be in range 0 < x <= 1", false;
        break;
      case 'v':
        version_ = true;
        return true;
      case 'a':
        write_all_ = true;
        break;
      default:
        return false;
      }
    }

    int remaining_arg_count = argc - optind;

    if (remaining_arg_count == 2)
    {
      geno_path_ = argv[optind];
      results_path_ = argv[optind + 1];
    }
    else if (remaining_arg_count < 2)
    {
      std::cerr << "Too few arguments\n";
      return false;
    }
    else
    {
      std::cerr << "Too many arguments\n";
      return false;
    }

    return true;
  }
};

typedef savvy::compressed_vector<std::int8_t> vec_t;

/*double compute_r2_v2(const std::vector<std::int8_t>& i_vec_dense, const vec_t& i_vec, const vec_t& j_vec, double min_r2_threshold)
{
  assert(i_vec_dense.size() == i_vec.size());
  assert(i_vec.size() == j_vec.size());
  std::size_t n = i_vec_dense.size();

  double s_xy = 0.;
  for (auto it = j_vec.begin(); it != j_vec.end(); ++it)
    s_xy += *it * i_vec_dense[it.offset()];

  double s_x = std::accumulate(i_vec_dense.begin(), i_vec_dense.end(), 0.);
  double s_y = std::accumulate(j_vec.begin(), j_vec.end(), 0.);

  double s_xx = std::inner_product(i_vec_dense.begin(), i_vec_dense.end(), i_vec_dense.begin(), 0.);
  double s_yy = std::inner_product(j_vec.begin(), j_vec.end(), j_vec.begin(), 0.);


  double x_mean = s_x / n;
  double y_mean = s_y / n;
  double r = (s_xy - n * x_mean * y_mean) / (std::sqrt(s_xx - x_mean * x_mean * n) * std::sqrt(s_yy - y_mean * y_mean * n));
  double r2 = r * r;

  double R = (s_xy - n * x_mean * y_mean) / ((n-1) * s_x * s_y);
  return r2;
}*/

double compute_r2(const std::vector<std::int8_t>& i_vec_dense, const vec_t& i_vec, const vec_t& j_vec, double min_r2_threshold)
{

  std::size_t n_haplotypes = i_vec.size();
  std::size_t ac_i = i_vec.non_zero_size();
  std::size_t ac_j = j_vec.non_zero_size();
  double freq_i = static_cast<double>(ac_i) / static_cast<double>(n_haplotypes);
  double freq_j = static_cast<double>(ac_j) / static_cast<double>(n_haplotypes);

  double min_p = std::max(0., static_cast<double>(ac_i) + static_cast<double>(ac_j) - static_cast<double>(n_haplotypes));
  double max_p = static_cast<double>(std::min(ac_i, ac_j));

  double freq_i_j = freq_i * freq_j;
  double denom = std::sqrt( freq_i_j * std::max(0., 1. - freq_i) * std::max(0., 1. - freq_j));

  if (denom > 0.)
  {

    double r_min_p = (min_p / static_cast<double>(n_haplotypes) - freq_i_j) / denom;
    double r_max_p = (max_p / static_cast<double>(n_haplotypes) - freq_i_j) / denom;

    double max_r2 = std::max(r_min_p * r_min_p, r_max_p * r_max_p);
    if (max_r2 < min_r2_threshold)
    {
      return 0.;
    }

    // Dot product
    double p = 0.;
    for (auto it = j_vec.begin(); it != j_vec.end(); ++it)
      p += *it * i_vec_dense[it.offset()];

    double r = (p / static_cast<double>(n_haplotypes) - freq_i * freq_j) / denom;

    return r * r;
  }

  return std::numeric_limits<float>::quiet_NaN();
}



int main(int argc, char** argv)
{
  clump_prog_args args;
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
    std::cout << "savant-clump v" << SAVANT_VERSION << std::endl;
    return EXIT_SUCCESS;
  }

  //========== Load input ==========//
  results_file input_results(args.results_path());
  if (!input_results)
    return std::cerr << "Error: opening association results input file failed\n", EXIT_FAILURE;

  std::list<results_file::record> records;
  std::unordered_map<std::string, std::vector<results_file::record*>> pheno_results;
  std::vector<variant_id_t> variant_ids;
  results_file::read(input_results, records, pheno_results, variant_ids, args.pval_threshold(), args.write_all());

  if (input_results.bad())
    return std::cerr << "Error: failed loading association results input file\n", EXIT_FAILURE;

  std::vector<savvy::compressed_vector<std::int8_t>> genotypes;
  if (!load_variant_id_genotypes(args.geno_path(), variant_ids, genotypes))
    return std::cerr << "Error: failed to load genotypes\n", EXIT_FAILURE;
  //========== END Load input ==========//

  //========== Run clumping ==========//
  std::vector<std::int8_t> dense_geno;
  for (auto it = pheno_results.begin(); it != pheno_results.end(); ++it)
  {
    if (it->second.size() == 1 && it->second[0]->pvalue() <= args.pval_threshold()) // TODO: this condition should be unnecessary
    {
      it->second[0]->set_group(1);
      it->second[0]->tophit();
    }
    else
    {
      assert(it->second.size() > 0);

      for (std::int32_t group = 1; it->second.size() > 0; ++group)
      {
        double min_pvalue = 2.;
        std::size_t min_idx = std::size_t(-1);

        for (std::size_t i = 0; i < it->second.size(); ++i)
        {
          if (it->second[i]->pvalue() < min_pvalue)
          {
            min_pvalue = it->second[i]->pvalue();
            min_idx = i;
          }
        }

        assert(min_idx < it->second.size());

        it->second[min_idx]->set_group(group);
        it->second[min_idx]->set_tophit();

        assert(it->second[min_idx]->genotype_index() < genotypes.size());
        auto& top_sparse_geno = genotypes[it->second[min_idx]->genotype_index()];
        dense_geno.clear();
        dense_geno.resize(top_sparse_geno.size());
        for (auto gt = top_sparse_geno.begin(); gt != top_sparse_geno.end(); ++gt)
          dense_geno[gt.offset()] = *gt;

        for (std::size_t i = 0; i < it->second.size(); ++i)
        {
          if (i != min_idx)
          {
            double r2 = compute_r2(dense_geno, top_sparse_geno, genotypes[it->second[i]->genotype_index()], args.r2_threshold());
            if (r2 >= args.r2_threshold())
              it->second[i]->set_group(group);
          }
        }

        // Remove records in current clump group
        std::size_t dest = 0;
        for (std::size_t i = 0; i < it->second.size(); ++i)
        {
          if (it->second[i]->group() != group)
          {
            if (i > dest)
              it->second[dest] = it->second[i];
            ++dest;
          }
        }
        it->second.resize(dest);
      }
    }
  }
  //========== END Run clumping ==========//

  //========== Write output ==========//
  shrinkwrap::bgzf::ostream output_file(args.output_path());
  if (!output_file)
    return std::cerr << "Error: opening output file failed\n", EXIT_FAILURE;

  output_file << input_results.header_line() << "\tclump_group" << std::endl;

  for (auto it = records.begin(); it != records.end() && output_file; ++it)
  {
    if (args.write_all() || it->tophit())
    {
      output_file << it->serialized_line() << "\t" << it->group() << "\n";
    }
  }

  if (!output_file)
    return std::cerr << "Error: failed to write output records\n", EXIT_FAILURE;
  //========== END Write output ==========//

  return EXIT_SUCCESS;
}

