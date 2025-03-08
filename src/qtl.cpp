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
#include "variable_input.hpp"

#include <cstdlib>
#include <random>
#include <list>
#include <unordered_set>
#include <string>
#include <utility>

class qtl_prog_args : public getopt_wrapper
{
private:
  //std::string sub_command_;
  //std::vector<option> long_options_;
  //std::string short_options_;
  std::unordered_set<std::string> consequences_;
  std::unordered_set<std::string> impacts_;
  std::list<std::pair<std::string, savvy::genomic_region>> collapse_regions_;
  std::string cov_path_;
  //std::string phenotype_field_;
  std::string geno_path_;
  std::string pheno_path_;
  std::string permutations_path_;
  std::string discovery_counts_path_;
  std::string output_path_ = "/dev/stdout";
  std::string debug_log_path_ = "/dev/null";
  std::string fmt_field_ = "";
  std::unique_ptr<savvy::genomic_region> region_;
  double rare_treshold_ = 0.01;
  double min_mac_ = 1.0;
  double min_maf_ = 0.f;
  double max_pval_ = 2.;
  double resid_geno_threshold_ = -1.;
  std::uint32_t seed_ = 7;
  bool split_output_ = false;
  bool help_ = false;
  bool invnorm_ = false;
  bool pass_only_ = false;
  bool print_model_fit_ = false;
public:
  qtl_prog_args() :
    getopt_wrapper("Usage: savant qtl [opts ...] <geno_file> <pheno_file>", {
      {"collapse-regions", "<file>", 'R', "BED file of regions to collapse for rare variant testing"},
      {"collapse-threshold", "<real>", 'T', "AF threshold for burden test (default: 0.01)"},
      {"collapse-anno", "<string>", 'A', "Comma-separated list of INFO/ANN annotation values to include in analysis (see https://pcingola.github.io/SnpEff/adds/VCFannotationformat_v1.0.pdf)"},
      {"collapse-impact", "<string>", 'I', "Comma-separated list of INFO/ANN impact values to include in analysis (see https://pcingola.github.io/SnpEff/adds/VCFannotationformat_v1.0.pdf)"},
      {"cov", "<file>", 'c', "Covariates file"},
      {"debug-log", "<file>", '\x02', "Enables debug logging and specifies log file"},
      {"discovery-counts", "<file>", '\x02', "Writes discovery counts to specified file for use in empirical FDR calculation"},
      {"fmt-field", "<string>", '\x02', "Format field to use (DS, HDS, or GT)"},
      {"help", "", 'h', "Print Usage"},
      {"inv-norm", "", '\x01', "Inverse normalize response"},
      {"min-mac", "<int>", '\x02', "Minimum minor allele count (default: 1)"},
      {"min-maf", "<real>", '\x02', "Minimum minor allele frequency (default: 0.0)"},
      {"output", "<file>", 'o', "Output path (default: /dev/stdout)"},
      {"region", "<string>", 'r', "Genomic region to test (chrom:beg-end)"},
      {"resid-geno-threshold", "<real>", '\x02', "Associations with p-values <= this threshold are retested with residualized genotypes (default: -1 a.k.a never)"},
      {"pass-only", "", '\x01', "Only test PASS variants"},
      {"max-pvalue", "<real>", '\x02', "Excludes association results from output when p-value is above this threshold"},
      {"split-output", "", '\x01', "Splits output into multiple files (one per phenotype)"},
      {"print-model-fit", "", '\x01', "Produces file with statistics for regressing the phenotype on covariates"},
      {"permutations", "<file>", 'p', "Path to file containing sample ID mappings for permutations"}})
  {
    //    long_options_.reserve(long_options_.size() + additional_options.size());
    //    long_options_.insert(--long_options_.end(), additional_options.begin(), additional_options.end());

    //    short_options_.reserve((long_options_.size() - 1) * 2);
    //    std::vector<bool> mask(256, false);
    //    for (const auto& o : long_options_)
    //    {
    //      if (!mask[unsigned(o.val)])
    //      {
    //        short_options_ += (char)o.val;
    //        if (o.has_arg == required_argument)
    //          short_options_ += ':';
    //        mask[(unsigned)o.val] = true;
    //      }
    //    }
  }

  virtual ~qtl_prog_args() {}

  const std::unordered_set<std::string>& consequences() const { return consequences_; }
  const std::unordered_set<std::string>& impacts() const { return impacts_; }
  const std::list<std::pair<std::string, savvy::genomic_region>>& collapse_regions() const { return collapse_regions_; }
  const std::string& cov_path() const { return cov_path_; }
  const std::string& geno_path() const { return geno_path_; }
  const std::string& pheno_path() const { return pheno_path_; }
  const std::string& perm_path() const { return permutations_path_; }
  const std::string& discovery_counts_path() const { return discovery_counts_path_; }
  const std::string& output_path() const { return output_path_; }
  const std::string& fmt_field() const { return fmt_field_; }
  const std::string& debug_log_path() const { return debug_log_path_; }
  const std::unique_ptr<savvy::genomic_region>& region() const { return region_; }
  double rare_threshold() const { return rare_treshold_; }
  double min_mac() const { return min_mac_; }
  double min_maf() const { return min_maf_; }
  double max_pval() const { return max_pval_; }
  double resid_geno_threshold() const { return resid_geno_threshold_; }
  std::int64_t window_size() const { return 48; } // TODO: remove
  std::uint32_t seed() const { return seed_; }
  bool split_output() const { return split_output_; }
  bool print_model_fit() const { return print_model_fit_; }
  bool help_is_set() const { return help_; }
  bool invnorm() const { return invnorm_; }
  bool pass_only() const { return pass_only_; }

  bool update_fmt_field(const savvy::reader& geno_file, const std::vector<std::string>& field_priority)
  {
    std::unordered_set<std::string> fmt_avail;

    for (const auto& h : geno_file.format_headers())
      fmt_avail.insert(h.id);

    if (fmt_field_.empty())
    {
      for (auto it = field_priority.begin(); fmt_field_.empty() && it != field_priority.end(); ++it)
      {
        if (fmt_avail.find(*it) != fmt_avail.end())
          fmt_field_ = *it;
      }

      if (fmt_field_.empty())
        return std::cerr << "Error: file must contain DS, HDS, or GT format fields\n", false;
      std::cerr << "Notice: --fmt-field not specified so auto selecting " << fmt_field_ << std::endl;
    }
    else
    {
      if (fmt_avail.find(fmt_field_) == fmt_avail.end())
        return std::cerr << "Error: requested format field (" << fmt_field_ << ") not found in file headers\n", false;
    }
    return true;
  }

  static bool parse_bed_file(const std::string& filepoath, std::list<std::pair<std::string, savvy::genomic_region>>& dest)
  {

    std::string line;
    std::ifstream ifs(filepoath);
    if (!ifs)
      return std::cerr << "Error: invalid file path\n", false;

    if (!std::getline(ifs, line))
      return std::cerr << "Error: BED file empty\n", false;

    while (std::getline(ifs, line))
    {
      auto fields = utility::split_string_to_vector(line, '\t');
      if (fields.size() < 4)
        return std::cerr << "Error: BED file must have at least 4 columns\n", false;

      dest.emplace_back(fields[3], savvy::genomic_region(fields[0], std::atoll(fields[1].c_str()) + 1, std::atoll(fields[2].c_str())));
    }

    return true;
  }

  bool parse(int argc, char** argv)
  {
    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, short_opt_string_.c_str() /*"\x01\x02:bc:ho:p:r:"*/, long_options_.data(), &long_index )) != -1)
    {
      char copt = char(opt & 0xFF);
      switch (copt)
      {
      case '\x01':
        if (std::string("inv-norm") == long_options_[long_index].name)
        {
          invnorm_ = true;
        }
        else if (std::string("pass-only") == long_options_[long_index].name)
        {
          pass_only_ = true;
        }
        else if (std::string("split-output") == long_options_[long_index].name)
        {
          split_output_ = true;
        }
        else if (std::string("print-model-fit") == long_options_[long_index].name)
        {
          print_model_fit_ = true;
        }
        else
        {
          return std::cerr << "Error: invalid option " << long_options_[long_index].name << std::endl, false;
        }
        break;
      case '\x02':
        if (std::string("min-mac") == long_options_[long_index].name)
        {
          min_mac_ = std::atof(optarg ? optarg : "");
        }
        else if (std::string("min-maf") == long_options_[long_index].name)
        {
          min_maf_ = std::atof(optarg ? optarg : "");
        }
        else if (std::string("fmt-field") == long_options_[long_index].name)
        {
          fmt_field_ = optarg ? optarg : "";
          if (fmt_field_ != "DS" && fmt_field_ != "HDS" && fmt_field_ != "GT")
            return std::cerr << "Error: --fmt-field must be DS, HDS, or GT\n", false;
        }
        else if (std::string("debug-log") == long_options_[long_index].name)
        {
          debug_log_path_ = optarg ? optarg : "";
        }
        else if (std::string("discovery-counts") == long_options_[long_index].name)
        {
          discovery_counts_path_ = optarg ? optarg : "";
        }
        else if (std::string("max-pvalue") == long_options_[long_index].name)
        {
          max_pval_ = std::atof(optarg ? optarg : "");
        }
        else if (std::string("resid-geno-threshold") == long_options_[long_index].name)
        {
          resid_geno_threshold_ = std::atof(optarg ? optarg : "");
        }
        else
        {
          return std::cerr << "Error: invalid option " << long_options_[long_index].name << std::endl, false;
        }
        break;
      case 'h':
        help_ = true;
        return true;
      case 'c':
        cov_path_ = optarg ? optarg : "";
        break;
      case 'o':
        output_path_ = optarg ? optarg : "";
        break;
      case 'p':
        permutations_path_ = optarg ? optarg : "";
        break;
      case 'r':
        region_.reset(new savvy::genomic_region(utility::string_to_region(optarg ? optarg : "")));
        break;
      case 'R':
        if (!parse_bed_file(optarg ? optarg : "", collapse_regions_))
          return std::cerr << "Error: failed to parse --collapse-regions file\n", false;
        break;
      default:
        return false;
      }
    }

    int remaining_arg_count = argc - optind;

    if (remaining_arg_count == 2)
    {
      geno_path_ = argv[optind];
      pheno_path_ = argv[optind + 1];
      //phenotype_field_ = argv[optind + 2];
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

bool parse_permutation_file(const qtl_prog_args& args, const std::vector<std::string>& ids, std::vector<std::vector<std::size_t>>& dest)
{
  std::unordered_map<std::string, std::size_t> id_map;
  id_map.reserve(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i)
    id_map[ids[i]] = i;
  std::size_t match_count = 0, line_count = 0;

  if (args.cov_path().empty())
    return std::cerr << "Error: must pass separate covariates file\n", false;
  std::string path = args.perm_path();
  std::ifstream perm_file(path, std::ios::binary);

  dest.clear();
  std::string line;
  char* p = nullptr;
  while (std::getline(perm_file, line))
  {
    auto str_fields = utility::split_string_to_vector(line.c_str(), '\t');
    if (str_fields.size() < 2)
      return std::cerr << "Error: permutation file must have at least two columns\n", false;


    if (dest.empty())
      dest.resize(str_fields.size() - 1, std::vector<std::size_t>(id_map.size()));
    else if (dest.size() != str_fields.size() - 1)
      return std::cerr << "Error: permutation file must same number of columns for each line\n", false;

    auto row_idx_it = id_map.find(str_fields[0]);
    if (row_idx_it != id_map.end())
    {
      ++match_count;

      for (std::size_t i = 1; i < str_fields.size(); ++i)
      {
        assert((i - 1) < dest.size());
        auto perm_idx_it = id_map.find(str_fields[i]);
        if (perm_idx_it == id_map.end())
          return std::cerr << "Error: permutation file should contain only IDs from intersection of genotypes and phenotypes", false;
        else
          dest[i-1][row_idx_it->second] = perm_idx_it->second;
      }
    }
  }

  if (match_count != ids.size())
    return std::cerr << "Error: missing permuation mapping for " << (ids.size() - match_count) << " samples\n", false;
  return true;
}

bool process_cis_batch(const std::vector<bed_file::record>& phenos,  std::vector<residualizer>& residualizers, std::vector<std::vector<std::size_t>> subset_non_missing_map, /* std::deque<std::vector<std::int8_t>>& genos,*/ savvy::reader& geno_file, std::ostream& output_file, const qtl_prog_args& args)
{
  if (phenos.empty())
    return false;

  auto window_size = args.window_size();
  if (window_size >=0)
  {
    std::int64_t window_start = std::max(std::int64_t(1), phenos.front().beg() + 1 - window_size);
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
    pheno_resids[i] = residualizers[residualizers.size() == 1 ? 0 : i](phenos[i].data(), args.invnorm());
    s_y[i] = std::accumulate(pheno_resids[i].begin(),  pheno_resids[i].end(), scalar_type());
    s_yy[i] = std::inner_product(pheno_resids[i].begin(),  pheno_resids[i].end(), pheno_resids[i].begin(), scalar_type());
  }

  const std::vector<std::string> pass{"PASS"};
  savvy::compressed_vector<std::int8_t> geno;
  savvy::compressed_vector<scalar_type> geno_sub;
  std::vector<scalar_type> geno_sub_dense;
  savvy::variant var; std::size_t progress = 0;
  while (geno_file.read(var))
  {
    if (args.pass_only() && var.filters() != pass) continue;
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
          geno_sub.resize(phenos[pheno_idx].data().size() * ploidy);
          for (auto it = geno.begin(); it != geno.end(); ++it)
          {
            std::size_t sub_offset = subset_non_missing_map[pheno_idx][it.offset()/ploidy] + (it.offset() % ploidy);
            if (sub_offset < geno_sub.size())
            {
              if (*it == alt_idx)
              {
                geno_sub[sub_offset] = 1;
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

          savvy::stride_reduce(geno_sub, ploidy, savvy::plus_eov<scalar_type>());

          // TODO: implement sparse residualize
          // geno_sub = residualizers[residualizers.size() == 1 ? 0 : pheno_idx](geno_sub);

//          double mean = std::accumulate(geno_sub.begin(), geno_sub.end(), 0.) / geno_sub.size();
//          double stdev = std::sqrt(std::max(0., std::inner_product(geno_sub.begin(), geno_sub.end(), geno_sub.begin(), 0.0) / geno_sub.size() - mean*mean));
//          double stdev2 = 0.;
//          for (auto it = geno_sub.begin(); it != geno_sub.end(); ++it)
//            stdev2 += (mean - *it) * (mean - *it);
//          stdev2 = std::sqrt(stdev2 / geno_sub.size());
//
//          for(auto& element : geno_sub)
//          {
//            element = (element - mean);
//            element = (element / stdev) * stdev_before;
//          }

          linear_model::stats_t stats = linear_model::ols(geno_sub, xt::adapt(pheno_resids[pheno_idx], {pheno_resids[pheno_idx].size()}), std::accumulate(geno_sub.begin(), geno_sub.end(), scalar_type()), s_y[pheno_idx], s_yy[pheno_idx], geno_sub.size() - (residualizers[residualizers.size() == 1 ? 0 : pheno_idx].n_predictors()) - 2);
#if 0
          geno_sub_dense.clear();
          geno_sub_dense.resize(geno_sub.size());
          for (auto it = geno_sub.begin(); it != geno_sub.end(); ++it)
            geno_sub_dense[it.offset()] = *it;

          scalar_type s_x_dense = std::accumulate(geno_sub_dense.begin(), geno_sub_dense.end(), scalar_type());
          scalar_type mean = s_x_dense / geno_sub_dense.size();
          for(auto& element : geno_sub_dense)
          {
            element = (element - mean);
          }

          linear_model::stats_t stats_dense = linear_model::ols(geno_sub_dense, xt::adapt(pheno_resids[pheno_idx], {pheno_resids[pheno_idx].size()}), s_x_dense, s_y[pheno_idx], s_yy[pheno_idx], geno_sub_dense.size() - (residualizers[residualizers.size() == 1 ? 0 : pheno_idx].n_variables() + 2));
#endif
          output_file << var.chromosome()
                      << "\t" << var.position()
                      << "\t" << var.ref()
                      << "\t" << (var.alts().empty() ? "." : var.alts()[0])
                      << "\t" << var.id()
                      << "\t" << af
                      << "\t" << ac
                      << "\t" << an/ploidy
                      << "\t" << phenos[pheno_idx].pheno_id()
                      << "\t" << phenos[pheno_idx].chrom()
                      << "\t" << phenos[pheno_idx].beg() + 1
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

int cis_qtl_main(int argc, char** argv)
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

  std::vector<std::string> cov_names;
  xt::xtensor<scalar_type, 2> cov_mat;
  if (!parse_covariates_file(args.cov_path(), sample_intersection, cov_mat, cov_names))
    return std::cerr << "Error: failed parsing covariates file\n", EXIT_FAILURE;

  //cov_mat = (cov_mat - xt::mean(cov_mat, {0})) / xt::stddev(cov_mat, {0});
  cov_mat = cov_mat - xt::mean(cov_mat, {0});

  shrinkwrap::bgzf::ostream output_file(args.output_path());
  output_file << "geno_chrom\tgeno_pos\tref\talt\tvariant_id\taf\tac\tns\tpheno_id\tpheno_chrom\tpheno_beg\tpheno_end\t" << linear_model::stats_t::header_column_names() << std::endl;

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
        return std::cerr << "Error: size mismatch at " << __FILE__ << ":" << __LINE__ << " (this should not happen)\n", false;

      //pheno_sub[i].resize(phenos[i].data().size());
      subset_non_missing_map[i] = phenos[i].remove_missing();
      for (std::size_t j = 0; j < subset_non_missing_map[i].size(); ++j)
      {
        if (subset_non_missing_map[i][j] <= j)
          keep_samples.push_back(j);
      }
//      if (keep_samples.size() != subset_non_missing_map[i].size())
//        throw std::runtime_error("Missing detected at " + std::to_string(i));
      if (i == 0 || keep_samples.size() != subset_non_missing_map[i].size() || keep_samples.size() != residualizers[0].n_samples())
      {
        residualizers[i] = residualizer(xt::view(cov_mat, xt::keep(keep_samples), xt::all()));
      }
      else
      {
        residualizers[i] = residualizers[0];
      }
    }

    for (std::size_t i = 0; i < residualizers.size(); ++i)
    {
      if (residualizers[i].n_samples() != subset_non_missing_map[0].size())
        break;
      if (i + 1 == residualizers.size())
        residualizers.resize(1);
    }

    if (!process_cis_batch(phenos, residualizers, subset_non_missing_map, /* genos,*/ geno_file, output_file, args))
      return std::cerr << "Error: processing batch failed\n", EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

class output_wrapper
{
private:
  std::vector<std::unique_ptr<shrinkwrap::bgzf::ostream>> out_files;
  std::vector<std::string> pheno_names_;
  bool split_;
public:
  output_wrapper(const std::string& output_path, std::vector<std::string> pheno_names, bool split_output) :
    pheno_names_(std::move(pheno_names)),
    split_(split_output)
  {
    if (split_)
    {
      out_files.resize(pheno_names_.size());
      for (std::size_t i = 0; i < out_files.size(); ++i)
        out_files[i] = std::make_unique<shrinkwrap::bgzf::ostream>(output_path + pheno_names_[i] + ".tsv.gz");
    }
    else
    {
      out_files.emplace_back(std::make_unique<shrinkwrap::bgzf::ostream>(output_path));
    }

    for (std::size_t i = 0; i < out_files.size(); ++i)
    {
      *out_files[i] << "chrom\tpos\tref\talt\tvariant_id\taf\tac\tns\t" << linear_model::stats_t::header_column_names();
      if(!split_)
        *out_files[i] << "\tpheno_id";
      *out_files[i] << std::endl;
    }
  }

  bool write(const savvy::site_info& var, float af, std::int64_t ac, std::int64_t ns, const linear_model::stats_t& stats, std::size_t pheno_idx)
  {
    std::size_t file_idx = 0;
    if (split_)
      file_idx = pheno_idx;
    (*out_files[file_idx]) << var.chromosome()
       << "\t" << var.position()
       << "\t" << var.ref()
       << "\t" << (var.alts().empty() ? "." : var.alts()[0])
       << "\t" << var.id()
       << "\t" << af
       << "\t" << ac
       << "\t" << ns
       << "\t" << stats;
    if (!split_)
      (*out_files[file_idx]) << "\t" << pheno_names_[pheno_idx];
    (*out_files[file_idx]) << "\n";

    return out_files[file_idx]->good();
  }

  bool close()
  {
    for (std::size_t i = 0; i < out_files.size(); ++i)
    {
      out_files[i]->flush();
      if (!out_files[i]->good())
        return std::cerr << "Error: output file close failure\n", false;
    }

    return true;
  }
};

class collapse_output_wrapper
{
private:
  std::vector<std::unique_ptr<shrinkwrap::bgzf::ostream>> out_files;
  std::vector<std::string> pheno_names_;
  bool split_;
public:
  collapse_output_wrapper(const std::string& output_path, std::vector<std::string> pheno_names, bool split_output) :
    pheno_names_(std::move(pheno_names)),
    split_(split_output)
  {
    if (split_)
    {
      out_files.resize(pheno_names_.size());
      for (std::size_t i = 0; i < out_files.size(); ++i)
        out_files[i] = std::make_unique<shrinkwrap::bgzf::ostream>(output_path + pheno_names_[i] + ".tsv.gz");
    }
    else
    {
      out_files.emplace_back(std::make_unique<shrinkwrap::bgzf::ostream>(output_path));
    }

    for (std::size_t i = 0; i < out_files.size(); ++i)
    {
      *out_files[i] << "chrom\tbeg\tend\tregion_id\trare_threshold\tn_carriers\tns\t" << linear_model::stats_t::header_column_names();
      if(!split_)
        *out_files[i] << "\tpheno_id";
      *out_files[i] << std::endl;
    }
  }

  bool write(const savvy::genomic_region& reg, const std::string& reg_id, float rare_threshold, std::int64_t n_carriers, std::int64_t ns, const linear_model::stats_t& stats, std::size_t pheno_idx)
  {
    std::size_t file_idx = 0;
    if (split_)
      file_idx = pheno_idx;
    (*out_files[file_idx]) << reg.chromosome()
                           << "\t" << reg.from()
                           << "\t" << reg.to()
                           << "\t" << reg_id
                           << "\t" << rare_threshold
                           << "\t" << n_carriers
                           << "\t" << ns
                           << "\t" << stats;
    if (!split_)
      (*out_files[file_idx]) << "\t" << pheno_names_[pheno_idx];
    (*out_files[file_idx]) << "\n";

    return out_files[file_idx]->good();
  }

  bool close()
  {
    for (std::size_t i = 0; i < out_files.size(); ++i)
    {
      out_files[i]->flush();
      if (!out_files[i]->good())
        return std::cerr << "Error: output file close failure\n", false;
    }

    return true;
  }
};

class discovery_counter
{
private:
  std::vector<std::vector<std::size_t>> counts_;
public:
//  discovery_counter(double min_maf)
//  {
//    assert(min_maf <= 1.);
//    assert(min_maf > 0.);
//    counts_.resize(1 + std::size_t(std::floor(-std::log10(min_maf))));
//  }

  template <typename T>
  void operator()(double maf, const T& stats)
  {
    assert(maf <= 1.);
    assert(maf > 0.);
    std::size_t maf_idx = std::size_t(std::floor(-std::log10(maf)));
    std::size_t pval_idx = std::size_t(std::floor(-std::log10(stats.pvalue)));
    
    if (maf_idx >= counts_.size())
      counts_.resize(maf_idx + 1);

    if (pval_idx >= counts_[maf_idx].size())
      counts_[maf_idx].resize(pval_idx + 1);

    ++counts_[maf_idx][pval_idx];
  }

  static bool write(const std::vector<discovery_counter>& vec, std::ostream& ofs)
  {
    if (vec.empty())
       return false;
    std::size_t n_maf_bins = vec[0].counts_.size();
    for (std::size_t i = 1; i < vec.size(); ++i)
    {
      if (n_maf_bins != vec[i].counts_.size())
        return false;
    }

    ofs << "#maf_bin\tpval_bin";
    for (std::size_t i = 0; i < vec.size(); ++i)
      ofs << "\tcount" << i;
    ofs << std::endl;

    for (std::size_t maf_idx = 0; maf_idx < n_maf_bins; ++maf_idx)
    {
      bool keep_going = true;
      for (std::size_t pval_idx = 0; keep_going; ++pval_idx)
      {
        keep_going = false;
        bool has_non_zero = false;
        for (auto it = vec.begin(); it != vec.end(); ++it)
        {
          if (pval_idx < it->counts_[maf_idx].size())
          {
            keep_going = true;
            if (it->counts_[maf_idx][pval_idx] != 0)
            {
              has_non_zero = true;
              break;
            }
          }
        }

        if (has_non_zero)
        {
          ofs << maf_idx << "\t" << pval_idx;
          for (auto it = vec.begin(); it != vec.end(); ++it)
          {
            ofs.put('\t');
            if (pval_idx < it->counts_[maf_idx].size())
              ofs << it->counts_[maf_idx][pval_idx];
            else
              ofs.put('0');
          }
          ofs.put('\n');
        }
      }
    }

    return ofs.good();
  }
};

bool process_burden_vector(const savvy::genomic_region& reg, const std::string& reg_id, const std::vector<std::vector<scalar_type>>& pheno_resids, const std::vector<linear_model::variable_stats<scalar_type>>& pheno_stats,  const std::vector<residualizer>& residualizers, const std::vector<std::vector<std::size_t>>& subset_non_missing_map, std::vector<scalar_type>& burden, collapse_output_wrapper& output_file, const qtl_prog_args& args, std::size_t ploidy, std::vector<discovery_counter>& discovery_counts)
{
  std::size_t n_perm = 0;
  if (!discovery_counts.empty())
    n_perm = discovery_counts.size() - 1;
  std::size_t stride = n_perm + 1;

  std::vector<scalar_type> burden_sub;

  for (std::size_t pheno_idx = 0; pheno_idx < pheno_resids.size(); ++pheno_idx)
  {
    const std::vector<scalar_type>& pheno_ref = pheno_resids[pheno_idx];
    const std::vector<std::size_t>& pheno_map_ref = subset_non_missing_map[pheno_idx];

    std::size_t n = pheno_resids[pheno_idx].size() / stride;

    //double ac = geno_stats_sub.sum();
    std::size_t an = n * ploidy;
    //double af = ac / an;
    //double mac = (ac > (an/2.f) ? an - ac : ac);
    //double maf = (af > 0.5f ? 1.f - af : af);

    if (an == 0) continue;
//    if (mac < args.min_mac()) continue;
//    if (maf < args.min_maf()) continue;

    burden_sub.resize(n);

    for (std::size_t i = 0; i < burden.size(); ++i)
    {
      std::size_t off = pheno_map_ref[i];
      if (off < n)
      {
        burden_sub[off] = burden[i];
      }
    }

    std::int64_t n_carriers = n - std::count(burden_sub.begin(), burden_sub.end(), scalar_type());

    burden_sub = residualizers[residualizers.size() == 1 ? 0 : pheno_idx](burden_sub, false);

    scalar_type s_x = 0;
    scalar_type s_xx = 0;
    std::vector<scalar_type> s_xy(stride);
    for (std::size_t i = 0; i < burden_sub.size(); ++i)
    {
      s_x += burden_sub[i];
      s_xx += burden_sub[i] * burden_sub[i];
      for (std::size_t j = 0; j < stride; ++j)
        s_xy[j] += burden_sub[i] * pheno_ref[i * stride + j];
    }


    std::vector<linear_model::stats_t> stats_dense(stride);
    for (std::size_t i = 0; i < stride; ++i)
      stats_dense[i] = linear_model::ols(n, s_xy[i], linear_model::variable_stats<scalar_type>(burden_sub), pheno_stats[pheno_idx], burden_sub.size() - (residualizers[residualizers.size() == 1 ? 0 : pheno_idx].n_predictors()) - 2);

    if (stride > 1)
    {
      for (std::size_t i = 0; i < stride; ++i)
        discovery_counts[i](args.rare_threshold(), stats_dense[i]);
    }

    if (stats_dense[0].pvalue > args.max_pval()) continue;
    if (!output_file.write(reg, reg_id, args.rare_threshold(), n_carriers, n, stats_dense[0], pheno_idx))
      return std::cerr << "Error: failed writing to output file\n", false;
  }
  return true;
}

template <typename GenoT>
bool process_variant(const savvy::site_info& var, const std::vector<std::vector<scalar_type>>& pheno_resids, const std::vector<linear_model::variable_stats<scalar_type>>& pheno_stats,  const std::vector<residualizer>& residualizers, const std::vector<std::vector<std::size_t>>& subset_non_missing_map, const savvy::compressed_vector<GenoT>& geno, std::vector<scalar_type>& geno_dense, const linear_model::variable_stats<scalar_type>& geno_stats, output_wrapper& output_file, const qtl_prog_args& args, std::size_t ploidy, std::vector<discovery_counter>& discovery_counts)
{
  std::size_t n_perm = 0;
  if (!discovery_counts.empty())
    n_perm = discovery_counts.size() - 1;
  std::size_t stride = n_perm + 1;

  for (std::size_t pheno_idx = 0; pheno_idx < pheno_resids.size(); ++pheno_idx)
  {
    const std::vector<scalar_type>& pheno_ref = pheno_resids[pheno_idx];
    const std::vector<std::size_t>& pheno_map_ref = subset_non_missing_map[pheno_idx];

    std::size_t n = pheno_resids[pheno_idx].size() / stride;

    scalar_type s_x = geno_stats.sum();
    scalar_type s_xx = geno_stats.sum_squared();
    for (auto it = geno.begin(); it != geno.end(); ++it)
    {
      if (pheno_map_ref[it.offset()] >= n)
      {
        s_x -= *it;
        s_xx -= (*it) * (*it);
      }
    }

    linear_model::variable_stats<scalar_type> geno_stats_sub(s_x, s_xx);

    double ac = geno_stats_sub.sum();
    std::size_t an = n * ploidy;
    double af = ac / an;
    double mac = (ac > (an/2.f) ? an - ac : ac);
    double maf = (af > 0.5f ? 1.f - af : af);

    if (an == 0) continue;
    if (mac < args.min_mac()) continue;
    if (maf < args.min_maf()) continue;

    //savvy::stride_reduce(geno_sub, ploidy, savvy::plus_eov<scalar_type>());



    std::vector<scalar_type> s_xy(stride);
    for (auto it = geno.begin(); it != geno.end(); ++it)
    {
      std::size_t off = pheno_map_ref[it.offset()];
      if (off < n)
      {
        for (std::size_t j = 0; j < stride; ++j)
          s_xy[j] += (*it) * pheno_ref[off * stride + j];
      }
    }


    //          double mean = std::accumulate(geno_sub.begin(), geno_sub.end(), 0.) / geno_sub.size();
    //          double stdev = std::sqrt(std::max(0., std::inner_product(geno_sub.begin(), geno_sub.end(), geno_sub.begin(), 0.0) / geno_sub.size() - mean*mean));
    //          double stdev2 = 0.;
    //          for (auto it = geno_sub.begin(); it != geno_sub.end(); ++it)
    //            stdev2 += (mean - *it) * (mean - *it);
    //          stdev2 = std::sqrt(stdev2 / geno_sub.size());
    //
    //          for(auto& element : geno_sub)
    //          {
    //            element = (element - mean);
    //            element = (element / stdev) * stdev_before;
    //          }

    std::vector<linear_model::stats_t> stats(stride);
    if (args.resid_geno_threshold() < 1.)
    {
      for (std::size_t i = 0; i < stride; ++i)
        stats[i] = linear_model::ols(n, s_xy[i], geno_stats_sub, pheno_stats[pheno_idx], n - (residualizers[residualizers.size() == 1 ? 0 : pheno_idx].n_predictors()) - 2);
    }
    if (args.resid_geno_threshold() >= 1. || std::any_of(stats.begin(), stats.end(), [&args](const linear_model::stats_t& s) { return s.pvalue <= args.resid_geno_threshold(); }))
    {
      geno_dense.clear();
      geno_dense.resize(n);
      for (auto it = geno.begin(); it != geno.end(); ++it)
      {
        std::size_t off = pheno_map_ref[it.offset()];
        if (off < n)
          geno_dense[off] = *it;
      }

      geno_dense = residualizers[residualizers.size() == 1 ? 0 : pheno_idx](geno_dense);
      std::fill(s_xy.begin(), s_xy.end(), scalar_type());
      for (std::size_t i = 0; i < geno_dense.size(); ++i)
      {
          for (std::size_t j = 0; j < stride; ++j)
            s_xy[j] += geno_dense[i] * pheno_ref[i * stride + j];
      }

      std::vector<linear_model::stats_t> stats_dense(stride);
      for (std::size_t i = 0; i < stride; ++i)
        stats_dense[i] = linear_model::ols(n, s_xy[i], linear_model::variable_stats<scalar_type>(geno_dense), pheno_stats[pheno_idx], geno_dense.size() - (residualizers[residualizers.size() == 1 ? 0 : pheno_idx].n_predictors()) - 2);

      for (std::size_t i = 0; i < stride; ++i)
      {
        if (args.resid_geno_threshold() >= 1. || stats[i].pvalue <= args.resid_geno_threshold())
          stats[i] = stats_dense[i];
      }
    }

    if (stride > 1)
    {
      for (std::size_t i = 0; i < stride; ++i)
        discovery_counts[i](maf, stats[i]);
    }

    if (stats[0].pvalue > args.max_pval()) continue;
    if (!output_file.write(var, af, ac, n, stats[0], pheno_idx))
      return std::cerr << "Error: failed writing to output file\n", false;
  }
  return true;
}

bool process_collapse(const std::vector<std::vector<scalar_type>>& pheno_resids, const std::vector<linear_model::variable_stats<scalar_type>>& pheno_stats, std::vector<discovery_counter>& discovery_counts, const std::vector<residualizer>& residualizers, const std::vector<std::vector<std::size_t>>& subset_non_missing_map, savvy::reader& geno_file, collapse_output_wrapper& output_file, const qtl_prog_args& args)
{
  if (pheno_resids.empty())
    return false;

  std::unordered_set<std::string> impacts = {"HIGH"}; // TODO:
  savvy::genomic_region reg = args.collapse_regions().front().second; // TODO:
  std::string reg_id = args.collapse_regions().front().first;

  if (!geno_file.reset_bounds(reg))
    return std::cerr << "Could not open genomic region\n", false;

  auto total_start = std::chrono::high_resolution_clock::now();

  savvy::compressed_vector<std::int8_t> geno;
  std::vector<scalar_type> burden_dense;
  savvy::variant var;
  std::string ann;
  std::vector<std::string> ann_vec;
  while (geno_file.read(var))
  {
    double af = std::numeric_limits<double>::quiet_NaN();
    std::int64_t ac, an;
    if (!var.get_info("AF", af))
    {
      if (var.get_info("AC", ac) && var.get_info("AN", an))
        af = double(ac) / double(an);
    }

    if (std::isnan(af))
      return std::cerr << "Error: AF or AC/AN INFO fields must be present for burden testing\n", false;

    if (af >= args.rare_threshold()) continue;

    bool geno_ready = false;

    bool fetch_ann = true;
    if (fetch_ann)
    {
      if (!var.get_info("ANN", ann))
        return std::cerr << "Error: INFO/ANN must be present when using --collapse-anno or --collapse-impact\n", false;
      ann_vec = utility::split_string_to_vector(ann, ',');
    }

    for (std::size_t alt_idx = 1; alt_idx <= var.alts().size(); ++alt_idx)
    {
      if (fetch_ann)
      {
        std::string allele = var.alts()[alt_idx-1];
        bool process_allele = false;
        for (auto it = ann_vec.begin(); it != ann_vec.end() && !process_allele; ++it)
        {
          auto ann_fields = utility::split_string_to_vector(*it, '|');
          if (ann_fields.size() > 2 && ann_fields[0] == allele)
          {
            if (impacts.size() && impacts.find(ann_fields[2]) != impacts.end())
              process_allele = true;
          }
        }

        if (process_allele)
        {
          if (!geno_ready)
          {
            var.get_format("GT", geno);
            if (burden_dense.empty())
              burden_dense.resize(geno.size());
            else if (burden_dense.size() != geno.size())
              return std::cerr << "Error: inconsistent ploidy in FORMAT field\n", false;
            geno_ready = true;
          }

          for (auto gt = geno.begin(); gt != geno.end(); ++gt)
          {
            if (*gt == alt_idx)
              burden_dense[gt.offset()] += 1; // TODO: maybe add weight by AF option
            else if (savvy::typed_value::is_end_of_vector(*gt))
              burden_dense[gt.offset()] = *gt;
          }
        }
      }
    }
  }

  assert(!subset_non_missing_map.empty());
  std::size_t ploidy = burden_dense.size() / subset_non_missing_map[0].size();
  savvy::stride_reduce(burden_dense, ploidy, savvy::plus_eov<scalar_type>());

  if (!process_burden_vector(reg, reg_id, pheno_resids, pheno_stats, residualizers, subset_non_missing_map, burden_dense, output_file, args, ploidy, discovery_counts))
    return false;

  std::cerr << "total_duration:" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - total_start).count() << std::endl;

  if (geno_file.bad())
    return std::cerr << "Error: input file read error\n", false;

  //  std::cerr << "subset_duration:" << std::chrono::duration_cast<std::chrono::seconds>(subset_duration).count() << std::endl;
  //  std::cerr << "stride_reduce_duration:" << std::chrono::duration_cast<std::chrono::seconds>(stride_reduce_duration).count() << std::endl;
  //  std::cerr << "stats_duration:" << std::chrono::duration_cast<std::chrono::seconds>(stats_duration).count() << std::endl;
  //  std::cerr << "ols_duration:" << std::chrono::duration_cast<std::chrono::seconds>(ols_duration).count() << std::endl;
  //  std::cerr << "ols_dense_duration:" << std::chrono::duration_cast<std::chrono::seconds>(ols_dense_duration).count() << std::endl;
  //  std::cerr << "write_duration (ms):" << std::chrono::duration_cast<std::chrono::microseconds>(write_duration).count() << std::endl;

  if (!discovery_counts.empty())
  {
    std::ofstream discovery_file(args.discovery_counts_path());
    if (!discovery_counter::write(discovery_counts, discovery_file))
      return std::cerr << "Error: failed writing discovery counts file\n", false;
    discovery_file.close();
    if (!discovery_file.good())
      return std::cerr << "Error: failed closing discovery counts file\n", false;
  }

  return true;
}

bool process_trans_batch(const std::vector<std::vector<scalar_type>>& pheno_resids, const std::vector<linear_model::variable_stats<scalar_type>>& pheno_stats, std::vector<discovery_counter>& discovery_counts, const std::vector<residualizer>& residualizers, const std::vector<std::vector<std::size_t>>& subset_non_missing_map, savvy::reader& geno_file, /*std::ostream*/output_wrapper& output_file, const qtl_prog_args& args)
{
  if (pheno_resids.empty())
    return false;

  if (args.region() && !geno_file.reset_bounds(*args.region()))
    return std::cerr << "Could not open genomic region\n", false;

  auto total_start = std::chrono::high_resolution_clock::now();

  savvy::compressed_vector<std::int8_t> geno;
  //savvy::compressed_vector<std::int8_t> geno_sub;
  savvy::compressed_vector<std::int8_t> geno_biallele;
  std::vector<scalar_type> geno_dense;
  savvy::variant var;
  while (geno_file.read(var))
  {
    var.get_format("GT", geno);

    assert(!subset_non_missing_map.empty());
    std::size_t ploidy = geno.size() / subset_non_missing_map[0].size();

    if (var.alts().size() > 1)
    {
      geno_biallele.resize(0);
      geno_biallele.resize(geno.size());
      for (std::size_t alt_idx = 1; alt_idx <= var.alts().size(); ++alt_idx)
      {
        for (auto it = geno.begin(); it != geno.end(); ++it)
        {
          if (*it == alt_idx)
            geno_biallele[it.offset()] = 1;
          else if (savvy::typed_value::is_end_of_vector(*it))
            geno_biallele[it.offset()] = *it;
        }
        savvy::stride_reduce(geno_biallele, ploidy, savvy::plus_eov<scalar_type>());
        linear_model::variable_stats<scalar_type> geno_stats(geno_biallele);
        if (!process_variant(var, pheno_resids, pheno_stats, residualizers, subset_non_missing_map, geno_biallele, geno_dense, geno_stats, output_file, args, ploidy, discovery_counts))
          return false;
      }
    }
    else
    {
      savvy::stride_reduce(geno, ploidy, savvy::plus_eov<scalar_type>());
      linear_model::variable_stats<scalar_type> geno_stats(geno);
      if (!process_variant(var, pheno_resids, pheno_stats, residualizers, subset_non_missing_map, geno, geno_dense, geno_stats, output_file, args, ploidy, discovery_counts))
        return false;
    }
  }

  std::cerr << "total_duration:" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - total_start).count() << std::endl;

  if (geno_file.bad())
    return std::cerr << "Error: input file read error\n", false;

//  std::cerr << "subset_duration:" << std::chrono::duration_cast<std::chrono::seconds>(subset_duration).count() << std::endl;
//  std::cerr << "stride_reduce_duration:" << std::chrono::duration_cast<std::chrono::seconds>(stride_reduce_duration).count() << std::endl;
//  std::cerr << "stats_duration:" << std::chrono::duration_cast<std::chrono::seconds>(stats_duration).count() << std::endl;
//  std::cerr << "ols_duration:" << std::chrono::duration_cast<std::chrono::seconds>(ols_duration).count() << std::endl;
//  std::cerr << "ols_dense_duration:" << std::chrono::duration_cast<std::chrono::seconds>(ols_dense_duration).count() << std::endl;
//  std::cerr << "write_duration (ms):" << std::chrono::duration_cast<std::chrono::microseconds>(write_duration).count() << std::endl;

  if (!discovery_counts.empty())
  {
    std::ofstream discovery_file(args.discovery_counts_path());
    if (!discovery_counter::write(discovery_counts, discovery_file))
      return std::cerr << "Error: failed writing discovery counts file\n", false;
    discovery_file.close();
    if (!discovery_file.good())
      return std::cerr << "Error: failed closing discovery counts file\n", false;
  }

  return true;
}

int trans_qtl_main(int argc, char** argv)
{
//  xt::xtensor<double, 2> X = {{ 1.,  -7.833333,     -9.5, -11.383333},
//    { 1.,  19.166667,    -19.5,  11.916667},
//    {1.,  50.166667,     12.5,  -1.583333},
//    {1., -53.833333,    -17.5, -13.083333},
//    {1.,  32.166667,     23.5,   8.016667},
//    {1., -39.833333,     12.,   3.116},
//    {1., -9.833333,     100.5,   2.116},
//    {1., -3.833333,     1.5,   7.116}};
//
//  std::vector<double> y = { 3.7,  7.833333,     9.5, 11.383333, 12., 100., 32., 12.};
//  std::vector<double> x = { 3.9,  6.,     8.5, 15.333, 10., 120., 50., 2.};
//
//  linear_model::variable_stats<double> sy(y);
//  linear_model::variable_stats<double> sx(x);
//
//  linear_model::stats_t a = linear_model::ols(x,  xt::adapt(y, {y.size()}), sx, sy, x.size() - (X.shape()[1] + 1));
//
//  mean_center(y);
//  sy = linear_model::variable_stats<double>(y);
//  linear_model::stats_t b = linear_model::ols(x,  xt::adapt(y, {y.size()}), sx, sy, x.size() - (X.shape()[1] + 1));
//
//  mean_center(x);
//  sx = linear_model::variable_stats<double>(x);
//  linear_model::stats_t c = linear_model::ols(x,  xt::adapt(y, {y.size()}), sx, sy, x.size() - (X.shape()[1] + 1));
//
//  residualizer r(X);
//  y = r(y);
//  x = r(x);
//  sy = linear_model::variable_stats<double>(y);
//  sx = linear_model::variable_stats<double>(x);
//
//  linear_model::stats_t d = linear_model::ols(x,  xt::adapt(y, {y.size()}), sx, sy, x.size() - (X.shape()[1] + 1));
//
//  mean_center(y);
//  sy = linear_model::variable_stats<double>(y);
//  linear_model::stats_t e = linear_model::ols(x,  xt::adapt(y, {y.size()}), sx, sy, x.size() - (X.shape()[1] + 1));
//
//  mean_center(x);
//  sx = linear_model::variable_stats<double>(x);
//  linear_model::stats_t f = linear_model::ols(x,  xt::adapt(y, {y.size()}), sx, sy, x.size() - (X.shape()[1] + 1));

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

  savvy::reader geno_file(args.geno_path());
  if (!geno_file)
    return std::cerr << "Could not open geno file\n", EXIT_FAILURE;
  geno_file.phasing_status(savvy::phasing::none); // Faster parsing of BCF/VCF

  std::vector<std::string> sample_intersection;
  std::vector<std::string> pheno_names;
  std::vector<std::vector<scalar_type>> phenos;
  if (!parse_phenotypes_file(args.pheno_path(), geno_file, sample_intersection, phenos, pheno_names))
    return std::cerr << "Error: failed parsing phenotypes file\n", EXIT_FAILURE;

  std::vector<std::string> cov_names;
  xt::xtensor<scalar_type, 2> cov_mat;
  if (args.cov_path().empty())
    cov_mat = xt::ones<scalar_type, std::vector<std::size_t>>({sample_intersection.size(), 1}); //TODO: test
  else if (!parse_covariates_file(args.cov_path(), sample_intersection, cov_mat, cov_names))
    return std::cerr << "Error: failed parsing covariates file\n", EXIT_FAILURE;

  std::vector<std::vector<std::size_t>> permutation_matrix;
  if (!args.discovery_counts_path().empty())
  {
    if (args.perm_path().empty())
    {
      permutation_matrix.resize(1, {sample_intersection.size()});
      std::iota(permutation_matrix.front().begin(), permutation_matrix.front().end(), 0);
      std::default_random_engine prng{args.seed()};
      std::shuffle(permutation_matrix.front().begin(), permutation_matrix.front().end(), prng);
    }
    else
    {
      if (!parse_permutation_file(args, sample_intersection, permutation_matrix))
        return std::cerr << "Error: failed parsing permutation file file\n", EXIT_FAILURE;
    }
  }


  if (args.print_model_fit() && false)
  {
    // scale covariates
    auto sd = xt::eval(xt::stddev(cov_mat, {0}));
    sd(0) = 1.;
    auto mu = xt::eval(xt::mean(cov_mat, {0}));
    mu(0) = 0.;
    cov_mat = (cov_mat - mu) / sd;
  }
  //cov_mat = (cov_mat - xt::mean(cov_mat, {0})) / xt::stddev(cov_mat, {0});
  //cov_mat = cov_mat - xt::mean(cov_mat, {0});

  for (auto it = pheno_names.begin(); args.split_output() && it != pheno_names.end(); ++it)
  {
    if (it->find('/') != std::string::npos)
      return std::cerr << "Error: forward slash encountered in phenotype column name which is not supported when using --split-output\n", false;
  }

  std::vector<std::vector<std::size_t>> subset_non_missing_map(phenos.size(), std::vector<std::size_t>(phenos.front().size()));
  std::vector<std::size_t> keep_samples(sample_intersection.size()); // reserve max number of samples
  std::vector<residualizer> residualizers(phenos.size());
  for (std::size_t i = 0; i < phenos.size(); ++i)
  {
    keep_samples.resize(0);
    subset_non_missing_map[i] = utility::remove_missing(phenos[i]);
    for (std::size_t j = 0; j < subset_non_missing_map[i].size(); ++j)
    {
      if (subset_non_missing_map[i][j] <= j)
        keep_samples.push_back(j);
    }

    if (i == 0 || keep_samples.size() != subset_non_missing_map[i].size() || keep_samples.size() != residualizers[0].n_samples())
      residualizers[i] = residualizer(xt::view(cov_mat, xt::keep(keep_samples), xt::all()));
    else
      residualizers[i] = residualizers[0];
  }

  for (std::size_t i = 0; i < residualizers.size(); ++i)
  {
    if (residualizers[i].n_samples() != subset_non_missing_map[0].size())
      break;
    if (i + 1 == residualizers.size())
      residualizers.resize(1);
  }



  if (args.print_model_fit())
  {
    std::ofstream ofs(args.output_path());
    //xt::xtensor<scalar_type, 2> m = xt::eval(xt::linalg::dot(xt::linalg::pinv(xt::linalg::dot(xt::transpose(cov_mat), cov_mat)), transpose(cov_mat)));
    ofs << "pheno_id\tr2\tadjusted_r2\tintercept";
    for (auto it = cov_names.begin(); it != cov_names.end(); ++it)
      ofs << "\t" << *it;
    ofs << std::endl;

    for (std::size_t i = 0; ofs.good() && i < phenos.size(); ++i)
    {
      ofs << pheno_names[i] << "\t";
      residualizers[i < residualizers.size() ? i : 0].print_model_fit(ofs, phenos[i]);
    }

    if (ofs.good())
      return EXIT_SUCCESS;
    return EXIT_FAILURE;
  }

  std::vector<discovery_counter> discovery_counts;
  std::size_t n_perm = permutation_matrix.size();
  if (n_perm)
    discovery_counts.resize(n_perm + 1);
  std::vector<linear_model::variable_stats<scalar_type>> pheno_stats(phenos.size());
  std::vector<std::vector<scalar_type>> pheno_resids(phenos.size());
  for (std::size_t pheno_idx = 0; pheno_idx < phenos.size(); ++pheno_idx)
  {
    pheno_resids[pheno_idx] = residualizers[residualizers.size() == 1 ? 0 : pheno_idx](phenos[pheno_idx], args.invnorm());
    pheno_stats[pheno_idx] = linear_model::variable_stats<scalar_type>(pheno_resids[pheno_idx]);

    if (n_perm)
    {
      assert(!subset_non_missing_map.empty());
      std::size_t stride = n_perm + 1;
      std::vector<scalar_type> tmp_pheno_with_perm(pheno_resids[pheno_idx].size() * stride, std::numeric_limits<scalar_type>::quiet_NaN());

      for (std::size_t i = 0; i < pheno_resids[pheno_idx].size(); ++i)
        tmp_pheno_with_perm[i * stride] = pheno_resids[pheno_idx][i];

      for (std::size_t perm_idx = 0; perm_idx < n_perm; ++perm_idx)
      {
        std::size_t dest_idx = 0;
        for (std::size_t i = 0; i < permutation_matrix[perm_idx].size(); ++i)
        {
          std::size_t primary_idx = subset_non_missing_map[pheno_idx][permutation_matrix[perm_idx][i]];
          if (primary_idx < pheno_resids[pheno_idx].size())
          {
            tmp_pheno_with_perm[(dest_idx++) * stride + (perm_idx+1)] = pheno_resids[pheno_idx][primary_idx];
          }
        }
        assert(dest_idx == pheno_resids[pheno_idx].size());
      }
      std::swap(pheno_resids[pheno_idx], tmp_pheno_with_perm);
    }
  }

  if (args.collapse_regions().size())
  {
    collapse_output_wrapper output(args.output_path(), pheno_names, args.split_output());

    if (!process_collapse(pheno_resids, pheno_stats, discovery_counts, residualizers, subset_non_missing_map, geno_file, output, args))
      return std::cerr << "Error: processing batch failed\n", EXIT_FAILURE;

    return output.close() ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  else
  {
    // shrinkwrap::bgzf::ostream output_file(args.output_path());
    // output_file << "geno_chrom\tgeno_pos\tref\talt\tvariant_id\tmaf\tmac\tns\t" << linear_model::stats_t::header_column_names() << "\tpheno_id" << std::endl;
    output_wrapper output(args.output_path(), pheno_names, args.split_output());

    if (!process_trans_batch(pheno_resids, pheno_stats, discovery_counts, residualizers, subset_non_missing_map, geno_file, output, args))
      return std::cerr << "Error: processing batch failed\n", EXIT_FAILURE;

    return output.close() ? EXIT_SUCCESS : EXIT_FAILURE;
  }
}
