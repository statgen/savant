/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SAVANT_ASSOC_HPP
#define SAVANT_ASSOC_HPP

#include "utility.hpp"
#include "getopt_wrapper.hpp"

#include <savvy/reader.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <getopt.h>

typedef double scalar_type;

class assoc_prog_args : public getopt_wrapper
{
private:
  //std::string sub_command_;
  //std::vector<option> long_options_;
  //std::string short_options_;
  std::vector<std::string> covariate_fields_;
  std::string id_field_;
  std::string phenotype_field_;
  std::string geno_path_;
  std::string pheno_path_;
  std::string output_path_ = "/dev/stdout";
  std::string debug_log_path_ = "/dev/null";
  std::string wgeno_file_path_;
  std::string kinship_file_path_;
  std::string fmt_field_ = "";
  std::unique_ptr<savvy::genomic_region> region_;
  double min_mac_ = 1.0;
  double min_maf_ = 0.f;
  bool no_sparse_ = false;
  bool always_sparse_ = false;
  bool logit_ = false;
  bool trust_info_ = false;
  bool help_ = false;
  bool invnorm_ = false;
  bool pass_only_ = false;
private:
  static std::vector<option_with_desc>& merge_longopts(std::vector<option_with_desc>& additional_options)
  {
    additional_options.insert(additional_options.end(), {
      {"cov", "<columns>", 'c', "Comma separated list of covariate columns"},
      {"debug-log", "<file>", '\x02', "Enables debug logging and specifies log file"},
      {"fmt-field", "<string>", '\x02', "Format field to use (DS, HDS, or GT)"},
      {"help", "", 'h', "Print Usage"},
      {"id", "<string>", 'i', "Sample ID column (defaults to first column)"},
      {"inv-norm", "", '\x01', "Inverse normalize response"},
      {"kinship", "<file>", 'k', "Kinship file"},
      {"logit", "", 'b', "Enable logistic model"},
      {"min-mac", "<int>", '\x02', "Minimum minor allele count (default: 1)"},
      {"min-maf", "<real>", '\x02', "Minimum minor allele frequency (default: 0.0)"},
      {"never-sparse", "", '\x01', "Disables sparse optimizations"},
      {"no-sparse", "", '\x01', ""},
      {"always-sparse", "", '\x01', "Forces sparse optimizations even for dense file records"},
      {"output", "<file>", 'o', "Output path (default: /dev/stdout)"},
      {"pheno", "<column>", 'p', "Phenotype column"},
      {"region", "<string>", 'r', "Genomic region to test (chrom:beg-end)"},
      {"trust-info", "", '\x01', "Uses AC and AN INFO fields instead of computing values"},
      {"pass-only", "", '\x01', "Only test PASS variants"},
      {"wgeno", "<file>", '\x02', "Path to thinned genotypes used for fitting whole genome model"}
    });

    return additional_options;
  }
public:
  assoc_prog_args(const std::string& sub_command, std::vector<option_with_desc>&& additional_options) :
    getopt_wrapper("Usage: savant " + sub_command + " [opts ...] <geno_file> <pheno_file>",
      std::move(merge_longopts(additional_options)))
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

  virtual ~assoc_prog_args() {}

  const std::vector<std::string>& cov_columns() const { return covariate_fields_; }
  const std::string& id_column() const { return id_field_; }
  const std::string& pheno_column() const { return phenotype_field_; }
  const std::string& geno_path() const { return geno_path_; }
  const std::string& pheno_path() const { return pheno_path_; }
  const std::string& output_path() const { return output_path_; }
  const std::string& fmt_field() const { return fmt_field_; }
  const std::string& debug_log_path() const { return debug_log_path_; }
  const std::string& whole_genome_file_path() const { return wgeno_file_path_; }
  const std::string& kinship_file_path() const { return kinship_file_path_; }
  const std::unique_ptr<savvy::genomic_region>& region() const { return region_; }
  double min_mac() const { return min_mac_; }
  double min_maf() const { return min_maf_; }
  bool sparse_disabled() const { return no_sparse_; }
  bool force_sparse() const { return always_sparse_; }
  bool logit_enabled() const { return logit_; }
  bool trust_info() const { return trust_info_; }
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
        if (std::string("never-sparse") == long_options_[long_index].name || std::string("no-sparse") == long_options_[long_index].name)
        {
          no_sparse_ = true;
        }
        else if (std::string("always-sparse") == long_options_[long_index].name)
        {
          always_sparse_ = true;
        }
        else if (std::string("trust-info") == long_options_[long_index].name)
        {
          trust_info_ = true;
        }
        else if (std::string("inv-norm") == long_options_[long_index].name)
        {
          invnorm_ = true;
        }
        else if (std::string("pass-only") == long_options_[long_index].name)
        {
          pass_only_ = true;
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
        else if (std::string("wgeno") == long_options_[long_index].name)
        {
          wgeno_file_path_ = optarg ? optarg : "";
        }
        else
        {
          return std::cerr << "Error: invalid option " << long_options_[long_index].name << std::endl, false;
        }
        break;
      case 'b':
        logit_ = true;
        break;
      case 'h':
        help_ = true;
        return true;
      case 'k':
        kinship_file_path_ = optarg ? optarg : "";
        break;
      case 'c':
        covariate_fields_ = utility::split_string_to_vector(optarg ? optarg : "", ',');
        break;
      case 'o':
        output_path_ = optarg ? optarg : "";
        break;
      case 'p':
        phenotype_field_ = optarg ? optarg : "";
        break;
      case 'r':
        region_.reset(new savvy::genomic_region(utility::string_to_region(optarg ? optarg : "")));
        break;
      case '?':
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

struct phenotype_file_data
{
  std::vector<std::string> ids;
  std::vector<scalar_type> resp_data;
  std::vector<std::vector<scalar_type>> cov_data;
};

bool load_phenotypes(const assoc_prog_args& args, savvy::reader& geno_file, xt::xtensor<scalar_type, 1>& pheno_vec, xt::xtensor<scalar_type, 2>& cov_mat, std::vector<std::string>& sample_intersection);

#endif // SAVANT_ASSOC_HPP
