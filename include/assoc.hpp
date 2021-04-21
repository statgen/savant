/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SAVANT_ASSOC_HPP
#define SAVANT_ASSOC_HPP

#include "utility.hpp"

#include <savvy/reader.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <getopt.h>

class assoc_prog_args
{
private:
  std::string sub_command_;
  std::vector<option> long_options_;
  std::string short_options_;
  std::vector<std::string> covariate_fields_;
  std::string id_field_;
  std::string phenotype_field_;
  std::string geno_path_;
  std::string pheno_path_;
  std::string output_path_ = "/dev/stdout";
  std::string debug_log_path_ = "/dev/stdnull";
  std::string fmt_field_ = "";
  std::unique_ptr<savvy::genomic_region> region_;
  double min_mac_ = 1.0;
  bool no_sparse_ = false;
  bool always_sparse_ = false;
  bool logit_ = false;
  bool trust_info_ = false;
  bool help_ = false;
public:
  assoc_prog_args(const std::string& sub_command, std::vector<option>&& additional_options) :
    sub_command_(sub_command),
    long_options_(
      {
        {"cov", required_argument, 0, 'c'},
        {"debug-log", required_argument, 0, '\x02'},
        {"fmt-field", required_argument, 0, '\x02'},
        {"help", no_argument, 0, 'h'},
        {"id", required_argument, 0, 'i'},
        {"logit", no_argument, 0, 'b'},
        {"min-mac", required_argument, 0, '\x02'},
        {"never-sparse", no_argument, 0, '\x01'},
        {"no-sparse", no_argument, 0, '\x01'},
        {"always-sparse", no_argument, 0, '\x01'},
        {"output", required_argument, 0, 'o'},
        {"pheno", required_argument, 0, 'p'},
        {"region", required_argument, 0, 'r'},
        {"trust-info", no_argument, 0, '\x01'},
        {0, 0, 0, 0}
      })
  {
    long_options_.reserve(long_options_.size() + additional_options.size());
    long_options_.insert(--long_options_.end(), additional_options.begin(), additional_options.end());

    short_options_.reserve((long_options_.size() - 1) * 2);
    std::vector<bool> mask(256, false);
    for (const auto& o : long_options_)
    {
      if (!mask[unsigned(o.val)])
      {
        short_options_ += (char)o.val;
        if (o.has_arg == required_argument)
          short_options_ += ':';
        mask[(unsigned)o.val] = true;
      }
    }
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
  const std::unique_ptr<savvy::genomic_region>& region() const { return region_; }
  double min_mac() const { return min_mac_; }
  bool sparse_disabled() const { return no_sparse_; }
  bool force_sparse() const { return always_sparse_; }
  bool logit_enabled() const { return logit_; }
  bool trust_info() const { return trust_info_; }
  bool help_is_set() const { return help_; }

  bool update_fmt_field(const savvy::reader& geno_file)
  {
    std::unordered_set<std::string> fmt_avail;

    for (const auto& h : geno_file.format_headers())
      fmt_avail.insert(h.id);

    if (fmt_field_.empty())
    {
      if (fmt_avail.find("DS") != fmt_avail.end()) fmt_field_ = "DS";
      else if (fmt_avail.find("HDS") != fmt_avail.end()) fmt_field_ = "HDS";
      else if (fmt_avail.find("GT") != fmt_avail.end()) fmt_field_ = "GT";
      else return std::cerr << "Error: file must contain DS, HDS, or GT format fields\n", false;
      std::cerr << "Notice: --fmt-field not specified so auto selecting " << fmt_field_ << std::endl;
    }
    else
    {
      if (fmt_avail.find(fmt_field_) == fmt_avail.end())
        return std::cerr << "Error: requested format field (" << fmt_field_ << ") not found in file headers\n", false;
    }
    return true;
  }

  void print_usage(std::ostream& os)
  {
    os << "Usage: savant " << sub_command_ << " [opts ...] <geno_file> <pheno_file> \n";
    os << "\n";
    os << " -c, --cov            Comma separated list of covariate columns\n";
    os << " -h, --help           Print usage\n";
    os << " -i, --id             Sample ID column (defaults to first column)\n";
    os << " -b, --logit          Enable logistic model\n";
    os << " -o, --output         Output path (default: /dev/stdout)\n";
    os << " -p, --pheno          Phenotype column\n";
    os << " -r, --region         Genomic region to test (chrom:beg-end)\n";
    os << "     --min-mac        Minimum minor allele count (default: 1)\n";
    os << "     --never-sparse   Disables sparse optimizations\n";
    os << "     --always-sparse  Forces sparse optimizations even for dense file records\n";
    os << "     --fmt-field      Format field to use (DS, HDS, or GT)\n";
    os << "     --debug-log      Enables debug logging and specifies log file\n";
    os << "     --trust-info     Uses AC and AN INFO fields instead of computing values\n";
    os << std::flush;
  }

  virtual bool process_opt(char copt) { return false; }

  bool parse(int argc, char** argv)
  {
    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, short_options_.c_str() /*"\x01\x02:bc:ho:p:r:"*/, long_options_.data(), &long_index )) != -1)
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
      default:
        if (!process_opt(copt))
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

#endif // SAVANT_ASSOC_HPP