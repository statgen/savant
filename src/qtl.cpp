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
#include <random>

class qtl_prog_args : public getopt_wrapper
{
private:
  //std::string sub_command_;
  //std::vector<option> long_options_;
  //std::string short_options_;
  std::string cov_path_;
  std::string phenotype_field_;
  std::string geno_path_;
  std::string pheno_path_;
  std::string output_path_ = "/dev/stdout";
  std::string debug_log_path_ = "/dev/null";
  std::string fmt_field_ = "";
  std::unique_ptr<savvy::genomic_region> region_;
  double min_mac_ = 1.0;
  double min_maf_ = 0.f;
  double max_pval_ = 2.;
  double resid_geno_threshold_ = -1.;
  bool split_output_ = false;
  bool help_ = false;
  bool invnorm_ = false;
  bool pass_only_ = false;
  bool print_model_fit_ = false;
public:
  qtl_prog_args() :
    getopt_wrapper("Usage: savant qtl [opts ...] <geno_file> <pheno_file>", {
      {"cov", "<file>", 'c', "Covariates file"},
      {"debug-log", "<file>", '\x02', "Enables debug logging and specifies log file"},
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
      {"print-model-fit", "", '\x01', "Produces file with statistics for regressing the phenotype on covariates"}})
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

  const std::string& cov_path() const { return cov_path_; }
  const std::string& geno_path() const { return geno_path_; }
  const std::string& pheno_path() const { return pheno_path_; }
  const std::string& output_path() const { return output_path_; }
  const std::string& fmt_field() const { return fmt_field_; }
  const std::string& debug_log_path() const { return debug_log_path_; }
  const std::unique_ptr<savvy::genomic_region>& region() const { return region_; }
  double min_mac() const { return min_mac_; }
  double min_maf() const { return min_maf_; }
  double max_pval() const { return max_pval_; }
  double resid_geno_threshold() const { return resid_geno_threshold_; }
  std::int64_t window_size() const { return 48; } // TODO: remove
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
        phenotype_field_ = optarg ? optarg : "";
        break;
      case 'r':
        region_.reset(new savvy::genomic_region(utility::string_to_region(optarg ? optarg : "")));
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


bool parse_covariates_file(const qtl_prog_args& args, const std::vector<std::string>& ids, xt::xtensor<scalar_type, 2>& dest, std::vector<std::string>& predictor_names)
{
  std::unordered_map<std::string, std::size_t> id_map;
  id_map.reserve(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i)
    id_map[ids[i]] = i;
  std::size_t match_count = 0, line_count = 0;

  if (args.cov_path().empty())
    return std::cerr << "Error: must pass separate covariates file\n", false;
  std::string path = args.cov_path();
  std::ifstream cov_file(path, std::ios::binary);

  std::string line;
  if (!std::getline(cov_file, line))
    return std::cerr << "Error: empty covariates file\n", false;

  auto str_fields = utility::split_string_to_vector(line.c_str(), '\t');
  if (str_fields.empty())
    return std::cerr << "Error: first line in covariates file is empty\n", false;
  predictor_names.assign(str_fields.begin() + 1, str_fields.end());

  dest = xt::xtensor<scalar_type, 2>::from_shape({ids.size(), str_fields.size()});

  char* p = nullptr;
  while (std::getline(cov_file, line))
  {
    str_fields = utility::split_string_to_vector(line.c_str(), '\t');
    if (str_fields.empty())
      return std::cerr << "Error: empty line in covariates file\n", false;

    auto row_idx_it = id_map.find(str_fields[0]);
    if (row_idx_it != id_map.end())
    {
      ++match_count;
      dest(row_idx_it->second, 0) = scalar_type(1);
      for (std::size_t i = 1; i < str_fields.size(); ++i)
      {
        assert(i < dest.shape()[1]);
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

bool parse_phenotypes_file(const qtl_prog_args& args, savvy::reader& geno_file, std::vector<std::string>& sample_intersection, std::vector<std::vector<scalar_type>>& dest, std::vector<std::string>& pheno_names)
{
  std::ifstream pheno_file(args.pheno_path(), std::ios::binary);
  if (!pheno_file)
    return std::cerr << "Error: could not open pheno file\n", false;

  std::vector<std::string> pheno_sample_ids;
  std::vector<std::vector<std::string>> pheno_str_vals;

  std::string line;
  if (!std::getline(pheno_file, line)) // skipping header
    return std::cerr << "Error: Pheno file empty\n", false;


  auto str_fields = utility::split_string_to_vector(line.c_str(), '\t');
  if (str_fields.size() < 2)
    return std::cerr << "Error: pheno file contains less than two columns\n", false;

  std::size_t n_cols = str_fields.size();

  pheno_names.assign(str_fields.begin() + 1, str_fields.end());

  while (std::getline(pheno_file, line))
  {
    str_fields = utility::split_string_to_vector(line.c_str(), '\t');
    if (str_fields.size() < n_cols)
      return std::cerr << "Error: pheno file contains inconsistent number of columns\n", false;

    pheno_sample_ids.push_back(str_fields[0]);
    pheno_str_vals.emplace_back(str_fields.begin() + 1, str_fields.end());
  }

  sample_intersection = geno_file.subset_samples({pheno_sample_ids.begin(), pheno_sample_ids.end()});
  if (sample_intersection.size() == 0)
    return std::cerr << "Error: no phenotype sample IDs overlap IDs in genotype file\n", false;
  if (sample_intersection.size() == 1)
    return std::cerr << "Error: only one phenotype sample ID overlaps IDs in genotype file\n", false;

  std::unordered_map<std::string, std::size_t> id_map;
  id_map.reserve(pheno_sample_ids.size());
  for (std::size_t i = 0; i < pheno_sample_ids.size(); ++i)
    id_map[pheno_sample_ids[i]] = i;

  char* p = nullptr;
  dest = std::vector<std::vector<scalar_type>>(n_cols - 1, std::vector<scalar_type>(sample_intersection.size()));
  for (std::size_t i = 0; i < sample_intersection.size(); ++i)
  {
    std::size_t src_idx = id_map[sample_intersection[i]];
    for (std::size_t j = 0; j < dest.size(); ++j)
    {
      assert(src_idx < pheno_str_vals.size());
      assert(j < pheno_str_vals[src_idx].size());
      const char* s = pheno_str_vals[src_idx][j].c_str();
      if (std::tolower(*s) == 'n')
      {
        dest[j][i] = savvy::typed_value::missing_value<scalar_type>();
      }
      else
      {
        scalar_type v = std::strtod(s, &p);
        if (p != s)
          dest[j][i] = v;
        else
          return std::cerr << "Error: encountered non-numeric phenotype value\n", false;
      }
    }
  }

  return true;
}

class residualizer_pinv
{
public:
  typedef double scalar_type;
  typedef xt::xtensor<scalar_type, 1> res_t;
  typedef xt::xtensor<scalar_type, 2> cov_t;
private:
  cov_t x_;
  cov_t m_;
public:
  residualizer_pinv() {}
  residualizer_pinv(const cov_t& x_orig)
  {
    using namespace xt;
    using namespace xt::linalg;
    //cov_t x = concatenate(xtuple(xt::ones<scalar_type>({y.size(), std::size_t(1)}), x_orig), 1);
    x_ = x_orig;
    /*auto it = std::find_if(x_orig.begin(), x_orig.end(), [](auto&x) { return std::isnan(x); });
    auto idx = it - x_orig.begin();
    cov_t c = xt::eval(dot(transpose(x_), x_));
    std::cerr << c << std::endl;
    std::cerr << xt::sum(x_orig) << std::endl;
    m_ = xt::eval(dot(pinv(c), transpose(x_)));*/
    cov_t c = dot(transpose(x_), x_);
    cov_t c_i;
    try
    {
      c_i = inv(c);
    }
    catch (...)
    {
      c_i = pinv(c);
    }

    m_ = xt::eval(dot(c_i, transpose(x_)));
  }

  std::size_t n_predictors() const { return x_.shape()[1] - 1; } // excludes intercept term.
  std::size_t n_samples() const { return x_.shape()[0]; }

  template <typename T>
  void print_model_fit(std::ostream& ofs, const std::vector<T>& v_std) const
  {
    using namespace xt;
    using namespace xt::linalg;
    //cov_t x = concatenate(xtuple(xt::ones<scalar_type>({y.size(), std::size_t(1)}), x_orig), 1);
    auto v = xt::adapt(v_std, {v_std.size()});
    auto pbetas = dot(m_, v);
    res_t residuals = v - dot(x_, pbetas);

//    double sse = xt::eval(xt::sum(residuals * residuals))();
//    double sst = xt::eval(xt::sum(xt::square(v - xt::mean(v))))();
    double sse = xt::eval(xt::linalg::dot(residuals, residuals))();
    double sst = xt::eval(xt::variance(v))() * v.size();
    double r2 = 1. - sse / sst;
    std::size_t n = v.size();
    std::size_t k = n_predictors();
    double r2_adj = 1. - (sse / (n - k - 1.)) / (sst / (n - 1.));

    ofs << r2 << "\t" << r2_adj;
    for (auto it = pbetas.begin(); it != pbetas.end(); ++it)
      ofs << "\t" << *it;
    ofs << "\n";
  }

  template <typename T>
  res_t operator()(const xt::xtensor<T, 1>& v, bool invnorm = false) const
  {
    using namespace xt;
    using namespace xt::linalg;

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

  template <typename T>
  std::vector<scalar_type> operator()(const savvy::compressed_vector<T>& v, bool invnorm = false) const // TODO: this method needs to be tested.
  {
    using namespace xt;
    using namespace xt::linalg;

    if (v.size() != m_.shape()[1]) throw std::runtime_error("length mismatch");

    xt::xtensor<scalar_type, 1> pbetas = xt::zeros<scalar_type>({m_.shape()[0]});
    for (auto it = v.begin(); it != v.end(); ++it)
    {
      for (std::size_t j = 0; j < pbetas.size(); ++j)
        pbetas[j] += (*it) * m_(j, it.offset()); // TODO: consider making m_ column-major
    }

    xt::xtensor<scalar_type, 1> pred = dot(x_, pbetas);
    std::vector<scalar_type> residuals(v.size());
    for (auto it = v.begin(); it != v.end(); ++it)
      residuals[it.offset()] = *it;

    for (std::size_t i = 0; i < pred.size(); ++i)
      residuals[i] -= pred[i];

    if (invnorm)
      inverse_normalize(residuals);
    return residuals;
  }
};

class residualizer_inv
{
public:
  typedef double scalar_type;
  typedef xt::xtensor<scalar_type, 1> res_t;
  typedef xt::xtensor<scalar_type, 2> cov_t;
private:
  cov_t x_;
  cov_t m_;
public:
  residualizer_inv() {}
  residualizer_inv(const cov_t& x_orig)
  {
    using namespace xt;
    using namespace xt::linalg;
    //cov_t x = concatenate(xtuple(xt::ones<scalar_type>({y.size(), std::size_t(1)}), x_orig), 1);
    x_ = x_orig;

    m_ = xt::eval(dot(inv(dot(transpose(x_), x_)), transpose(x_)));
  }

  std::size_t n_variables() const { return x_.shape()[1]; }
  std::size_t n_samples() const { return x_.shape()[0]; }

  template <typename T>
  res_t operator()(const xt::xtensor<T, 1>& v, bool invnorm = false) const
  {
    using namespace xt;
    using namespace xt::linalg;

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

class residualizer_qr
{
public:
  typedef double scalar_type;
  typedef xt::xtensor<scalar_type, 1> res_t;
  typedef xt::xtensor<scalar_type, 2> cov_t;
private:
  cov_t q_;
public:
  residualizer_qr() {}
  residualizer_qr(const cov_t& x_orig)
  {
    using namespace xt;
    using namespace xt::linalg;
    //cov_t x = concatenate(xtuple(xt::ones<scalar_type>({y.size(), std::size_t(1)}), x_orig), 1);
    cov_t r;
    std::tie(q_, r) = qr(x_orig);
  }

  std::size_t n_variables() const { return q_.shape()[1]; }
  std::size_t n_samples() const { return q_.shape()[0]; }

  template <typename T>
  res_t operator()(const xt::xtensor<T, 1>& v, bool invnorm = false) const
  {
    using namespace xt;
    using namespace xt::linalg;

    res_t residuals = v - dot(dot(v, q_), transpose(q_));
    if (invnorm)
      inverse_normalize(residuals);
    return residuals;
  }

  template <typename T>
  std::vector<scalar_type> operator()(const std::vector<T>& v_std, bool invnorm = false) const
  {
    using namespace xt;
    using namespace xt::linalg;

    auto v = xt::adapt(v_std, {v_std.size()});

    res_t residuals = v - dot(dot(v, q_), transpose(q_));
    if (invnorm)
      inverse_normalize(residuals);
    return std::vector<scalar_type> (residuals.begin(), residuals.end());
  }

};

typedef residualizer_pinv residualizer;

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
                      << "\t" << maf
                      << "\t" << mac
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
  if (!parse_covariates_file(args, sample_intersection, cov_mat, cov_names))
    return std::cerr << "Error: failed parsing covariates file\n", EXIT_FAILURE;

  //cov_mat = (cov_mat - xt::mean(cov_mat, {0})) / xt::stddev(cov_mat, {0});
  cov_mat = cov_mat - xt::mean(cov_mat, {0});

  shrinkwrap::bgzf::ostream output_file(args.output_path());
  output_file << "geno_chrom\tgeno_pos\tref\talt\tvariant_id\tmaf\tmac\tns\tpheno_id\tpheno_chrom\tpheno_beg\tpheno_end\t" << linear_model::stats_t::header_column_names() << std::endl;

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

template <typename T>
std::vector<std::size_t> remove_missing(std::vector<T>& vec)
{
  std::vector<std::size_t> subset_mask(vec.size());

  std::size_t dest = 0;
  for (std::size_t i = 0; i < vec.size(); ++i)
  {
    if (std::isnan(vec[i]))
    {
      subset_mask[i] = std::size_t(-1);
    }
    else
    {
      vec[dest] = vec[i];
      subset_mask[i] = dest++;
    }
  }
  vec.resize(dest);

  return subset_mask;
}

template <typename T>
void mean_center(std::vector<T>& vec)
{
  T s = std::accumulate(vec.begin(), vec.end(), T());
  T m = s / vec.size();

  for (auto it = vec.begin(); it != vec.end(); ++it)
    *it = *it - m;
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
      *out_files[i] << "chrom\tpos\tref\talt\tvariant_id\tmaf\tmac\tns\t" << linear_model::stats_t::header_column_names();
      if(!split_)
        *out_files[i] << "\tpheno_id";
      *out_files[i] << std::endl;
    }
  }

  bool write(const savvy::site_info& var, float maf, std::int64_t mac, std::int64_t ns, const linear_model::stats_t& stats, std::size_t pheno_idx)
  {
    std::size_t file_idx = 0;
    if (split_)
      file_idx = pheno_idx;
    (*out_files[file_idx]) << var.chromosome()
       << "\t" << var.position()
       << "\t" << var.ref()
       << "\t" << (var.alts().empty() ? "." : var.alts()[0])
       << "\t" << var.id()
       << "\t" << maf
       << "\t" << mac
       << "\t" << ns
       << "\t" << stats;
    if (!split_)
      (*out_files[file_idx]) << "\t" << pheno_names_[pheno_idx];
    (*out_files[file_idx]) << "\n";

    return out_files[file_idx]->good();
  }
};

template <typename GenoT>
bool process_variant(const savvy::site_info& var, const std::vector<std::vector<scalar_type>>& pheno_resids, const std::vector<linear_model::variable_stats<scalar_type>>& pheno_stats,  const std::vector<residualizer>& residualizers, const std::vector<std::vector<std::size_t>>& subset_non_missing_map, const savvy::compressed_vector<GenoT>& geno, std::vector<scalar_type>& geno_dense, const linear_model::variable_stats<scalar_type>& geno_stats, output_wrapper& output_file, const qtl_prog_args& args, std::size_t ploidy)
{
  for (std::size_t pheno_idx = 0; pheno_idx < pheno_resids.size(); ++pheno_idx)
  {
    const std::vector<scalar_type>& pheno_ref = pheno_resids[pheno_idx];
    const std::vector<std::size_t>& pheno_map_ref = subset_non_missing_map[pheno_idx];

    std::size_t n = pheno_resids[pheno_idx].size();

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

    if (an == 0) return true;
    if (mac < args.min_mac()) return true;
    if (maf < args.min_maf()) return true;

    //savvy::stride_reduce(geno_sub, ploidy, savvy::plus_eov<scalar_type>());



    scalar_type s_xy = 0;
    for (auto it = geno.begin(); it != geno.end(); ++it)
    {
      std::size_t off = pheno_map_ref[it.offset()];
      if (off < n)
      {
        s_xy += (*it) * pheno_ref[off];
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

    linear_model::stats_t stats{};
    if (args.resid_geno_threshold() < 1.)
    {
      stats = linear_model::ols(n, s_xy, geno_stats_sub, pheno_stats[pheno_idx], n - (residualizers[residualizers.size() == 1 ? 0 : pheno_idx].n_predictors()) - 2);
    }
    if (args.resid_geno_threshold() >= 1. || stats.pvalue <= args.resid_geno_threshold())
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
      stats = linear_model::ols(geno_dense, pheno_resids[pheno_idx], linear_model::variable_stats<scalar_type>(geno_dense), pheno_stats[pheno_idx], geno_dense.size() - (residualizers[residualizers.size() == 1 ? 0 : pheno_idx].n_predictors()) - 2);
    }

    if (stats.pvalue > args.max_pval()) continue;
    if (!output_file.write(var, maf, mac, n, stats, pheno_idx))
      return std::cerr << "Error: failed writing to output file\n", false;
  }
  return true;
}

bool process_trans_batch(const std::vector<std::vector<scalar_type>>& phenos,  const std::vector<residualizer>& residualizers, const std::vector<std::vector<std::size_t>>& subset_non_missing_map, savvy::reader& geno_file, /*std::ostream*/output_wrapper& output_file, const qtl_prog_args& args)
{
  if (phenos.empty())
    return false;
#if 0 // PERMUTATION
  std::size_t n_perm = 1;
  std::vector<std::size_t> random_map(subset_non_missing_map.front().size());
  std::iota(random_map.begin(), random_map.end(), 0);
  std::default_random_engine prng{7};
  std::shuffle(random_map.begin(), random_map.end(), prng);
  std::vector<scalar_type> tmp_resid;
#endif
  std::vector<linear_model::variable_stats<scalar_type>> pheno_stats(phenos.size());
  std::vector<std::vector<scalar_type>> pheno_resids(phenos.size());
  for (std::size_t i = 0; i < phenos.size(); ++i)
  {
    pheno_resids[i] = residualizers[residualizers.size() == 1 ? 0 : i](phenos[i], args.invnorm());
#if 0 // PERMUTATION
    std::size_t dest_idx = 0;
    pheno_resids[i] = residualizers[residualizers.size() == 1 ? 0 : i](phenos[i], args.invnorm());
    for (std::size_t j = 0; j < subset_non_missing_map[i].size(); ++j)
    {
      if (subset_non_missing_map[i][random_map[j]] < pheno_resids[i].size())
        tmp_resid[dest_idx++] = pheno_resids[i][subset_non_missing_map[i][random_map[j]]];
    }
    assert(dest_idx == pheno_resids.size());
    std::swap(pheno_resids[i], tmp_resid);
    pheno_stats[i] = linear_model::variable_stats<scalar_type>::gen_strided_stats(pheno_resids[i], 1 + n_perm);

//    assert(i < pheno_resids.size());
//    assert(i < subset_non_missing_map.size());
//    tmp_resid.clear();
//    tmp_resid.reserve(pheno_resids[i].size());
//    std::vector<std::size_t> shuffled_subset_non_missing_map(subset_non_missing_map[i].size());
//    for (std::size_t j = 0; j < shuffled_subset_non_missing_map.size(); ++j)
//      shuffled_subset_non_missing_map[random_map[j]] = subset_non_missing_map[i][j];
//
//    for (std::size_t j = 0; j < shuffled_subset_non_missing_map.size(); ++j)
//    {
//      std::size_t src_idx = shuffled_subset_non_missing_map[j];
//      if (src_idx < pheno_resids[i].size())
//        tmp_resid.push_back(pheno_resids[i][src_idx]);
//    }
//    pheno_resids[i] = tmp_resid;
#else
    pheno_stats[i] = linear_model::variable_stats<scalar_type>(pheno_resids[i]);
#endif
  }

  savvy::compressed_vector<std::int8_t> geno;
  savvy::compressed_vector<std::int8_t> geno_sub;
  savvy::compressed_vector<std::int8_t> geno_biallele;
  std::vector<scalar_type> geno_dense;
  savvy::variant var; std::size_t progress = 0;
  while (geno_file.read(var))
  {
    var.get_format("GT", geno);

    std::int64_t an = geno.size();
    assert(!subset_non_missing_map.empty());
    std::size_t ploidy = an / subset_non_missing_map[0].size();

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
        if (!process_variant(var, pheno_resids, pheno_stats, residualizers, subset_non_missing_map, geno_biallele, geno_dense, geno_stats, output_file, args, ploidy))
          return false;
      }
    }
    else
    {
      savvy::stride_reduce(geno, ploidy, savvy::plus_eov<scalar_type>());
      linear_model::variable_stats<scalar_type> geno_stats(geno);
      if (!process_variant(var, pheno_resids, pheno_stats, residualizers, subset_non_missing_map, geno, geno_dense, geno_stats, output_file, args, ploidy))
        return false;
    }
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

  if (args.region() && !geno_file.reset_bounds(*args.region()))
    return std::cerr << "Could not open genomic region\n", EXIT_FAILURE;

  std::vector<std::string> sample_intersection;
  std::vector<std::string> pheno_names;
  std::vector<std::vector<scalar_type>> phenos;
  if (!parse_phenotypes_file(args, geno_file, sample_intersection, phenos, pheno_names))
    return std::cerr << "Error: failed parsing phenotypes file\n", EXIT_FAILURE;

  std::vector<std::string> cov_names;
  xt::xtensor<scalar_type, 2> cov_mat;
  if (args.cov_path().empty())
    cov_mat = xt::ones<scalar_type, std::vector<std::size_t>>({sample_intersection.size(), 1}); //TODO: test
  else if (!parse_covariates_file(args, sample_intersection, cov_mat, cov_names))
    return std::cerr << "Error: failed parsing covariates file\n", EXIT_FAILURE;

  std::vector<std::vector<std::string>> permute_id_matrix;
  if (!args.perm_path().empty() && !parse_permutation_file(args, sample_intersection, permute_id_matrix))
    return std::cerr << "Error: failed parsing permutation file file\n", EXIT_FAILURE;


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
    subset_non_missing_map[i] = remove_missing(phenos[i]);
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

  //shrinkwrap::bgzf::ostream output_file(args.output_path());
  //output_file << "geno_chrom\tgeno_pos\tref\talt\tvariant_id\tmaf\tmac\tns\t" << linear_model::stats_t::header_column_names() << "\tpheno_id" << std::endl;
  output_wrapper output(args.output_path(), pheno_names, args.split_output());

  if (!process_trans_batch(phenos, residualizers, subset_non_missing_map, /* genos,*/ geno_file, output, args))
    return std::cerr << "Error: processing batch failed\n", EXIT_FAILURE;


  return EXIT_SUCCESS;
}
