#include "debug_log.hpp"
#include "single.hpp"
#include "utility.hpp"
#include "linear_model.hpp"
#include "logistic_score_model.hpp"

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

class single_prog_args
{
private:
  std::vector<option> long_options_;
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
  single_prog_args() :
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
  }

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
    os << "Usage: savant single [opts ...] <geno_file> <pheno_file> \n";
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

  bool parse(int argc, char** argv)
  {
    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "\x01\x02:bc:ho:p:r:", long_options_.data(), &long_index )) != -1)
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

typedef double scalar_type;

struct phenotype_file_data
{
  std::vector<std::string> ids;
  std::vector<scalar_type> resp_data;
  std::vector<std::vector<scalar_type>> cov_data;
};

bool parse_pheno_file(const single_prog_args& args, phenotype_file_data& dest)
{
  dest = phenotype_file_data();
//  xt::xtensor<scalar_type, 1> resp_data;
//  xt::xtensor<scalar_type, 2> cov_data;

  //std::size_t id_idx = args.id_column().empty() ? 0 : std::size_t(-1);
  //std::vector<std::size_t> covariate_idxs(args.cov_columns().size(), std::size_t(-1));

  std::ifstream pheno_file(args.pheno_path(), std::ios::binary);

  const std::size_t id_code = 1;
  const std::size_t resp_code = 2;
  const std::size_t cov_code = 3;

  std::string line;
  if (std::getline(pheno_file, line))
  {
    auto header_names = utility::split_string_to_vector(line.c_str(), '\t');
    if (header_names.empty())
      return std::cerr << "Error: empty header\n", false;

    if (header_names[0].size() && header_names[0][0] == '#')
      header_names[0].erase(header_names[0].begin());

    std::vector<std::size_t> mask(header_names.size());
    if (args.id_column().empty())
    {
      std::size_t default_id_idx = args.pheno_path().rfind(".ped") == (args.pheno_path().size() - 4) ? 1 : 0;
      std::cerr << "Notice: using column " << (default_id_idx + 1) << " for sample ID column since --id not specified\n";
      mask[default_id_idx] = id_code;
    }

    for (std::size_t i = 0; i < header_names.size(); ++i)
    {
      if (header_names[i] == args.id_column())
      {
        mask[i] = id_code;
      }
      else if (header_names[i] == args.pheno_column())
      {
        mask[i] = resp_code;
      }
      else
      {
        for (std::size_t j = 0; j < args.cov_columns().size(); ++j)
        {
          if (header_names[i] == args.cov_columns()[j])
          {
            mask[i] = cov_code;
            break;
          }
        }
      }
    }

    if (std::count(mask.begin(), mask.end(), id_code) == 0)
      return std::cerr << "Error: missing identifier column\n", false; // TODO: better error message
    if (std::count(mask.begin(), mask.end(), resp_code) == 0)
      return std::cerr << "Error: missing response column\n", false; // TODO: better error message
    if (std::count(mask.begin(), mask.end(), cov_code) != args.cov_columns().size())
      return std::cerr << "Error: could not find all covariate columns\n", false; // TODO: better error message

    char* p = nullptr;
    while (std::getline(pheno_file, line))
    {
      auto str_fields = utility::split_string_to_vector(line.c_str(), '\t');
      dest.cov_data.emplace_back(args.cov_columns().size(), std::numeric_limits<scalar_type>::quiet_NaN());
      dest.resp_data.emplace_back(std::numeric_limits<scalar_type>::quiet_NaN());
      std::size_t j = 0;
      for (std::size_t i = 0; i < str_fields.size(); ++i)
      {
        if (mask[i] == id_code)
        {
          dest.ids.emplace_back(std::move(str_fields[i]));
        }
        else if (mask[i] == resp_code)
        {
          scalar_type v = std::strtod(str_fields[i].c_str(), &p);
          if (p == str_fields[i].c_str() && !str_fields[i].empty() && str_fields[i][0] != '.' && std::tolower(str_fields[i][0]) != 'n')
            return std::cerr << "Error: encountered non-numeric phenotype\n", false;
          else
            dest.resp_data.back() = v;
        }
        else if (mask[i] == cov_code)
        {
          scalar_type v = std::strtod(str_fields[i].c_str(), &p);
          if (p == str_fields[i].c_str() && !str_fields[i].empty() && str_fields[i][0] != '.' && std::tolower(str_fields[i][0]) != 'n')
            return std::cerr << "Error: encountered non-numeric covariate\n", false;
          else
            dest.cov_data.back()[j] = v;
          ++j;
        }
      }
    }
  }

  return true;
}

bool load_phenotypes(const single_prog_args& args, savvy::reader& geno_file, xt::xtensor<scalar_type, 1>& pheno_vec, xt::xtensor<scalar_type, 2>& cov_mat)
{
  phenotype_file_data full_pheno;
  if (!parse_pheno_file(args, full_pheno))
    return false;



  std::unordered_set<std::string> samples_with_phenotypes;
  std::unordered_map<std::string, std::size_t> id_map;
  samples_with_phenotypes.reserve(full_pheno.ids.size());
  id_map.reserve(full_pheno.ids.size());
  for (std::size_t i = 0; i < full_pheno.resp_data.size(); ++i)
  {
    if (std::isnan(full_pheno.resp_data[i]) /*|| std::find_if(covariate_data[i].begin(); covariate_data[i].end(), std::isnan) != covariate_data[i].end())*/)
    {
      // missing

    }
    else
    {
      id_map[full_pheno.ids[i]] = i;
      samples_with_phenotypes.emplace(full_pheno.ids[i]);
    }
  }

  auto sample_intersection = geno_file.subset_samples(samples_with_phenotypes);

  pheno_vec = xt::xtensor<scalar_type, 1>::from_shape({sample_intersection.size()});
  cov_mat = xt::xtensor<scalar_type, 2>::from_shape({sample_intersection.size(), args.cov_columns().size()});
  for (std::size_t i = 0; i < sample_intersection.size(); ++i)
  {
    std::size_t src_idx = id_map[sample_intersection[i]];
    pheno_vec(i) = full_pheno.resp_data[src_idx];
    for (std::size_t j = 0; j < args.cov_columns().size() /*TODO: change if bias added*/; ++j)
      cov_mat(i, j) = full_pheno.cov_data[src_idx][j];
  }

  return true;
}

template <typename ModelT>
int run_single(const single_prog_args& args, savvy::reader& geno_file, const ModelT& mdl)
{
  auto start = std::chrono::steady_clock::now();
  std::ofstream output_file(args.output_path(), std::ios::binary);
  //output_file <<  "#chrom\tpos\tmaf\tmac\tbeta\tse\tt\tpval\n";
  output_file << "#chrom\tpos\tmaf\tmac\t" << mdl << std::endl;

  savvy::variant var;
  savvy::compressed_vector<scalar_type> sparse_geno;
  std::vector<scalar_type> dense_geno;
  while (geno_file >> var)
  {
    std::size_t ploidy = 0;
    bool is_sparse = false;
    bool found = false;
    for (const auto& f : var.format_fields())
    {
      if (f.first == args.fmt_field())
      {
        found = true;
        is_sparse = args.force_sparse() || (!args.sparse_disabled() && f.second.is_sparse());
        is_sparse ? f.second.get(sparse_geno) : f.second.get(dense_geno);
        ploidy = is_sparse ? sparse_geno.size() / mdl.sample_size() : dense_geno.size() / mdl.sample_size();
        assert(is_sparse ? sparse_geno.size() % mdl.sample_size() == 0 : dense_geno.size() % mdl.sample_size() == 0);
        is_sparse ? savvy::stride_reduce(sparse_geno, ploidy) : savvy::stride_reduce(dense_geno, ploidy);
        break;
      }
    }

    if (!found)
    {
      std::cerr << "Warning: skipping variant with not GT field\n";
      continue;
    }

    assert(ploidy != 0);

    float ac = 0.f, af = 0.f;
    std::int64_t an = 0;
    // For now, we are pulling from INFO fields but will likely always compute AC (along with case/ctrl AC) in the future.
    if (args.trust_info() && mdl.sample_size() == geno_file.samples().size() && var.get_info("AC", ac) && var.get_info("AN", an) && an > 0)
    {
      af = float(ac) / an;
    }
    else //if (!var.get_info("AF", af))
    {
      // For computing ac and af we use AN of sample subset.
      an = mdl.sample_size() * ploidy;
      ac = is_sparse ? std::accumulate(sparse_geno.begin(), sparse_geno.end(), 0.f) : std::accumulate(dense_geno.begin(), dense_geno.end(), 0.f);
      af = ac / an;
    }
    //else
    //{
    //  an = (geno_file.samples().size() * ploidy);
    //  ac = af * an;
    //}

    float mac = (ac > (an/2) ? an - ac : ac);
    float maf = (af > 0.5 ? 1.f - af : af);

    if (mac < args.min_mac()) continue;

    auto stats = is_sparse ? mdl.test_single(sparse_geno, ac) : mdl.test_single(dense_geno, ac);
    output_file << var.chromosome()
                << "\t" << var.position()
                << "\t" << maf
                << "\t" << mac
                << "\t" << stats << "\n";
  }
  std::size_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  std::cerr << "Analysis time (ms): " << elapsed_ms << std::endl;

  // gzip -cd sp-reg-results-chr19-ldl.tsv | tail -n+2 | awk -F'\t' '$4>5  {print $2"\t"$8}' | gnuplot --persist -e "set logscale y; set yrange [0.99:5e-32] reverse; set xrange [1:65000000]; plot '-' using 1:2 w points"

  return geno_file.bad() ? EXIT_FAILURE : EXIT_SUCCESS;
}

int single_main(int argc, char** argv)
{
  single_prog_args args;
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

  geno_file.phasing_status(savvy::phasing::none);

  if (!args.update_fmt_field(geno_file))
    return EXIT_FAILURE;

  xt::xtensor<scalar_type, 1> xresp;
  xt::xtensor<scalar_type, 2> xcov;
  if (!load_phenotypes(args, geno_file, xresp, xcov))
    return std::cerr << "Could not load phenotypes\n", EXIT_FAILURE;

  if (args.logit_enabled())
  {
    return run_single(args, geno_file, logistic_score_model(xresp, xcov));
  }
  else
  {
    return run_single(args, geno_file, linear_model(xresp, xcov));
  }

  // gzip -cd sp-reg-results-chr19-ldl.tsv | tail -n+2 | awk -F'\t' '$4>5  {print $2"\t"$8}' | gnuplot --persist -e "set logscale y; set yrange [0.99:5e-32] reverse; set xrange [1:65000000]; plot '-' using 1:2 w points"
}