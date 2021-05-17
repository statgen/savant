#include "debug_log.hpp"
#include "assoc.hpp"
#include "single.hpp"
#include "utility.hpp"
#include "linear_model.hpp"
#include "logistic_score_model.hpp"
#include "whole_genome_model.hpp"

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

class single_prog_args : public assoc_prog_args
{
public:
  single_prog_args() : assoc_prog_args("single", {})
  {

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
      std::cerr << "Warning: skipping variant with no " << args.fmt_field() << " field\n";
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

// [CHROM]:[POS]_[REF]/[ALT]
static savvy::site_info marker_id_to_site_info(std::string::const_iterator beg, std::string::const_iterator end, std::array<char, 3> delims = {':', '_', '/'})
{
  auto colon_it = std::find(beg, end, delims[0]);
  std::string chrom(beg, colon_it);
  if (colon_it != end)
  {
    auto underscore_it = std::find(++colon_it, end, delims[1]);
    std::uint64_t pos = static_cast<std::uint64_t>(std::atoll(std::string(colon_it, underscore_it).c_str()));
    if (underscore_it != end)
    {
      auto slash_it = std::find(++underscore_it, end, delims[2]);
      std::string ref(underscore_it, slash_it);
      if (slash_it != end)
      {
        std::string alt(++slash_it, end);
        return savvy::site_info{std::move(chrom), std::uint32_t(pos), std::move(ref), {std::move(alt)}};
      }
    }
  }

  return savvy::site_info{};
}

static std::tuple<std::string, std::list<savvy::site_info>> parse_marker_group_line(const std::string& input)
{
  std::tuple<std::string, std::list<savvy::site_info>> ret;
  auto delim_it = std::find(input.begin(), input.end(), '\t');
  if (delim_it != input.end())
  {
    std::get<0>(ret) = std::string(input.begin(), delim_it);
    ++delim_it;

    std::string::const_iterator next_delim_it;
    while ((next_delim_it = std::find(delim_it, input.end(), '\t')) != input.end())
    {
      std::get<1>(ret).emplace_back(marker_id_to_site_info(delim_it, next_delim_it));
      delim_it = next_delim_it + 1;
    }

    std::get<1>(ret).emplace_back(marker_id_to_site_info(delim_it, input.end()));
  }

  return ret;
}

static std::vector<savvy::genomic_region> sites_to_regions(const std::list<savvy::site_info>& un_merged_regions)
{
  std::vector<savvy::genomic_region> ret;

  for (auto it = un_merged_regions.begin(); it != un_merged_regions.end(); ++it)
  {
    if (ret.empty() || ret.back().chromosome() != it->chromosome())
    {
      ret.emplace_back(it->chromosome(), it->position(), it->position());
    }
    else
    {
      std::uint64_t from = std::min<std::uint64_t>(ret.back().from(), it->position());
      std::uint64_t to = std::max<std::uint64_t>(ret.back().to(), it->position());
      ret.back() = savvy::genomic_region(ret.back().chromosome(), from, to);
    }
  }

  return ret;
}

std::vector<savvy::genomic_region> prune_regions(const std::vector<savvy::genomic_region>& regions, const std::list<savvy::site_info>& target_sites, savvy::s1r::reader& index_reader)
{
  std::vector<savvy::genomic_region> ret;

  auto site_it = target_sites.begin();
  for (auto reg_it = regions.begin(); reg_it != regions.end() && site_it != target_sites.end(); ++reg_it)
  {
    for (const auto& entry : index_reader.create_query(*reg_it))
    {
      entry.region_start();
    }
  }

  return ret;
}

template <typename ModelT>
int run_collapse(const single_prog_args& args, savvy::reader& geno_file, const ModelT& mdl)
{
  savvy::s1r::reader index_reader(args.geno_path());
  savvy::variant var;
  savvy::compressed_vector<float> genos;

  std::string group_file_path = "../../EPACTS/install/share/EPACTS/1000G_exome_chr20_example_softFiltered.calls.anno.grp";
  std::ifstream group_file(group_file_path, std::ios::binary);
  if (!group_file)
    return std::cerr << "Error: could not open group file " << group_file_path << "\n", EXIT_FAILURE;

  std::string line;
  while (group_file >> line)
  {
    std::vector<float> counts(mdl.sample_size());
    std::string group_name;
    std::list<savvy::site_info> target_sites;
    std::tie(group_name, target_sites) = parse_marker_group_line(line);
    auto regions = sites_to_regions(target_sites);
    regions = prune_regions(regions, target_sites, index_reader);

    for (const auto& reg : regions)
    {
      geno_file.reset_bounds(reg);
      while (geno_file >> var)
      {
        if(!var.get_format(args.fmt_field(), genos))
        {
          std::cerr << "Warning: skipping variant with no " << args.fmt_field() << " field\n";
          continue;
        }

        for (auto gt = genos.begin(); gt != genos.end(); ++gt)
        {
          counts[gt.offset()] += *gt; // TODO: optional weights
        }
      }
    }

    auto stats = mdl.test_single(counts, std::accumulate(counts.begin(), counts.end(), 0.));
    // TODO: write stats
  }

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

  if (false)
    return run_collapse(args, geno_file, linear_model(xresp, xcov));

  if (args.whole_genome_file_path().size())
  {
    savvy::reader whole_genome_file(args.whole_genome_file_path());
    if (!whole_genome_file)
      return std::cerr << "Could not open --wgeno\n", EXIT_FAILURE;

    savvy::variant var;
    std::size_t i = 0;
    while (whole_genome_file >> var && i <= 1000/**/)
      ++i;

    xt::xtensor<float, 2> xgeno = xt::zeros<float>({whole_genome_file.samples().size(), i + 1});
    xt::col(xgeno, 0) = xt::ones<float>({whole_genome_file.samples().size()});

    whole_genome_file = savvy::reader(args.whole_genome_file_path());
    if (!whole_genome_file)
      return std::cerr << "Could not reopen --wgeno\n", EXIT_FAILURE;

    savvy::compressed_vector<float> geno_vec;
    i = 1;
    while (whole_genome_file >> var && i <= xgeno.shape(1))
    {
      var.get_format("GT", geno_vec);

      std::size_t ploidy = geno_vec.size() / whole_genome_file.samples().size();
      std::size_t an = geno_vec.size();
      float ac = 0;
      for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
      {
        if (std::isnan(*it))
          --an;
        else
          ac += *it;
      }

      float af = ac / an;
      float denom = std::sqrt(2. * af * (1. - af));
      if (denom > 0.f)
      {
        for (auto it = geno_vec.begin(); it != geno_vec.end(); ++it)
        {
          if (std::isnan(*it))
            xgeno(it.offset() / ploidy, i) += af / denom;
          else
            xgeno(it.offset() / ploidy, i) += *it / denom;
        }
      }

      ++i;
    }


    return run_single(args, geno_file, whole_genome_model(xresp, xcov, xgeno, 10000, 0.0001, 1e-5, 1.0));
  }
  else
  {
    if (args.logit_enabled())
    {
      return run_single(args, geno_file, logistic_score_model(xresp, xcov));
    }
    else
    {
      return run_single(args, geno_file, linear_model(xresp, xcov));
    }
  }

  // gzip -cd sp-reg-results-chr19-ldl.tsv | tail -n+2 | awk -F'\t' '$4>5  {print $2"\t"$8}' | gnuplot --persist -e "set logscale y; set yrange [0.99:5e-32] reverse; set xrange [1:65000000]; plot '-' using 1:2 w points"
}