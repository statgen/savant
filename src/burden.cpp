#include "assoc.hpp"
#include "burden.hpp"
#include "debug_log.hpp"
#include "linear_model.hpp"

#include <savvy/reader.hpp>
#include <shrinkwrap/istream.hpp>

class burden_prog_args : public assoc_prog_args
{
private:
  std::string groups_file_;
  std::string regions_file_;
  float threshold_ = 1.f;
  std::uint32_t permutations_ = 10000;
  bool frequency_weighted_ = false;
public:
  burden_prog_args() :
    assoc_prog_args("burden", {
      {"groups-file", "<file>", 'G', "Path to file listing grouped identifiers of variants to collapse"},
      {"permutations", "<int>", 'n', "Number of permutations to use for variable threshold"},
      {"regions-file", "<file>", 'R', "Path to file listing regions of variants to collapse"},
      {"threshold", "<arg>", 't', "MAF threshold for rare variants (a fixed number in the range (0:0.5] or 'variable')"},
      {"frequency-weighted", "", 'w', "Weight allele contribution using 1/sqrt(MAF(1-MAF))"}
    })
  {

  }

  const std::string& groups_file() const { return groups_file_; }
  const std::string& regions_file() const { return regions_file_; }
  float threshold() const { return threshold_; }
  std::uint32_t permutations() const { return permutations_; }
  bool frequency_weighted() const { return frequency_weighted_; }

  bool parse(int argc, char** argv)
  {
    if (!assoc_prog_args::parse(argc, argv))
      return false;

    optind = 1; // reset getopt for second loop
    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, short_opt_string_.c_str(), long_options_.data(), &long_index)) != -1)
    {
      char copt = char(opt & 0xFF);
      switch (copt)
      {
      case 'G':
        groups_file_ = optarg ? optarg : "";
        break;
      case 'n':
        permutations_ = std::atol(optarg ? optarg : "");
        break;
      case 'R':
        regions_file_ = optarg ? optarg : "";
        break;
      case 't':
        threshold_ = std::atof(optarg ? optarg : "");
        break;
      case '?':
        return false;
      }
    }

    if (groups_file_.empty() && regions_file_.empty())
      return std::cerr << "Error: either --groups-file or --regions-file is required to run burden tests\n", false;

    return true;
  }
};

std::int64_t collapse(std::vector<scalar_type>& burden_scores, savvy::reader& geno_file, const std::string& fmt, float threshold, bool freq_weighted, const std::list<savvy::site_info>& targets)
{
  const std::int64_t ploidy = 2; // TODO: make configurable for non-diploid
  std::size_t n_samples = burden_scores.size();
  std::int64_t n_burden = 0;
  auto target_it = targets.begin();
  savvy::compressed_vector<float> genos;
  savvy::compressed_vector<float> genos2;
  savvy::variant var;
  while (geno_file >> var)
  {
    while (target_it != targets.end() && target_it->pos() < var.pos())
      ++target_it;

    bool target_found = targets.empty() ? true : false;
    for (auto match_it = target_it; !target_found && match_it != targets.end() && match_it->pos() == var.pos(); ++match_it)
    {
      if (match_it->ref() == var.ref() && match_it->alts() == var.alts()) // TODO: support multiallelics
        target_found = true;
    }

    if (!target_found)
      continue;

    if(!var.get_format(fmt, genos))
    {
      std::cerr << "Warning: skipping variant with no " << fmt << " field\n"; // TODO: suppress multiple warnings
      continue;
    }

    std::size_t stride = genos.size() / n_samples;

    std::int64_t an = genos.size();
    if (fmt == "DS")
      an *= ploidy;

    scalar_type ac = 0;
    for (auto it = genos.begin(); it != genos.end(); ++it)
    {
      if (!std::isnan(*it))
        ac += *it;
    }

    scalar_type af = ac / scalar_type(an);
    scalar_type maf = af < 0.5 ? af : scalar_type(1) - af;

    if (maf >= threshold)
      continue;

    if (af > 0.5)
    {
      scalar_type max_val(fmt == "DS" ? ploidy : 1);
      genos2.clear();
      genos2.resize(genos.size());
      std::size_t last_off = 0;
      for (auto it = genos.begin(); it != genos.end(); ++it)
      {
        for (std::size_t i = last_off; i < it.offset(); ++i)
          genos2[i] = max_val;

        if (std::isnan(*it))
        {
          genos2[it.offset()] = *it;
        }
        else
        {
          auto v = max_val - *it;
          if (v != scalar_type(0))
            genos2[it.offset()] = v;
        }
        last_off = it.offset() + 1;
      }

      for (std::size_t i = last_off; i < genos.size(); ++i)
        genos2[i] = max_val;

      std::swap(genos, genos2);
    }

    if (freq_weighted)
    {
      // weight = 1/sqrt(MAF(1-MAF))
      for (auto gt = genos.begin(); gt != genos.end(); ++gt)
        burden_scores[gt.offset() / stride] += *gt / std::sqrt(maf * (scalar_type(1) - maf));
    }
    else
    {
      for (auto gt = genos.begin(); gt != genos.end(); ++gt)
        burden_scores[gt.offset() / stride] += *gt;
    }

    ++n_burden;
  }

  return n_burden;
}

int burden_main(int argc, char** argv)
{
  burden_prog_args args;
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

  geno_file.phasing_status(savvy::phasing::none);
  if (!args.update_fmt_field(geno_file, {"HDS","GT", "DS"}))
    return EXIT_FAILURE;

  std::vector<std::string> sample_intersection;
  xt::xtensor<scalar_type, 1> xresp;
  xt::xtensor<scalar_type, 2> xcov;
  if (!load_phenotypes(args, geno_file, xresp, xcov, sample_intersection))
    return std::cerr << "Could not load phenotypes\n", EXIT_FAILURE;

  std::ofstream output_file(args.output_path(), std::ios::binary);

  linear_model mdl(xresp, xcov, args.invnorm());

  output_file << "#group_id\tn_variants\t" << mdl << std::endl;
  shrinkwrap::istream groups_file(args.groups_file().empty() ? args.regions_file() : args.groups_file());
  if (!groups_file)
    return std::cerr << "Error: could not open regions file (" << args.regions_file() << ")\n", EXIT_FAILURE;

  std::vector<scalar_type> burden_scores;
  std::string line;
  while (std::getline(groups_file, line))
  {
    burden_scores.clear();
    burden_scores.resize(mdl.sample_size());

    std::string group_name;
    std::list<savvy::site_info> target_sites;
    std::vector<savvy::genomic_region> regions;
    if (!args.groups_file().empty())
    {
      std::tie(group_name, target_sites) = utility::parse_marker_group_line(line);
      regions = utility::sites_to_regions(target_sites);
    }
    else
    {
      group_name = line;
      regions = {utility::string_to_region(line)};
    }

    std::size_t n_variants = 0;
    for (const auto& reg : regions)
    {
      geno_file.reset_bounds(reg, savvy::bounding_point::any);
      std::list<savvy::site_info> chrom_targets;
      for (auto it = target_sites.begin(); it != target_sites.end(); ++it)
      {
        if (it->chromosome() == reg.chromosome())
          chrom_targets.emplace_back(*it);
      }

      std::int64_t res = collapse(burden_scores, geno_file, args.fmt_field(), args.threshold(), args.frequency_weighted(), chrom_targets);
      if (res < 0)
        return std::cerr << "Error: Collapse failed\n", EXIT_FAILURE;
      n_variants += res;
    }

    if (n_variants > 0)
    {
      auto stats = mdl.test_single(burden_scores, std::accumulate(burden_scores.begin(), burden_scores.end(), 0.));
      output_file << group_name
        << "\t" << n_variants
        << "\t" << stats << "\n";
    }
  }


  return EXIT_SUCCESS;
}