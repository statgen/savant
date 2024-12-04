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

#include "getopt_wrapper.hpp"

class prune_prog_args : public getopt_wrapper
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
  prune_prog_args() :
    getopt_wrapper("Usage: savant-prune [opts ...] <geno_file> <results_file>", {
      {"help", "", 'h', "Print usage"},
      {"output", "<file>", 'o', "Output path (default: /dev/stdout)"},
      {"max-pvalue", "<real>", 'p', "Max p-value to clump"},
      {"r2-threshold", "<real>", 's', "R-squared threshold for clumping (default: 0.2)"},
      {"top-hits", "", 'a', "Only write most significant association from each LD group to output"},
      {"version", "", 'v', "Print version"}})
  {
  }

  virtual ~prune_prog_args() {}
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
    double p = 0.f;
    for (auto it = j_vec.begin(); it != j_vec.end(); ++it)
      p += *it * i_vec_dense[it.offset()];

    double r = (p / static_cast<double>(n_haplotypes) - freq_i * freq_j) / denom;

    return r * r;
  }

  return std::numeric_limits<float>::quiet_NaN();
}

class variant_id_t
{
public:
  std::string chrom;
  std::int64_t pos = 0;
  std::string ref;
  std::string alt;

  std::string to_string() const { return chrom + ":" + std::to_string(pos) + ":" + ref + ":" + alt; }

  bool matches(const savvy::site_info& s) const
  {
    if (s.chrom() == chrom && s.pos() == pos && s.ref() == ref && (s.alts().empty() ? "" : s.alts()[0]) == alt) // TODO: support multiallelics
      return true;
    return false;
  }

  bool operator==(const variant_id_t& other) const
  {
    return (chrom == other.chrom
            && pos == other.pos
            && ref == other.ref
            && alt == other.alt);
  }

  bool operator!=(const variant_id_t& other) const
  {
    return !(operator==(other));
  }
};

template <typename T>
static std::size_t hash_combine(std::size_t seed, const T& val)
{
  seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed<<6) + (seed>>2);
  return seed;
}

namespace std
{
template <>
struct hash<variant_id_t>
{
  size_t operator()(const variant_id_t& k) const
  {
    size_t ret = 7;
    ret = hash_combine(ret, k.chrom);
    ret = hash_combine(ret, k.pos);
    ret = hash_combine(ret, k.ref);
    ret = hash_combine(ret, k.alt);
    return ret;
  }
};
}

class results_file
{
private:
  shrinkwrap::gz::istream ifs_;
  std::string header_line_;
  bool has_pheno_id_column_ = false;
public:
  class record
  {
  private:
    std::string line_;
    variant_id_t variant_id_;
    double pvalue_ = 2.;
    std::string pheno_id_;
    std::int32_t prune_group_ = 0;
    std::size_t genotype_idx_ = std::size_t(-1);
    bool tophit_ = false;
  public:
    double pvalue() const { return pvalue_; }
    const std::string& pheno_id() const { return pheno_id_; }
    void set_group(std::int32_t v) { prune_group_ = v; }
    std::int32_t group() const { return prune_group_; }
    void set_tophit(bool v = true) { tophit_ = v; }
    bool tophit() const { return tophit_; }
    const variant_id_t& variant_id() const { return variant_id_; }
    void set_genotype_index(std::size_t idx) { genotype_idx_ = idx; }
    std::size_t genotype_index() const {  return genotype_idx_; }
    const std::string& serialized_line() const { return line_; }

    bool matches(const savvy::site_info& s) const
    {
      return variant_id_.matches(s);
    }

    static bool deserialize(record& self, std::istream& ifs, bool expect_pheno_column)
    {
      if (!std::getline(ifs, self.line_))
        return false;

      std::size_t end_i = expect_pheno_column ? 14 : 9;
      std::size_t start_pos = 0;
      for (std::size_t i = 0; i < end_i; ++i)
      {
        std::size_t pos = self.line_.find('\t', start_pos);
        if (pos == std::string::npos && i + 1 != end_i)
        {
          std::cerr << "Error: results file missing " << (i + 1) << " column" << std::endl;
          ifs.setstate(ifs.rdstate() | std::ios::badbit);
          return false;
        }

        if (i == 0)
          self.variant_id_.chrom = self.line_.substr(start_pos, pos - start_pos);
        else if (i == 1)
          self.variant_id_.pos = std::atoll(self.line_.substr(start_pos, pos - start_pos).c_str());
        else if (i == 2)
          self.variant_id_.ref = self.line_.substr(start_pos, pos - start_pos);
        else if (i == 3)
          self.variant_id_.alt = self.line_.substr(start_pos, pos - start_pos);
        else if (i == 8)
          self.pvalue_ = std::atof(self.line_.substr(start_pos, pos - start_pos).c_str());
        else if (i == 13)
          self.pheno_id_ = self.line_.substr(start_pos, pos - start_pos);

        start_pos = pos + 1;
      }

      return true;
    }
  };

  results_file(const std::string& file_path) :
    ifs_(file_path)
  {
    std::getline(ifs_, header_line_);
    if (header_line_.size() > 8 && header_line_.substr(header_line_.size() - 8) == "pheno_id")
      has_pheno_id_column_ = true;
  }

  results_file& operator>>(record& rec)
  {
    record::deserialize(rec, ifs_, has_pheno_id_column_);
    return *this;
  }

  const std::string& header_line() const { return header_line_; }

  explicit operator bool() const { return (bool)ifs_; }
  bool bad() const { return ifs_.bad(); }
  bool good() const { return ifs_.good(); }
  bool fail() const { return ifs_.fail(); }
  bool eof() const { return ifs_.eof(); }
};

int main(int argc, char** argv)
{
  prune_prog_args args;
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
    std::cout << "savant-prune v" << SAVANT_VERSION << std::endl;
    return EXIT_SUCCESS;
  }

  //========== Load results ==========//
  results_file input_results(args.results_path());
  if (!input_results)
    return std::cerr << "Error: opening association results input file failed\n", EXIT_FAILURE;

  shrinkwrap::bgzf::ostream output_file(args.output_path());
  if (!output_file)
    return std::cerr << "Error: opening output file failed\n", EXIT_FAILURE;

  output_file << input_results.header_line() << "\tprune_group" << std::endl;

  std::list<results_file::record> records;
  std::unordered_map<std::string, std::vector<results_file::record*>> pheno_results;


  //std::unordered_map<variant_id, savvy::compressed_vector<std::int8_t>> variant_data;
  std::vector<variant_id_t> variant_ids;

  records.emplace_back();
  while (input_results >> records.back())
  {
    if (!args.write_all() && records.back().pvalue() > args.pval_threshold()) continue;

    if (records.back().pvalue() <= args.pval_threshold())
    {
      pheno_results[records.back().pheno_id()].push_back(&records.back());
      if (variant_ids.empty() || variant_ids.back() != records.back().variant_id())
        variant_ids.push_back(records.back().variant_id());
      records.back().set_genotype_index(variant_ids.size() - 1);
      // TODO: try to detect if records are not sorted by genomic position.=
    }
    records.emplace_back();
  }
  records.pop_back();

  if (input_results.bad())
    return std::cerr << "Error: failed loading association results input file\n", EXIT_FAILURE;
  //========== END Load results ==========//


  //========== Determine most efficient regions for querying genotypes ==========//
  savvy::s1r::reader index_file(args.geno_path());
  if (!index_file.good())
    return std::cerr << "Error: could not open SAV index\n", EXIT_FAILURE;

  std::list<savvy::genomic_region> regions;


  std::size_t i = 0;
  while (i < variant_ids.size())
  {
    savvy::genomic_region r(variant_ids[i].chrom, variant_ids[i].pos, variant_ids[i].pos);
    auto q = index_file.create_query(savvy::genomic_region(variant_ids[i].chrom, variant_ids[i].pos, variant_ids[i].pos));
    ++i;
    for (auto it = q.begin(); it != q.end(); ++it)
    {
      while (i < variant_ids.size() && variant_ids[i].pos >= it->region_start() && variant_ids[i].pos <= it->region_end())
      {
        r = savvy::genomic_region(r.chromosome(), r.from(), variant_ids[i].pos);
        assert(r.from() <= r.to());
        ++i;
      }
    }
    regions.push_back(r);

    if (!index_file.good())
      return std::cerr << "Error: failure during SAV index query\n", EXIT_FAILURE;
  }
  //========== END Determine most efficient regions for querying genotypes ==========//


  //========== Load genotypes ==========//
  std::vector<savvy::compressed_vector<std::int8_t>> genotypes(variant_ids.size());
  savvy::reader geno_file(args.geno_path());
  if (!geno_file)
    return std::cerr << "Error: failed to open genotype file\n", EXIT_FAILURE;

  auto id_it = variant_ids.begin();
  savvy::variant var;
  for (const auto& r : regions)
  {
    if (!geno_file.reset_bounds(r))
      return std::cerr << "Error: failed to query region from genotype file\n", EXIT_FAILURE;

    while (id_it != variant_ids.end() && geno_file >> var)
    {
      while(id_it != variant_ids.end() && var.pos() > id_it->pos)
      {
        if (genotypes[id_it - variant_ids.begin()].size() == 0)
          return std::cerr << "Error: could not find " << id_it->to_string() << " in genotype file\n", EXIT_FAILURE;
        ++id_it;
      }

      for  (auto lit = id_it; lit != variant_ids.end() && var.pos() == lit->pos; ++lit)
      {
        if (lit->matches(var))
        {
          std::size_t gidx = lit - variant_ids.begin();
          var.get_format("GT", genotypes[gidx]);
          assert(genotypes[gidx].size());
        }
      }
    }

    if (geno_file.bad())
      return std::cerr << "Error: failed to reading from genotype file\n", EXIT_FAILURE;
  }
  //========== END Load genotypes ==========//


  //========== Run clumping ==========//
  std::vector<std::int8_t> dense_geno;
  for (auto it = pheno_results.begin(); it != pheno_results.end(); ++it)
  {
    if (it->second.size() == 1 && it->second[0]->pvalue() <= args.pval_threshold())
    {
      it->second[0]->set_group(1);
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

