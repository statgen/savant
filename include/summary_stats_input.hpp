/*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef SAVANT_SUMMARY_STATS_INPUT_HPP
#define SAVANT_SUMMARY_STATS_INPUT_HPP

#include <string>
#include <cstdint>

#include <savvy/reader.hpp>
#include <shrinkwrap/gz.hpp>

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
    std::string string_variant_id_;
    double pvalue_ = 2.;
    std::string pheno_id_;
    std::int32_t clump_group_ = 0;
    std::size_t genotype_idx_ = std::size_t(-1);
    double score_ = std::numeric_limits<double>::quiet_NaN();
    bool tophit_ = false;
  public:
    virtual ~record() {}
    double pvalue() const { return pvalue_; }
    const std::string& pheno_id() const { return pheno_id_; }
    void set_group(std::int32_t v) { clump_group_ = v; }
    std::int32_t group() const { return clump_group_; }
    void set_score(double v) { score_ = v; }
    double score() const { return score_; }
    void set_tophit(bool v = true) { tophit_ = v; }
    bool tophit() const { return tophit_; }
    const variant_id_t& variant_id() const { return variant_id_; }
    const std::string& string_variant_id() const { return string_variant_id_; }
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
        else if (i == 4)
          self.string_variant_id_ = self.line_.substr(start_pos, pos - start_pos);
        else if (i == 8)
          self.pvalue_ = std::atof(self.line_.substr(start_pos, pos - start_pos).c_str());
        else if (i == 13)
          self.pheno_id_ = self.line_.substr(start_pos, pos - start_pos);

        start_pos = pos + 1;
      }

      if (expect_pheno_column && self.pheno_id_.empty())
        return std::cerr << "Empty pheno_id: " << self.line_ << std::endl, false;

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

  static void read(results_file& input_file, std::list<results_file::record>& records, std::unordered_map<std::string, std::vector<results_file::record*>>& records_by_pheno, std::vector<variant_id_t>& variant_ids, double pval_threshold, bool keep_all)
  {
    records.clear();
    records.emplace_back();
    while (input_file >> records.back())
    {
      if (!keep_all && records.back().pvalue() > pval_threshold) continue;

      if (records.back().pvalue() <= pval_threshold)
      {
        records_by_pheno[records.back().pheno_id()].push_back(&records.back());
        if (variant_ids.empty() || variant_ids.back() != records.back().variant_id())
          variant_ids.push_back(records.back().variant_id());
        records.back().set_genotype_index(variant_ids.size() - 1);
        // TODO: try to detect if records are not sorted by genomic position.=
      }
      records.emplace_back();
    }
    records.pop_back();
  }
};

template <typename GenoT>
bool load_variant_id_genotypes(savvy::reader& geno_file, const std::string& geno_file_path, const std::vector<variant_id_t>& variant_ids,  std::vector<savvy::compressed_vector<GenoT>>& genotypes)
{
  //========== Determine most efficient regions for querying genotypes ==========//
  savvy::s1r::reader index_file(geno_file_path);
  if (!index_file.good())
    return std::cerr << "Error: could not open SAV index\n", false;

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
      return std::cerr << "Error: failure during SAV index query\n", false;
  }
  //========== END Determine most efficient regions for querying genotypes ==========//


  //========== Load genotypes ==========//
  genotypes.clear();
  genotypes.resize(variant_ids.size());

  if (!geno_file)
    return std::cerr << "Error: failed to open genotype file\n", false;

  auto id_it = variant_ids.begin();
  savvy::variant var;
  for (const auto& r : regions)
  {
    if (!geno_file.reset_bounds(r))
      return std::cerr << "Error: failed to query region from genotype file\n", false;

    while (id_it != variant_ids.end() && geno_file >> var)
    {
      while(id_it != variant_ids.end() && var.pos() > id_it->pos)
      {
        if (genotypes[id_it - variant_ids.begin()].size() == 0)
          return std::cerr << "Error: could not find " << id_it->to_string() << " in genotype file\n", false;
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
      return std::cerr << "Error: failed to reading from genotype file\n", false;
  }
  //========== END Load genotypes ==========//

  return true;
}

template <typename GenoT>
bool load_variant_id_genotypes(const std::string& geno_file_path, const std::vector<variant_id_t>& variant_ids,  std::vector<savvy::compressed_vector<GenoT>>& genotypes)
{
  savvy::reader geno_file(geno_file_path);
  return load_variant_id_genotypes(geno_file, geno_file_path, variant_ids, genotypes);
}

#endif // SAVANT_SUMMARY_STATS_INPUT_HPP
