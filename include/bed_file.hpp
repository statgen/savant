/*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SAVANT_BED_FILE_HPP
#define SAVANT_BED_FILE_HPP

#include <string>
#include <vector>

#include <shrinkwrap/gz.hpp>

class bed_file
{
private:
  shrinkwrap::gz::istream is_;
  std::vector<std::string> sample_ids_;
public:
  typedef float data_type;

  class record
  {
  private:
    std::string chrom_;
    std::string pheno_id_;
    std::int64_t beg_ = 0;
    std::int64_t end_ = 0;
    std::vector<data_type> data_;
  public:
    record();
    ~record();

    const std::string& chrom() const { return chrom_; }
    const std::string& pheno_id() const { return pheno_id_; }
    std::int64_t beg() const { return beg_ + 1; }
    std::int64_t end() const { return end_; }
    const std::vector<data_type>& data() const { return data_; }
    std::vector<std::size_t> remove_missing();

    static bool deserialize(record& dest, std::istream& is, const std::vector<std::size_t>& pheno_to_geno_map);
  };

  bed_file(const std::string& file_path);
  ~bed_file();

  const std::vector<std::string>& sample_ids() const { return sample_ids_; }
  bool read(std::vector<record>& dest, const std::vector<std::size_t>& pheno_to_geno_map, std::size_t n_records = 1);
};

#endif // SAVANT_BED_FILE_HPP