/*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "bed_file.hpp"
#include "utility.hpp"

#include <algorithm>

bed_file::record::record()
{

}

bed_file::record::~record()
{

}

std::vector<std::size_t> bed_file::record::remove_missing()
{
  std::vector<std::size_t> subset_mask(data_.size());

  std::size_t dest = 0;
  for (std::size_t i = 0; i < data_.size(); ++i)
  {
    if (std::isnan(data_[i]))
    {
      subset_mask[i] = std::size_t(-1);
    }
    else
    {
      data_[dest] = data_[i];
      subset_mask[i] = dest++;
    }
  }
  data_.resize(dest);

  return subset_mask;
}

bool bed_file::record::deserialize(bed_file::record& dest, std::istream& is, const std::vector<std::size_t>& pheno_to_geno_map)
{
  is >> dest.chrom_;
  is >> dest.beg_;
  is >> dest.end_;
  is >> dest.pheno_id_;
  is.get(); // remove \t

  std::string data_line;
  if (!std::getline(is, data_line))
    return false;

  dest.data_.resize(0);
  dest.data_.resize(pheno_to_geno_map.size());

  std::size_t bed_idx = 0;
  std::size_t max_idx = 0;

  const char* d = nullptr;
  char delim = '\t';
  const char* s = data_line.data();
  const char*const e = s + data_line.size();
  while ((d = std::find(s, e,  delim)) != e)
  {
    if (bed_idx >= pheno_to_geno_map.size())
      return std::cerr << "Error: inconsistent number of columns in BED file\n", false;

    //dest.data_.emplace_back(std::string(s, d));
    //*d = '\0';
    if (pheno_to_geno_map[bed_idx] < dest.data_.size())
    {
      if (std::tolower(*s) == 'n')
        dest.data_[pheno_to_geno_map[bed_idx]] = savvy::typed_value::missing_value<data_type>();
      else
        dest.data_[pheno_to_geno_map[bed_idx]] = std::atof(s);
      max_idx = std::max(max_idx, pheno_to_geno_map[bed_idx]);
    }
    s = d ? d + 1 : d;
    ++bed_idx;
  }

  if (pheno_to_geno_map[bed_idx] < dest.data_.size())
  {
    dest.data_[pheno_to_geno_map[bed_idx]] = std::atof(s);
    max_idx = std::max(max_idx, pheno_to_geno_map[bed_idx]);
  }

  dest.data_.resize(max_idx + 1);
  ++bed_idx;

  if (bed_idx != pheno_to_geno_map.size())
    return std::cerr << "Error: inconsistent number of columns in BED file\n", false;

  return !(is.bad());
}

bed_file::bed_file(const std::string& file_path) :
  is_(file_path)
{
  std::string str;
  is_ >> str; // chrom
  is_ >> str; // bed
  is_ >> str; // end
  is_ >> str; // pheno_id
  is_.get(); // remove \t

  if (std::getline(is_, str))
  {
    sample_ids_ = utility::split_string_to_vector(str, '\t');
  }

}

bed_file::~bed_file()
{
}

bool bed_file::read(std::vector<bed_file::record>& dest, const std::vector<std::size_t>& pheno_to_geno_map, std::size_t n_records)
{
  dest.resize(0);
  dest.reserve(n_records);

  if (pheno_to_geno_map.size() != sample_ids_.size())
    return std::cerr << "Error: bed file sample count does not match that of sample map\n", false;

  while (dest.size() < n_records)
  {
    dest.emplace_back();
    if (!record::deserialize(dest.back(), is_, pheno_to_geno_map))
    {
      dest.pop_back();
      break;
    }
  }

  return !(is_.bad());
}
