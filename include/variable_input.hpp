/*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SAVANT_VARIABLE_INPUT_HPP
#define SAVANT_VARIABLE_INPUT_HPP

#include "linear_model.hpp"
#include "utility.hpp"

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <fstream>

template <typename ScalarType>
bool parse_covariates_file(const std::string& cov_path, const std::vector<std::string>& ids, xt::xtensor<ScalarType, 2>& dest, std::vector<std::string>& predictor_names)
{
  std::unordered_map<std::string, std::size_t> id_map;
  id_map.reserve(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i)
    id_map[ids[i]] = i;
  std::size_t match_count = 0, line_count = 0;

  if (cov_path.empty())
    return std::cerr << "Error: must pass separate covariates file\n", false;
  std::ifstream cov_file(cov_path, std::ios::binary);

  std::string line;
  if (!std::getline(cov_file, line))
    return std::cerr << "Error: empty covariates file\n", false;

  auto str_fields = utility::split_string_to_vector(line.c_str(), '\t');
  if (str_fields.empty())
    return std::cerr << "Error: first line in covariates file is empty\n", false;
  predictor_names.assign(str_fields.begin() + 1, str_fields.end());

  dest = xt::xtensor<ScalarType, 2>::from_shape({ids.size(), str_fields.size()});

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
      dest(row_idx_it->second, 0) = ScalarType(1);
      for (std::size_t i = 1; i < str_fields.size(); ++i)
      {
        assert(i < dest.shape()[1]);
        ScalarType v = std::strtod(str_fields[i].c_str(), &p);
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

template <typename ScalarType>
bool parse_phenotypes_file(const std::string& pheno_path, savvy::reader& geno_file, std::vector<std::string>& sample_intersection, std::vector<std::vector<ScalarType>>& dest, std::vector<std::string>& pheno_names)
{
  std::ifstream pheno_file(pheno_path, std::ios::binary);
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
  dest = std::vector<std::vector<ScalarType>>(n_cols - 1, std::vector<ScalarType>(sample_intersection.size()));
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
        dest[j][i] = savvy::typed_value::missing_value<ScalarType>();
      }
      else
      {
        ScalarType v = std::strtod(s, &p);
        if (p != s)
          dest[j][i] = v;
        else
          return std::cerr << "Error: encountered non-numeric phenotype value\n", false;
      }
    }
  }

  return true;
}

#endif // SAVANT_VARIABLE_INPUT_HPP