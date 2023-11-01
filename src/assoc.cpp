#include "assoc.hpp"

bool parse_pheno_file(const assoc_prog_args& args, phenotype_file_data& dest)
{
  dest = phenotype_file_data();
//  xt::xtensor<scalar_type, 1> resp_data;
//  xt::xtensor<scalar_type, 2> cov_data;

  //std::size_t id_idx = args.id_column().empty() ? 0 : std::size_t(-1);
  //std::vector<std::size_t> covariate_idxs(args.cov_columns().size(), std::size_t(-1));

  std::ifstream pheno_file(args.pheno_path(), std::ios::binary);
  if (!pheno_file)
    return std::cerr << "Error: could not open pheno file\n", false;

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
          else if (p != str_fields[i].c_str())
            dest.resp_data.back() = v;
        }
        else if (mask[i] == cov_code)
        {
          scalar_type v = std::strtod(str_fields[i].c_str(), &p);
          if (p == str_fields[i].c_str() && !str_fields[i].empty() && str_fields[i][0] != '.' && std::tolower(str_fields[i][0]) != 'n')
            return std::cerr << "Error: encountered non-numeric covariate\n", false;
          else if (p != str_fields[i].c_str())
            dest.cov_data.back()[j] = v;
          ++j;
        }
      }
    }
    return true;
  }

  return std::cerr << "Error: Pheno file empty\n", false;
}

bool load_phenotypes(const assoc_prog_args& args, savvy::reader& geno_file, xt::xtensor<scalar_type, 1>& pheno_vec, xt::xtensor<scalar_type, 2>& cov_mat, std::vector<std::string>& sample_intersection)
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
    if (std::isnan(full_pheno.resp_data[i]) || std::find_if(full_pheno.cov_data[i].begin(), full_pheno.cov_data[i].end(), [](scalar_type v) { return std::isnan(v); }) != full_pheno.cov_data[i].end())
    {
      // missing

    }
    else
    {
      id_map[full_pheno.ids[i]] = i;
      samples_with_phenotypes.emplace(full_pheno.ids[i]);
    }
  }

  sample_intersection = geno_file.subset_samples(samples_with_phenotypes);
  if (sample_intersection.size() == 0)
    return std::cerr << "Error: no phenotype sample IDs overlap IDs in genotype file\n", false;
  if (sample_intersection.size() == 1)
    return std::cerr << "Error: only one phenotype sample ID overlaps IDs in genotype file\n", false;

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