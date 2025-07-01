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
#include <numeric>
#include <chrono>

#include "getopt_wrapper.hpp"
#include "summary_stats_input.hpp"
#include "linear_model.hpp"
#include "variable_input.hpp"

class condition_prog_args : public getopt_wrapper
{
private:
  std::string geno_path_;
  std::string pheno_path_;
  std::string cov_path_;
  std::string results_path_;
  std::string output_path_ = "/dev/stdout";
  std::unique_ptr<savvy::genomic_region> region_;
  double pval_threshold_ = 2.;
  std::uint32_t seed_ = 7;
  int progress_unit_ = 100;
  bool two_pass_ = false;
  bool invnorm_ = false;
  //bool write_all_ = false;
  bool help_ = false;
  bool version_ = false;
public:
  condition_prog_args() :
    getopt_wrapper("Usage: savant-condition [opts ...] <geno_file> <pheno_file> <results_file>", {
      {"cov", "<file>", 'c', "Covariates file"},
      {"help", "", 'h', "Print usage"},
      {"inv-norm", "", '\x01', "Inverse normalize response"},
      {"output", "<file>", 'o', "Output path (default: /dev/stdout)"},
      {"max-pvalue", "<real>", 'p', "Max p-value to clump"},
      {"method", "<string>", 'm', "Either forward or two-pass (default: forward)"},
      {"progress", "<integer>", '\x02', "Number of phenotypes to group for progress logging (default: 100; disable: 0)"},
      {"version", "", 'v', "Print version"}})
  {
  }

  virtual ~condition_prog_args() {}
  const std::string& geno_path() const { return geno_path_; }
  const std::string& pheno_path() const { return pheno_path_; }
  const std::string& cov_path() const { return cov_path_; }
  const std::string& output_path() const { return output_path_; }
  const std::string& results_path() const { return results_path_; }
  double pval_threshold() const { return pval_threshold_; }
  std::uint32_t seed() const { return seed_; }
  int progress_unit() const { return progress_unit_; }
  bool help_is_set() const { return help_; }
  bool version_is_set() const { return version_; }
  //bool write_all() const { return write_all_; }
  bool invnorm() const { return invnorm_; }
  bool two_pass() const { return two_pass_; }

  bool parse(int argc, char** argv)
  {
    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, short_opt_string_.c_str(), long_options_.data(), &long_index )) != -1)
    {
      char copt = char(opt & 0xFF);
      switch (copt)
      {
      case '\x01':
        if (std::string("inv-norm") == long_options_[long_index].name)
        {
          invnorm_ = true;
        }
        break;
      case '\x02':
        if (std::string("progress") == long_options_[long_index].name)
        {
          progress_unit_ = std::atol(optarg ? optarg : "");
        }
        break;
      case 'c':
        cov_path_ = optarg ? optarg : "";
        break;
      case 'h':
        help_ = true;
        return true;
      case 'm':
        two_pass_ = std::string(optarg ? optarg : "") == "two-pass";
        break;
      case 'o':
        output_path_ = optarg ? optarg : "";
        break;
      case 'p':
        pval_threshold_ = std::atof(optarg ? optarg : "");
        break;
      case 'v':
        version_ = true;
        return true;
      default:
        return false;
      }
    }

    int remaining_arg_count = argc - optind;

    if (remaining_arg_count == 3)
    {
      geno_path_ = argv[optind];
      pheno_path_ = argv[optind + 1];
      results_path_ = argv[optind + 2];
    }
    else if (remaining_arg_count < 3)
    {
      std::cerr << "Too few arguments\n";
      return false;
    }
    else
    {
      std::cerr << "Too many arguments\n";
      return false;
    }

    if (pval_threshold_ >= 1. || pval_threshold_ <= 0.)
      return std::cerr << "Error: must set valid significance threshold for --max-pvalue\n", false;

    return true;
  }
};

typedef double scalar_type;
//typedef savvy::compressed_vector<std::int8_t> vec_t;




std::ostream& write_header(std::ostream& ofs, bool write_pheno_id)
{
  ofs << "chrom\tpos\tref\talt\tid\taf\tac\tns\t" << linear_model::stats_t::header_column_names();
  if (write_pheno_id)
    ofs << "\tpheno_id";
  ofs << "\trank" << std::endl;
  return ofs; 
}

std::ostream& write_conditioned(std::ostream& ofs, const variant_id_t& var_id, const std::string& str_id, float af, std::int64_t ac, std::int64_t ns, const linear_model::stats_t& stats, const std::string& pheno_id, std::size_t rank)
{
  ofs << var_id.chrom
    << "\t" << var_id.pos
    << "\t" << var_id.ref
    << "\t" << var_id.alt
    << "\t" << str_id
    << "\t" << af
    << "\t" << ac
    << "\t" << ns
    << "\t" << stats;

  if (pheno_id.size())
    ofs << "\t" << pheno_id;

  ofs << "\t" << rank << "\n";

  return ofs;
}

int main(int argc, char** argv)
{
  condition_prog_args args;
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
    std::cout << "savant-condition v" << SAVANT_VERSION << std::endl;
    return EXIT_SUCCESS;
  }

  typedef std::chrono::high_resolution_clock hrc;

  //========== Load phenotypes and covariates ==========//
  std::cerr << "Loading phenotypes and covariates ... " << std::flush;
  auto t = hrc::now();
  savvy::reader geno_file(args.geno_path());
  if (!geno_file)
    return std::cerr << "Could not open geno file\n", EXIT_FAILURE;

  std::vector<std::string> sample_intersection;
  std::vector<std::string> pheno_names;
  std::vector<std::vector<scalar_type>> phenos;
  if (!parse_phenotypes_file(args.pheno_path(), geno_file, sample_intersection, phenos, pheno_names))
    return std::cerr << "Error: failed parsing phenotypes file\n", EXIT_FAILURE;

  std::unordered_map<std::string, std::size_t> pheno_name_to_idx;
  for (std::size_t i = 0; i < pheno_names.size(); ++i)
    pheno_name_to_idx[pheno_names[i]] = i;

  std::vector<std::string> cov_names;
  xt::xtensor<scalar_type, 2> cov_mat;
  if (args.cov_path().empty())
    cov_mat = xt::ones<scalar_type, std::vector<std::size_t>>({sample_intersection.size(), 1}); //TODO: test
  else if (!parse_covariates_file(args.cov_path(), sample_intersection, cov_mat, cov_names))
    return std::cerr << "Error: failed parsing covariates file\n", EXIT_FAILURE;
  std::cerr << "took " << std::chrono::duration_cast<std::chrono::seconds>(hrc::now() - t).count() << " seconds" << std::endl;
  //========== END Load phenotypes and covariates ==========//

  //========== Load results input ==========//
  std::cerr << "Loading summary statistics input ... " << std::flush;
  t = hrc::now();

  results_file input_results(args.results_path());
  if (!input_results)
    return std::cerr << "Error: opening association results input file failed\n", EXIT_FAILURE;

  std::list<results_file::record> records;
  std::unordered_map<std::string, std::vector<results_file::record*>> pheno_results;
  std::vector<variant_id_t> variant_ids;
  results_file::read(input_results, records, pheno_results, variant_ids, 2., true);

  if (input_results.bad())
    return std::cerr << "Error: failed loading association results input file\n", EXIT_FAILURE;
  std::cerr << "took " << std::chrono::duration_cast<std::chrono::seconds>(hrc::now() - t).count() << " seconds" << std::endl;

  std::cerr << "Loading genotypes ... " << std::flush;
  t = hrc::now();

  std::vector<savvy::compressed_vector<scalar_type>> genotypes;
  if (!load_variant_id_genotypes(geno_file, args.geno_path(), variant_ids, genotypes))
    return std::cerr << "Error: failed to load genotypes\n", EXIT_FAILURE;
  
  std::vector<std::size_t> genotype_strides(genotypes.size());
  for (std::size_t i = 0; i < genotypes.size(); ++i)
  {
    genotype_strides[i] = genotypes[i].size() / sample_intersection.size();
    savvy::stride_reduce(genotypes[i], genotype_strides[i], savvy::plus_eov<scalar_type>());
  }
  std::cerr << "took " << std::chrono::duration_cast<std::chrono::seconds>(hrc::now() - t).count() << " seconds" << std::endl;
  //========== END Load results input ==========//

  //========== Open output file  ==========//
  shrinkwrap::bgzf::ostream output_file(args.output_path());
  if (!output_file)
    return std::cerr << "Error: opening output file failed\n", EXIT_FAILURE;

  const std::string& l = input_results.header_line();
  if (!write_header(output_file, l.size() >= 8 && l.substr(l.size() - 8) == "pheno_id"))
    return std::cerr << "Error: writing to output file failed\n", EXIT_FAILURE;
  //========== END Open output file  ==========//

  //========== Run conditional analyses ==========//
  std::vector<std::vector<scalar_type>> dense_genos;
  std::vector<std::uint8_t> dense_genos_mask;
  std::vector<double> dense_genos_ac;
  std::size_t progress = 0;
  //std::int64_t t_sub, t_pheno_res_cov, t_geno_res_cov, t_res, t_ols, t_write;
  for (auto it = pheno_results.begin(); it != pheno_results.end(); ++it)
  {
    if (args.progress_unit() > 0 && progress % args.progress_unit() == 0)
    {
      t = hrc::now();
      std::cerr << "Processing phenotypes " << (progress + 1) << "-" <<  std::min(pheno_results.size(), progress + args.progress_unit()) << " ... " << std::flush;
    }
    auto pf = pheno_name_to_idx.find(it->first);
    if (pf == pheno_name_to_idx.end())
      return std::cerr << "Error: results file contains phenotype (" << it->first << ") not in phenotype file\n", EXIT_FAILURE;

    //~~~~~~~~~~ subset pheno and geno vector ~~~~~~~~~~//
    std::size_t pheno_idx = pf->second;
    assert(pheno_idx < phenos.size());
    std::vector<std::size_t> missing_map = utility::remove_missing(phenos[pheno_idx]);
    std::vector<std::size_t> keep_samples;
    for (std::size_t i = 0; i < missing_map.size(); ++i)
    {
      if (missing_map[i] <= i)
        keep_samples.push_back(i);
    }

    dense_genos.resize(it->second.size());
    dense_genos_mask.resize(it->second.size());
    dense_genos_ac.resize(it->second.size());
    
    for (std::size_t i = 0; i < dense_genos.size(); ++i)
    {
      dense_genos_mask[i] = 0;
      dense_genos[i].clear();
      dense_genos[i].resize(phenos[pheno_idx].size());
      dense_genos_ac[i] = 0;
      const auto& g = genotypes[it->second[i]->genotype_index()];

      for (auto gt = g.begin(); gt != g.end(); ++gt)
      {
        if (missing_map[gt.offset()] < dense_genos[i].size())
        {
          dense_genos[i][missing_map[gt.offset()]] = *gt;
          dense_genos_ac[i] += *gt;
        }
        /*else
        {
          dense_genos_counts[i].an -= stride;
        }*/
      }

      //it->second[i]->set_genotype_index(i); // switch to using dense_geno index instead of genotypes index.
    }
    //~~~~~~~~~~ END subset pheno and geno vector ~~~~~~~~~~//

    //~~~~~~~~~~ Regress out covariates ~~~~~~~~~~//
    residualizer covariate_residualizer(xt::view(cov_mat, xt::keep(keep_samples), xt::all()));
    std::vector<scalar_type> pheno_resid = covariate_residualizer(phenos[pheno_idx], false);
    std::vector<scalar_type> pheno_resid_invnorm = pheno_resid;
    if (args.invnorm())
      inverse_normalize(pheno_resid_invnorm);

    std::size_t dof_subtrahend = covariate_residualizer.n_predictors() + 2;
    std::vector<linear_model::variable_stats<scalar_type>> dense_geno_stats(dense_genos.size());
    for (std::size_t i = 0; i < dense_genos.size(); ++i)
    {
      dense_genos[i] = covariate_residualizer(dense_genos[i], false);
      dense_geno_stats[i] =  linear_model::variable_stats<scalar_type>(dense_genos[i]);
    }
    //~~~~~~~~~~ END Regress out covariates ~~~~~~~~~~//

    /*for (std::size_t i = 0; i < it->second.size(); ++i)
    {
      it->second[i]->set_score(it->second[i]->pvalue());
      if (it->second[i]->pvalue() <= args.pval_threshold())
        it->second[i]->set_group(1);
    }*/

    std::vector<std::size_t> geno_cov_indices;
    assert(it->second.size() > 0);
    const std::size_t ns = phenos[pheno_idx].size(); 
    std::size_t max_idx = std::size_t(-1);
    linear_model::stats_t max_assoc, prev_assoc;
    for (std::int32_t group = 1; max_assoc.pvalue <= args.pval_threshold(); ++group)
    {
      /*double min_pvalue = 2.;
      std::size_t min_idx = std::size_t(-1);

      for (std::size_t i = 0; i < it->second.size(); ++i)
      {
        if (it->second[i]->score() < min_pvalue)
        {
          min_pvalue = it->second[i]->score();
          min_idx = i;
        }
      }

      assert(min_idx < it->second.size());

      //it->second[min_idx]->set_group(group);
      it->second[min_idx]->set_tophit();

      assert(it->second[min_idx]->genotype_index() < genotypes.size());
      auto& top_geno = dense_genos[it->second[min_idx]->genotype_index()];

      //~~~~~~~~~~ regress out top genotype ~~~~~~~~~~//
      ++dof_subtrahend;
      if (dof_subtrahend >= pheno_resid_invnorm.size())
      {
        std::cerr << "Warning: ran out of degrees of freedom for " << it->first << std::endl;
        break;
      }

      linear_model::variable_stats<scalar_type> top_geno_summary(top_geno);
      linear_model::residualize(pheno_resid, linear_model::variable_stats<scalar_type>(pheno_resid), top_geno, top_geno_summary);
      pheno_resid_invnorm = pheno_resid;
      if (args.invnorm())
        inverse_normalize(pheno_resid_invnorm);
      linear_model::variable_stats<scalar_type> pheno_resid_invnorm_summary(pheno_resid_invnorm);
      //~~~~~~~~~~ END regress out top genotype ~~~~~~~~~~//
      */
      max_assoc.pvalue = 2.;
      scalar_type max_abs_t = -1.;
      linear_model::variable_stats<scalar_type> pheno_resid_invnorm_summary(pheno_resid_invnorm);
      for (std::size_t i = 0; i < dense_genos.size(); ++i)
      {
        if (!dense_genos_mask[i])
        {
          //std::size_t g_idx = it->second[i]->genotype_index();
          //linear_model::residualize(dense_genos[g_idx], dense_geno_stats[g_idx], top_geno, top_geno_summary);
          //dense_geno_stats[g_idx] = linear_model::variable_stats<scalar_type>(dense_genos[g_idx]);

          auto s = linear_model::ols(pheno_resid_invnorm.size(),
            std::inner_product(dense_genos[i].begin(), dense_genos[i].end(), pheno_resid_invnorm.begin(), scalar_type()),
            dense_geno_stats[i],
            pheno_resid_invnorm_summary,
            pheno_resid_invnorm.size() - dof_subtrahend);

          if (std::abs(s.t) > max_abs_t)
          {
            max_assoc = s;
            max_abs_t = std::abs(s.t);
            max_idx = i;
          }

          /*if (s.pvalue <= args.pval_threshold())
          {
            it->second[i]->set_group(group);
            it->second[i]->set_score(s.t);
          }*/  
        }
      }

      /*// Remove records in previous clump group
      std::size_t dest = 0;
      for (std::size_t i = 0; i < it->second.size(); ++i)
      {
        if (it->second[i]->group() == group || !it->second[i]->group())
        {
          if (i > dest)
            it->second[dest] = it->second[i];
          ++dest;
        }
      }
      it->second.resize(dest);*/

      if (max_assoc.pvalue <= args.pval_threshold())
      {
        /*double max_tstat = -1.;
        //std::size_t min_idx = std::size_t(-1);

        for (std::size_t i = 0; i < it->second.size(); ++i)
        {
          if (std::abs(it->second[i]->score()) > max_tstat)
          {
            max_tstat = it->second[i]->score();
            max_idx = i;
          }
        }*/
        assert(max_idx < it->second.size());
        if (args.two_pass())
        {
          geno_cov_indices.push_back(max_idx);
          prev_assoc = max_assoc;
        }
        else
        {
          if (!write_conditioned(output_file,
              variant_ids[it->second[max_idx]->genotype_index()],
              it->second[max_idx]->string_variant_id(),
              dense_genos_ac[max_idx] / (genotype_strides[it->second[max_idx]->genotype_index()] * ns),
              dense_genos_ac[max_idx],
              ns,
              max_assoc,
              it->first,
              group))
          {
            return std::cerr << "Error: writing output record failed\n", EXIT_FAILURE;
          }
        }
        dense_genos_mask[max_idx] = 1; // do not test this variant anymore

        assert(max_idx < dense_genos.size());
        std::vector<scalar_type> top_geno = dense_genos[max_idx];

        //~~~~~~~~~~ regress out top genotype ~~~~~~~~~~//
        ++dof_subtrahend;
        if (dof_subtrahend >= pheno_resid_invnorm.size())
        {
          std::cerr << "Warning: ran out of degrees of freedom for " << it->first << std::endl;
          break;
        }

        linear_model::variable_stats<scalar_type> top_geno_summary(top_geno);
        linear_model::residualize(pheno_resid, linear_model::variable_stats<scalar_type>(pheno_resid), top_geno, top_geno_summary);
        pheno_resid_invnorm = pheno_resid;
        if (args.invnorm())
          inverse_normalize(pheno_resid_invnorm);

        for (std::size_t i = 0; i < it->second.size(); ++i)
        {
          //std::size_t g_idx = it->second[i]->genotype_index();
          linear_model::residualize(dense_genos[i], dense_geno_stats[i], top_geno, top_geno_summary);
          dense_geno_stats[i] = linear_model::variable_stats<scalar_type>(dense_genos[i]);
        }
        //~~~~~~~~~~ END regress out top genotype ~~~~~~~~~~//
      }
    }

    if (args.two_pass() && geno_cov_indices.size())
    {
      if (geno_cov_indices.size() == 1)
      {
        std::size_t idx = geno_cov_indices[0];
        if (!write_conditioned(output_file,
          variant_ids[it->second[idx]->genotype_index()],
          it->second[idx]->string_variant_id(),
          dense_genos_ac[idx] / (genotype_strides[it->second[idx]->genotype_index()] * ns),
          dense_genos_ac[idx],
          ns,
          prev_assoc,
          it->first,
          1))
        {
          return std::cerr << "Error: writing output record failed\n", EXIT_FAILURE;
        }
      }
      else
      {
        for (std::size_t i = 0; i < dense_genos.size(); ++i)
        {
          dense_genos_mask[i] = 0;
          dense_genos[i].clear();
          dense_genos[i].resize(phenos[pheno_idx].size());
          //dense_genos_ac[i] = 0;
          const auto& g = genotypes[it->second[i]->genotype_index()];

          for (auto gt = g.begin(); gt != g.end(); ++gt)
          {
            if (missing_map[gt.offset()] < dense_genos[i].size())
            {
              dense_genos[i][missing_map[gt.offset()]] = *gt;
              //dense_genos_ac[i] += *gt;
            }
          }
        }

        xt::xtensor<scalar_type, 2> cond_cov_mat = xt::concatenate(xt::xtuple(xt::view(cov_mat, xt::keep(keep_samples), xt::all()), xt::zeros<scalar_type>({keep_samples.size(), geno_cov_indices.size()})), 1);
        for (std::size_t i = 0; i < geno_cov_indices.size(); ++i)
        {
          const auto& g = genotypes[it->second[geno_cov_indices[i]]->genotype_index()];
          for (auto gt = g.begin(); gt != g.end(); ++gt)
          {
            if (missing_map[gt.offset()] < ns)
              cond_cov_mat(missing_map[gt.offset()], cov_mat.shape()[1] + i) = *gt;
          }
        }

        xt::xtensor<scalar_type, 1> rms = xt::sqrt(xt::mean(xt::square(cond_cov_mat), {0}));
        cond_cov_mat = cond_cov_mat / rms;

        std::vector<linear_model::stats_t> top_assoc_vec;
        std::vector<std::size_t> top_assoc_idx_vec;
        top_assoc_vec.reserve(geno_cov_indices.size());
        top_assoc_idx_vec.reserve(geno_cov_indices.size());

        for (std::int32_t c = 0; c < geno_cov_indices.size(); ++c)
        {
          residualizer covariate_residualizer(xt::view(cond_cov_mat, xt::all(), xt::drop(cov_mat.shape()[1] + c)));

          std::vector<scalar_type> pheno_resid = covariate_residualizer(phenos[pheno_idx], false);
          std::vector<scalar_type> pheno_resid_invnorm = pheno_resid;
          if (args.invnorm())
            inverse_normalize(pheno_resid_invnorm);

          max_assoc.pvalue = 2.;
          scalar_type max_abs_t = -1.;
          linear_model::variable_stats<scalar_type> pheno_resid_invnorm_summary(pheno_resid_invnorm);
          std::vector<scalar_type> dense_geno_resid;
          for (std::size_t i = 0; i < dense_genos.size(); ++i)
          {
            if (!dense_genos_mask[i])
            {
              // std::size_t g_idx = it->second[i]->genotype_index();
              // linear_model::residualize(dense_genos[g_idx], dense_geno_stats[g_idx], top_geno, top_geno_summary);
              // dense_geno_stats[g_idx] = linear_model::variable_stats<scalar_type>(dense_genos[g_idx]);
              dense_geno_resid = covariate_residualizer(dense_genos[i], false);
              scalar_type lrms = std::sqrt(std::inner_product(dense_geno_resid.begin(), dense_geno_resid.end(), dense_geno_resid.begin(), scalar_type()) / dense_geno_resid.size());
              std::transform(dense_geno_resid.begin(), dense_geno_resid.end(), dense_geno_resid.begin(), std::bind(std::divides<scalar_type>(), std::placeholders::_1, lrms));

              auto s = linear_model::ols(pheno_resid_invnorm.size(),
                std::inner_product(dense_geno_resid.begin(), dense_geno_resid.end(), pheno_resid_invnorm.begin(), scalar_type()),
                linear_model::variable_stats<scalar_type>(dense_geno_resid),
                pheno_resid_invnorm_summary,
                pheno_resid_invnorm.size() - (covariate_residualizer.n_predictors() + 2));

              if (std::abs(s.t) > max_abs_t)
              {
                max_assoc = s;
                max_abs_t = std::abs(s.t);
                max_idx = i;
              }
            }
          }

          if (max_assoc.pvalue <= args.pval_threshold())
          {
            top_assoc_vec.push_back(max_assoc);
            top_assoc_idx_vec.push_back(max_idx);
            dense_genos_mask[max_idx] = 1; // do not test this variant anymore
          }
        }

        // rank top associations by absolute t-statistic.
        std::vector<std::size_t> rank_map(top_assoc_vec.size());
        std::iota(rank_map.begin(), rank_map.end(), 0);
        std::sort(rank_map.begin(), rank_map.end(), [&top_assoc_vec](std::size_t l, std::size_t r) { return std::abs(top_assoc_vec[l].t) > std::abs(top_assoc_vec[r].t); });

        for (std::size_t rank = 0; rank < rank_map.size(); ++rank)
        {
          std::size_t idx = top_assoc_idx_vec[rank_map[rank]];
          assert(idx < it->second.size());
          assert(idx < dense_genos_ac.size());
          assert(idx < dense_genos_mask.size());

          if (!write_conditioned(output_file,
            variant_ids[it->second[idx]->genotype_index()],
            it->second[idx]->string_variant_id(),
            dense_genos_ac[idx] / (genotype_strides[it->second[idx]->genotype_index()] * ns),
            dense_genos_ac[idx],
            ns,
            top_assoc_vec[rank_map[rank]],
            it->first,
            rank + 1))
          {
            return std::cerr << "Error: writing output record failed\n", EXIT_FAILURE;
          }
        }
      }
    }

    
    if (args.progress_unit() > 0 && (++progress % args.progress_unit() == 0 || progress == pheno_results.size()))
      std::cerr << "took " << std::chrono::duration_cast<std::chrono::seconds>(hrc::now() - t).count() << " seconds" << std::endl;
  }
  //========== END Run clumping ==========//

  //========== Write output ==========//
  /*shrinkwrap::bgzf::ostream output_file(args.output_path());
  if (!output_file)
    return std::cerr << "Error: opening output file failed\n", EXIT_FAILURE;

  output_file << input_results.header_line() << "\tclump_group\tconditioned_pvalue" << std::endl;

  for (auto it = records.begin(); it != records.end() && output_file; ++it)
  {
    if (args.write_all() || it->tophit())
    {
      output_file << it->serialized_line() << "\t" << it->group()  << "\t" << linear_model::calculate_pvalue(it->score(), - (2 + covariate_residualizer.n_predictors() + std::max(it->group(), 1) - 1)) << "\n";
    }
  }*/

  if (!output_file)
    return std::cerr << "Error: failed to write output records\n", EXIT_FAILURE;
  //========== END Write output ==========//

  return EXIT_SUCCESS;
}

