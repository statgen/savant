#include "debug_log.hpp"
#include "assoc.hpp"
#include "single.hpp"
#include "utility.hpp"
#include "linear_model.hpp"
#include "logistic_score_model.hpp"
#include "whole_genome_model.hpp"
#include "mixed_effects_model.hpp"

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

template <typename T>
typename std::enable_if<std::is_floating_point<typename T::value_type>::value, std::tuple<double, std::int64_t>>::type
generate_ac_an(const T& genos)
{
  double ac = 0.;
  std::int64_t an = genos.size();
  for (auto it = genos.begin(); it != genos.end(); ++it)
  {
    if (std::isnan(*it))
      --an;
    else
      ac += *it;
  }

  return std::make_tuple(ac, an);
}

template <typename T>
typename std::enable_if<std::is_integral<typename T::value_type>::value, std::tuple<double, std::int64_t>>::type
generate_ac_an(const T& genos)
{
  double ac = 0.;
  std::int64_t an = genos.size();
  for (auto it = genos.begin(); it != genos.end(); ++it)
  {
    if (*it < 0)
      --an;
    else
      ac += *it;
  }

  return std::make_tuple(ac, an);
}

template <typename T>
typename std::enable_if<std::is_floating_point<typename T::value_type>::value, void>::type
mean_impute(T& genos, typename T::value_type mean)
{
  for (auto it = genos.begin(); it != genos.end(); ++it)
  {
    if (std::isnan(*it))
      *it = mean;
  }
}

template <typename T>
typename std::enable_if<std::is_integral<typename T::value_type>::value, void>::type
mean_impute(T& genos, typename T::value_type mean)
{
  for (auto it = genos.begin(); it != genos.end(); ++it)
  {
    if (*it < 0)
      *it = mean > 0.5 ? 1 : 0;
  }
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
    std::int64_t an = 0;
    std::int64_t missing_cnt = 0;
    float ac = 0.;
    float af = 0.;
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
        if (!args.trust_info() || mdl.sample_size() != geno_file.samples().size() || !var.get_info("AC", ac) || !var.get_info("AN", an) || an == 0)
          std::tie(ac, an) = is_sparse ? generate_ac_an(sparse_geno) : generate_ac_an(dense_geno);
        af = ac / float(an);
        missing_cnt = std::int64_t(is_sparse ? sparse_geno.size() : dense_geno.size()) - an;
        assert(std::int64_t(missing_cnt) >= 0);
        if (missing_cnt > 0)
          is_sparse ? mean_impute(sparse_geno, af) : mean_impute(dense_geno, af);
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

    float mac = (ac > (an/2.f) ? an - ac : ac);
    float maf = (af > 0.5f ? 1.f - af : af);

    if (an == 0) continue;
    if (mac < args.min_mac()) continue;

    auto stats = is_sparse ? mdl.test_single(sparse_geno, ac + missing_cnt * af) : mdl.test_single(dense_geno, ac + missing_cnt * af);
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

  if (!args.update_fmt_field(geno_file, {"DS", "HDS", "GT"}))
    return EXIT_FAILURE;

  std::vector<std::string> sample_intersection;
  xt::xtensor<scalar_type, 1> xresp;
  xt::xtensor<scalar_type, 2> xcov;
  if (!load_phenotypes(args, geno_file, xresp, xcov, sample_intersection))
    return std::cerr << "Could not load phenotypes\n", EXIT_FAILURE;

#if 0
  if ((true || std::string(argv[0]) == "simulate") && args.kinship_file_path().size())
  {
    Eigen::Map<Eigen::VectorXd> mapped_resp(xresp.data(), xresp.size());
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped_cov(xcov.data(), xcov.shape(0), xcov.shape(1));
    Eigen::MatrixXd kinship(geno_file.samples().size(), geno_file.samples().size());
    if (!mixed_effects_model::load_kinship(args.kinship_file_path(), kinship, geno_file.samples())) // sample_intersection))
      return std::cerr << "Error: could not load kinship\n", EXIT_FAILURE;

    auto K_ldlt = kinship.ldlt(); //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> K_ldlt; K_ldlt.compute(kinship);
    assert(K_ldlt.info() == Eigen::Success);

    double log_det_V = K_ldlt.vectorD().array().log().sum();
    double K_det = kinship.determinant(); // K_ldlt.determinant();

    Eigen::VectorXd d = kinship.diagonal();
    for (std::size_t i = 0; i < d.size(); ++i)
    {
      double v = d[i];
      if (v == 0.)
      {
        auto a = 0;
      }
      else if (v < 0.05)
      {
        auto a = 0;
      }

    }
    std::srand(891437); //(unsigned int) std::time(NULL));
    Eigen::MatrixXd rand_vec(geno_file.samples().size(), 8);
    std::default_random_engine rng(891437);
    std::normal_distribution<double> nd(0., 1.);
    int foo = rand_vec.cols();
    for(std::size_t i = 0; i < rand_vec.rows(); ++i)
    {
      for (std::size_t j = 0; j < rand_vec.cols(); ++j)
        rand_vec(i, j) = nd(rng);
    }
    Eigen::MatrixXd simulated_pheno = K_ldlt.matrixL() * rand_vec;
    std::cout << "#iid";
    for (std::size_t j = 0; j < simulated_pheno.cols(); ++j)
      std::cout << "\tsimulated_pheno_" << j;
    std::cout << std::endl;
    for (std::size_t i = 0; i < simulated_pheno.rows(); ++i)
    {
      std::cout << geno_file.samples()[i];
      for (std::size_t j = 0; j < simulated_pheno.cols(); ++j)
        std::cout << "\t" << simulated_pheno(i, j);
      std::cout << std::endl;
    }
    return EXIT_SUCCESS;
  }
#endif
  if (false)
    return run_collapse(args, geno_file, linear_model(xresp, xcov));

  if (args.whole_genome_file_path().size())
  {
    savvy::reader whole_genome_file(args.whole_genome_file_path());
    if (!whole_genome_file)
      return std::cerr << "Could not open --wgeno\n", EXIT_FAILURE;

    savvy::variant var;
    std::size_t i = 0;
    while (whole_genome_file >> var /*&& i <= 1000*/)
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
  else if (args.kinship_file_path().size())
  {
//    xt::xtensor<float, 1> a = {0.f, 1.0002f, 0.9f, 0.f};
//    xt::xtensor<float, 1> b = {0.f, 0.9f, 1.001f, 0.f};
//    std::cerr << xt::linalg::dot(a - xt::mean(a), b - xt::mean(b)) << std::endl;
//    auto cov_a_b = xt::linalg::dot(a - xt::mean(a), b - xt::mean(b)) / a.size();
//    float r = (cov_a_b / std::sqrt(xt::variance(a)() * xt::variance(b)()))();


//    xt::xtensor<float, 2> X = {
//      {0.f, 1.f, 1.f, 0.f, 1.f},
//      {1.f, 1.f, 0.f, 0.f, 2.f},
//      {0.f, 2.f, 0.f, 0.f, 0.f},
//      {0.f, 2.f, 0.f, 2.f, 1.f},
//      {2.f, 0.f, 1.f, 0.f, 0.f},
//      {1.f, 0.f, 0.f, 2.f, 2.f},
//      {1.f, 2.f, 0.f, 0.f, 1.f},
//      {1.f, 0.f, 0.f, 2.f, 1.f},
//      {0.f, 2.f, 0.f, 0.f, 0.f},
//    };
//
//    xt::xtensor<float, 2> X_norm = (X - xt::mean(X)) / xt::variance(X);
//    xt::xtensor<float, 2> A = xt::linalg::dot(X_norm, xt::transpose(X_norm)) / X.shape(1);
//    std::cerr << A << std::endl;
//    A += 0.00001 * xt::eye(A.shape(0));
//    float d = xt::linalg::det(A);
//    xt::linalg::cholesky(A);

//    std::cerr << "X_norm: " << X_norm << std::endl;
//    std::cerr << "XtX: " << xt::linalg::dot(xt::transpose(X), X) << std::endl;
//    std::cerr << "XntXn: " << xt::linalg::dot(xt::transpose(X_norm), X_norm) << std::endl;
//    std::cerr << "XntXn/Xdim0: " << xt::linalg::dot(xt::transpose(X_norm), X_norm) / X.shape(0) << std::endl;
    //std::cerr << "chol: " << xt::linalg::cholesky(xt::linalg::dot(xt::transpose(X_norm), X_norm) / X.shape(0)) << std::endl;
//    std::cerr << "----------------------" << std::endl;
//    std::cerr << "X_norm: " << X_norm << std::endl;
//    std::cerr << "XntXn: " << xt::linalg::dot(X_norm, xt::transpose(X_norm)) << std::endl;
//    std::cerr << "XntXn/Xdim0: " << xt::linalg::dot(X_norm, xt::transpose(X_norm)) / X.shape(0) << std::endl;
//    std::cerr << "chol: " << xt::linalg::cholesky(xt::linalg::dot(X_norm, xt::transpose(X_norm)) / X.shape(0)) << std::endl;
//    std::cerr << "----------------------" << std::endl;
//    X_norm = (X - xt::mean(X, {0})) / xt::stddev(X, {0});
//    std::cerr << "mean/stddev: " << xt::mean(X, {0}) << " / " << xt::stddev(X, {0}) << std::endl;
//    std::cerr << "XntXn/Xdim0: " << xt::linalg::dot(xt::transpose(X_norm), X_norm) / X.shape(0) << std::endl;
//    std::cerr << "chol: " << xt::linalg::cholesky(xt::linalg::dot(xt::transpose(X_norm), X_norm) / X.shape(0)) << std::endl;
//    std::cerr << "----------------------" << std::endl;
//    std::cerr << "BOLT-LMM" << std::endl;
//    X_norm = ((X - xt::mean(X, {0})) / xt::stddev(X, {0}));
//    std::cerr << "mean/stddev: " << xt::mean(X, {0}) << " / " << xt::stddev(X, {0}) << std::endl;
//    std::cerr << "XnXnt/Xdim0: " << xt::linalg::dot(X_norm, xt::transpose(X_norm)) / X.shape(1) << std::endl;
//    std::cerr << "chol: " << xt::linalg::cholesky(xt::linalg::dot(X_norm, xt::transpose(X_norm)) / X.shape(1)) << std::endl;
//    std::cerr << "----------------------" << std::endl;
//    X_norm = (X) / xt::stddev(X, {0});
//    std::cerr << "stddev: " << xt::stddev(X, {0}) << std::endl;
//    std::cerr << "XntXn/Xdim0: " << xt::linalg::dot(X_norm, xt::transpose(X_norm)) / X.shape(1) << std::endl;
//    std::cerr << "chol: " << xt::linalg::cholesky(xt::linalg::dot(X_norm, xt::transpose(X_norm)) / X.shape(1)) << std::endl;



    Eigen::Map<Eigen::VectorXd> mapped_resp(xresp.data(), xresp.size());
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped_cov(xcov.data(), xcov.shape(0), xcov.shape(1));
    Eigen::SparseMatrix<double> kinship(xresp.size(), xresp.size());
    if (!mixed_effects_model::load_kinship(args.kinship_file_path(), kinship, sample_intersection))
      return std::cerr << "Error: could not load kinship\n", EXIT_FAILURE;


    std::vector<savvy::compressed_vector<float>> grammar_genotypes;
    if (!mixed_effects_model::load_grammar_variants(geno_file, args.fmt_field(), mapped_resp, grammar_genotypes))
      return std::cerr << "Error: could not load grammar variants\n", EXIT_FAILURE;

    geno_file.reset_bounds(0);

    return run_single(args, geno_file, mixed_effects_model(mapped_resp, mapped_cov, kinship, grammar_genotypes, sample_intersection));
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