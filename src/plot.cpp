
#include "plot.hpp"
#include "utility.hpp"

#include <shrinkwrap/gz.hpp>

#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <sstream>

int plot_pca_main(int argc, char** argv)
{
  if (argc < 3)
    return std::cerr << "Error: missing argument (path to results file)\n", EXIT_FAILURE;

  std::string results_file_path = argv[1];
  std::string population_map_file_path = argv[2];

  std::string line;
  std::unordered_map<std::string, std::string> sample_to_population;
  std::ifstream population_map_file(population_map_file_path);
  if (!population_map_file || !std::getline(population_map_file, line))
    return std::cerr << "Error: could not open population map file ("<< population_map_file_path << ")\n", EXIT_FAILURE;

  while (std::getline(population_map_file, line))
  {
    auto fields = utility::split_string_to_vector(line, '\t');
    if (fields.size() < 2)
      return std::cerr << "Error: not enough columns in " << population_map_file_path << "\n", EXIT_FAILURE;
    sample_to_population[fields[0]] = fields[1];
  }

  shrinkwrap::gz::istream results_file(results_file_path);
  if (!results_file || !std::getline(results_file, line))
    return std::cerr << "Error: could not open results file ("<< results_file_path << ")\n", EXIT_FAILURE;


  auto fields = utility::split_string_to_vector(line, '\t');

  if (fields.empty())
    return std::cerr << "Error: empty header line\n", EXIT_FAILURE;

  std::size_t sample_idx = 0; // TODO: add option
  std::size_t pc1_idx = 0;
  for ( ; pc1_idx < fields.size(); ++pc1_idx)
    if (fields[pc1_idx] == "pc1") break;

  if (pc1_idx == fields.size())
    return std::cerr << "Error: 'pc1' missing from header line\n", EXIT_FAILURE;

  // from color brewer sequential
  std::unordered_map<std::string, std::string> population_to_color = {
    // super populations
    {"AFR", "0x6A51A3"}, // purple
    {"AMR", "0xEC7014"}, // orange
    {"EAS", "0xCB181D"}, // red
    {"EUR", "0x238B45"}, // green
    {"SAS", "0x2171B5"}, // blue

    // AFR
    {"ACB", "0xEFEDF5"}, //
    {"ASW", "0xDADAEB"}, //
    {"ESN", "0xBCBDDC"}, // light purple
    {"GWD", "0x9E9AC8"}, //
    {"LWK", "0x807DBA"}, // medium purple
    {"MSL", "0x6A51A3"}, //
    {"YRI", "0x4A1486"}, // dark purple

    // AMR
    {"CLM", "0xFEC44F"}, //  light yellow-orange-brown
    {"MXL", "0xFE9929"}, //
    {"PEL", "0xEC7014"}, //  medium yellow-orange-brown
    {"PUR", "0xCC4C02"}, //

    // EAS
    {"CDX", "0xFC9272"}, // light red
    {"CHB", "0xFB6A4A"}, //
    {"CHS", "0xEF3B2C"}, // medium red
    {"JPT", "0xCB181D"}, //
    {"KHV", "0x99000D"}, // dark red

    // EUR
    {"CEU", "0xA1D99B"}, // light green
    {"FIN", "0x74C476"}, //
    {"GBR", "0x41AB5D"}, // medium green
    {"IBS", "0x238B45"}, //
    {"TSI", "0x005A32"}, // dark green

    // SAS
    {"BEB", "0x9ECAE1"}, // light blue
    {"GIH", "0x6BAED6"}, //
    {"ITU", "0x4292C6"}, // medium blue
    {"PJL", "0x2171B5"}, //
    {"STU", "0x084594"}  // dark blue
  };

  // ================ //
  // TODO: allow for custom population => color map files
  //std::unordered_set<std::string>  unique_populations =  {"AFR", "AMR", "EAS", "EUR", "SAS"};
  //std::vector<std::string> unique_populations_sorted;
  //unique_populations_sorted.reserve(unique_populations.size());
  //for (const auto& p : unique_populations)
  //  unique_populations_sorted.insert(std::upper_bound(unique_populations_sorted.begin(), unique_populations_sorted.end(), p), p);
  std::vector<std::string> unique_populations_sorted = {"AFR", "AMR", "EAS", "EUR", "SAS"};
  // ================ //

  //  std::string plot_cmd = "ancestries=\"AFR AMR EAS EUR SAS\""
  //                          "anc_to_color(name) = sum [i=1:words(ancestries)] (name eq word(ancestries, i) ? i : 0)"
  //                          "plot '/dev/stdin' u 1:2:(anc_to_color(strcol(3))) w points pt 6 lc variable, for [i=1:words(ancestries)] NaN title word(ancestries, i) lc i";


  std::stringstream plot_cmd;
  plot_cmd << "gnuplot --persist -e \"";
  plot_cmd << "plot '/dev/stdin' u 1:2:3 w points pt 6 lc rgb variable notitle";
  for (const auto& p : unique_populations_sorted)
    plot_cmd << ", NaN w circles title '" << p << "' lc rgb " << population_to_color[p];
  plot_cmd << "\"";

  std::FILE* pipe = popen(plot_cmd.str().c_str(), "w");
  if (!pipe)
    return perror("popen"), EXIT_FAILURE;

  std::size_t cnt = 1;
  while (std::getline(results_file, line))
  {
    fields = utility::split_string_to_vector(line, '\t');

    if (pc1_idx + 1 >= fields.size())
      return std::cerr << "Error: number of columns do not match header line at line " << cnt << std::endl, EXIT_FAILURE;

    std::fprintf(pipe, "%s\t%s\t%s\n", fields[pc1_idx].c_str(), fields[pc1_idx + 1].c_str(), population_to_color[sample_to_population[fields[sample_idx]]].c_str()); // TODO: handle parse failure.
    ++cnt;
  }

  pclose(pipe);

  return EXIT_SUCCESS;
}

int plot_qq_main(int argc, char** argv)
{
  if (argc < 2)
    return std::cerr << "Error: missing argument (path to results file)\n", EXIT_FAILURE;

  std::string results_file_path = argv[1];

  std::string line;
  shrinkwrap::gz::istream results_file(results_file_path);
  if (!results_file || !std::getline(results_file, line))
    return std::cerr << "Error: could not open results file ("<< results_file_path << ")\n", EXIT_FAILURE;


  auto fields = utility::split_string_to_vector(line, '\t');

  if (fields.empty())
    return std::cerr << "Error: empty header line\n", EXIT_FAILURE;

  std::size_t pval_idx = 0;
  for ( ; pval_idx < fields.size(); ++pval_idx)
    if (fields[pval_idx] == "pvalue") break;

  std::size_t maf_idx = 0;
  for ( ; maf_idx < fields.size(); ++maf_idx)
    if (fields[maf_idx] == "maf") break;

  if (pval_idx == fields.size())
    return std::cerr << "Error: 'pvalue' missing from header line\n", EXIT_FAILURE;

  if (maf_idx == fields.size())
    return std::cerr << "Error: 'maf' missing from header line\n", EXIT_FAILURE;

  struct datum
  {
    double maf;
    double pval;
    datum(double m, double p) : maf(m), pval(p) {}
  };

  std::vector<datum> pvalues;
  while (std::getline(results_file, line))
  {
    fields = utility::split_string_to_vector(line, '\t');

    if (pval_idx >= fields.size())
      return std::cerr << "Error: number of columns do not match header line at line " << (pvalues.size() + 2) << std::endl, EXIT_FAILURE;

    pvalues.emplace_back(std::atof(fields[maf_idx].c_str()), std::atof(fields[pval_idx].c_str())); // TODO: handle parse failure.
//    if (pvalues.size() > 500)
//      break;
  }

  std::sort(pvalues.begin(), pvalues.end(), [](const auto& l, const auto& r) { return l.pval < r.pval; });


  std::string plot_cmd = "gnuplot --persist -e \"set key bottom right; "
                         "plot '-' using 1:2:3 with points pt 6 lc variable notitle"
                         ", x lc 'gray' notitle"
                         ", NaN w p pt 6 lc 1 title 'MAF <= 0.5' "
                         ", NaN w p pt 6 lc 2 title 'MAF < 0.1' "
                         ", NaN w p pt 6 lc 3 title 'MAF < 0.01' "
                         ", NaN w p pt 6 lc 4 title 'MAF < 0.001' "
                         ", NaN w p pt 6 lc 5 title 'MAF < 0.0001' \"";

  // TODO: determine min MAF bin

  std::FILE* pipe = popen(plot_cmd.c_str(), "w");
  if (!pipe)
    return perror("popen"), EXIT_FAILURE;

  int i = 0;
  for (auto d : pvalues)
  {
//    auto a =  -std::log10( (1. + i++) / pvalues.size());
//    auto b = -log10(p);
    auto a = -std::log10(d.maf);
    int maf_group = int(a) + 1;
    std::fprintf(pipe, "%f\t%f\t%d\n", -std::log10( (1. + i++) / pvalues.size()), -log10(d.pval), std::min(5, maf_group));
  }

  pclose(pipe);

  return EXIT_SUCCESS;
}

int plot_main(int argc, char** argv)
{
  if (argc < 1)
    return std::cerr << "Error: missing plot type  (qq, pca, or manhattan)\n", EXIT_FAILURE;

  std::string plot_type(argv[1]);
  if (plot_type == "qq")
    return plot_qq_main(--argc, ++argv);
  else if (plot_type == "pca")
    return plot_pca_main(--argc, ++argv);
  else
    return std::cerr << "Error: unsupported plot type " << plot_type << "\n", EXIT_FAILURE;
}