
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

int plot_qq_main(int argc, char** argv)
{
  if (argc < 2)
    return std::cerr << "Error: missing argument (path to results file)\n", EXIT_FAILURE;

  if (std::string(argv[0]) != "qq")
    return std::cerr << "Error: unsupported plot type " << argv[0] << " (only QQ plots are supported)\n", EXIT_FAILURE;

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