
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
#include <sys/stat.h>
#include <getopt.h>

class plot_pca_prog_args
{
private:
  std::vector<option> long_options_;
  std::string input_path_;
  std::string pop_map_path_;
  std::string custom_plot_commands_;
  std::string output_path_ = "/dev/stdout";
  int first_pc_ = 1;
  int num_pcs_ = 2;
  bool help_ = false;
public:
  plot_pca_prog_args() :
    long_options_(
      {
        {"gnuplot-opts", required_argument, 0, 'g'},
        {"help", no_argument, 0, 'h'},
        {"output", required_argument, 0, 'o'},
        {"pcs", required_argument, 0, 'p'},
        {0, 0, 0, 0}
      })
  {
  }

  const std::string& input_path() const { return input_path_; }
  const std::string& pop_map_path() const { return pop_map_path_; }
  const std::string& output_path() const { return output_path_; }
  const std::string& custom_plot_commands() const { return custom_plot_commands_; }

  int first_pc() const { return first_pc_; }
  int num_pcs() const { return num_pcs_; }
  bool help_is_set() const { return help_; }

  void print_usage(std::ostream& os)
  {
    os << "Usage: savant plot pca [opts ...] <results_file> [<id_to_pop_file>] \n";
    os << "\n";
    os << " -g, --gnuplot-opts  Custom gnuplot commands to include\n";
    os << " -h, --help          Print usage\n";
    os << " -o, --output        Output path (default: /dev/stdout)\n";
    os << " -p, --pcs           PC range to plot in the format <FIRST_PC>:<LAST_PC> (default: 1:2)\n";
    os << std::flush;
  }

  bool parse(int argc, char** argv)
  {
    int long_index = 0;
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "g:ho:p:", long_options_.data(), &long_index )) != -1)
    {
      char copt = char(opt & 0xFF);
      switch (copt)
      {
      case 'g':
        custom_plot_commands_ = optarg ? optarg : "";
        custom_plot_commands_.erase(custom_plot_commands_.find_last_not_of(" \t\r\n;") + 1);
        custom_plot_commands_ += "; ";
        break;
      case 'h':
        help_ = true;
        return true;
      case 'o':
        output_path_ = optarg ? optarg : "";
        break;
      case 'p':
      {
        auto tokens = utility::split_string_to_vector(optarg ? optarg : "", ':');
        if (tokens.size() < 2 || std::atoi(tokens[0].c_str()) >= std::atoi(tokens[1].c_str()))
          return std::cerr << "Error: invalid pc range ("<< (optarg ? optarg : "") << ")\n", false;

        first_pc_ = std::atoi(tokens[0].c_str());
        int last_pc = std::atoi(tokens[1].c_str());
        num_pcs_ = (last_pc - first_pc_) + 1;
        break;
      }
      default:
        return false;
      }
    }

    int remaining_arg_count = argc - optind;

    if (remaining_arg_count == 1)
    {
      input_path_ = argv[optind];
    }
    else if (remaining_arg_count == 2)
    {
      input_path_ = argv[optind];
      pop_map_path_ = argv[optind + 1];
    }
    else if (remaining_arg_count < 1)
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

int plot_pca_main(int argc, char** argv)
{
  plot_pca_prog_args args;
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


  std::string line;
  std::unordered_map<std::string, std::uint8_t> sample_to_color;
  std::unordered_map<std::string, std::uint8_t> population_to_color = {{"", 0}};
  std::list<std::string> ordered_populations;


  if (!args.pop_map_path().empty())
  {
    std::ifstream population_map_file(args.pop_map_path());
    if (!population_map_file)
      return std::cerr << "Error: could not open population map file ("<< args.pop_map_path() << ")\n", EXIT_FAILURE;

    while (std::getline(population_map_file, line))
    {
      auto fields = utility::split_string_to_vector(line, '\t');
      if (fields.size() < 2)
        return std::cerr << "Error: not enough columns in " << args.pop_map_path() << "\n", EXIT_FAILURE;
      std::uint8_t& c = population_to_color[fields[1]];
      if (c == 0 && fields[1].size())
      {
        c = population_to_color.size() - 1;
        ordered_populations.emplace_back(fields[1]);
      }

      sample_to_color[fields[0]] = c;
    }
  }




  shrinkwrap::gz::istream results_file(args.input_path());
  if (!results_file || !std::getline(results_file, line))
    return std::cerr << "Error: could not open results file ("<< args.input_path() << ")\n", EXIT_FAILURE;


  auto fields = utility::split_string_to_vector(line, '\t');

  if (fields.empty())
    return std::cerr << "Error: empty header line\n", EXIT_FAILURE;

  std::string first_pc_header = "pc" + std::to_string(args.first_pc());
  std::size_t sample_idx = 0; // TODO: add option
  std::size_t first_pc_idx = 0;
  for ( ; first_pc_idx < fields.size(); ++first_pc_idx)
    if (fields[first_pc_idx] == first_pc_header) break;

  if (first_pc_idx == fields.size())
    return std::cerr << "Error: '" << first_pc_header << "' missing from header line\n", EXIT_FAILURE;

  if (fields.size() <= first_pc_idx + args.num_pcs())
    return std::cerr << "Error: last pc (" << (args.first_pc() + args.num_pcs()) - 1 << ") out of header line range\n", EXIT_FAILURE;

  // ================ //
  // TODO: allow for custom population => color map files
  //std::unordered_set<std::string>  unique_populations =  {"AFR", "AMR", "EAS", "EUR", "SAS"};
  //std::vector<std::string> unique_populations_sorted;
  //unique_populations_sorted.reserve(unique_populations.size());
  //for (const auto& p : unique_populations)
  //  unique_populations_sorted.insert(std::upper_bound(unique_populations_sorted.begin(), unique_populations_sorted.end(), p), p);
  //std::vector<std::string> unique_populations_sorted = {"AFR", "AMR", "EAS", "EUR", "SAS"};
  // ================ //

  //  std::string plot_cmd = "ancestries=\"AFR AMR EAS EUR SAS\""
  //                          "anc_to_color(name) = sum [i=1:words(ancestries)] (name eq word(ancestries, i) ? i : 0)"
  //                          "plot '/dev/stdin' u 1:2:(anc_to_color(strcol(3))) w points pt 6 lc variable, for [i=1:words(ancestries)] NaN title word(ancestries, i) lc i";


  if (args.num_pcs() == 2)
  {
    // set linetype 1 linecolor rgb '
    // set style line 1 lt 1 lc 'black; set style increment user;'
    // load '~/.gnuplot-colorbrewer/qualitative/Dark2.plt';
    // set terminal jpeg size 1600,1200 enhanced font ',18'; set xtics ('0' 0); set ytics ('0' 0)
    std::stringstream plot_cmd;
    plot_cmd << "gnuplot --persist -e \"";
    if (args.custom_plot_commands().size())
      plot_cmd << args.custom_plot_commands();
    plot_cmd << "set style increment user; ";
    plot_cmd << "plot '/dev/stdin' u 1:2:3 w points pt 6 lc variable notitle";
    for (const auto& p : ordered_populations)
      plot_cmd << ", NaN w circles title '" << p << "' ls " << 1 + population_to_color[p];
    plot_cmd << "\"";

    std::FILE* pipe = popen(plot_cmd.str().c_str(), "w");
    if (!pipe)
      return perror("popen"), EXIT_FAILURE;

    std::size_t cnt = 1;
    while (std::getline(results_file, line))
    {
      fields = utility::split_string_to_vector(line, '\t');

      if (first_pc_idx + 1 >= fields.size())
        return std::cerr << "Error: number of columns do not match header line at line " << cnt << std::endl, EXIT_FAILURE;

      std::fprintf(pipe, "%s\t%s\t%u\n", fields[first_pc_idx].c_str(), fields[first_pc_idx + 1].c_str(), 1 + sample_to_color[fields[sample_idx]]);
      ++cnt;
    }

    pclose(pipe);
  }
  else
  {
    char tmpl[] = "/tmp/tmp_savant_XXXXXX";
    std::string temp_prefix = mkdtemp(tmpl); //std::tmpnam(nullptr);
    std::cerr << "Created temp directory " << temp_prefix << std::endl;

    temp_prefix += '/';

    //double max_x = 0.;
    //double max_y = 0.;
    std::size_t cnt = 1;
    std::list<std::ofstream> temp_files;
    std::list<std::string> temp_file_names;
    for (std::size_t j = 0; j < args.num_pcs(); ++j)
    {
      for (std::size_t i = j + 1; i < args.num_pcs(); ++i)
      {
        temp_file_names.emplace_back(temp_prefix + std::to_string(i) + "_" + std::to_string(j) + ".tsv");
        temp_files.emplace_back(temp_file_names.back());
      }
    }

    while (std::getline(results_file, line))
    {
      fields = utility::split_string_to_vector(line, '\t');

      if (first_pc_idx + args.num_pcs() >= fields.size())
        return std::cerr << "Error: number of columns do not match header line at line " << cnt << std::endl, EXIT_FAILURE;

      auto tmp_it = temp_files.begin();
      for (std::size_t j = 0; j < args.num_pcs(); ++j)
      {
        for (std::size_t i = j + 1; i < args.num_pcs(); ++i)
        {
          *tmp_it << fields[first_pc_idx + j].c_str() << "\t" << fields[first_pc_idx + i] << "\t" << 1 + sample_to_color[fields[sample_idx]] << "\n";
          ++tmp_it;
        }
      }
      ++cnt;
    }

    //std::ofstream plot_cmd("/dev/stdout");
    std::stringstream plot_cmd;
    plot_cmd << "gnuplot --persist -e \"";
    if (args.custom_plot_commands().size())
      plot_cmd << args.custom_plot_commands();
    plot_cmd << "set multiplot layout " << args.num_pcs() << "," << args.num_pcs() << " rowsfirst; ";
    plot_cmd << "set style increment user; ";
    plot_cmd << "save set '| grep -v range | grep -v origin > " << temp_prefix << "settings.gnu'; ";
    for (std::size_t i = 0; i < args.num_pcs(); ++i)
    {
      for (std::size_t j = 0; j < args.num_pcs(); ++j)
      {
        if (i < j)
        {
          plot_cmd << "unset border; unset tics; ";
          if (i == 0 && j + 1 == args.num_pcs())
          {
            plot_cmd << "set key center center; ";
            plot_cmd << "plot [-1:1][-1:1] '< echo 0 0' w labels notitle";
            for (const auto& p : ordered_populations)
              plot_cmd << ", NaN w circles title '" << p << "' ls " << 1 + population_to_color[p];
            plot_cmd << "; ";
          }
          else
          {
            plot_cmd << "plot [-1:1][-1:1] '< echo 0 0' w labels notitle; ";
          }
          plot_cmd << "load '" << temp_prefix << "settings.gnu'; ";
        }
        else if (j == i)
        {
          plot_cmd << "unset border; unset tics; ";
          plot_cmd << "plot [-1:1][-1:1] '< echo 0 0 PC" << i + args.first_pc() << "' w labels notitle; ";
          plot_cmd << "load '" << temp_prefix << "settings.gnu'; ";
        }
        else
        {
          plot_cmd << "plot '" << temp_prefix << i << "_" << j << ".tsv' u 1:2:3 w points pt 6 lc variable notitle; ";
        }
      }
    }
    plot_cmd << "unset multiplot\"";

    std::FILE* pipe = popen(plot_cmd.str().c_str(), "w");
    if (!pipe)
      return perror("popen"), EXIT_FAILURE;
    pclose(pipe);

    for (const auto& t: temp_file_names)
      std::remove(t.c_str());
    std::remove((temp_prefix + "settings.gnu").c_str());

    if (::rmdir(temp_prefix.c_str()))
      std::cerr << "Warning:: failed to remove temp directory\n";
  }

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