#include "assoc.hpp"
#include "group.hpp"

#include <savvy/reader.hpp>

class group_prog_args : public assoc_prog_args
{
private:
  std::string group_file_;
public:
  group_prog_args() :
    assoc_prog_args("group", {
      {"group-file", required_argument, 0, '\x02'}
    })
  {

  }

  void print_usage(std::ostream& os)
  {
    assoc_prog_args::print_usage(os);
    os << "\n";
    os << "Group options:\n";
    os << "     --group-file     Path to group file\n";
    os << std::flush;
  }

  bool process_opt(char copt)
  {
    switch (copt)
    {
    case '\x02':
      if (std::string("group-file") == long_options_[long_index].name)
      {
        group_file_ = optarg ? optarg : "";
      }
      break;
    default:
      return false;
    }
    return true;
  }
};

// [CHROM]:[POS]_[REF]/[ALT]
static savvy::site_info marker_id_to_site_info(std::string::const_iterator beg, std::string::const_iterator end)
{
  auto colon_it = std::find(beg, end, ':');
  std::string chrom(beg, colon_it);
  if (colon_it != end)
  {
    auto underscore_it = std::find(++colon_it, end, '_');
    std::uint64_t pos = static_cast<std::uint64_t>(std::atoll(std::string(colon_it, underscore_it).c_str()));
    if (underscore_it != end)
    {
      auto slash_it = std::find(++underscore_it, end, '/');
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

int group_main(int argc, char** argv)
{
  group_prog_args args;
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



  return EXIT_SUCCESS;
}