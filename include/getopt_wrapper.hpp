#ifndef GETOPT_WRAPPER_HPP
#define GETOPT_WRAPPER_HPP

#include <cstdint>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <getopt.h>

class getopt_wrapper
{
public:
  struct option_with_desc
  {
    std::string long_opt;
    std::string argument;
    int short_opt;
    std::string description;
    option_with_desc(const std::string& _long_opt, const std::string& _argument, int _short_opt, const std::string& _description)
    {
      long_opt = _long_opt;
      argument = _argument;
      short_opt = _short_opt;
      description = _description;
    }
  };
protected:
  std::vector<option> long_options_;
  std::vector<option_with_desc> opts_;
  std::string usage_str_;
  std::string short_opt_string_;
  std::size_t max_long_opt_length_ = 0;
public:
  getopt_wrapper(std::string usage_str, std::vector<option_with_desc>&& long_opts) :
    usage_str_(std::move(usage_str)),
    opts_(std::move(long_opts))
  {
    std::unordered_set<char> unique_vals;
    long_options_.resize(opts_.size() + 1, {0, 0, 0, 0});
    auto lit = long_options_.begin();
    for (auto it = opts_.begin(); it != opts_.end(); ++it)
    {
      *(lit++) = {it->long_opt.c_str(), it->argument.empty() ? no_argument : required_argument, NULL, it->short_opt};
      max_long_opt_length_ = std::max(max_long_opt_length_, it->long_opt.size() + it->argument.size());
      if (it->short_opt && unique_vals.insert(it->short_opt).second)
      {
        short_opt_string_ += (char)it->short_opt;
        if (!it->argument.empty())
          short_opt_string_ += ':';
      }
    }
  }

  void print_usage(std::ostream& os)
  {
    os << usage_str_ << '\n';
    os << '\n';
    for (auto it = opts_.begin(); it != opts_.end(); ++it)
    {
      if (it->description.empty())
        continue;

      if (std::isprint(it->short_opt))
      {
        if (it->long_opt.empty())
          os << " -" << (char)it->short_opt << "  ";
        else
          os << " -" << (char)it->short_opt << ", ";
      }
      else
        os << "     ";

      std::size_t n_spaces = 2;
      if (it->long_opt.empty())
        n_spaces += 2;
      else
        os << "--" << it->long_opt;

      os.put(' ');
      if (!it->argument.empty())
        os << it->argument;

      n_spaces += max_long_opt_length_ - it->long_opt.size() - it->argument.size();
      for (std::size_t i = 0; i < n_spaces; ++i)
        os.put(' ');
      os << it->description << '\n';
    }

    os << std::flush;
  }
};

#endif // GETOPT_WRAPPER_HPP