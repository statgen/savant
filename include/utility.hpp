
/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SAVANT_UTILITY_HPP
#define SAVANT_UTILITY_HPP

#include <savvy/region.hpp>

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

struct utility
{
  static std::vector<std::string> split_string_to_vector(const char* in, char delim)
  {
    std::vector<std::string> ret;
    const char* d = nullptr;
    std::string token;
    const char* s = in;
    const char*const e = in + strlen(in);
    while ((d = std::find(s, e,  delim)) != e)
    {
      ret.emplace_back(std::string(s, d));
      s = d ? d + 1 : d;
    }
    ret.emplace_back(std::string(s,d));
    return ret;
  }

  static std::vector<std::string> split_string_to_vector(const std::string& in, char delim)
  {
    return split_string_to_vector(in.c_str(), delim);
  }

  static savvy::genomic_region string_to_region(const std::string& s)
  {
    const std::size_t colon_pos = s.find(':');
    if (colon_pos == std::string::npos)
    {
      return savvy::genomic_region(s);
    }
    else
    {
      std::string chr = s.substr(0, colon_pos);
      const std::size_t hyphen_pos = s.find('-', colon_pos + 1);
      if (hyphen_pos == std::string::npos)
      {
        std::string slocus = s.substr(colon_pos + 1);
        std::uint64_t ilocus = std::uint64_t(std::atoll(slocus.c_str()));
        return savvy::genomic_region(chr, ilocus, ilocus);
      }
      else
      {
        std::string sbeg = s.substr(colon_pos + 1, hyphen_pos - chr.size() - 1);
        std::string send = s.substr(hyphen_pos + 1);
        if (send.empty())
        {
          return savvy::genomic_region(chr, std::uint64_t(std::atoll(sbeg.c_str())));
        }
        else
        {
          return savvy::genomic_region(chr, std::uint64_t(std::atoll(sbeg.c_str())), std::uint64_t(std::atoll(send.c_str())));
        }
      }
    }

  }

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
};

#endif // SAVANT_UTILITY_HPP