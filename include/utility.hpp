
/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef SAVANT_UTILITY_HPP
#define SAVANT_UTILITY_HPP

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
};

#endif // SAVANT_UTILITY_HPP