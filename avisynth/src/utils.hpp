/*
  utils.hpp

  This file is part of TCannyMod

  Copyright (C) 2026 Oka Motofumi

  Authors: Oka Motofumi (chikuzen.mo at gmail dot com)

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
*/

#ifndef TCM_UTILS_HPP
#define TCM_UTILS_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <type_traits>

bool has_sse41(uint32_t info = 0) noexcept;

bool has_avx(uint32_t info = 0) noexcept;

bool has_avx2(uint32_t info = 0) noexcept;

bool has_avx512(uint32_t info = 0) noexcept;

bool has_avx512fp16(uint32_t info = 0) noexcept;


std::vector<std::string> split(const std::string& str, const char* separator) noexcept;

uint32_t get_halfvalue(int bits);

#endif // TCM_UTILS_HPP

