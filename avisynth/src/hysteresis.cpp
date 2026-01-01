/*
  hysteresis.cpp

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


#include <vector>
#include "tcannymod.hpp"

struct Pos {
    int x, y;
    Pos(int _x, int _y) : x(_x), y(_y) {}
    template <typename Td>
    void search(const int width, const int height, float* emaskp, Td* dstp,
        const int epitch, const int dpitch, const float th, const Td maxv,
        std::vector<Pos>& stack)
    {
        std::array<Pos, 8> coordinates{
            Pos(x - 1, y - 1), Pos(x, y - 1), Pos(x + 1, y - 1), Pos(x - 1, y),
            Pos(x + 1, y), Pos(x - 1, y + 1), Pos(x, y + 1), Pos(x + 1, y + 1),
        };
        for (const auto& p : coordinates) {
            if (p.x < 0 || p.x >= width || p.y < 0 || p.y >= height)
                continue;
            else {
                auto posD = p.x + p.y * dpitch;
                auto posE = p.x + p.y * epitch;
                if (dstp[posD] == 0 && emaskp[posE] >= th) {
                    dstp[posD] = maxv;
                    stack.emplace_back(p);
                }
            }
        }
    }
};


template <typename Td>
static void hysteresis(void* dstp, const int dpitch, float* emaskp,
    const int epitch, const int width, const int height, const float tmin,
    const float tmax, const float maxval)
{
    Td* d = reinterpret_cast<Td*>(dstp);
    const Td maxv = static_cast<Td>(maxval);

    memset(d, 0, dpitch * height * sizeof(Td));
    std::vector<Pos> stack;
    stack.reserve(512);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto posD = x + y * dpitch;
            if (d[posD] > 0 || emaskp[x + y * epitch] < tmax) {
                continue;
            }
            d[posD] = maxv;
            stack.emplace_back(x, y);

            do {
                auto pos = stack.back();
                stack.pop_back();
                pos.search<Td>(width, height, emaskp, d, epitch, dpitch, tmin,
                    maxv, stack);
            } while (!stack.empty());
        }
    }
}


hysteresis_t get_hysteresis(int bytes)
{
    if (bytes == 1) return hysteresis<uint8_t>;
    if (bytes == 2) return hysteresis<uint16_t>;
    return hysteresis<float>;
}

