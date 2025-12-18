/*
  edge_detection.cpp

  This file is part of TCannyMod

  Copyright (C) 2013 Oka Motofumi

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


#include <algorithm>
#include <vector>
#include <cmath>
#include <cfloat>
#include <array>
#include "tcannymod.h"


struct Pos {
    int32_t x, y;
    Pos(int32_t _x, int32_t _y) : x(_x), y(_y) {}
    void search(const int width, const int height, float* edge, uint8_t* hyst,
        const size_t epitch, const size_t hpitch, const float th,
        std::vector<Pos>& stack)
    {
        std::array<Pos, 8> coodinates{
            Pos(x - 1, y - 1), Pos(x , y - 1), Pos(x + 1, y - 1), Pos(x - 1, y),
            Pos(x + 1, y), Pos(x - 1, y + 1), Pos(x, y + 1), Pos(x + 1, y + 1),
        };
        for (const auto& p : coodinates) {
            if (p.x < 0) continue;
            else if (p.x == width) continue;
            else if (p.y < 0) continue;
            else if (p.y == height) continue;
            else {
                auto posh = p.x + p.y * hpitch;
                auto pose = p.x + p.y * epitch;
                if (hyst[posh] == 0 && edge[pose] >= th) {
                    edge[pose] = FLT_MAX;
                    hyst[posh] = 0xFF;
                    stack.emplace_back(p);
                }
            }
        }
    }
};


void __stdcall
hysteresis(uint8_t* hmap, const size_t hpitch, float* emap,
           const size_t epitch, const int width, const int height,
           const float tmin, const float tmax) noexcept
{
    memset(hmap, 0, hpitch * height);
    std::vector<Pos> stack;
    stack.reserve(512);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            auto posh = x + y * hpitch;
            auto posb = x + y * epitch;
            if (hmap[posh] != 0 || emap[posb] < tmax) {
                continue;
            } else {
                emap[posb] = FLT_MAX;
                hmap[posh] = 0xFF;
                stack.emplace_back(x, y);
            }
            while (!stack.empty()) {
                auto pos = stack.back();
                stack.pop_back();
                pos.search(width, height, emap, hmap, epitch, hpitch, tmin,
                    stack);
            }
        }
    }
}
