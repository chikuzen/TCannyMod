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
#include "edge_detection.h"

void __stdcall
non_max_suppress_c(const float* emaskp, const size_t em_pitch,
                 const int32_t* dirp, const size_t dir_pitch,
                 float* blurp, const size_t blr_pitch, const int width,
                 const int height)
{
    using std::max;

    memset(blurp - 4, 0, blr_pitch * sizeof(float));
    for (int y = 1; y < height - 1; ++y) {
        memcpy(blurp + blr_pitch * y, emaskp + em_pitch * y,
               width * sizeof(float));
    }
    blurp += blr_pitch;
    emaskp += em_pitch;

    for (int y = 1; y < height - 1; ++y) {
        dirp += dir_pitch;
        blurp[-1] = blurp[0] = -FLT_MAX;
        for (int x = 1; x < width - 1; ++x) {
            float p0;
            if (dirp[x] == 255) {
                p0 = max(emaskp[x - 1 - em_pitch], emaskp[x + 1 + em_pitch]);
            } else if (dirp[x] == 127) {
                p0 = max(emaskp[x - em_pitch], emaskp[x + em_pitch]);
            } else if (dirp[x] == 63) {
                p0 = max(emaskp[x + 1 - em_pitch], emaskp[x - 1 + em_pitch]);
            } else {
                p0 = max(emaskp[x + 1], emaskp[x - 1]);
            }
            if (emaskp[x] < p0) {
                blurp[x] = -FLT_MAX;
            }
        }
        blurp[width - 1] = blurp[width] = -FLT_MAX;
        emaskp += em_pitch;
        blurp += blr_pitch;
    }

    memset(blurp - 4, 0, blr_pitch * sizeof(float));
}


struct Pos {
    int32_t x, y;
    Pos(int32_t _x, int32_t _y) {
        x = _x;
        y = _y;
    }
};

static __forceinline void
hystfunc(const int32_t x, const int32_t y, float* edge, uint8_t* hyst,
         const size_t epitch, const size_t hpitch, const float th,
         std::vector<Pos>& stack)
{
    if (!hyst[x + y * hpitch] && edge[x + y * epitch] > th) {
        edge[x + y * epitch] = FLT_MAX;
        hyst[x + y * hpitch] = 0xFF;
        stack.emplace_back(x, y);
    }
}


void __stdcall
hysteresis(uint8_t* hystp, const size_t hpitch, float* blurp,
           const size_t bpitch, const int width, const int height,
           const float tmin, const float tmax)
{
    memset(hystp, 0, hpitch * height);
    std::vector<Pos> stack;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            if (hystp[x + y * hpitch] || blurp[x + y * bpitch] < tmax) {
                continue;
            }
            blurp[x + y * bpitch] = FLT_MAX;
            hystp[x + y * hpitch] = 0xFF;
            stack.emplace_back(x, y);

            while (!stack.empty()) {
                Pos  pos = stack.back();
                stack.pop_back();
                int32_t xmin = std::max(pos.x - 1, 0);
                int32_t xmax = std::min(pos.x + 1, width - 1);
                int32_t ymin = std::max(pos.y - 1, 0);
                int32_t ymax = std::min(pos.y + 1, height - 1);
                hystfunc(xmin, ymin, blurp, hystp, bpitch, hpitch, tmin, stack);
                hystfunc(xmin + 1, ymin, blurp, hystp, bpitch, hpitch, tmin, stack);
                if (xmin + 2 == xmax) {
                    hystfunc(xmax, ymin, blurp, hystp, bpitch, hpitch, tmin, stack);
                }
                hystfunc(xmin, ymin + 1, blurp, hystp, bpitch, hpitch, tmin, stack);
                if (xmin + 2 == xmax) {
                    hystfunc(xmax, ymin + 1, blurp, hystp, bpitch, hpitch, tmin, stack);
                }
                if (ymin + 2 == ymax) {
                    hystfunc(xmin, ymax, blurp, hystp, bpitch, hpitch, tmin, stack);
                    hystfunc(xmin + 1, ymax, blurp, hystp, bpitch, hpitch, tmin, stack);
                    if (xmin + 2 == xmax) {
                        hystfunc(xmax, ymax, blurp, hystp, bpitch, hpitch, tmin, stack);
                    }
                }
            }
        }
    }
}
