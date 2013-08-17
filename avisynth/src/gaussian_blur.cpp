#include <emmintrin.h>
#include "tcannymod.hpp"


static inline void __stdcall
horizontal_blur(float* srcp, const int radius, const int length, int width,
                float* kernel, float* dstp)
{
    for (int i = 1; i <= radius ; i++) {
        srcp[-i] = srcp[i];
        srcp[width - 1 + i] = srcp[width - 1 - i];
    }

    __declspec(align(16)) float ar_kernel[GB_MAX_LENGTH][4];
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < 4; j++) {
            ar_kernel[i][j] = kernel[i];
        }
    }

    for (int x = 0; x < width; x += 4) {
        __m128 sum = _mm_setzero_ps();

        for (int i = -radius; i <= radius; i++) {
            __m128 k = _mm_load_ps(ar_kernel[i + radius]);
            __m128 xmm0 = _mm_loadu_ps(srcp + x + i);
            sum = _mm_add_ps(sum, _mm_mul_ps(xmm0, k));
        }
        _mm_store_ps(dstp + x, sum);
    }
}


void __stdcall TCannyM::
gaussian_blur(const uint8_t* srcp, int src_pitch, int width, int height)
{
    const int length = gb_radius * 2 + 1;

    const uint8_t *p[GB_MAX_LENGTH];
    for (int i = -gb_radius; i <= gb_radius; i++) {
        p[i + gb_radius] = srcp + abs(i) * src_pitch;
    }

    float *dstp = buff + 8;

    __m128i zero = _mm_setzero_si128();
    __m128 zerof = _mm_castsi128_ps(zero);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 16) {
            __m128 sum[4] = {zerof, zerof, zerof, zerof};

            for (int i = 0; i < length; i++) {
                __m128 input[4];
                __m128i xmm0 = _mm_load_si128((__m128i*)(p[i] + x));
                __m128i xmm1 = _mm_unpackhi_epi8(xmm0, zero);
                xmm0 = _mm_unpacklo_epi8(xmm0, zero);
                input[0] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(xmm0, zero));
                input[1] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(xmm0, zero));
                input[2] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(xmm1, zero));
                input[3] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(xmm1, zero));
                __m128 k = _mm_set1_ps(gb_kernel[i]);

                for (int j = 0; j < 4; j++) {
                    sum[j] = _mm_add_ps(sum[j], _mm_mul_ps(k, input[j]));
                }
            }
            _mm_store_ps(dstp + x,      sum[0]);
            _mm_store_ps(dstp + x +  4, sum[1]);
            _mm_store_ps(dstp + x +  8, sum[2]);
            _mm_store_ps(dstp + x + 12, sum[3]);
        }
        horizontal_blur(dstp, gb_radius, length, width, gb_kernel,
                        blur_frame + frame_pitch * y);

        for (int i = 0; i < length - 1; i++) {
            p[i] = p[i + 1];
        }
        p[length - 1] += (y < height - gb_radius - 1 ? 1 : -1) * src_pitch;
    }
}
