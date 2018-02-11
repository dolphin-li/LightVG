/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "lightvg/image/Image.h"

/*
 * This file includes the code, contributed by Simon Perreault
 * (the function icvMedianBlur_8u_O1)
 *
 * Constant-time median filtering -- http://nomis80.org/ctmf.html
 * Copyright (C) 2006 Simon Perreault
 *
 * Contact:
 *  Laboratoire de vision et systemes numeriques
 *  Pavillon Adrien-Pouliot
 *  Universite Laval
 *  Sainte-Foy, Quebec, Canada
 *  G1K 7P4
 *
 *  perreaul@gel.ulaval.ca
 */

namespace lvg
{
#ifdef LVG_ENABLE_SSE
#define CV_SSE2 1
#endif
#ifdef LVG_ENABLE_NEON
#define CV_NEON 1
#endif

	/****************************************************************************************\
											 Box Filter
	\****************************************************************************************/
	template<typename T, typename ST>
	struct RowSum
	{
		int ksize;

		RowSum(int _ksize)
		{
			ksize = _ksize;
		}

		void operator()(const uchar* src, uchar* dst, int width, int cn)
		{
			const T* S = (const T*)src;
			ST* D = (ST*)dst;
			const int kL = ksize / 2 - (ksize % 2==0);
			const int kR = ksize / 2;
			const int kL_cn = kL * cn;
			const int kR_cn = kR * cn;
			const int ksz_cn = ksize * cn;
			int i = 0;

			width = (width - 1)*cn;
			if (ksize == 3)
			{
				for (i = 0; i < cn; i++)
					D[i] = (ST)S[i] + (ST)2 * (ST)S[i + cn];
				for (; i < width - 3; i += 8)
				{
					D[i + 0] = (ST)S[i + 0] + (ST)S[i + 0 + cn] + (ST)S[i + 0 - cn];
					D[i + 1] = (ST)S[i + 1] + (ST)S[i + 1 + cn] + (ST)S[i + 1 - cn];
					D[i + 2] = (ST)S[i + 2] + (ST)S[i + 2 + cn] + (ST)S[i + 2 - cn];
					D[i + 3] = (ST)S[i + 3] + (ST)S[i + 3 + cn] + (ST)S[i + 3 - cn];
					D[i + 4] = (ST)S[i + 4] + (ST)S[i + 4 + cn] + (ST)S[i + 4 - cn];
					D[i + 5] = (ST)S[i + 5] + (ST)S[i + 5 + cn] + (ST)S[i + 5 - cn];
					D[i + 6] = (ST)S[i + 6] + (ST)S[i + 6 + cn] + (ST)S[i + 6 - cn];
					D[i + 7] = (ST)S[i + 7] + (ST)S[i + 7 + cn] + (ST)S[i + 7 - cn];
				}
				for (; i < width - 3; i += 4)
				{
					D[i + 0] = (ST)S[i + 0] + (ST)S[i + 0 + cn] + (ST)S[i + 0 - cn];
					D[i + 1] = (ST)S[i + 1] + (ST)S[i + 1 + cn] + (ST)S[i + 1 - cn];
					D[i + 2] = (ST)S[i + 2] + (ST)S[i + 2 + cn] + (ST)S[i + 2 - cn];
					D[i + 3] = (ST)S[i + 3] + (ST)S[i + 3 + cn] + (ST)S[i + 3 - cn];
				}
				for (; i < width; i ++)
					D[i] = (ST)S[i] + (ST)S[i + cn] + (ST)S[i - cn];
				for (; i < width + cn; i++)
					D[i] = (ST)S[i] + (ST)2 * (ST)S[i - cn];	
			}
			else if (cn == 1)
			{
				ST s = 0;
				for (i = 1; i <= kL_cn; i++)
					s += (ST)S[i];
				for (i = 0; i <= kR_cn; i++)
					s += (ST)S[i];
				D[0] = s;
				for (i = 0; i < kL_cn; i++)
				{
					s += (ST)S[i + kR_cn + 1] - (ST)S[kL_cn + i];
					D[i + 1] = s;
				}
				for (; i < width - kR_cn - 3; i+= 4)
				{
					s += (ST)S[i + kR_cn + 1] - (ST)S[i - kL_cn];
					D[i + 1] = s;
					s += (ST)S[i + kR_cn + 2] - (ST)S[i - kL_cn + 1];
					D[i + 2] = s;
					s += (ST)S[i + kR_cn + 3] - (ST)S[i - kL_cn + 2];
					D[i + 3] = s;
					s += (ST)S[i + kR_cn + 4] - (ST)S[i - kL_cn + 3];
					D[i + 4] = s;
				}
				for (; i < width - kR_cn; i++)
				{
					s += (ST)S[i + kR_cn + 1] - (ST)S[i - kL_cn];
					D[i + 1] = s;
				}
				for (; i < width; i++)
				{
					s += (ST)S[i - kR_cn + 1] - (ST)S[i - kL_cn];
					D[i + 1] = s;
				}
			}
			else if (cn == 3)
			{
				ST s0 = 0, s1 = 0, s2 = 0;
				for (i = 3; i <= kL_cn; i += 3)
				{
					s0 += (ST)S[i];
					s1 += (ST)S[i + 1];
					s2 += (ST)S[i + 2];
				}
				for (i = 0; i <= kR_cn; i += 3)
				{
					s0 += (ST)S[i];
					s1 += (ST)S[i + 1];
					s2 += (ST)S[i + 2];
				}
				D[0] = s0;
				D[1] = s1;
				D[2] = s2;
				for (i = 0; i < kL_cn; i += 3)
				{
					s0 += (ST)S[i + kR_cn + 3] - (ST)S[kL_cn + i];
					s1 += (ST)S[i + kR_cn + 4] - (ST)S[kL_cn + i + 1];
					s2 += (ST)S[i + kR_cn + 5] - (ST)S[kL_cn + i + 2];
					D[i + 3] = s0;
					D[i + 4] = s1;
					D[i + 5] = s2;
				}
				for (; i < width - kR_cn - 9; i += 12)
				{
					s0 += (ST)S[i + kR_cn + 3] - (ST)S[i - kL_cn];
					s1 += (ST)S[i + kR_cn + 4] - (ST)S[i - kL_cn + 1];
					s2 += (ST)S[i + kR_cn + 5] - (ST)S[i - kL_cn + 2];
					D[i + 3] = s0;
					D[i + 4] = s1;
					D[i + 5] = s2;
					s0 += (ST)S[i + kR_cn + 6] - (ST)S[i - kL_cn + 3];
					s1 += (ST)S[i + kR_cn + 7] - (ST)S[i - kL_cn + 4];
					s2 += (ST)S[i + kR_cn + 8] - (ST)S[i - kL_cn + 5];
					D[i + 6] = s0;
					D[i + 7] = s1;
					D[i + 8] = s2;
					s0 += (ST)S[i + kR_cn + 9] - (ST)S[i - kL_cn + 6];
					s1 += (ST)S[i + kR_cn + 10] - (ST)S[i - kL_cn + 7];
					s2 += (ST)S[i + kR_cn + 11] - (ST)S[i - kL_cn + 8];
					D[i + 9] = s0;
					D[i + 10] = s1;
					D[i + 11] = s2;
					s0 += (ST)S[i + kR_cn + 12] - (ST)S[i - kL_cn + 9];
					s1 += (ST)S[i + kR_cn + 13] - (ST)S[i - kL_cn + 10];
					s2 += (ST)S[i + kR_cn + 14] - (ST)S[i - kL_cn + 11];
					D[i + 12] = s0;
					D[i + 13] = s1;
					D[i + 14] = s2;
				}
				for (; i < width - kR_cn; i += 3)
				{
					s0 += (ST)S[i + kR_cn + 3] - (ST)S[i - kL_cn];
					s1 += (ST)S[i + kR_cn + 4] - (ST)S[i - kL_cn + 1];
					s2 += (ST)S[i + kR_cn + 5] - (ST)S[i - kL_cn + 2];
					D[i + 3] = s0;
					D[i + 4] = s1;
					D[i + 5] = s2;
				}
				for (; i < width; i += 3)
				{
					s0 += (ST)S[i - kR_cn + 3] - (ST)S[i - kL_cn];
					s1 += (ST)S[i - kR_cn + 4] - (ST)S[i - kL_cn + 1];
					s2 += (ST)S[i - kR_cn + 5] - (ST)S[i - kL_cn + 2];
					D[i + 3] = s0;
					D[i + 4] = s1;
					D[i + 5] = s2;
				}
			}
			else if (cn == 4)
			{
				ST s0 = 0, s1 = 0, s2 = 0, s3 = 0;
				for (i = 4; i <= kL_cn; i += 4)
				{
					s0 += (ST)S[i];
					s1 += (ST)S[i + 1];
					s2 += (ST)S[i + 2];
					s3 += (ST)S[i + 3];
				}
				for (i = 0; i <= kR_cn; i += 4)
				{
					s0 += (ST)S[i];
					s1 += (ST)S[i + 1];
					s2 += (ST)S[i + 2];
					s3 += (ST)S[i + 3];
				}
				D[0] = s0;
				D[1] = s1;
				D[2] = s2;
				D[3] = s3;
				for (i = 0; i < kL_cn; i += 4)
				{
					s0 += (ST)S[i + kR_cn + 4] - (ST)S[kL_cn + i];
					s1 += (ST)S[i + kR_cn + 5] - (ST)S[kL_cn + i + 1];
					s2 += (ST)S[i + kR_cn + 6] - (ST)S[kL_cn + i + 2];
					s3 += (ST)S[i + kR_cn + 7] - (ST)S[kL_cn + i + 3];
					D[i + 4] = s0;
					D[i + 5] = s1;
					D[i + 6] = s2;
					D[i + 7] = s3;
				}
				for (; i < width - kR_cn - 12; i += 16)
				{
					s0 += (ST)S[i + kR_cn + 4] - (ST)S[i - kL_cn];
					s1 += (ST)S[i + kR_cn + 5] - (ST)S[i - kL_cn + 1];
					s2 += (ST)S[i + kR_cn + 6] - (ST)S[i - kL_cn + 2];
					s3 += (ST)S[i + kR_cn + 7] - (ST)S[i - kL_cn + 3];
					D[i + 4] = s0;
					D[i + 5] = s1;
					D[i + 6] = s2;
					D[i + 7] = s3;
					s0 += (ST)S[i + kR_cn + 8] - (ST)S[i - kL_cn + 4];
					s1 += (ST)S[i + kR_cn + 9] - (ST)S[i - kL_cn + 5];
					s2 += (ST)S[i + kR_cn + 10] - (ST)S[i - kL_cn + 6];
					s3 += (ST)S[i + kR_cn + 11] - (ST)S[i - kL_cn + 7];
					D[i + 8] = s0;
					D[i + 9] = s1;
					D[i + 10] = s2;
					D[i + 11] = s3;
					s0 += (ST)S[i + kR_cn + 12] - (ST)S[i - kL_cn + 8];
					s1 += (ST)S[i + kR_cn + 13] - (ST)S[i - kL_cn + 9];
					s2 += (ST)S[i + kR_cn + 14] - (ST)S[i - kL_cn + 10];
					s3 += (ST)S[i + kR_cn + 15] - (ST)S[i - kL_cn + 11];
					D[i + 12] = s0;
					D[i + 13] = s1;
					D[i + 14] = s2;
					D[i + 15] = s3;
					s0 += (ST)S[i + kR_cn + 16] - (ST)S[i - kL_cn + 12];
					s1 += (ST)S[i + kR_cn + 17] - (ST)S[i - kL_cn + 13];
					s2 += (ST)S[i + kR_cn + 18] - (ST)S[i - kL_cn + 14];
					s3 += (ST)S[i + kR_cn + 19] - (ST)S[i - kL_cn + 15];
					D[i + 16] = s0;
					D[i + 17] = s1;
					D[i + 18] = s2;
					D[i + 19] = s3;
				}
				for (; i < width - kR_cn; i += 4)
				{
					s0 += (ST)S[i + kR_cn + 4] - (ST)S[i - kL_cn];
					s1 += (ST)S[i + kR_cn + 5] - (ST)S[i - kL_cn + 1];
					s2 += (ST)S[i + kR_cn + 6] - (ST)S[i - kL_cn + 2];
					s3 += (ST)S[i + kR_cn + 7] - (ST)S[i - kL_cn + 3];
					D[i + 4] = s0;
					D[i + 5] = s1;
					D[i + 6] = s2;
					D[i + 7] = s3;
				}
				for (; i < width; i += 4)
				{
					s0 += (ST)S[i - kR_cn + 4] - (ST)S[i - kL_cn];
					s1 += (ST)S[i - kR_cn + 5] - (ST)S[i - kL_cn + 1];
					s2 += (ST)S[i - kR_cn + 6] - (ST)S[i - kL_cn + 2];
					s3 += (ST)S[i - kR_cn + 7] - (ST)S[i - kL_cn + 3];
					D[i + 4] = s0;
					D[i + 5] = s1;
					D[i + 6] = s2;
					D[i + 7] = s3;
				}
			}
			else
				LVG_LOGE("non-supported channel num: %d", cn);
		}
	};

	template<typename ST, typename T>
	struct ColumnSum
	{
		int ksize;
		double scale;
		int sumCount;
		std::vector<ST> sum;

		ColumnSum(int _ksize, double _scale)
		{
			ksize = _ksize;
			scale = _scale;
			sumCount = 0;
		}

		void reset() { sumCount = 0; }

		void operator()(const uchar* src, int srcstep, uchar* dst, int dststep, int count, int width)
		{
			int i;
			ST* SUM;
			bool haveScale = scale != 1;
			double _scale = scale;

			if (width != (int)sum.size())
			{
				sum.resize(width);
				sumCount = 0;
			}

			SUM = &sum[0];
			if (sumCount == 0)
			{
				memset((void*)SUM, 0, width * sizeof(ST));

				for (; sumCount < ksize - 1; sumCount++, src+=srcstep)
				{
					const ST* Sp = (const ST*)src;

					for (i = 0; i < width; i++)
						SUM[i] += Sp[i];
				}
			}
			else
			{
				assert(sumCount == ksize - 1);
				src += (ksize - 1)*srcstep;
			}

			for (; count--; src+=srcstep)
			{
				const ST* Sp = (const ST*)src;
				const ST* Sm = (const ST*)(src + (1 - ksize)*srcstep);
				T* D = (T*)dst;
				if (haveScale)
				{
					for (i = 0; i <= width - 2; i += 2)
					{
						ST s0 = SUM[i] + Sp[i], s1 = SUM[i + 1] + Sp[i + 1];
						D[i] = T(s0*_scale);
						D[i + 1] = T(s1*_scale);
						s0 -= Sm[i]; s1 -= Sm[i + 1];
						SUM[i] = s0; SUM[i + 1] = s1;
					}

					for (; i < width; i++)
					{
						ST s0 = SUM[i] + Sp[i];
						D[i] = T(s0*_scale);
						SUM[i] = s0 - Sm[i];
					}
				}
				else
				{
					for (i = 0; i <= width - 2; i += 2)
					{
						ST s0 = SUM[i] + Sp[i], s1 = SUM[i + 1] + Sp[i + 1];
						D[i] = T(s0);
						D[i + 1] = T(s1);
						s0 -= Sm[i]; s1 -= Sm[i + 1];
						SUM[i] = s0; SUM[i + 1] = s1;
					}

					for (; i < width; i++)
					{
						ST s0 = SUM[i] + Sp[i];
						D[i] = T(s0);
						SUM[i] = s0 - Sm[i];
					}
				}
				dst += dststep;
			}
		}
	};

	template<>
	struct ColumnSum<int, uchar>
	{
		int ksize;
		double scale;
		int sumCount;
		std::vector<int> sum;

		ColumnSum(int _ksize, double _scale)
		{
			ksize = _ksize;
			scale = _scale;
			sumCount = 0;
		}

		void reset() { sumCount = 0; }

		void operator()(const uchar* src, int srcstep, uchar* dst, int dststep, int count, int width)
		{
			int* SUM;
			bool haveScale = scale != 1;
			double _scale = scale;

			if (width != (int)sum.size())
			{
				sum.resize(width);
				sumCount = 0;
			}

			SUM = &sum[0];
			if (sumCount == 0)
			{
				memset((void*)SUM, 0, width * sizeof(int));
				for (; sumCount < ksize - 1; sumCount++, src+=srcstep)
				{
					const int* Sp = (const int*)src;
					int i = 0;
#if CV_SSE2
					for (; i <= width - 4; i += 4)
					{
						__m128i _sum = _mm_loadu_si128((const __m128i*)(SUM + i));
						__m128i _sp = _mm_loadu_si128((const __m128i*)(Sp + i));
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_add_epi32(_sum, _sp));
					}
#elif CV_NEON
					for (; i <= width - 4; i += 4)
						vst1q_s32(SUM + i, vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i)));
#endif
					for (; i < width; i++)
						SUM[i] += Sp[i];
				}
			}
			else
			{
				assert(sumCount == ksize - 1);
				src += (ksize - 1)*srcstep;
			}

			for (; count--; src+=srcstep)
			{
				const int* Sp = (const int*)src;
				const int* Sm = (const int*)(src + (1 - ksize)*srcstep);
				uchar* D = (uchar*)dst;
				if (haveScale)
				{
					int i = 0;
#if CV_SSE2
					const __m128 scale4 = _mm_set1_ps((float)_scale);
					for (; i <= width - 8; i += 8)
					{
						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _sm1 = _mm_loadu_si128((const __m128i*)(Sm + i + 4));

						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));
						__m128i _s01 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i + 4)),
							_mm_loadu_si128((const __m128i*)(Sp + i + 4)));

						__m128i _s0T = _mm_cvtps_epi32(_mm_mul_ps(scale4, _mm_cvtepi32_ps(_s0)));
						__m128i _s0T1 = _mm_cvtps_epi32(_mm_mul_ps(scale4, _mm_cvtepi32_ps(_s01)));

						_s0T = _mm_packs_epi32(_s0T, _s0T1);

						_mm_storel_epi64((__m128i*)(D + i), _mm_packus_epi16(_s0T, _s0T));

						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
						_mm_storeu_si128((__m128i*)(SUM + i + 4), _mm_sub_epi32(_s01, _sm1));
					}
#elif CV_NEON
					float32x4_t v_scale = vdupq_n_f32((float)_scale);
					for (; i <= width - 8; i += 8)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));
						int32x4_t v_s01 = vaddq_s32(vld1q_s32(SUM + i + 4), vld1q_s32(Sp + i + 4));

						uint32x4_t v_s0d = cv_vrndq_u32_f32(vmulq_f32(vcvtq_f32_s32(v_s0), v_scale));
						uint32x4_t v_s01d = cv_vrndq_u32_f32(vmulq_f32(vcvtq_f32_s32(v_s01), v_scale));

						uint16x8_t v_dst = vcombine_u16(vqmovn_u32(v_s0d), vqmovn_u32(v_s01d));
						vst1_u8(D + i, vqmovn_u16(v_dst));

						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
						vst1q_s32(SUM + i + 4, vsubq_s32(v_s01, vld1q_s32(Sm + i + 4)));
					}
#endif
					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = uchar(s0*_scale);
						SUM[i] = s0 - Sm[i];
					}
				}
				else
				{
					int i = 0;
#if CV_SSE2
					for (; i <= width - 8; i += 8)
					{
						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _sm1 = _mm_loadu_si128((const __m128i*)(Sm + i + 4));

						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));
						__m128i _s01 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i + 4)),
							_mm_loadu_si128((const __m128i*)(Sp + i + 4)));

						__m128i _s0T = _mm_packs_epi32(_s0, _s01);

						_mm_storel_epi64((__m128i*)(D + i), _mm_packus_epi16(_s0T, _s0T));

						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
						_mm_storeu_si128((__m128i*)(SUM + i + 4), _mm_sub_epi32(_s01, _sm1));
					}
#elif CV_NEON
					for (; i <= width - 8; i += 8)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));
						int32x4_t v_s01 = vaddq_s32(vld1q_s32(SUM + i + 4), vld1q_s32(Sp + i + 4));

						uint16x8_t v_dst = vcombine_u16(vqmovun_s32(v_s0), vqmovun_s32(v_s01));
						vst1_u8(D + i, vqmovn_u16(v_dst));

						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
						vst1q_s32(SUM + i + 4, vsubq_s32(v_s01, vld1q_s32(Sm + i + 4)));
					}
#endif

					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = uchar(s0);
						SUM[i] = s0 - Sm[i];
					}
				}
				dst += dststep;
			}
		}
	};

	template<>
	struct ColumnSum<ushort, uchar>
	{
		int ksize;
		double scale;
		int sumCount;
		int divDelta;
		int divScale;
		std::vector<ushort> sum;
		ColumnSum(int _ksize, double _scale)
		{
			ksize = _ksize;
			scale = _scale;
			sumCount = 0;
			divDelta = 0;
			divScale = 1;
			if (scale != 1)
			{
				int d = int(1.f / scale + 0.5f);
				double scalef = ((double)(1 << 16)) / d;
				divScale = int(scalef);
				scalef -= divScale;
				divDelta = d / 2;
				if (scalef < 0.5)
					divDelta++;
				else
					divScale++;
			}
		}

		void reset() { sumCount = 0; }

		void operator()(const uchar* src, int srcstep, uchar* dst, int dststep, int count, int width)
		{
			const int ds = divScale;
			const int dd = divDelta;
			ushort* SUM;
			const bool haveScale = scale != 1;

			if (width != (int)sum.size())
			{
				sum.resize(width);
				sumCount = 0;
			}

			SUM = &sum[0];
			if (sumCount == 0)
			{
				memset((void*)SUM, 0, width * sizeof(SUM[0]));
				for (; sumCount < ksize - 1; sumCount++, src+=srcstep)
				{
					const ushort* Sp = (const ushort*)src;
					int i = 0;
#if CV_SSE2
					for (; i <= width - 8; i += 8)
					{
						__m128i _sum = _mm_loadu_si128((const __m128i*)(SUM + i));
						__m128i _sp = _mm_loadu_si128((const __m128i*)(Sp + i));
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_add_epi16(_sum, _sp));
					}
#elif CV_NEON
					for (; i <= width - 8; i += 8)
						vst1q_u16(SUM + i, vaddq_u16(vld1q_u16(SUM + i), vld1q_u16(Sp + i)));
#endif
					for (; i < width; i++)
						SUM[i] += Sp[i];
				}
			}
			else
			{
				assert(sumCount == ksize - 1);
				src += (ksize - 1)*srcstep;
			}

			for (; count--; src+=srcstep)
			{
				const ushort* Sp = (const ushort*)src;
				const ushort* Sm = (const ushort*)(src+(1 - ksize)*srcstep);
				uchar* D = (uchar*)dst;
				if (haveScale)
				{
					int i = 0;
#if CV_SSE2
					__m128i ds8 = _mm_set1_epi16((short)ds);
					__m128i dd8 = _mm_set1_epi16((short)dd);

					for (; i <= width - 16; i += 16)
					{
						__m128i _sm0 = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _sm1 = _mm_loadu_si128((const __m128i*)(Sm + i + 8));

						__m128i _s0 = _mm_add_epi16(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));
						__m128i _s1 = _mm_add_epi16(_mm_loadu_si128((const __m128i*)(SUM + i + 8)),
							_mm_loadu_si128((const __m128i*)(Sp + i + 8)));
						__m128i _s2 = _mm_mulhi_epu16(_mm_adds_epu16(_s0, dd8), ds8);
						__m128i _s3 = _mm_mulhi_epu16(_mm_adds_epu16(_s1, dd8), ds8);
						_s0 = _mm_sub_epi16(_s0, _sm0);
						_s1 = _mm_sub_epi16(_s1, _sm1);
						_mm_storeu_si128((__m128i*)(D + i), _mm_packus_epi16(_s2, _s3));
						_mm_storeu_si128((__m128i*)(SUM + i), _s0);
						_mm_storeu_si128((__m128i*)(SUM + i + 8), _s1);
					}
#endif
					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = (uchar)((s0 + dd)*ds >> 16);
						SUM[i] = (ushort)(s0 - Sm[i]);
					}
				}
				else
				{
					int i = 0;
					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = uchar(s0);
						SUM[i] = (ushort)(s0 - Sm[i]);
					}
				}
				dst += dststep;
			}
		}
	};

	template<>
	struct ColumnSum<int, short>
	{
		int ksize;
		double scale;
		int sumCount;
		std::vector<int> sum;
		ColumnSum(int _ksize, double _scale)
		{
			ksize = _ksize;
			scale = _scale;
			sumCount = 0;
		}

		void reset() { sumCount = 0; }

		void operator()(const uchar* src, int srcstep, uchar* dst, int dststep, int count, int width)
		{
			int i;
			int* SUM;
			bool haveScale = scale != 1;
			double _scale = scale;

			if (width != (int)sum.size())
			{
				sum.resize(width);
				sumCount = 0;
			}

			SUM = &sum[0];
			if (sumCount == 0)
			{
				memset((void*)SUM, 0, width * sizeof(int));
				for (; sumCount < ksize - 1; sumCount++, src+=srcstep)
				{
					const int* Sp = (const int*)src;
					i = 0;
#if CV_SSE2
					for (; i <= width - 4; i += 4)
					{
						__m128i _sum = _mm_loadu_si128((const __m128i*)(SUM + i));
						__m128i _sp = _mm_loadu_si128((const __m128i*)(Sp + i));
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_add_epi32(_sum, _sp));
					}
#elif CV_NEON
					for (; i <= width - 4; i += 4)
						vst1q_s32(SUM + i, vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i)));
#endif
					for (; i < width; i++)
						SUM[i] += Sp[i];
				}
			}
			else
			{
				assert(sumCount == ksize - 1);
				src += (ksize - 1)*srcstep;
			}

			for (; count--; src+=srcstep)
			{
				const int* Sp = (const int*)src;
				const int* Sm = (const int*)(src+(1 - ksize)*srcstep);
				short* D = (short*)dst;
				if (haveScale)
				{
					i = 0;
#if CV_SSE2
					const __m128 scale4 = _mm_set1_ps((float)_scale);
					for (; i <= width - 8; i += 8)
					{
						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _sm1 = _mm_loadu_si128((const __m128i*)(Sm + i + 4));

						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));
						__m128i _s01 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i + 4)),
							_mm_loadu_si128((const __m128i*)(Sp + i + 4)));

						__m128i _s0T = _mm_cvtps_epi32(_mm_mul_ps(scale4, _mm_cvtepi32_ps(_s0)));
						__m128i _s0T1 = _mm_cvtps_epi32(_mm_mul_ps(scale4, _mm_cvtepi32_ps(_s01)));

						_mm_storeu_si128((__m128i*)(D + i), _mm_packs_epi32(_s0T, _s0T1));

						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
						_mm_storeu_si128((__m128i*)(SUM + i + 4), _mm_sub_epi32(_s01, _sm1));
					}
#elif CV_NEON
					float32x4_t v_scale = vdupq_n_f32((float)_scale);
					for (; i <= width - 8; i += 8)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));
						int32x4_t v_s01 = vaddq_s32(vld1q_s32(SUM + i + 4), vld1q_s32(Sp + i + 4));

						int32x4_t v_s0d = cv_vrndq_s32_f32(vmulq_f32(vcvtq_f32_s32(v_s0), v_scale));
						int32x4_t v_s01d = cv_vrndq_s32_f32(vmulq_f32(vcvtq_f32_s32(v_s01), v_scale));
						vst1q_s16(D + i, vcombine_s16(vqmovn_s32(v_s0d), vqmovn_s32(v_s01d)));

						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
						vst1q_s32(SUM + i + 4, vsubq_s32(v_s01, vld1q_s32(Sm + i + 4)));
					}
#endif
					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = short(s0*_scale);
						SUM[i] = s0 - Sm[i];
					}
				}
				else
				{
					i = 0;
#if CV_SSE2
					for (; i <= width - 8; i += 8)
					{

						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _sm1 = _mm_loadu_si128((const __m128i*)(Sm + i + 4));

						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));
						__m128i _s01 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i + 4)),
							_mm_loadu_si128((const __m128i*)(Sp + i + 4)));

						_mm_storeu_si128((__m128i*)(D + i), _mm_packs_epi32(_s0, _s01));

						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
						_mm_storeu_si128((__m128i*)(SUM + i + 4), _mm_sub_epi32(_s01, _sm1));
					}
#elif CV_NEON
					for (; i <= width - 8; i += 8)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));
						int32x4_t v_s01 = vaddq_s32(vld1q_s32(SUM + i + 4), vld1q_s32(Sp + i + 4));

						vst1q_s16(D + i, vcombine_s16(vqmovn_s32(v_s0), vqmovn_s32(v_s01)));

						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
						vst1q_s32(SUM + i + 4, vsubq_s32(v_s01, vld1q_s32(Sm + i + 4)));
					}
#endif

					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = short(s0);
						SUM[i] = s0 - Sm[i];
					}
				}
				dst += dststep;
			}
		}
	};

	template<>
	struct ColumnSum<int, ushort>
	{
		int ksize;
		double scale;
		int sumCount;
		std::vector<int> sum;
		ColumnSum(int _ksize, double _scale)
		{
			ksize = _ksize;
			scale = _scale;
			sumCount = 0;
		}

		void reset() { sumCount = 0; }

		void operator()(const uchar* src, int srcstep, uchar* dst, int dststep, int count, int width)
		{
			int* SUM;
			bool haveScale = scale != 1;
			double _scale = scale;

			if (width != (int)sum.size())
			{
				sum.resize(width);
				sumCount = 0;
			}

			SUM = &sum[0];
			if (sumCount == 0)
			{
				memset((void*)SUM, 0, width * sizeof(int));
				for (; sumCount < ksize - 1; sumCount++, src+=srcstep)
				{
					const int* Sp = (const int*)src;
					int i = 0;
#if CV_SSE2
					for (; i <= width - 4; i += 4)
					{
						__m128i _sum = _mm_loadu_si128((const __m128i*)(SUM + i));
						__m128i _sp = _mm_loadu_si128((const __m128i*)(Sp + i));
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_add_epi32(_sum, _sp));
					}
#elif CV_NEON
					for (; i <= width - 4; i += 4)
						vst1q_s32(SUM + i, vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i)));
#endif
					for (; i < width; i++)
						SUM[i] += Sp[i];
				}
			}
			else
			{
				assert(sumCount == ksize - 1);
				src += (ksize - 1)*srcstep;
			}

			for (; count--; src+=srcstep)
			{
				const int* Sp = (const int*)src;
				const int* Sm = (const int*)(src+(1 - ksize)*srcstep);
				ushort* D = (ushort*)dst;
				if (haveScale)
				{
					int i = 0;
#if CV_SSE2
					const __m128 scale4 = _mm_set1_ps((float)_scale);
					const __m128i delta0 = _mm_set1_epi32(0x8000);
					const __m128i delta1 = _mm_set1_epi32(0x80008000);

					for (; i < width - 4; i += 4)
					{
						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));

						__m128i _res = _mm_cvtps_epi32(_mm_mul_ps(scale4, _mm_cvtepi32_ps(_s0)));

						_res = _mm_sub_epi32(_res, delta0);
						_res = _mm_add_epi16(_mm_packs_epi32(_res, _res), delta1);

						_mm_storel_epi64((__m128i*)(D + i), _res);
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
					}
#elif CV_NEON
					float32x4_t v_scale = vdupq_n_f32((float)_scale);
					for (; i <= width - 8; i += 8)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));
						int32x4_t v_s01 = vaddq_s32(vld1q_s32(SUM + i + 4), vld1q_s32(Sp + i + 4));

						uint32x4_t v_s0d = cv_vrndq_u32_f32(vmulq_f32(vcvtq_f32_s32(v_s0), v_scale));
						uint32x4_t v_s01d = cv_vrndq_u32_f32(vmulq_f32(vcvtq_f32_s32(v_s01), v_scale));
						vst1q_u16(D + i, vcombine_u16(vqmovn_u32(v_s0d), vqmovn_u32(v_s01d)));

						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
						vst1q_s32(SUM + i + 4, vsubq_s32(v_s01, vld1q_s32(Sm + i + 4)));
					}
#endif
					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = ushort(s0*_scale);
						SUM[i] = s0 - Sm[i];
					}
				}
				else
				{
					int i = 0;
#if CV_SSE2
					const __m128i delta0 = _mm_set1_epi32(0x8000);
					const __m128i delta1 = _mm_set1_epi32(0x80008000);

					for (; i < width - 4; i += 4)
					{
						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));

						__m128i _res = _mm_sub_epi32(_s0, delta0);
						_res = _mm_add_epi16(_mm_packs_epi32(_res, _res), delta1);

						_mm_storel_epi64((__m128i*)(D + i), _res);
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
					}
#elif CV_NEON
					for (; i <= width - 8; i += 8)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));
						int32x4_t v_s01 = vaddq_s32(vld1q_s32(SUM + i + 4), vld1q_s32(Sp + i + 4));

						vst1q_u16(D + i, vcombine_u16(vqmovun_s32(v_s0), vqmovun_s32(v_s01)));

						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
						vst1q_s32(SUM + i + 4, vsubq_s32(v_s01, vld1q_s32(Sm + i + 4)));
					}
#endif

					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = ushort(s0);
						SUM[i] = s0 - Sm[i];
					}
				}
				dst += dststep;
			}
		}
	};

	template<>
	struct ColumnSum<int, int>
	{
		int ksize;
		double scale;
		int sumCount;
		std::vector<int> sum;
		ColumnSum(int _ksize, double _scale)
		{
			ksize = _ksize;
			scale = _scale;
			sumCount = 0;
		}

		void reset() { sumCount = 0; }

		void operator()(const uchar* src, int srcstep, uchar* dst, int dststep, int count, int width)
		{
			int* SUM;
			bool haveScale = scale != 1;
			double _scale = scale;

			if (width != (int)sum.size())
			{
				sum.resize(width);
				sumCount = 0;
			}

			SUM = &sum[0];
			if (sumCount == 0)
			{
				memset((void*)SUM, 0, width * sizeof(int));
				for (; sumCount < ksize - 1; sumCount++, src+=srcstep)
				{
					const int* Sp = (const int*)src;
					int i = 0;
#if CV_SSE2
					for (; i <= width - 4; i += 4)
					{
						__m128i _sum = _mm_loadu_si128((const __m128i*)(SUM + i));
						__m128i _sp = _mm_loadu_si128((const __m128i*)(Sp + i));
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_add_epi32(_sum, _sp));
					}
#elif CV_NEON
					for (; i <= width - 4; i += 4)
						vst1q_s32(SUM + i, vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i)));
#endif
					for (; i < width; i++)
						SUM[i] += Sp[i];
				}
			}
			else
			{
				assert(sumCount == ksize - 1);
				src += (ksize - 1)*srcstep;
			}

			for (; count--; src+=srcstep)
			{
				const int* Sp = (const int*)src;
				const int* Sm = (const int*)(src+(1 - ksize)*srcstep);
				int* D = (int*)dst;
				if (haveScale)
				{
					int i = 0;
#if CV_SSE2
					const __m128 scale4 = _mm_set1_ps((float)_scale);
					for (; i <= width - 4; i += 4)
					{
						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));

						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));

						__m128i _s0T = _mm_cvtps_epi32(_mm_mul_ps(scale4, _mm_cvtepi32_ps(_s0)));

						_mm_storeu_si128((__m128i*)(D + i), _s0T);
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
					}
#elif CV_NEON
					float32x4_t v_scale = vdupq_n_f32((float)_scale);
					for (; i <= width - 4; i += 4)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));

						int32x4_t v_s0d = cv_vrndq_s32_f32(vmulq_f32(vcvtq_f32_s32(v_s0), v_scale));
						vst1q_s32(D + i, v_s0d);

						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
					}
#endif
					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = int(s0*_scale);
						SUM[i] = s0 - Sm[i];
					}
				}
				else
				{
					int i = 0;
#if CV_SSE2
					for (; i <= width - 4; i += 4)
					{
						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));

						_mm_storeu_si128((__m128i*)(D + i), _s0);
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
					}
#elif CV_NEON
					for (; i <= width - 4; i += 4)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));

						vst1q_s32(D + i, v_s0);
						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
					}
#endif

					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = s0;
						SUM[i] = s0 - Sm[i];
					}
				}
				dst += dststep;
			}
		}
	};


	template<>
	struct ColumnSum<int, float>
	{
		int ksize;
		double scale;
		int sumCount;
		std::vector<int> sum;
		ColumnSum(int _ksize, double _scale)
		{
			ksize = _ksize;
			scale = _scale;
			sumCount = 0;
		}

		void reset() { sumCount = 0; }

		void operator()(const uchar* src, int srcstep, uchar* dst, int dststep, int count, int width)
		{
			int* SUM;
			bool haveScale = scale != 1;
			double _scale = scale;

			if (width != (int)sum.size())
			{
				sum.resize(width);
				sumCount = 0;
			}

			SUM = &sum[0];
			if (sumCount == 0)
			{
				memset((void*)SUM, 0, width * sizeof(int));
				for (; sumCount < ksize - 1; sumCount++, src+=srcstep)
				{
					const int* Sp = (const int*)src;
					int i = 0;
#if CV_SSE2
					for (; i <= width - 4; i += 4)
					{
						__m128i _sum = _mm_loadu_si128((const __m128i*)(SUM + i));
						__m128i _sp = _mm_loadu_si128((const __m128i*)(Sp + i));
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_add_epi32(_sum, _sp));
					}
#elif CV_NEON
					for (; i <= width - 4; i += 4)
						vst1q_s32(SUM + i, vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i)));
#endif

					for (; i < width; i++)
						SUM[i] += Sp[i];
				}
			}
			else
			{
				assert(sumCount == ksize - 1);
				src += ksize - 1;
			}

			for (; count--; src+=srcstep)
			{
				const int * Sp = (const int*)src;
				const int * Sm = (const int*)(src+(1 - ksize)*srcstep);
				float* D = (float*)dst;
				if (haveScale)
				{
					int i = 0;

#if CV_SSE2
					const __m128 scale4 = _mm_set1_ps((float)_scale);

					for (; i < width - 4; i += 4)
					{
						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));

						_mm_storeu_ps(D + i, _mm_mul_ps(scale4, _mm_cvtepi32_ps(_s0)));
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
					}
#elif CV_NEON
					float32x4_t v_scale = vdupq_n_f32((float)_scale);
					for (; i <= width - 8; i += 8)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));
						int32x4_t v_s01 = vaddq_s32(vld1q_s32(SUM + i + 4), vld1q_s32(Sp + i + 4));

						vst1q_f32(D + i, vmulq_f32(vcvtq_f32_s32(v_s0), v_scale));
						vst1q_f32(D + i + 4, vmulq_f32(vcvtq_f32_s32(v_s01), v_scale));

						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
						vst1q_s32(SUM + i + 4, vsubq_s32(v_s01, vld1q_s32(Sm + i + 4)));
					}
#endif

					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = (float)(s0*_scale);
						SUM[i] = s0 - Sm[i];
					}
				}
				else
				{
					int i = 0;

#if CV_SSE2
					for (; i < width - 4; i += 4)
					{
						__m128i _sm = _mm_loadu_si128((const __m128i*)(Sm + i));
						__m128i _s0 = _mm_add_epi32(_mm_loadu_si128((const __m128i*)(SUM + i)),
							_mm_loadu_si128((const __m128i*)(Sp + i)));

						_mm_storeu_ps(D + i, _mm_cvtepi32_ps(_s0));
						_mm_storeu_si128((__m128i*)(SUM + i), _mm_sub_epi32(_s0, _sm));
					}
#elif CV_NEON
					for (; i <= width - 8; i += 8)
					{
						int32x4_t v_s0 = vaddq_s32(vld1q_s32(SUM + i), vld1q_s32(Sp + i));
						int32x4_t v_s01 = vaddq_s32(vld1q_s32(SUM + i + 4), vld1q_s32(Sp + i + 4));

						vst1q_f32(D + i, vcvtq_f32_s32(v_s0));
						vst1q_f32(D + i + 4, vcvtq_f32_s32(v_s01));

						vst1q_s32(SUM + i, vsubq_s32(v_s0, vld1q_s32(Sm + i)));
						vst1q_s32(SUM + i + 4, vsubq_s32(v_s01, vld1q_s32(Sm + i + 4)));
					}
#endif

					for (; i < width; i++)
					{
						int s0 = SUM[i] + Sp[i];
						D[i] = (float)(s0);
						SUM[i] = s0 - Sm[i];
					}
				}
				dst += dststep;
			}
		}
	};

	template<>
	struct ColumnSum<float, float>
	{
		int ksize;
		double scale;
		int sumCount;
		std::vector<float> sum;
		ColumnSum(int _ksize, double _scale)
		{
			ksize = _ksize;
			scale = _scale;
			sumCount = 0;
		}

		void reset() { sumCount = 0; }

		void operator()(const uchar* src, int srcstep, uchar* dst, int dststep, int count, int width)
		{
			float* SUM;
			bool haveScale = scale != 1;
			double _scale = scale;

			if (width != (int)sum.size())
			{
				sum.resize(width);
				sumCount = 0;
			}

			SUM = &sum[0];
			if (sumCount == 0)
			{
				memset((void*)SUM, 0, width * sizeof(float));
				for (; sumCount < ksize - 1; sumCount++, src += srcstep)
				{
					const float* Sp = (const float*)src;
					int i = 0;
#if CV_SSE2
					for (; i <= width - 4; i += 4)
					{
						__m128 _sum = _mm_loadu_ps(SUM + i);
						__m128 _sp = _mm_loadu_ps(Sp + i);
						_mm_storeu_ps(SUM + i, _mm_add_ps(_sum, _sp));
					}
#elif CV_NEON
					for (; i <= width - 4; i += 4)
						vst1q_f32(SUM + i, vaddq_f32(vld1q_f32(SUM + i), vld1q_f32(Sp + i)));
#endif

					for (; i < width; i++)
						SUM[i] += Sp[i];
				}
			}
			else
			{
				assert(sumCount == ksize - 1);
				src += ksize - 1;
			}

			for (; count--; src += srcstep)
			{
				const float * Sp = (const float*)src;
				const float * Sm = (const float*)(src + (1 - ksize)*srcstep);
				float* D = (float*)dst;
				if (haveScale)
				{
					int i = 0;

#if CV_SSE2
					const __m128 scale4 = _mm_set1_ps((float)_scale);

					for (; i < width - 4; i += 4)
					{
						__m128 _sm = _mm_loadu_ps(Sm + i);
						__m128 _s0 = _mm_add_ps(_mm_loadu_ps(SUM + i), _mm_loadu_ps(Sp + i));
						_mm_storeu_ps(D + i, _mm_mul_ps(scale4, _s0));
						_mm_storeu_ps(SUM + i, _mm_sub_ps(_s0, _sm));
					}
#elif CV_NEON
					float32x4_t v_scale = vdupq_n_f32((float)_scale);
					for (; i <= width - 8; i += 8)
					{
						float32x4_t v_s0  = vaddq_f32(vld1q_f32(SUM + i), vld1q_f32(Sp + i));
						float32x4_t v_s01 = vaddq_f32(vld1q_f32(SUM + i + 4), vld1q_f32(Sp + i + 4));

						vst1q_f32(D + i, vmulq_f32(v_s0, v_scale));
						vst1q_f32(D + i + 4, vmulq_f32(v_s01, v_scale));

						vst1q_f32(SUM + i, vsubq_f32(v_s0, vld1q_f32(Sm + i)));
						vst1q_f32(SUM + i + 4, vsubq_f32(v_s01, vld1q_f32(Sm + i + 4)));
					}
#endif

					for (; i < width; i++)
					{
						float s0 = SUM[i] + Sp[i];
						D[i] = (float)(s0*_scale);
						SUM[i] = s0 - Sm[i];
					}
				}
				else
				{
					int i = 0;

#if CV_SSE2
					for (; i < width - 4; i += 4)
					{
						__m128 _sm = _mm_loadu_ps(Sm + i);
						__m128 _s0 = _mm_add_ps(_mm_loadu_ps(SUM + i), _mm_loadu_ps(Sp + i));

						_mm_storeu_ps(D + i, _s0);
						_mm_storeu_ps(SUM + i, _mm_sub_ps(_s0, _sm));
					}
#elif CV_NEON
					for (; i <= width - 8; i += 8)
					{
						float32x4_t v_s0  = vaddq_f32(vld1q_f32(SUM + i), vld1q_f32(Sp + i));
						float32x4_t v_s01 = vaddq_f32(vld1q_f32(SUM + i + 4), vld1q_f32(Sp + i + 4));

						vst1q_f32(D + i, v_s0);
						vst1q_f32(D + i + 4, v_s01);

						vst1q_f32(SUM + i, vsubq_f32(v_s0, vld1q_f32(Sm + i)));
						vst1q_f32(SUM + i + 4, vsubq_f32(v_s01, vld1q_f32(Sm + i + 4)));
					}
#endif

					for (; i < width; i++)
					{
						float s0 = SUM[i] + Sp[i];
						D[i] = (float)(s0);
						SUM[i] = s0 - Sm[i];
					}
				}
				dst += dststep;
			}
		}
	};

	template<typename SrcDataType, typename DstDataType, typename SumDataType, 
		int Channels, int SrcAlign, int DstAlign>
	void boxFilterT(const Image<SrcDataType, Channels, SrcAlign>& src, 
		Image<DstDataType, Channels, DstAlign>& dst, int ksize, ByteImage* workBuffer = nullptr)
	{
		const int W = src.width();
		const int H = src.height();
		const int C = src.channels();
		if (W < ksize)
		{
			LVG_LOGE("kernel size must > image width: %d > %d", ksize, W);
			return;
		}
		if (dst.width() != W || dst.height() != H)
			dst.create(W, H);

		// create row/column filter
		RowSum<SrcDataType, SumDataType> rowSum(ksize);
		ColumnSum<SumDataType, DstDataType> colSum(ksize, 1. / (ksize*ksize));

		// allocate tempory work buffer
		bool allocWorkBuffer = false;
		if (workBuffer == nullptr)
		{
			workBuffer = new ByteImage();
			allocWorkBuffer = true;
		}
		if (workBuffer->width() < sizeof(SumDataType)*W*Channels || workBuffer->height() < H + ksize)
			workBuffer->create(sizeof(SumDataType)*W*Channels, H + ksize);

		// perform row filter and then padding
		const int L = ksize / 2 - (0 == ksize % 2);
		const int R = ksize / 2;
		for (int y = 0; y < H; y++)
			rowSum((uchar*)src.rowPtr(y), workBuffer->rowPtr(y + L), W, C);
		for (int y = 0; y <= L; y++)
			memcpy(workBuffer->rowPtr(y), workBuffer->rowPtr(std::min(H - 1, 2 * L - y)),
				sizeof(SumDataType)*W*Channels);
		for (int y = 0; y <= R; y++)
			memcpy(workBuffer->rowPtr(y + H + L), workBuffer->rowPtr(std::max(0, H + L - 2 - y)),
				sizeof(SumDataType)*W*Channels);

		// perform column filter
		colSum(workBuffer->data(), workBuffer->stride(), (uchar*)dst.data(), dst.stride(), H, W*C);

		// deallocate
		if (allocWorkBuffer)
			delete workBuffer;
	}

	void boxFilter(const ByteImage& src, ByteImage& dst, int nKernel, ByteImage* workBuffer = nullptr)
	{
		if (nKernel < 16)
			boxFilterT<uchar, uchar, ushort, 1, ByteImage::ALIGN_BYTES, ByteImage::ALIGN_BYTES>(
				src, dst, nKernel, workBuffer
				);
		else
			boxFilterT<uchar, uchar, int, 1, ByteImage::ALIGN_BYTES, ByteImage::ALIGN_BYTES>(
				src, dst, nKernel, workBuffer
				);
	}

	void boxFilter(const RgbImage& src, RgbImage& dst, int nKernel, ByteImage* workBuffer = nullptr)
	{
		if (nKernel < 16)
			boxFilterT<uchar, uchar, ushort, 3, RgbImage::ALIGN_BYTES, RgbImage::ALIGN_BYTES>(
				src, dst, nKernel, workBuffer
				);
		else
			boxFilterT<uchar, uchar, int, 3, RgbImage::ALIGN_BYTES, RgbImage::ALIGN_BYTES>(
				src, dst, nKernel, workBuffer
				);
	}

	void boxFilter(const RgbaImage& src, RgbaImage& dst, int nKernel, ByteImage* workBuffer = nullptr)
	{
		if(nKernel < 16)
			boxFilterT<uchar, uchar, ushort, 4, RgbaImage::ALIGN_BYTES, RgbaImage::ALIGN_BYTES>(
				src, dst, nKernel, workBuffer
				);
		else
			boxFilterT<uchar, uchar, int, 4, RgbaImage::ALIGN_BYTES, RgbaImage::ALIGN_BYTES>(
				src, dst, nKernel, workBuffer
				);
	}

	void boxFilter(const FloatImage& src, FloatImage& dst, int nKernel, ByteImage* workBuffer = nullptr)
	{
		boxFilterT<float, float, float, 1, ByteImage::ALIGN_BYTES, ByteImage::ALIGN_BYTES>(
			src, dst, nKernel, workBuffer
			);
	}

	void boxFilter(const RgbFloatImage& src, RgbFloatImage& dst, int nKernel, ByteImage* workBuffer = nullptr)
	{
		boxFilterT<float, float, float, 3, RgbFloatImage::ALIGN_BYTES, RgbFloatImage::ALIGN_BYTES>(
			src, dst, nKernel, workBuffer
			);
	}

	void boxFilter(const RgbaFloatImage& src, RgbaFloatImage& dst, int nKernel, ByteImage* workBuffer = nullptr)
	{
		boxFilterT<float, float, float, 4, RgbaFloatImage::ALIGN_BYTES, RgbaFloatImage::ALIGN_BYTES>(
			src, dst, nKernel, workBuffer
			);
	}

} // namespace lvg
