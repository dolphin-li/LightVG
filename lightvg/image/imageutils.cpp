#include "imageutils.h"
#include "lightvg/common/logger.h"
#include "convhelper.h"
#include "opencvdebug.h"
namespace lvg
{
	void gaussianBlur_7x7(const ByteImage& src, ByteImage& dst, std::vector<int>* workBuffer)
	{
		if (src.channels() != 1) {
			LVG_LOG(LVG_LOG_ERROR, "only one-channel image supported!");
			return;
		}
		if (src.width() < 7 || src.height() < 7) {
			LVG_LOG(LVG_LOG_ERROR, "image must be large than 7!");
			return;
		}
		const static int kernel[7] = { 9, 17, 25, 26, 25, 17, 9 };
		//const static int kernel_sum = 128;
		const static int kernel_shift = 7; // sum(vkernel)=128=2^7, shift can be used for division
#ifdef CV_SIMD128
		const static v_uint16x8 vkernel[7] = { v_setall_u16(9), v_setall_u16(17), v_setall_u16(25),
			v_setall_u16(26),v_setall_u16(25),v_setall_u16(17),v_setall_u16(9) };
		const static uint vkernel_shift = 7; // sum(vkernel)=128=2^7, shift can be used for division
#endif
		if (dst.width() != src.width() || dst.height() != src.height())
			dst.create(src.width(), src.height());

		const int sW = src.width();
		const int sH = src.height();
		const int sS = src.stride();
		std::vector<int> localWorkBuffer;
		if (workBuffer == nullptr)
			workBuffer = &localWorkBuffer;
		workBuffer->resize((sS + sizeof(int)) * sH / sizeof(int));

		// 1. row filter
		for (int y = 0; y < sH; y++)
		{
			const uchar* pSrcRow = src.rowPtr(y);
			const uchar* pSrcRowBack = pSrcRow + sW - 1;
			uchar *tmp = (uchar*)workBuffer->data() + sS * y;
			uchar *tmpBack = tmp + sW;

			// left
			tmp[0] = (pSrcRow[0] * kernel[3] + 2 * pSrcRow[1] * kernel[2] +
				2 * pSrcRow[2] * kernel[1] + 2 * pSrcRow[3] * kernel[3]) >> kernel_shift;
			tmp[1] = (pSrcRow[1] * kernel[3] + (pSrcRow[0] + pSrcRow[2]) * kernel[2] +
				(pSrcRow[1] + pSrcRow[3]) * kernel[1] + (pSrcRow[2] + pSrcRow[4]) * kernel[3]) >> kernel_shift;
			tmp[2] = (pSrcRow[2] * kernel[3] + (pSrcRow[1] + pSrcRow[3]) * kernel[2] +
				(pSrcRow[0] + pSrcRow[4]) * kernel[1] + (pSrcRow[1] + pSrcRow[5]) * kernel[3]) >> kernel_shift;

#define KERNEL_SUM(x,k) (pSrcRow[x-3+k]*kernel[0]+pSrcRow[x-2+k]*kernel[1]+pSrcRow[x-1+k]*kernel[2]+\
						pSrcRow[x-0+k]*kernel[3]+pSrcRow[x+1+k]*kernel[4]+pSrcRow[x+2+k]*kernel[5]+\
						pSrcRow[x+3+k]*kernel[6])>>kernel_shift

			// middle
			int x = 3;
			for (; x < sW - 3 - 15; x += 16)
			{
#ifdef CV_SIMD128
#define VKSUM(x, k) \
                vin = v_load(pSrcRow+x-3+k);\
                v_expand(vin, vin_low, vin_high);\
                v_mul_expand(vin_low, vkernel[k], vout[0], vout[1]);\
                v_mul_expand(vin_high, vkernel[k], vout[2], vout[3]);\
                vsum[0] += vout[0];\
                vsum[1] += vout[1];\
                vsum[2] += vout[2];\
                vsum[3] += vout[3];\

				v_uint8x16 vin;
				v_uint16x8 vin_low, vin_high;
				v_uint32x4 vout[4];
				v_uint32x4 vsum[4] = { v_setzero_u32(), v_setzero_u32(), v_setzero_u32(), v_setzero_u32() };

				VKSUM(x, 0);
				VKSUM(x, 1);
				VKSUM(x, 2);
				VKSUM(x, 3);
				VKSUM(x, 4);
				VKSUM(x, 5);
				VKSUM(x, 6);

				vsum[0] = (vsum[0] >> vkernel_shift);
				vsum[1] = (vsum[1] >> vkernel_shift);
				vsum[2] = (vsum[2] >> vkernel_shift);
				vsum[3] = (vsum[3] >> vkernel_shift);

				v_uint8x16 result(
					vsum[0].get<0>(), vsum[0].get<1>(), vsum[0].get<2>(), vsum[0].get<3>(),
					vsum[1].get<0>(), vsum[1].get<1>(), vsum[1].get<2>(), vsum[1].get<3>(),
					vsum[2].get<0>(), vsum[2].get<1>(), vsum[2].get<2>(), vsum[2].get<3>(),
					vsum[3].get<0>(), vsum[3].get<1>(), vsum[3].get<2>(), vsum[3].get<3>());
				v_store(tmp + x, result);
#undef VKSUM
#else
				tmp[x + 0] = KERNEL_SUM(x, 0);
				tmp[x + 1] = KERNEL_SUM(x, 1);
				tmp[x + 2] = KERNEL_SUM(x, 2);
				tmp[x + 3] = KERNEL_SUM(x, 3);
				tmp[x + 4] = KERNEL_SUM(x, 4);
				tmp[x + 5] = KERNEL_SUM(x, 5);
				tmp[x + 6] = KERNEL_SUM(x, 6);
				tmp[x + 7] = KERNEL_SUM(x, 7);
				tmp[x + 8] = KERNEL_SUM(x, 8);
				tmp[x + 9] = KERNEL_SUM(x, 9);
				tmp[x + 10] = KERNEL_SUM(x, 10);
				tmp[x + 11] = KERNEL_SUM(x, 11);
				tmp[x + 12] = KERNEL_SUM(x, 12);
				tmp[x + 13] = KERNEL_SUM(x, 13);
				tmp[x + 14] = KERNEL_SUM(x, 14);
				tmp[x + 15] = KERNEL_SUM(x, 15);
#endif
			} // x

			for (; x < sW - 3; x++)
			{
				tmp[x] = KERNEL_SUM(x, 0);
			} // x

#undef KERNEL_SUM

			  // right
			tmpBack[0] = (pSrcRowBack[0] * kernel[3] + 2 * pSrcRowBack[-1] * kernel[2] +
				2 * pSrcRowBack[-2] * kernel[1] + 2 * pSrcRowBack[-3] * kernel[3]) >> kernel_shift;
			tmpBack[-1] = (pSrcRowBack[-1] * kernel[3] + (pSrcRowBack[0] + pSrcRowBack[-2]) * kernel[2] +
				(pSrcRowBack[-1] + pSrcRowBack[-3]) * kernel[1] + (pSrcRowBack[-2] + pSrcRowBack[-4]) * kernel[3]) >> kernel_shift;
			tmpBack[-2] = (pSrcRowBack[-2] * kernel[3] + (pSrcRowBack[-1] + pSrcRowBack[-3]) * kernel[2] +
				(pSrcRowBack[0] + pSrcRowBack[-4]) * kernel[1] + (pSrcRowBack[-1] + pSrcRowBack[-5]) * kernel[3]) >> kernel_shift;
		} // end for y

		  // 2. col filter, now we start from dst
		for (int y = 0; y < sH; y++)
		{
			const uchar* pSrc3 = ((uchar*)workBuffer->data()) + y * sS;
			const uchar* pSrc2 = pSrc3 - sS;
			const uchar* pSrc1 = pSrc2 - sS;
			const uchar* pSrc0 = pSrc1 - sS;
			const uchar* pSrc4 = pSrc3 + sS;
			const uchar* pSrc5 = pSrc4 + sS;
			const uchar* pSrc6 = pSrc5 + sS;
			uchar *tmp = dst.rowPtr(y);
			if (y == 0) {
				pSrc2 = pSrc4;
				pSrc1 = pSrc5;
				pSrc0 = pSrc6;
			}
			else if (y == 1) {
				pSrc1 = pSrc3;
				pSrc0 = pSrc4;
			}
			else if (y == 2) {
				pSrc0 = pSrc3;
			}
			else if (y == sH - 1) {
				pSrc4 = pSrc2;
				pSrc5 = pSrc1;
				pSrc6 = pSrc0;
			}
			else if (y == sH - 2) {
				pSrc5 = pSrc3;
				pSrc6 = pSrc2;
			}
			else if (y == sH - 3) {
				pSrc6 = pSrc3;
			}

#define KERNEL_SUM(x, k) (pSrc0[x+k] * kernel[0] + pSrc1[x+k] * kernel[1] + pSrc2[x+k] * kernel[2]\
							+ pSrc3[x+k] * kernel[3] + pSrc4[x+k] * kernel[4] + pSrc5[x+k] * kernel[5] \
							+ pSrc6[x+k] * kernel[6]) >>kernel_shift
			int x = 0;

			for (; x < sW - 15; x += 16) {
#ifdef CV_SIMD128
#define VKSUM(x, k) \
                vin = v_load(pSrc##k + x);\
                v_expand(vin, vin_low, vin_high);\
                v_mul_expand(vin_low, vkernel[k], vout[0], vout[1]);\
                v_mul_expand(vin_high, vkernel[k], vout[2], vout[3]);\
                vsum[0] += vout[0];\
                vsum[1] += vout[1];\
                vsum[2] += vout[2];\
                vsum[3] += vout[3];\

				v_uint8x16 vin;
				v_uint16x8 vin_low, vin_high;
				v_uint32x4 vout[4];
				v_uint32x4 vsum[4] = { v_setzero_u32(), v_setzero_u32(), v_setzero_u32(), v_setzero_u32() };

				VKSUM(x, 0);
				VKSUM(x, 1);
				VKSUM(x, 2);
				VKSUM(x, 3);
				VKSUM(x, 4);
				VKSUM(x, 5);
				VKSUM(x, 6);

				vsum[0] = (vsum[0] >> vkernel_shift);
				vsum[1] = (vsum[1] >> vkernel_shift);
				vsum[2] = (vsum[2] >> vkernel_shift);
				vsum[3] = (vsum[3] >> vkernel_shift);

				v_uint8x16 result(
					vsum[0].get<0>(), vsum[0].get<1>(), vsum[0].get<2>(), vsum[0].get<3>(),
					vsum[1].get<0>(), vsum[1].get<1>(), vsum[1].get<2>(), vsum[1].get<3>(),
					vsum[2].get<0>(), vsum[2].get<1>(), vsum[2].get<2>(), vsum[2].get<3>(),
					vsum[3].get<0>(), vsum[3].get<1>(), vsum[3].get<2>(), vsum[3].get<3>());
				v_store(tmp + x, result);
#undef VKSUM
#else
				tmp[x + 0] = KERNEL_SUM(x, 0);
				tmp[x + 1] = KERNEL_SUM(x, 1);
				tmp[x + 2] = KERNEL_SUM(x, 2);
				tmp[x + 3] = KERNEL_SUM(x, 3);
				tmp[x + 4] = KERNEL_SUM(x, 4);
				tmp[x + 5] = KERNEL_SUM(x, 5);
				tmp[x + 6] = KERNEL_SUM(x, 6);
				tmp[x + 7] = KERNEL_SUM(x, 7);
				tmp[x + 8] = KERNEL_SUM(x, 8);
				tmp[x + 9] = KERNEL_SUM(x, 9);
				tmp[x + 10] = KERNEL_SUM(x, 10);
				tmp[x + 11] = KERNEL_SUM(x, 11);
				tmp[x + 12] = KERNEL_SUM(x, 12);
				tmp[x + 13] = KERNEL_SUM(x, 13);
				tmp[x + 14] = KERNEL_SUM(x, 14);
				tmp[x + 15] = KERNEL_SUM(x, 15);
#endif
			}
			for (; x < sW; x++) {
				tmp[x] = KERNEL_SUM(x, 0);
			}
#undef KERNEL_SUM
		} // end for y
	}

	void gaussianBlur_5x5(const ByteImage& src, ByteImage& dst, std::vector<int>* workBuffer)
	{
		if (src.channels() != 1) {
			LVG_LOG(LVG_LOG_ERROR, "only one-channel image supported!");
			return;
		}
		if (src.width() < 5 || src.height() < 5) {
			LVG_LOG(LVG_LOG_ERROR, "image must be large than 7!");
			return;
		}
		const static int kernel[5] = { 20, 28, 32, 28, 20 };
		const static int kernel_shift = 7; //2^7=128
#ifdef CV_SIMD128
		const static v_uint16x8 vkernel[5] = { v_setall_u16(20), v_setall_u16(28), v_setall_u16(32),
			v_setall_u16(28),v_setall_u16(20) };
		const static uint vkernel_shift = 7; // sum(vkernel)=128=2^7, shift can be used for division
#endif
		if (dst.width() != src.width() || dst.height() != src.height())
			dst.create(src.width(), src.height());

		const int sW = src.width();
		const int sH = src.height();
		const int sS = src.stride();
		std::vector<int> localWorkBuffer;
		if (workBuffer == nullptr)
			workBuffer = &localWorkBuffer;
		workBuffer->resize((sS + sizeof(int)) * sH / sizeof(int));

		// 1. row filter
		for (int y = 0; y < sH; y++)
		{
			const uchar* pSrcRow = src.rowPtr(y);
			const uchar* pSrcRowBack = pSrcRow + sW - 1;
			uchar *tmp = ((uchar*)workBuffer->data()) + sS * y;
			uchar* tmpBack = tmp + sW - 1;
			int x = 0;

			// left
			tmp[0] = (pSrcRow[0] * kernel[2] + 2 * pSrcRow[1] * kernel[1] + 2 * pSrcRow[2] * kernel[0]) >> kernel_shift;
			tmp[1] = (pSrcRow[1] * kernel[2] + (pSrcRow[0] + pSrcRow[2]) * kernel[1] +
				(pSrcRow[1] + pSrcRow[3]) * kernel[0]) >> kernel_shift;
#define KERNEL_SUM(x, k) (pSrcRow[x - 2 + k] * kernel[0] + pSrcRow[x - 1 + k] * kernel[1]\
							+ pSrcRow[x - 0 + k] * kernel[2] + pSrcRow[x + 1 + k] * kernel[3]\
							+ pSrcRow[x + 2 + k] * kernel[4]) >>kernel_shift
			// middle
			x = 2;
			for (; x < sW - 2 - 15; x += 16)
			{
#ifdef CV_SIMD128
#define VKSUM(x, k) \
                vin = v_load(pSrcRow+x-2+k);\
                v_expand(vin, vin_low, vin_high);\
                v_mul_expand(vin_low, vkernel[k], vout[0], vout[1]);\
                v_mul_expand(vin_high, vkernel[k], vout[2], vout[3]);\
                vsum[0] += vout[0];\
                vsum[1] += vout[1];\
                vsum[2] += vout[2];\
                vsum[3] += vout[3];\

				v_uint8x16 vin;
				v_uint16x8 vin_low, vin_high;
				v_uint32x4 vout[4];
				v_uint32x4 vsum[4] = { v_setzero_u32(), v_setzero_u32(), v_setzero_u32(), v_setzero_u32() };

				VKSUM(x, 0);
				VKSUM(x, 1);
				VKSUM(x, 2);
				VKSUM(x, 3);
				VKSUM(x, 4);

				vsum[0] = (vsum[0] >> vkernel_shift);
				vsum[1] = (vsum[1] >> vkernel_shift);
				vsum[2] = (vsum[2] >> vkernel_shift);
				vsum[3] = (vsum[3] >> vkernel_shift);

				v_uint8x16 result(
					vsum[0].get<0>(), vsum[0].get<1>(), vsum[0].get<2>(), vsum[0].get<3>(),
					vsum[1].get<0>(), vsum[1].get<1>(), vsum[1].get<2>(), vsum[1].get<3>(),
					vsum[2].get<0>(), vsum[2].get<1>(), vsum[2].get<2>(), vsum[2].get<3>(),
					vsum[3].get<0>(), vsum[3].get<1>(), vsum[3].get<2>(), vsum[3].get<3>());
				v_store(tmp + x, result);
#undef VKSUM
#else
				tmp[x + 0] = KERNEL_SUM(x, 0);
				tmp[x + 1] = KERNEL_SUM(x, 1);
				tmp[x + 2] = KERNEL_SUM(x, 2);
				tmp[x + 3] = KERNEL_SUM(x, 3);
				tmp[x + 4] = KERNEL_SUM(x, 4);
				tmp[x + 5] = KERNEL_SUM(x, 5);
				tmp[x + 6] = KERNEL_SUM(x, 6);
				tmp[x + 7] = KERNEL_SUM(x, 7);
				tmp[x + 8] = KERNEL_SUM(x, 8);
				tmp[x + 9] = KERNEL_SUM(x, 9);
				tmp[x + 10] = KERNEL_SUM(x, 10);
				tmp[x + 11] = KERNEL_SUM(x, 11);
				tmp[x + 12] = KERNEL_SUM(x, 12);
				tmp[x + 13] = KERNEL_SUM(x, 13);
				tmp[x + 14] = KERNEL_SUM(x, 14);
				tmp[x + 15] = KERNEL_SUM(x, 15);
#endif
			} // x

			for (; x < sW - 2; x++)
			{
				tmp[x] = (pSrcRow[x - 2] * kernel[0] + pSrcRow[x - 1] * kernel[1]
					+ pSrcRow[x - 0] * kernel[2] + pSrcRow[x + 1] * kernel[3]
					+ pSrcRow[x + 2] * kernel[4]) >> kernel_shift;
			} // x
#undef KERNEL_SUM

			  // right
			tmpBack[0] = (pSrcRowBack[0] * kernel[2] + 2 * pSrcRowBack[-1] * kernel[1] +
				2 * pSrcRowBack[-2] * kernel[0]) >> kernel_shift;
			tmpBack[-1] = (pSrcRowBack[-1] * kernel[2] + (pSrcRowBack[0] + pSrcRowBack[-2]) * kernel[1] +
				(pSrcRowBack[-1] + pSrcRowBack[-3]) * kernel[0]) >> kernel_shift;
		} // end for y

		  // 2. col filter, now we start from dst
		for (int y = 0; y < sH; y++)
		{
			const uchar* pSrc2 = ((uchar*)workBuffer->data()) + y * sS;
			const uchar* pSrc1 = pSrc2 - sS;
			const uchar* pSrc0 = pSrc1 - sS;
			const uchar* pSrc3 = pSrc2 + sS;
			const uchar* pSrc4 = pSrc3 + sS;
			uchar *tmp = dst.rowPtr(y);
			if (y == 0) {
				pSrc1 = pSrc3;
				pSrc0 = pSrc4;
			}
			else if (y == 1) {
				pSrc0 = pSrc1;
			}
			else if (y == sH - 1) {
				pSrc3 = pSrc1;
				pSrc4 = pSrc0;
			}
			else if (y == sH - 2) {
				pSrc4 = pSrc3;
			}
			int x = 0;
#define KERNEL_SUM(x, k) (pSrc0[x + k] * kernel[0] + pSrc1[x + k] * kernel[1] + pSrc2[x + k] * kernel[2]\
						 + pSrc3[x + k] * kernel[3] + pSrc4[x + k] * kernel[4])>>kernel_shift
			for (; x < sW - 15; x += 16) {
#ifdef CV_SIMD128
#define VKSUM(x, k) \
                vin = v_load(pSrc##k + x);\
                v_expand(vin, vin_low, vin_high);\
                v_mul_expand(vin_low, vkernel[k], vout[0], vout[1]);\
                v_mul_expand(vin_high, vkernel[k], vout[2], vout[3]);\
                vsum[0] += vout[0];\
                vsum[1] += vout[1];\
                vsum[2] += vout[2];\
                vsum[3] += vout[3];\

				v_uint8x16 vin;
				v_uint16x8 vin_low, vin_high;
				v_uint32x4 vout[4];
				v_uint32x4 vsum[4] = { v_setzero_u32(), v_setzero_u32(), v_setzero_u32(), v_setzero_u32() };

				VKSUM(x, 0);
				VKSUM(x, 1);
				VKSUM(x, 2);
				VKSUM(x, 3);
				VKSUM(x, 4);

				vsum[0] = (vsum[0] >> vkernel_shift);
				vsum[1] = (vsum[1] >> vkernel_shift);
				vsum[2] = (vsum[2] >> vkernel_shift);
				vsum[3] = (vsum[3] >> vkernel_shift);

				v_uint8x16 result(
					vsum[0].get<0>(), vsum[0].get<1>(), vsum[0].get<2>(), vsum[0].get<3>(),
					vsum[1].get<0>(), vsum[1].get<1>(), vsum[1].get<2>(), vsum[1].get<3>(),
					vsum[2].get<0>(), vsum[2].get<1>(), vsum[2].get<2>(), vsum[2].get<3>(),
					vsum[3].get<0>(), vsum[3].get<1>(), vsum[3].get<2>(), vsum[3].get<3>());
				v_store(tmp + x, result);
#undef VKSUM
#else
				tmp[x + 0] = KERNEL_SUM(x, 0);
				tmp[x + 1] = KERNEL_SUM(x, 1);
				tmp[x + 2] = KERNEL_SUM(x, 2);
				tmp[x + 3] = KERNEL_SUM(x, 3);
				tmp[x + 4] = KERNEL_SUM(x, 4);
				tmp[x + 5] = KERNEL_SUM(x, 5);
				tmp[x + 6] = KERNEL_SUM(x, 6);
				tmp[x + 7] = KERNEL_SUM(x, 7);
				tmp[x + 8] = KERNEL_SUM(x, 8);
				tmp[x + 9] = KERNEL_SUM(x, 9);
				tmp[x + 10] = KERNEL_SUM(x, 10);
				tmp[x + 11] = KERNEL_SUM(x, 11);
				tmp[x + 12] = KERNEL_SUM(x, 12);
				tmp[x + 13] = KERNEL_SUM(x, 13);
				tmp[x + 14] = KERNEL_SUM(x, 14);
				tmp[x + 15] = KERNEL_SUM(x, 15);
#endif
			}
			for (; x < sW; x++) {
				tmp[x] = KERNEL_SUM(x, 0);
			}
#undef KERNEL_SUM
		} // end for y
	}

#pragma region --fast detector
	template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
	{
		return (_Tp*)(((size_t)ptr + n - 1) & -n);
	}
	inline void makeOffsets(int pixel[25], int rowStride, int patternSize)
	{
		static const int offsets16[][2] =
		{
			{ 0,  3 },{ 1,  3 },{ 2,  2 },{ 3,  1 },{ 3, 0 },{ 3, -1 },{ 2, -2 },{ 1, -3 },
			{ 0, -3 },{ -1, -3 },{ -2, -2 },{ -3, -1 },{ -3, 0 },{ -3,  1 },{ -2,  2 },{ -1,  3 }
		};

		static const int offsets12[][2] =
		{
			{ 0,  2 },{ 1,  2 },{ 2,  1 },{ 2, 0 },{ 2, -1 },{ 1, -2 },
			{ 0, -2 },{ -1, -2 },{ -2, -1 },{ -2, 0 },{ -2,  1 },{ -1,  2 }
		};

		static const int offsets8[][2] =
		{
			{ 0,  1 },{ 1,  1 },{ 1, 0 },{ 1, -1 },
			{ 0, -1 },{ -1, -1 },{ -1, 0 },{ -1,  1 }
		};

		const int(*offsets)[2] = patternSize == 16 ? offsets16 :
			patternSize == 12 ? offsets12 :
			patternSize == 8 ? offsets8 : 0;

		int k = 0;
		for (; k < patternSize; k++)
			pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
		for (; k < 25; k++)
			pixel[k] = pixel[k - patternSize];
	}

	inline void makeThreshould(uchar threshold_tab[512], int threshold)
	{
		for (int i = -255; i <= 255; i++)
			threshold_tab[i + 255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);
	}

	template<int N> inline int cornerScore(const uchar* ptr, const int pixel[], int threshold);

	template<>
	inline int cornerScore<16>(const uchar* ptr, const int pixel[], int threshold)
	{
		const int K = 8, N = K * 3 + 1;
		int k, v = ptr[0];
		short d[N];
		for (k = 0; k < N; k++)
			d[k] = (short)(v - ptr[pixel[k]]);

		int a0 = threshold;
		for (k = 0; k < 16; k += 2)
		{
			int a = std::min((int)d[k + 1], (int)d[k + 2]);
			a = std::min(a, (int)d[k + 3]);
			if (a <= a0)
				continue;
			a = std::min(a, (int)d[k + 4]);
			a = std::min(a, (int)d[k + 5]);
			a = std::min(a, (int)d[k + 6]);
			a = std::min(a, (int)d[k + 7]);
			a = std::min(a, (int)d[k + 8]);
			a0 = std::max(a0, std::min(a, (int)d[k]));
			a0 = std::max(a0, std::min(a, (int)d[k + 9]));
		}

		int b0 = -a0;
		for (k = 0; k < 16; k += 2)
		{
			int b = std::max((int)d[k + 1], (int)d[k + 2]);
			b = std::max(b, (int)d[k + 3]);
			b = std::max(b, (int)d[k + 4]);
			b = std::max(b, (int)d[k + 5]);
			if (b >= b0)
				continue;
			b = std::max(b, (int)d[k + 6]);
			b = std::max(b, (int)d[k + 7]);
			b = std::max(b, (int)d[k + 8]);

			b0 = std::min(b0, std::max(b, (int)d[k]));
			b0 = std::min(b0, std::max(b, (int)d[k + 9]));
		}

		threshold = -b0 - 1;
		return threshold;
	}

	template<>
	inline int cornerScore<12>(const uchar* ptr, const int pixel[], int threshold)
	{
		const int K = 6, N = K * 3 + 1;
		int k, v = ptr[0];
		short d[N + 4];
		for (k = 0; k < N; k++)
			d[k] = (short)(v - ptr[pixel[k]]);

		int a0 = threshold;
		for (k = 0; k < 12; k += 2)
		{
			int a = std::min((int)d[k + 1], (int)d[k + 2]);
			if (a <= a0)
				continue;
			a = std::min(a, (int)d[k + 3]);
			a = std::min(a, (int)d[k + 4]);
			a = std::min(a, (int)d[k + 5]);
			a = std::min(a, (int)d[k + 6]);
			a0 = std::max(a0, std::min(a, (int)d[k]));
			a0 = std::max(a0, std::min(a, (int)d[k + 7]));
		}

		int b0 = -a0;
		for (k = 0; k < 12; k += 2)
		{
			int b = std::max((int)d[k + 1], (int)d[k + 2]);
			b = std::max(b, (int)d[k + 3]);
			b = std::max(b, (int)d[k + 4]);
			if (b >= b0)
				continue;
			b = std::max(b, (int)d[k + 5]);
			b = std::max(b, (int)d[k + 6]);

			b0 = std::min(b0, std::max(b, (int)d[k]));
			b0 = std::min(b0, std::max(b, (int)d[k + 7]));
		}

		threshold = -b0 - 1;
		return threshold;
	}

	template<>
	inline int cornerScore<8>(const uchar* ptr, const int pixel[], int threshold)
	{
		const int K = 4, N = K * 3 + 1;
		int k, v = ptr[0];
		short d[N];
		for (k = 0; k < N; k++)
			d[k] = (short)(v - ptr[pixel[k]]);


		int a0 = threshold;
		for (k = 0; k < 8; k += 2)
		{
			int a = std::min((int)d[k + 1], (int)d[k + 2]);
			if (a <= a0)
				continue;
			a = std::min(a, (int)d[k + 3]);
			a = std::min(a, (int)d[k + 4]);
			a0 = std::max(a0, std::min(a, (int)d[k]));
			a0 = std::max(a0, std::min(a, (int)d[k + 5]));
		}

		int b0 = -a0;
		for (k = 0; k < 8; k += 2)
		{
			int b = std::max((int)d[k + 1], (int)d[k + 2]);
			b = std::max(b, (int)d[k + 3]);
			if (b >= b0)
				continue;
			b = std::max(b, (int)d[k + 4]);

			b0 = std::min(b0, std::max(b, (int)d[k]));
			b0 = std::min(b0, std::max(b, (int)d[k + 5]));
		}

		threshold = -b0 - 1;
		return threshold;
	}

	template<int patternSize>
	void FAST_t(const ByteImage& img, std::vector<KeyPoint>& keypoints, int threshold,
		bool nonmax_suppression, std::vector<unsigned char>* workBuffer)
	{
		const int K = patternSize / 2, N = patternSize + K + 1;
#if CV_SIMD128
		const int quarterPatternSize = patternSize / 4;
		v_uint8x16 delta = v_setall_u8(0x80), t = v_setall_u8((char)threshold), K16 = v_setall_u8((char)K);
		bool hasSimd = hasSIMD128();
#endif

		int pixel[25];
		makeOffsets(pixel, img.stride(), patternSize);

		keypoints.clear();

		threshold = std::min(std::max(threshold, 0), 255);

		uchar threshold_tab[512];
		makeThreshould(threshold_tab, threshold);

		std::vector<uchar> tmpBuf;
		if (workBuffer == nullptr)
			workBuffer = &tmpBuf;
		workBuffer->resize((img.cols() + 16) * 3 * (sizeof(int) + sizeof(uchar)) + 128);

		uchar* buf[3];
		buf[0] = workBuffer->data();
		buf[1] = buf[0] + img.cols();
		buf[2] = buf[1] + img.cols();

		int* cpbuf[3];
		cpbuf[0] = (int*)alignPtr(buf[2] + img.cols(), sizeof(int)) + 1;
		cpbuf[1] = cpbuf[0] + img.cols() + 1;
		cpbuf[2] = cpbuf[1] + img.cols() + 1;
		memset(buf[0], 0, img.cols() * 3);

		for (int i = 3; i < img.rows() - 2; i++)
		{
			const uchar* ptr = img.rowPtr(i) + 3;
			uchar* curr = buf[(i - 3) % 3];
			int* cornerpos = cpbuf[(i - 3) % 3];
			memset(curr, 0, img.cols());
			int ncorners = 0;

			if (i < img.rows() - 3)
			{
				int j = 3;
#ifdef CV_SIMD128
				if (patternSize == 16)
				{
					for (; j < img.cols() - 16 - 3; j += 16, ptr += 16)
					{
						v_uint8x16 v = v_load(ptr);
						v_int8x16 v0 = v_reinterpret_as_s8((v + t) ^ delta);
						v_int8x16 v1 = v_reinterpret_as_s8((v - t) ^ delta);

						v_int8x16 x0 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[0]), delta));
						v_int8x16 x1 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[quarterPatternSize]), delta));
						v_int8x16 x2 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[2 * quarterPatternSize]), delta));
						v_int8x16 x3 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[3 * quarterPatternSize]), delta));

						v_int8x16 m0, m1;
						m0 = (v0 < x0) & (v0 < x1);
						m1 = (x0 < v1) & (x1 < v1);
						m0 = m0 | ((v0 < x1) & (v0 < x2));
						m1 = m1 | ((x1 < v1) & (x2 < v1));
						m0 = m0 | ((v0 < x2) & (v0 < x3));
						m1 = m1 | ((x2 < v1) & (x3 < v1));
						m0 = m0 | ((v0 < x3) & (v0 < x0));
						m1 = m1 | ((x3 < v1) & (x0 < v1));
						m0 = m0 | m1;

						int mask = v_signmask(m0);
						if (mask == 0)
							continue;
						if ((mask & 255) == 0)
						{
							j -= 8;
							ptr -= 8;
							continue;
						}

						v_int8x16 c0 = v_setzero_s8();
						v_int8x16 c1 = v_setzero_s8();
						v_uint8x16 max0 = v_setzero_u8();
						v_uint8x16 max1 = v_setzero_u8();
						for (int k = 0; k < N; k++)
						{
							v_int8x16 x = v_reinterpret_as_s8(v_load((ptr + pixel[k])) ^ delta);
							m0 = v0 < x;
							m1 = x < v1;

							c0 = v_sub_wrap(c0, m0) & m0;
							c1 = v_sub_wrap(c1, m1) & m1;

							max0 = v_max(max0, v_reinterpret_as_u8(c0));
							max1 = v_max(max1, v_reinterpret_as_u8(c1));
						}

						max0 = v_max(max0, max1);
						int m = v_signmask(K16 < max0);

						for (int k = 0; m > 0 && k < 16; k++, m >>= 1)
						{
							if (m & 1)
							{
								cornerpos[ncorners++] = j + k;
								if (nonmax_suppression)
									curr[j + k] = (uchar)cornerScore<patternSize>(ptr + k, pixel, threshold);
							}
						}
					}
				}
#endif
				for (; j < img.cols() - 3; j++, ptr++)
				{
					int v = ptr[0];
					const uchar* tab = &threshold_tab[0] - v + 255;
					int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

					if (d == 0)
						continue;

					d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
					d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
					d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

					if (d == 0)
						continue;

					d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
					d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
					d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
					d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

					if (d & 1)
					{
						int vt = v - threshold, count = 0;

						for (int k = 0; k < N; k++)
						{
							int x = ptr[pixel[k]];
							if (x < vt)
							{
								if (++count > K)
								{
									cornerpos[ncorners++] = j;
									if (nonmax_suppression)
										curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
									break;
								}
							}
							else
								count = 0;
						}
					}

					if (d & 2)
					{
						int vt = v + threshold, count = 0;

						for (int k = 0; k < N; k++)
						{
							int x = ptr[pixel[k]];
							if (x > vt)
							{
								if (++count > K)
								{
									cornerpos[ncorners++] = j;
									if (nonmax_suppression)
										curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
									break;
								}
							}
							else
								count = 0;
						}
					}
				}
			}

			cornerpos[-1] = ncorners;

			if (i == 3)
				continue;

			const uchar* prev = buf[(i - 4 + 3) % 3];
			const uchar* pprev = buf[(i - 5 + 3) % 3];
			cornerpos = cpbuf[(i - 4 + 3) % 3];
			ncorners = cornerpos[-1];

			for (int k = 0; k < ncorners; k++)
			{
				const int j = cornerpos[k];
				int score = prev[j];
				if (!nonmax_suppression ||
					(score > prev[j + 1] && score > prev[j - 1] &&
						score > pprev[j - 1] && score > pprev[j] && score > pprev[j + 1] &&
						score > curr[j - 1] && score > curr[j] && score > curr[j + 1]))
				{
					keypoints.push_back(KeyPoint(float2((float)j, (float)(i - 1)), 0, -1.f, (float)score, 7));
				}
			} // end for k
		} // end for i
	}

	void fastCornerDetect(const ByteImage& src, std::vector<KeyPoint>& keyPoints,
		int threshold, bool bNonmaxSurpress, std::vector<uchar>* workBuffer)
	{
		//FAST_t<8>(src, keyPoints, threshold, bNonmaxSurpress, workBuffer);
		//FAST_t<12>(src, keyPoints, threshold, bNonmaxSurpress, workBuffer);
		FAST_t<16>(src, keyPoints, threshold, bNonmaxSurpress, workBuffer);
	}


	struct KeypointResponseGreater
	{
		inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const
		{
			return kp1.response > kp2.response;
		}
	};
	struct KeypointResponseGreaterThanThreshold
	{
		KeypointResponseGreaterThanThreshold(float _value) :
			value(_value)
		{
		}
		inline bool operator()(const KeyPoint& kpt) const
		{
			return kpt.response >= value;
		}
		float value;
	};
	void keyPointsFilter_retainBest(std::vector<KeyPoint>& keypoints, int n_points)
	{
		//this is only necessary if the keypoints size is greater than the number of desired points.
		if (n_points >= 0 && keypoints.size() > (size_t)n_points)
		{
			if (n_points == 0)
			{
				keypoints.clear();
				return;
			}
			//first use nth element to partition the keypoints into the best and worst.
			std::nth_element(keypoints.begin(), keypoints.begin() + n_points, keypoints.end(), KeypointResponseGreater());
			//this is the boundary response, and in the case of FAST may be ambigous
			float ambiguous_response = keypoints[n_points - 1].response;
			//use std::partition to grab all of the keypoints with the boundary response.
			std::vector<KeyPoint>::const_iterator new_end =
				std::partition(keypoints.begin() + n_points, keypoints.end(),
					KeypointResponseGreaterThanThreshold(ambiguous_response));
			//resize the keypoints, given this new end point. nth_element and partition reordered the points inplace
			keypoints.resize(new_end - keypoints.begin());
		}
	}
#pragma endregion

	void separableConv2(const FloatImage& src, FloatImage& dst, const float* kernel, int nKernel)
	{
		if (!dst.sameWith(src))
			dst = src.clone();
		else
			dst = src;
		switch (nKernel)
		{
		case 1:
			dst *= kernel[0];
			break;
		case 2:
			conv2<float, 2>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 3:
			conv2<float, 3>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 4:
			conv2<float, 4>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 5:
			conv2<float, 5>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 6:
			conv2<float, 6>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 7:
			conv2<float, 7>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 8:
			conv2<float, 8>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 9:
			conv2<float, 9>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 10:
			conv2<float, 10>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 11:
			conv2<float, 11>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 12:
			conv2<float, 12>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 13:
			conv2<float, 13>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 14:
			conv2<float, 14>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		case 15:
			conv2<float, 15>(dst.data(), kernel, dst.width(), dst.height(), dst.stride());
			break;
		default:
			LVG_LOG(LVG_LOG_ERROR, "conv2: non supported kernel size");
			break;
		}
	}

	template<class T> void maxFilterT(const Image<T, 1, 4>& src, Image<T, 1, 4>& dst, int nKernel)
	{
		if (!dst.sameWith(src))
			dst = src.clone();
		else
			dst = src;
		switch (nKernel)
		{
		case 1:
			break;
		case 2:
			max_filter2<T, 2>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 3:
			max_filter2<T, 3>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 4:
			max_filter2<T, 4>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 5:
			max_filter2<T, 5>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 6:
			max_filter2<T, 6>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 7:
			max_filter2<T, 7>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 8:
			max_filter2<T, 8>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 9:
			max_filter2<T, 9>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 10:
			max_filter2<T, 10>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 11:
			max_filter2<T, 11>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 12:
			max_filter2<T, 12>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 13:
			max_filter2<T, 13>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 14:
			max_filter2<T, 14>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 15:
			max_filter2<T, 15>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		default:
			LVG_LOG(LVG_LOG_ERROR, "conv2: non supported kernel size");
			break;
		}
	}

	void maxFilter(const FloatImage& src, FloatImage& dst, int nKernel)
	{
		maxFilterT(src, dst, nKernel);
	}
	void maxFilter(const ByteImage& src, ByteImage& dst, int nKernel)
	{
		maxFilterT(src, dst, nKernel);
	}
	void maxFilter(const IntImage& src, IntImage& dst, int nKernel)
	{
		maxFilterT(src, dst, nKernel);
	}

	template<class T> void minFilterT(const Image<T, 1, 4>& src, Image<T, 1, 4>& dst, int nKernel)
	{
		if (!dst.sameWith(src))
			dst = src.clone();
		else
			dst = src;
		switch (nKernel)
		{
		case 1:
			break;
		case 2:
			min_filter2<T, 2>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 3:
			min_filter2<T, 3>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 4:
			min_filter2<T, 4>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 5:
			min_filter2<T, 5>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 6:
			min_filter2<T, 6>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 7:
			min_filter2<T, 7>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 8:
			min_filter2<T, 8>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 9:
			min_filter2<T, 9>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 10:
			min_filter2<T, 10>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 11:
			min_filter2<T, 11>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 12:
			min_filter2<T, 12>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 13:
			min_filter2<T, 13>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 14:
			min_filter2<T, 14>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		case 15:
			min_filter2<T, 15>(dst.data(), dst.width(), dst.height(), dst.stride());
			break;
		default:
			LVG_LOG(LVG_LOG_ERROR, "conv2: non supported kernel size");
			break;
		}
	}

	void minFilter(const FloatImage& src, FloatImage& dst, int nKernel)
	{
		minFilterT(src, dst, nKernel);
	}
	void minFilter(const ByteImage& src, ByteImage& dst, int nKernel)
	{
		minFilterT(src, dst, nKernel);
	}
	void minFilter(const IntImage& src, IntImage& dst, int nKernel)
	{
		minFilterT(src, dst, nKernel);
	}

	FloatImage bwdist(const ByteImage& imMask)
	{
		FloatImage imDist;
		imDist.create(imMask.width(), imMask.height());
		IntImage imTmp;
		imTmp.create(imMask.width(), imMask.height());
		const int m = imDist.height(), n = imDist.width();
		const int inf = m + n;
		const int maxMN = std::max(m, n);

		//metric function
		struct EuclideMetric
		{
			static int f(int a, int b)
			{
				return a*a + b*b;
			}
			static int sep(int i, int u, int gi, int gu)
			{
				return (u*u - i*i + gu*gu - gi*gi) / (2 * (u - i));
			}
		};

		// phase 1
		for (int x = 0; x<n; x++)
		{
			imTmp.rowPtr(0)[x] = (imMask.rowPtr(0)[x] > 128) ? inf : 0;

			// scan 1
			for (int y = 1; y<m; y++)
				imTmp.rowPtr(y)[x] = (imMask.rowPtr(y)[x] > 128) ? 1 + imTmp.rowPtr(y - 1)[x] : 0;

			// scan 2
			for (int y = m - 2; y >= 0; y--)
				if (imTmp.rowPtr(y + 1)[x] < imTmp.rowPtr(y)[x])
					imTmp.rowPtr(y)[x] = 1 + imTmp.rowPtr(y + 1)[x];
		}//end for y

		 // phase 2
		std::vector<int> s(maxMN), t(maxMN);
		for (int y = 0; y<m; y++)
		{
			int* pTmp = imTmp.rowPtr(y);
			int q = 0;
			s[0] = 0;
			t[0] = 0;

			// scan 3
			for (int x = 1; x<n; x++)
			{
				while (q >= 0 && EuclideMetric::f(t[q] - s[q], pTmp[s[q]]) > EuclideMetric::f(t[q] - x, pTmp[x]))
					q--;
				if (q<0)
				{
					q = 0;
					s[0] = x;
				}
				else
				{
					int const w = 1 + EuclideMetric::sep(s[q], x, pTmp[s[q]], pTmp[x]);
					if (w < n)
					{
						++q;
						s[q] = x;
						t[q] = w;
					}
				}
			}//end for x

			 // scan 4
			for (int x = n - 1; x >= 0; x--)
			{
				int const d = EuclideMetric::f(x - s[q], pTmp[s[q]]);

				//output
				imDist.rowPtr(y)[x] = sqrt((float)d);

				if (x == t[q]) --q;
			}//end for x
		}//end for y

		return imDist;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// resize related
	/////////////////////////////////////////////////////////////////////////////////////////////////
	inline double lanczos3(double x)
	{
		double f = 0.0;
		const static double eps = 2.220446049250313e-16;
		double pix = MATH_PI*x;
		f = (sin(pix) * sin(pix / 3.0) + eps) / ((pix * pix) / 3.0 + eps);
		f = f * (abs(x) < 3);
		return f;
	}
	void Lanczos3KernelIdx(int srcLen, int dstLen, std::vector<std::vector<float>>& weights, std::vector<std::vector<int>>& indices)
	{
		float scale = float(srcLen) / float(dstLen);
		int kernel_size = 6;
		if (scale > 1.0) kernel_size = int(kernel_size*scale);
		weights.resize(dstLen);
		indices.resize(dstLen);
		for (int i = 0; i<dstLen; i++)
		{
			const float dpos = i*scale - 0.5f * (1.f - scale);
			int ipos = int(dpos);
			if (dpos < 0) ipos--;
			std::vector<float>& w = weights[i];
			std::vector<int>& id = indices[i];
			w.resize(kernel_size);
			id.resize(kernel_size);
			float sumW = 0.f;
			const int left = ipos - kernel_size / 2 + 1, right = ipos + kernel_size / 2;
			for (int j = left; j <= right; j++)
			{
				if (scale > 1.f)
					w[j - left] = float(lanczos3((dpos - j) / scale));
				else
					w[j - left] = float(lanczos3(dpos - j));
				id[j - left] = std::max(0, std::min(srcLen - 1, j));
				sumW += w[j - left];
			}
			for (int j = 0; j<kernel_size; j++)
				w[j] /= sumW;
		}
	}
	
	template<class T, int Channels, int alignBytes>
	void imresizeBilinear(const Image<T, Channels, alignBytes>& src, Image<T, Channels, alignBytes>& dst, int dstW, int dstH)
	{
		if (dst.memoryOverlap(src))
		{
			LVG_LOG(lvg::LVG_LOG_ERROR, "src and dst cannot share memory");
			return;
		}
		if(dstW != dst.width() || dstH != dst.height())
			dst.create(dstW, dstH);
		const int srcH = src.height();
		const int srcW = src.width();
		if (dstW == 0 || dstH == 0 || srcW == 0 || srcH == 0)
			return;
		const float scalex = (float)srcW / (float)dstW;
		const float scaley = (float)srcH / (float)dstH;
		std::vector<int> tmpBuffer(2 * dstW);
		int* xofs = (int*)tmpBuffer.data();
		float* xws = (float*)(xofs + dstW);
		for (int x = 0; x < dstW; x++)
		{
			float fx = std::max(0.f, float(x + 0.5f)*scalex - 0.5f);
			int sx = int(fx);
			fx -= sx;
			if (sx >= srcW - 1)
			{
				sx = std::max(0, srcW - 2);
				fx = 1.f;
			}
			xofs[x] = sx;
			xws[x] = fx;
		} // x

		for (int y = 0; y < dstH; y++)
		{
			float fy = std::max(0.f, float(y + 0.5f)*scaley - 0.5f);
			int sy = int(fy);
			fy -= sy;
			if (sy >= srcH - 1)
			{
				sy = std::max(0, srcH - 2);
				fy = 1.f;
			}

			T* pDst_row = dst.rowPtr(y);
			const T* pSrc0_row = src.rowPtr(sy);
			const T* pSrc1_row = src.rowPtr(sy + 1);
			for (int x = 0; x < dstW; x++)
			{
				const float fx = xws[x];
				const int sx = xofs[x];
				const float a00 = (1 - fx)*(1 - fy);
				const float a01 = (1 - fx)*fy;
				const float a10 = fx*(1 - fy);
				const float a11 = fx*fy;
				T* pdst = pDst_row + x * Channels;
				const T* psrc00 = pSrc0_row + sx * Channels;
				const T* psrc10 = psrc00 + Channels;
				const T* psrc01 = pSrc1_row + sx * Channels;
				const T* psrc11 = psrc01 + Channels;
				for (int c = 0; c < Channels; c++)
					pdst[c] = T(a00 * psrc00[c] + a01 * psrc01[c] + a11 * psrc11[c] + a10 * psrc10[c]);
			}//end for x
		}//end for y
	}

	template<class T, int Channels, int alignBytes> 
	Image<T, Channels, alignBytes> imresizeBilinear(const Image<T, Channels, alignBytes>& src, int dstW, int dstH)
	{
		Image<T, Channels, alignBytes> dst;
		dst.create(dstW, dstH);
		imresizeBilinear(src, dst, dstW, dstH);
		return dst;
	}

	template<class T, int Channels, int alignBytes>
	void imresizeNearest(const Image<T, Channels, alignBytes>& src, Image<T, Channels, alignBytes>& dst, int dstW, int dstH)
	{
		if (dst.memoryOverlap(src))
		{
			LVG_LOG(lvg::LVG_LOG_ERROR, "src and dst cannot share memory");
			return;
		}
		if(dstW != dst.width() || dstH != dst.height())
			dst.create(dstW, dstH);
		const int srcH = src.height();
		const int srcW = src.width();
		if (dstW == 0 || dstH == 0 || srcW == 0 || srcH == 0)
			return;
		const float scalex = (float)srcW / (float)dstW;
		const float scaley = (float)srcH / (float)dstH;

		std::vector<int> x_ofs(dstW);
		for (int x = 0; x < dstW; x++)
			x_ofs[x] = std::min(int(x*scalex), srcW - 1)*Channels;

		for (int y = 0; y < dstH; y++)
		{
			T* pDst_row = dst.rowPtr(y);
			const T* pSrc_row = src.rowPtr(std::min(int(y*scaley), srcH - 1));
			int x = 0;
#ifdef CV_SIMD128
			if (sizeof(T)*Channels == 4)
			{
				v_int32x4 vsrc;
				for (; x < dstW - 3; x += 4)
				{
					int* pdst = (int*)(pDst_row) + x;
					const int* psrc0 = (int*)(pSrc_row + x_ofs[x + 0]);
					const int* psrc1 = (int*)(pSrc_row + x_ofs[x + 1]);
					const int* psrc2 = (int*)(pSrc_row + x_ofs[x + 2]);
					const int* psrc3 = (int*)(pSrc_row + x_ofs[x + 3]);
					vsrc = v_int32x4(*psrc0, *psrc1, *psrc2, *psrc3);
					v_store(pdst, vsrc);
				}//end for x
			}
#endif
			for (; x < dstW; x++)
			{
				T* pdst = pDst_row + x * Channels;
				const T* psrc = pSrc_row + x_ofs[x];
				for (int c = 0; c < Channels; c++)
					pdst[c] = psrc[c];
			}//end for x
		}//end for y
	}

	template<class T, int Channels, int alignBytes> 
	Image<T, Channels, alignBytes> imresizeNearest(const Image<T, Channels, alignBytes>& src, int dstW, int dstH)
	{
		Image<T, Channels, alignBytes> dst;
		dst.create(dstW, dstH);
		imresizeNearest(src, dst, dstW, dstH);
		return dst;
	}
	
	template<class T, int Channels, int alignBytes>
	void imresizeLanczos3(const Image<T, Channels, alignBytes>& src, Image<T, Channels, alignBytes>& dst, int dstW, int dstH)
	{
		if (dst.memoryOverlap(src))
		{
			LVG_LOG(lvg::LVG_LOG_ERROR, "src and dst cannot share memory");
			return;
		}
		if(dstW != dst.width() || dstH != dst.height())
			dst.create(dstW, dstH);
		if (dst.width() == 0 || dst.height() == 0 || src.width() == 0 || src.height() == 0)
			return;

		Image<float, Channels, alignBytes> imTmp;
		imTmp.create(dstW, src.height());
		dst.setZero();
		imTmp.setZero();

		//prepare weights and indices
		std::vector<std::vector<float>> weightsX, weightsY;
		std::vector<std::vector<int>> idxX, idxY;
		Lanczos3KernelIdx(src.width(), dst.width(), weightsX, idxX);
		Lanczos3KernelIdx(src.height(), dst.height(), weightsY, idxY);

		//x-direction scale
		for (int y = 0; y<imTmp.height(); y++)
		{
			const T* pSrc = src.rowPtr(y);
			float* pTmp = imTmp.rowPtr(y);
			for (int x = 0; x<imTmp.width(); x++)
			{
				const std::vector<float>& weights = weightsX[x];
				const std::vector<int>& idx = idxX[x];
				for (size_t j = 0; j<weights.size(); j++)
				{
					const int ij = idx[j];
					for (int c = 0; c < Channels; c++)
						pTmp[x*Channels + c] += pSrc[ij*Channels + c] * weights[j];
				}
			}//end for x
		}//end for y

		 //y-direction scale
		for (int x = 0; x<dstW; x++)
		{
			for (int y = 0; y<dstH; y++)
			{
				const std::vector<float>& weights = weightsY[y];
				const std::vector<int>& idx = idxY[y];

				typename Image<float, Channels, alignBytes>::VecType t = Image<float, Channels, alignBytes>::VecType::Zero();
				for (size_t j = 0; j<weights.size(); j++)
				{
					const typename Image<float, Channels, alignBytes>::VecType& pixelTmp = imTmp.pixel(Point(x, idx[j]));
					for (int c = 0; c < Channels; c++)
						t[c] += pixelTmp[c] * weights[j];
				}
				typename Image<T, Channels, alignBytes>::VecType& pixelDst = dst.pixel(Point(x, y));
				if (typeid(T) == typeid(uchar))
				{
					for (int c = 0; c < Channels; c++)
						pixelDst[c] = (T)std::max(0.f, std::min(255.f, t[c]));
				}
				else
				{
					for (int c = 0; c < Channels; c++)
						pixelDst[c] = (T)t[c];
				}
			}//end for y
		}//end for x
	}

	template<class T, int Channels, int alignBytes> 
	Image<T, Channels, alignBytes> imresizeLanczos3(const Image<T, Channels, alignBytes>& src, int dstW, int dstH)
	{
		Image<T, Channels, alignBytes> dst;
		dst.create(dstW, dstH);

		imresizeLanczos3(src, dst, dstW, dstH);

		return dst;
	}
	
	template<class T, int C, int alignBytes> Image<T, C, alignBytes> 
	imresizeT(const Image<T, C, alignBytes>& src, int dstW, int dstH, ResizeMethod m)
	{
		switch (m)
		{
		case lvg::ResizeLinear:
			return imresizeBilinear<T, C, alignBytes>(src, dstW, dstH);
		case lvg::ResizeNearest:
			return imresizeNearest<T, C, alignBytes>(src, dstW, dstH);
		case lvg::ResizeLanczos3:
			return imresizeLanczos3<T, C, alignBytes>(src, dstW, dstH);
		default:
			LVG_LOG(LVG_LOG_ERROR, "non supported resize method");
			return Image<T, C, alignBytes>();
		}
	}

	template<class T, int C, int alignBytes> void
	imresizeT(const Image<T, C, alignBytes>& src, Image<T, C, alignBytes>& dst, int dstW, int dstH, ResizeMethod m)
	{
		switch (m)
		{
		case lvg::ResizeLinear:
			return imresizeBilinear<T, C, alignBytes>(src, dst, dstW, dstH);
		case lvg::ResizeNearest:
			return imresizeNearest<T, C, alignBytes>(src, dst, dstW, dstH);
		case lvg::ResizeLanczos3:
			return imresizeLanczos3<T, C, alignBytes>(src, dst, dstW, dstH);
		default:
			LVG_LOG(LVG_LOG_ERROR, "non supported resize method");
			return;
		}
	}

	ByteImage imresize(const ByteImage& src, int dstW, int dstH, ResizeMethod m)
	{
		return imresizeT(src, dstW, dstH, m);
	}
	IntImage imresize(const IntImage& src, int dstW, int dstH, ResizeMethod m)
	{
		return imresizeT(src, dstW, dstH, m);
	}
	FloatImage imresize(const FloatImage& src, int dstW, int dstH, ResizeMethod m)
	{
		return imresizeT(src, dstW, dstH, m);
	}
	RgbImage imresize(const RgbImage& src, int dstW, int dstH, ResizeMethod m)
	{
		return imresizeT(src, dstW, dstH, m);
	}
	RgbaImage imresize(const RgbaImage& src, int dstW, int dstH, ResizeMethod m)
	{
		return imresizeT(src, dstW, dstH, m);
	}
	RgbFloatImage imresize(const RgbFloatImage& src, int dstW, int dstH, ResizeMethod m)
	{
		return imresizeT(src, dstW, dstH, m);
	}
	RgbaFloatImage imresize(const RgbaFloatImage& src, int dstW, int dstH, ResizeMethod m)
	{
		return imresizeT(src, dstW, dstH, m);
	}

	void imresize(const ByteImage& src, ByteImage& dst, int dstW, int dstH, ResizeMethod m)
	{
		imresizeT(src, dst, dstW, dstH, m);
	}
	void imresize(const IntImage& src, IntImage& dst, int dstW, int dstH, ResizeMethod m)
	{
		imresizeT(src, dst, dstW, dstH, m);
	}
	void imresize(const FloatImage& src, FloatImage& dst, int dstW, int dstH, ResizeMethod m)
	{
		imresizeT(src, dst, dstW, dstH, m);
	}
	void imresize(const RgbImage& src, RgbImage& dst, int dstW, int dstH, ResizeMethod m)
	{
		imresizeT(src, dst, dstW, dstH, m);
	}
	void imresize(const RgbaImage& src, RgbaImage& dst, int dstW, int dstH, ResizeMethod m)
	{
		imresizeT(src, dst, dstW, dstH, m);
	}
	void imresize(const RgbFloatImage& src, RgbFloatImage& dst, int dstW, int dstH, ResizeMethod m)
	{
		imresizeT(src, dst, dstW, dstH, m);
	}
	void imresize(const RgbaFloatImage& src, RgbaFloatImage& dst, int dstW, int dstH, ResizeMethod m)
	{
		imresizeT(src, dst, dstW, dstH, m);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	//RGB <--> Lab
	//////////////////////////////////////////////////////////////////////////////////////////////////
	static const float D65[] = { 0.950456f, 1.f, 1.088754f };

	static const float sRGB2XYZ_D65[] =
	{
		0.412453f, 0.357580f, 0.180423f,
		0.212671f, 0.715160f, 0.072169f,
		0.019334f, 0.119193f, 0.950227f
	};

	static const float XYZ2sRGB_D65[] =
	{
		3.240479f, -1.53715f, -0.498535f,
		-0.969256f, 1.875991f, 0.041556f,
		0.055648f, -0.204043f, 1.057311f
	};

	enum { LAB_CBRT_TAB_SIZE = 1024, GAMMA_TAB_SIZE = 1024 };
	static float LabCbrtTab[LAB_CBRT_TAB_SIZE * 4];
	static const float LabCbrtTabScale = LAB_CBRT_TAB_SIZE / 1.5f;

	static float sRGBGammaTab[GAMMA_TAB_SIZE * 4], sRGBInvGammaTab[GAMMA_TAB_SIZE * 4];
	static const float GammaTabScale = (float)GAMMA_TAB_SIZE;

#undef min
#undef max
#undef lab_shift
#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)

	/* ************************************************************************** *\
	Fast cube root by Ken Turkowski
	(http://www.worldserver.com/turk/computergraphics/papers.html)
	\* ************************************************************************** */
	typedef union Cv32suf
	{
		int i;
		unsigned u;
		float f;
	}Cv32suf;
	inline float cubeRoot(float value)
	{
		float fr;
		Cv32suf v, m;
		int ix, s;
		int ex, shx;

		v.f = value;
		ix = v.i & 0x7fffffff;
		s = v.i & 0x80000000;
		ex = (ix >> 23) - 127;
		shx = ex % 3;
		shx -= shx >= 0 ? 3 : 0;
		ex = (ex - shx) / 3; /* exponent of cube root */
		v.i = (ix & ((1 << 23) - 1)) | ((shx + 127) << 23);
		fr = v.f;

		/* 0.125 <= fr < 1.0 */
		/* Use quartic rational polynomial with error < 2^(-24) */
		fr = (float)(((((45.2548339756803022511987494 * fr +
			192.2798368355061050458134625) * fr +
			119.1654824285581628956914143) * fr +
			13.43250139086239872172837314) * fr +
			0.1636161226585754240958355063) /
			((((14.80884093219134573786480845 * fr +
				151.9714051044435648658557668) * fr +
				168.5254414101568283957668343) * fr +
				33.9905941350215598754191872) * fr +
				1.0));

		/* fr *= 2^ex * sign */
		m.f = value;
		v.f = fr;
		v.i = (v.i + (ex << 23) + s) & (m.i * 2 != 0 ? -1 : 0);
		return v.f;
	}
	// computes cubic spline coefficients for a function: (xi=i, yi=f[i]), i=0..n
	template<typename _Tp> static void splineBuild(const _Tp* f, int n, _Tp* tab)
	{
		_Tp cn = 0;
		int i;
		tab[0] = tab[1] = (_Tp)0;

		for (i = 1; i < n - 1; i++)
		{
			_Tp t = 3 * (f[i + 1] - 2 * f[i] + f[i - 1]);
			_Tp l = 1 / (4 - tab[(i - 1) * 4]);
			tab[i * 4] = l; tab[i * 4 + 1] = (t - tab[(i - 1) * 4 + 1])*l;
		}

		for (i = n - 1; i >= 0; i--)
		{
			_Tp c = tab[i * 4 + 1] - tab[i * 4] * cn;
			_Tp b = f[i + 1] - f[i] - (cn + c * 2)*(_Tp)0.3333333333333333;
			_Tp d = (cn - c)*(_Tp)0.3333333333333333;
			tab[i * 4] = f[i]; tab[i * 4 + 1] = b;
			tab[i * 4 + 2] = c; tab[i * 4 + 3] = d;
			cn = c;
		}
	}
	// interpolates value of a function at x, 0 <= x <= n using a cubic spline.
	template<typename _Tp> static inline _Tp splineInterpolate(_Tp x, const _Tp* tab, int n)
	{
		int ix = (int)std::floor(x);
		ix = std::min(std::max(ix, 0), n - 1);
		x -= ix;
		tab += ix * 4;
		return ((tab[3] * x + tab[2])*x + tab[1])*x + tab[0];
	}

	static void initLabTabs()
	{
		static bool initialized = false;
		if (!initialized)
		{
			float f[LAB_CBRT_TAB_SIZE + 1], g[GAMMA_TAB_SIZE + 1], ig[GAMMA_TAB_SIZE + 1], scale = 1.f / LabCbrtTabScale;
			int i;
			for (i = 0; i <= LAB_CBRT_TAB_SIZE; i++)
			{
				float x = i*scale;
				f[i] = x < 0.008856f ? x*7.787f + 0.13793103448275862f : cubeRoot(x);
			}
			splineBuild(f, LAB_CBRT_TAB_SIZE, LabCbrtTab);

			scale = 1.f / GammaTabScale;
			for (i = 0; i <= GAMMA_TAB_SIZE; i++)
			{
				float x = i*scale;
				g[i] = x <= 0.04045f ? x*(1.f / 12.92f) : (float)pow((double)(x + 0.055)*(1. / 1.055), 2.4);
				ig[i] = x <= 0.0031308 ? x*12.92f : (float)(1.055*pow((double)x, 1. / 2.4) - 0.055);
			}
			splineBuild(g, GAMMA_TAB_SIZE, sRGBGammaTab);
			splineBuild(ig, GAMMA_TAB_SIZE, sRGBInvGammaTab);
			initialized = true;
		}
	}

	struct RGB2Lab_f
	{
		typedef float channel_type;

		RGB2Lab_f(int _srccn, int blueIdx, const float* _coeffs,
			const float* _whitept, bool _srgb)
			: srccn(_srccn), srgb(_srgb)
		{
			volatile int _3 = 3;
			initLabTabs();

			if (!_coeffs) _coeffs = sRGB2XYZ_D65;
			if (!_whitept) _whitept = D65;
			float scale[] = { LabCbrtTabScale / _whitept[0], LabCbrtTabScale, LabCbrtTabScale / _whitept[2] };

			for (int i = 0; i < _3; i++)
			{
				coeffs[i * 3 + (blueIdx ^ 2)] = _coeffs[i * 3] * scale[i];
				coeffs[i * 3 + 1] = _coeffs[i * 3 + 1] * scale[i];
				coeffs[i * 3 + blueIdx] = _coeffs[i * 3 + 2] * scale[i];
				assert(coeffs[i * 3] >= 0 && coeffs[i * 3 + 1] >= 0 && coeffs[i * 3 + 2] >= 0 &&
					coeffs[i * 3] + coeffs[i * 3 + 1] + coeffs[i * 3 + 2] < 1.5f*LabCbrtTabScale);
			}
		}

		void operator()(const uchar* src, float* dst, int n) const
		{
			int i, scn = srccn;
			float gscale = GammaTabScale;
			const float* gammaTab = srgb ? sRGBGammaTab : 0;
			const static float inv255 = 1.f / 255.f;
			float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
				C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
				C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
			n *= 3;

			for (i = 0; i < n; i += 3, src += scn)
			{
				float R = src[0], G = src[1], B = src[2];
				R *= inv255; G *= inv255; B *= inv255;
				if (gammaTab)
				{
					R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
					G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
					B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
				}
				float fX = splineInterpolate(R*C0 + G*C1 + B*C2, LabCbrtTab, LAB_CBRT_TAB_SIZE) / 1.f;
				float fY = splineInterpolate(R*C3 + G*C4 + B*C5, LabCbrtTab, LAB_CBRT_TAB_SIZE) / 1.f;
				float fZ = splineInterpolate(R*C6 + G*C7 + B*C8, LabCbrtTab, LAB_CBRT_TAB_SIZE) / 1.f;

				float L = 116.f*fY - 16.f;
				float a = 500.f*(fX - fY);
				float b = 200.f*(fY - fZ);

				dst[i] = L; dst[i + 1] = a; dst[i + 2] = b;
			}
		}

		int srccn;
		float coeffs[9];
		bool srgb;
	};

	struct Lab2RGB_f
	{
		typedef float channel_type;

		Lab2RGB_f(int _dstcn, int blueIdx, const float* _coeffs,
			const float* _whitept, bool _srgb)
			: dstcn(_dstcn), srgb(_srgb)
		{
			initLabTabs();

			if (!_coeffs) _coeffs = XYZ2sRGB_D65;
			if (!_whitept) _whitept = D65;

			for (int i = 0; i < 3; i++)
			{
				coeffs[i + (blueIdx ^ 2) * 3] = _coeffs[i] * _whitept[i];
				coeffs[i + 3] = _coeffs[i + 3] * _whitept[i];
				coeffs[i + blueIdx * 3] = _coeffs[i + 6] * _whitept[i];
			}
		}

		void operator()(const float* src, uchar* dst, int n) const
		{
			int i, dcn = dstcn;
			const float* gammaTab = srgb ? sRGBInvGammaTab : 0;
			float gscale = GammaTabScale;
			float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
				C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
				C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
			n *= 3;

			for (i = 0; i < n; i += 3, dst += dcn)
			{
				float L = src[i], a = src[i + 1], b = src[i + 2];
				float Y = (L + 16.f)*(1.f / 116.f);
				float X = (Y + a*0.002f);
				float Z = (Y - b*0.005f);
				Y = Y*Y*Y;
				X = X*X*X;
				Z = Z*Z*Z;

				float R = X*C0 + Y*C1 + Z*C2;
				float G = X*C3 + Y*C4 + Z*C5;
				float B = X*C6 + Y*C7 + Z*C8;

				if (gammaTab)
				{
					R = splineInterpolate(R*gscale, gammaTab, GAMMA_TAB_SIZE);
					G = splineInterpolate(G*gscale, gammaTab, GAMMA_TAB_SIZE);
					B = splineInterpolate(B*gscale, gammaTab, GAMMA_TAB_SIZE);
				}

				dst[0] = (uchar)std::min(255.f, std::max(0.f, R*255.f + 0.5f));
				dst[1] = (uchar)std::min(255.f, std::max(0.f, G*255.f + 0.5f));
				dst[2] = (uchar)std::min(255.f, std::max(0.f, B*255.f + 0.5f));
			}
		}

		int dstcn;
		float coeffs[9];
		bool srgb;
	};

	void sRgb2Lab(const RgbImage& imgRgb, RgbFloatImage& imgLab)
	{
		if(imgLab.width() != imgRgb.width() || imgLab.height() != imgRgb.height())
			imgLab.create(imgRgb.width(), imgRgb.height());

		RGB2Lab_f converter(imgRgb.channels(), 0, NULL, NULL, true);

		const int nWidth = imgRgb.width();
		const int nHeight = imgRgb.height();

#ifdef ENABLE_OPENMP
#pragma omp parallel for
#endif
		for (int y = 0; y<nHeight; y++)
			converter((const uchar*)imgRgb.rowPtr(y), (float*)imgLab.rowPtr(y), nWidth);
	}

	void Lab2sRgb(const RgbFloatImage& imgLab, RgbImage& imgRgb)
	{
		if (imgLab.width() != imgRgb.width() || imgLab.height() != imgRgb.height())
			imgRgb.create(imgLab.width(), imgLab.height());

		Lab2RGB_f converter(imgRgb.channels(), 0, NULL, NULL, true);

		const int nWidth = imgRgb.width();
		const int nHeight = imgRgb.height();

#pragma omp parallel for
		for (int y = 0; y<nHeight; y++)
		{
			converter((const float*)imgLab.rowPtr(y), (uchar*)imgRgb.rowPtr(y), nWidth);
		}
	}

	void rgba2bgra(const RgbaImage& src, RgbaImage& dst)
	{
		if (!dst.sameShape(src))
			dst.create(src.width(), src.height());
		const int W = src.width(), H = src.height();
		const int n = W * 4;
		for (int y = 0; y < H; y++)
		{
			const uchar* pSrc = src.rowPtr(y);
			uchar* pDst = dst.rowPtr(y);
			int x = 0;
#if 0
			for (; x <= n - 64; x += 64)
			{
				uint8x16x4_t v_src = vld4q_u8(src + i), v_dst;
				v_dst.val[0] = v_src.val[2];
				v_dst.val[1] = v_src.val[1];
				v_dst.val[2] = v_src.val[0];
				v_dst.val[3] = v_src.val[3];
				vst4q_u8(dst + x, v_dst);
			}
			for (; x <= n - 32; x += 32)
			{
				uint8x8x4_t v_src = vld4_u8(src + i), v_dst;
				v_dst.val[0] = v_src.val[2];
				v_dst.val[1] = v_src.val[1];
				v_dst.val[2] = v_src.val[0];
				v_dst.val[3] = v_src.val[3];
				vst4_u8(dst + x, v_dst);
			}
#endif
			for (; x < n; x += 4)
			{
				uchar c0, c1, c2, c3;
				c0 = pSrc[x + 0];
				c1 = pSrc[x + 1];
				c2 = pSrc[x + 2];
				c3 = pSrc[x + 3];
				pDst[x + 2] = c0;
				pDst[x + 1] = c1;
				pDst[x + 0] = c2;
				pDst[x + 3] = c3;
			} // x
		} // y
	}

	void imtranspose(const FloatImage& src, FloatImage& dst)
	{
		if (dst.memoryOverlap(src))
		{
			LVG_LOG(lvg::LVG_LOG_ERROR, "src and dst cannot share memory");
			return;
		}
		const int sW = src.width();
		const int sH = src.height();

		if(dst.width() != sH || dst.height() != sW)
			dst.create(sH, sW);
		int dy = 0;

#ifdef CV_SIMD128
		v_float32x4 vsrc;
		for (; dy < sW - 3; dy += 4)
		{
			float* pDstRow0 = dst.rowPtr(dy + 0);
			float* pDstRow1 = dst.rowPtr(dy + 1);
			float* pDstRow2 = dst.rowPtr(dy + 2);
			float* pDstRow3 = dst.rowPtr(dy + 3);
			for (int dx = 0; dx < sH; dx++)
			{
				vsrc = v_load(src.rowPtr(dx) + dy);
				pDstRow0[dx] = vsrc.get<0>();
				pDstRow1[dx] = vsrc.get<1>();
				pDstRow2[dx] = vsrc.get<2>();
				pDstRow3[dx] = vsrc.get<3>();
			} // end for x
		} // end for dy
#endif
		for (; dy < sW; dy++)
		{
			float* pDstRow = dst.rowPtr(dy);
			for (int dx = 0; dx < sH; dx++)
			{
				pDstRow[dx] = src.rowPtr(dx)[dy];
			} // end for x
		} // end for dy
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// guided image filter
	/////////////////////////////////////////////////////////////////////////////////////////////////
	void fastGuidedFilter(FloatImage& img, const FloatImage& guideImg, int r, float eps, int s)
	{
		if (s <= 0)
			s = r / 4;
		const int r_s = r / s;
		const int ksize = r_s * 2 + 1;
		const int W = img.width(), H = img.height();
		const int W_s = W / s, H_s = H / s;
		if (W != guideImg.width() || H != guideImg.height())
		{
			LVG_LOGE("image and guide size not matched: (%d, %d) != (%d, %d)",
				W, H, guideImg.width(), guideImg.height());
			return;
		}
		const FloatImage& I = guideImg;
		FloatImage& P = img;
		FloatImage I_s = imresize(I, W_s, H_s);
		FloatImage P_s = imresize(P, W_s, H_s);

		FloatImage mean_I, mean_P;
		FloatImage mean_IP = I_s.clone();
		mean_IP *= P_s;

		ByteImage workBuffer;
		boxFilter(I_s, mean_I, ksize, &workBuffer);
		boxFilter(P_s, mean_P, ksize, &workBuffer);
		boxFilter(mean_IP, mean_IP, ksize, &workBuffer);

		FloatImage cov_IP;
		cov_IP.create(W_s, H_s);
		for (int y = 0; y < H_s; y++)
		{
			float* pCov = cov_IP.rowPtr(y);
			const float* pI = mean_I.rowPtr(y);
			const float* pP = mean_P.rowPtr(y);
			const float* pIP = mean_IP.rowPtr(y);
			for (int x = 0; x < W_s; x++)
				pCov[x] = pIP[x] - pI[x] * pP[x];
		} // y

		FloatImage mean_II = I_s.clone();
		mean_II *= I_s;
		boxFilter(mean_II, mean_II, ksize, &workBuffer);

		FloatImage val_I, A, B;
		val_I.create(W_s, H_s);
		A.create(W_s, H_s);
		B.create(W_s, H_s);
		for (int y = 0; y < H_s; y++)
		{
			const float* pI = mean_I.rowPtr(y);
			const float* pP = mean_P.rowPtr(y);
			const float* pCov = cov_IP.rowPtr(y);
			float* pII = mean_II.rowPtr(y);
			float* pVal = val_I.rowPtr(y);
			float* pA = A.rowPtr(y);
			float* pB = B.rowPtr(y);
			for (int x = 0; x < W_s; x++)
			{
				const float vI = pI[x];
				pVal[x] = pII[x] - vI * vI;
				pA[x] = pCov[x] / (pVal[x] + eps);
				pB[x] = pP[x] - pA[x] * pI[x];
			} // x
		} // y

		boxFilter(A, A, ksize, &workBuffer);
		boxFilter(B, B, ksize, &workBuffer);
		A = imresize(A, W, H);
		B = imresize(B, W, H);

		for (int y = 0; y < H; y++)
		{
			const float* pA = A.rowPtr(y);
			const float* pB = B.rowPtr(y);
			const float* pI = I.rowPtr(y);
			float* pImg = img.rowPtr(y);
			int x = 0;
#ifdef CV_SIMD128
			for (; x < W - 3; x += 4)
			{
				lvg::v_store(pImg + x, lvg::v_muladd(lvg::v_load(pA+x), lvg::v_load(pI + x), lvg::v_load(pB + x)));
			} // x
#endif
			for (; x < W; x++)
			{
				pImg[x] = pA[x] * pI[x] + pB[x];
			} // x
		} // y
	}
}