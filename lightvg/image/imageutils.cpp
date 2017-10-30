#include "imageutils.h"
#include "lightvg\common\logger.h"
#include "convhelper.h"
namespace lvg
{
	void copyMakeBorderReflect(const ByteImage& src, ByteImage& dst, int bl, int br, int bt, int bb)
	{
		if (src.channels() != 1) {
			LVG_LOG(LVG_LOG_ERROR, "only one-channel image supported!");
			return;
		}
		const int sW = src.width();
		const int sH = src.height();
		const int dW = sW + bl + br;
		const int dH = sH + bt + bb;

		if (bl >= sW || br >= sW || bt >= sH || bb >= sH) {
			LVG_LOG(LVG_LOG_ERROR, "not supported border, too large than image size!");
			return;
		}
		if (dst.width() != dW || dst.height() != dH || dst.channels() != src.channels())
			dst.create(dW, dH);

		const int bt_sH2 = bt + (sH - 1) * 2;
		const int bl_sW2 = bl + (sW - 1) * 2;

		// copy middle
		for (int dy = 0; dy < dH; dy++)
		{
			int sy = 0;

			// top rows
			if (dy < bt)
				sy = bt - dy;
			// middle rows
			else if (dy < bt + sH)
				sy = dy - bt;
			// bottom rows
			else
				sy = bt_sH2 - dy;

			unsigned char* dstRowPtr = dst.rowPtr(dy);
			const unsigned char* srcRowPtr = src.rowPtr(sy);

			// left
			for (int dx = 0; dx < bl; dx++)
				dstRowPtr[dx] = srcRowPtr[bl - dx];

			// middle
			memcpy(dstRowPtr + bl, srcRowPtr, sW * sizeof(unsigned char));

			// right
			for (int dx = bl + sW; dx < dW; dx++)
				dstRowPtr[dx] = srcRowPtr[bl_sW2 - dx];
		} // end for dy
	}

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

				v_uint8x16 result(vgetq_lane_u32(vsum[0].val, 0), vgetq_lane_u32(vsum[0].val, 1), vgetq_lane_u32(vsum[0].val, 2), vgetq_lane_u32(vsum[0].val, 3),
					vgetq_lane_u32(vsum[1].val, 0), vgetq_lane_u32(vsum[1].val, 1), vgetq_lane_u32(vsum[1].val, 2), vgetq_lane_u32(vsum[1].val, 3),
					vgetq_lane_u32(vsum[2].val, 0), vgetq_lane_u32(vsum[2].val, 1), vgetq_lane_u32(vsum[2].val, 2), vgetq_lane_u32(vsum[2].val, 3),
					vgetq_lane_u32(vsum[3].val, 0), vgetq_lane_u32(vsum[3].val, 1), vgetq_lane_u32(vsum[3].val, 2), vgetq_lane_u32(vsum[3].val, 3));
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

				v_uint8x16 result(vgetq_lane_u32(vsum[0].val, 0), vgetq_lane_u32(vsum[0].val, 1), vgetq_lane_u32(vsum[0].val, 2), vgetq_lane_u32(vsum[0].val, 3),
					vgetq_lane_u32(vsum[1].val, 0), vgetq_lane_u32(vsum[1].val, 1), vgetq_lane_u32(vsum[1].val, 2), vgetq_lane_u32(vsum[1].val, 3),
					vgetq_lane_u32(vsum[2].val, 0), vgetq_lane_u32(vsum[2].val, 1), vgetq_lane_u32(vsum[2].val, 2), vgetq_lane_u32(vsum[2].val, 3),
					vgetq_lane_u32(vsum[3].val, 0), vgetq_lane_u32(vsum[3].val, 1), vgetq_lane_u32(vsum[3].val, 2), vgetq_lane_u32(vsum[3].val, 3));
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

				v_uint8x16 result(vgetq_lane_u32(vsum[0].val, 0), vgetq_lane_u32(vsum[0].val, 1), vgetq_lane_u32(vsum[0].val, 2), vgetq_lane_u32(vsum[0].val, 3),
					vgetq_lane_u32(vsum[1].val, 0), vgetq_lane_u32(vsum[1].val, 1), vgetq_lane_u32(vsum[1].val, 2), vgetq_lane_u32(vsum[1].val, 3),
					vgetq_lane_u32(vsum[2].val, 0), vgetq_lane_u32(vsum[2].val, 1), vgetq_lane_u32(vsum[2].val, 2), vgetq_lane_u32(vsum[2].val, 3),
					vgetq_lane_u32(vsum[3].val, 0), vgetq_lane_u32(vsum[3].val, 1), vgetq_lane_u32(vsum[3].val, 2), vgetq_lane_u32(vsum[3].val, 3));
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

				v_uint8x16 result(vgetq_lane_u32(vsum[0].val, 0), vgetq_lane_u32(vsum[0].val, 1), vgetq_lane_u32(vsum[0].val, 2), vgetq_lane_u32(vsum[0].val, 3),
					vgetq_lane_u32(vsum[1].val, 0), vgetq_lane_u32(vsum[1].val, 1), vgetq_lane_u32(vsum[1].val, 2), vgetq_lane_u32(vsum[1].val, 3),
					vgetq_lane_u32(vsum[2].val, 0), vgetq_lane_u32(vsum[2].val, 1), vgetq_lane_u32(vsum[2].val, 2), vgetq_lane_u32(vsum[2].val, 3),
					vgetq_lane_u32(vsum[3].val, 0), vgetq_lane_u32(vsum[3].val, 1), vgetq_lane_u32(vsum[3].val, 2), vgetq_lane_u32(vsum[3].val, 3));
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
}