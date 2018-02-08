#define LDP_ENABLE_OPENCV_DEBUG
#include "ConvolutionPyramid.h"
#include "convhelper.h"
#include "lightvg/common/mathutils.h"
#include "lightvg/common/logger.h"

namespace lvg
{
#undef min
#undef max
	ConvolutionPyramid::ConvolutionPyramid()
	{
	}

	ConvolutionPyramid::~ConvolutionPyramid()
	{
	}

	void ConvolutionPyramid::convolveBoundary(FloatImage& srcDst)
	{
		const float kernel5x5[5] = { 0.1507f, 0.6836f, 1.0334f, 0.6836f, 0.1507f };
		const float kernel3x3[3] = { 0.0312f, 0.7753f, 0.0312f };
		const float mul = sqrt(0.0270f);
		const float kernel5x5up[5] = { mul * 0.1507f, mul * 0.6836f, mul * 1.0334f, mul * 0.6836f, mul * 0.1507f };

		PyramidConvolve(srcDst, kernel5x5, kernel3x3, kernel5x5up);
	}

	void ConvolutionPyramid::solvePoisson(FloatImage& I, const FloatImage& dX, const FloatImage& dY)
	{
		const float kernel5x5[5] = { 0.15f, 0.5f, 0.7f, 0.5f, 0.15f };
		const float kernel3x3[3] = { 0.175f, 0.547f, 0.175f };

		ComputeDivergence(I, dX, dY, -1.f);
		PyramidConvolve(I, kernel5x5, kernel3x3, kernel5x5);
	}

	void ConvolutionPyramid::blendImage(ColorImage& dst, const ColorImage& src, const MaskImage& mask)
	{
		const int W = src.width();
		const int H = src.height();

		std::vector<FloatImage> srcChannels, dstChannels;
		SeparateChannels(srcChannels, src);
		SeparateChannels(dstChannels, dst);

		FloatImage boundary;
		MaskToBoundary(boundary, mask);
#pragma omp parallel for
		for (int i = 0; i < int(srcChannels.size()); i++)
		{
			AddImage(dstChannels[i], srcChannels[i], 1.f, -1.f);
			MultImage(dstChannels[i], boundary);
		}

#pragma omp parallel for
		for (int i = 0; i < int(dstChannels.size() + 1); i++)
		{
			if (i < int(dstChannels.size()))
				convolveBoundary(dstChannels[i]);
			else
				convolveBoundary(boundary);
		}

#pragma omp parallel for
		for (int i = 0; i < int(srcChannels.size()); i++)
		{
			DivImage(dstChannels[i], boundary);
			AddImage(srcChannels[i], dstChannels[i]);
		}

		MergeChannels(dst, srcChannels, mask);
	}

	void ConvolutionPyramid::blendImage(RgbFloatImage& dst, const RgbFloatImage& src, const MaskImage& mask)
	{
		const int W = src.width();
		const int H = src.height();

		std::vector<FloatImage> srcChannels, dstChannels;
		SeparateChannels(srcChannels, src);
		SeparateChannels(dstChannels, dst);

		FloatImage boundary;
		MaskToBoundary(boundary, mask);
#pragma omp parallel for
		for (int i = 0; i < int(srcChannels.size()); i++)
		{
			AddImage(dstChannels[i], srcChannels[i], 1.f, -1.f);
			MultImage(dstChannels[i], boundary);
		}

#pragma omp parallel for
		for (int i = 0; i < int(dstChannels.size() + 1); i++)
		{
			if (i < int(dstChannels.size()))
				convolveBoundary(dstChannels[i]);
			else
				convolveBoundary(boundary);
		}

#pragma omp parallel for
		for (int i = 0; i < int(srcChannels.size()); i++)
		{
			DivImage(dstChannels[i], boundary);
			AddImage(srcChannels[i], dstChannels[i]);
		}

		MergeChannels(dst, srcChannels, mask);
	}

	void ConvolutionPyramid::fillHole(ColorImage& srcDst, const MaskImage& mask)
	{
		const int W = srcDst.width();
		const int H = srcDst.height();

		std::vector<FloatImage> srcChannels;
		SeparateChannels(srcChannels, srcDst);

		FloatImage boundary;
		MaskToBoundary(boundary, mask);
		for (int i = 0; i < int(srcChannels.size()); i++)
			MultImage(srcChannels[i], boundary);

#pragma omp parallel for
		for (int i = 0; i < int(srcChannels.size() + 1); i++)
		{
			if (i < int(srcChannels.size()))
				convolveBoundary(srcChannels[i]);
			else
				convolveBoundary(boundary);
		}

		for (int i = 0; i < int(srcChannels.size()); i++)
			DivImage(srcChannels[i], boundary);

		MergeChannels(srcDst, srcChannels, mask);
	}

	void ConvolutionPyramid::fillHole(RgbFloatImage& srcDst, const MaskImage& mask)
	{
		const int W = srcDst.width();
		const int H = srcDst.height();

		std::vector<FloatImage> srcChannels;
		SeparateChannels(srcChannels, srcDst);

		FloatImage boundary;
		MaskToBoundary(boundary, mask);
		for (int i = 0; i < int(srcChannels.size()); i++)
			MultImage(srcChannels[i], boundary);

#pragma omp parallel for
		for (int i = 0; i < int(srcChannels.size() + 1); i++)
		{
			if (i < int(srcChannels.size()))
				convolveBoundary(srcChannels[i]);
			else
				convolveBoundary(boundary);
		}

		for (int i = 0; i < int(srcChannels.size()); i++)
			DivImage(srcChannels[i], boundary);

		MergeChannels(srcDst, srcChannels, mask);
	}

	void ConvolutionPyramid::fillHole(FloatImage& srcDst, const MaskImage& mask)
	{
		FloatImage boundary, src;
		src = srcDst.clone();
		MaskToBoundary(boundary, mask);
		MultImage(srcDst, boundary);

#pragma omp parallel for
		for (int i = 0; i < 2; i++)
		{
			if (i == 0)
				convolveBoundary(srcDst);
			else
				convolveBoundary(boundary);
		}
		DivImage(srcDst, boundary);

		// keep the origonal non-masked part
		const int W = srcDst.width();
		const int H = srcDst.height();
		for (int y = 0; y < H; y++)
		{
			float* pdst = srcDst.rowPtr(y);
			const uchar* pmask = mask.rowPtr(y);
			const float* psrc = src.rowPtr(y);
			for (int x = 0; x < W; x++)
			{
				if (pmask[x] < 128)
					pdst[x] = psrc[x];
			} // x
		} // y
	}

	void ConvolutionPyramid::PyramidConvolve(FloatImage& srcDst, const float* kernel5x5,
		const float* kernel3x3, const float* kernel5x5up)
	{
		const int nMaxLevel = (int)(ceil(log((float)std::max(srcDst.width(), srcDst.height()))) / log(2.0f));

		std::vector<FloatImage> pyramid(nMaxLevel);
		pyramid.resize(nMaxLevel);
		pyramid[0] = srcDst;

		/// down------------------------------------------------------------------
		for (int i = 1; i < nMaxLevel; i++)
		{
			FloatImage imConv = pyramid[i - 1].zeroPadding(PAD_SZ);
			conv2<float, 5>(imConv.data(), kernel5x5, imConv.width(), imConv.height(), imConv.stride());
			DownSamplex2(pyramid[i], imConv);
		}//i
		
		 /// up------------------------------------------------------------------------
		 // on the coarse level
		 // imCurrent = conv(pad(imCurrent), g_3x3)
		FloatImage imCurrent = pyramid[nMaxLevel - 1].zeroPadding(PAD_SZ);
		conv2<float, 3>(imCurrent.data(), kernel3x3, imCurrent.width(), imCurrent.height(), imCurrent.stride());
		
		for (int i = nMaxLevel - 2; i >= 0; i--)
		{
			// imTmpDown = unpad(imCurrent)
			FloatImage imTmpDown = imCurrent.range(PAD_SZ, imCurrent.height() - PAD_SZ, PAD_SZ, imCurrent.width() - PAD_SZ);

			// imTmpUp = conv(upscale(imTmpDown), h2_5x5)
			FloatImage imTmpUp;
			imTmpUp.create(pyramid[i].width() + PAD_SZ * 2, pyramid[i].height() + PAD_SZ * 2);
			VolumeUpscalex2_ZeroHalf(imTmpUp, imTmpDown);
			conv2<float, 5>(imTmpUp.data(), kernel5x5up, imTmpUp.width(), imTmpUp.height(), imTmpUp.stride());

			// imCurrent = conv(pad(imCurrent), g_3x3)
			imCurrent = pyramid[i].zeroPadding(5);
			conv2<float, 3>(imCurrent.data(), kernel3x3, imCurrent.width(), imCurrent.height(), imCurrent.stride());

			// imCurrent += imTmpUp
			AddImage(imCurrent, imTmpUp);
		}

		// unpad
		srcDst.copyFrom(imCurrent.range(PAD_SZ, imCurrent.height() - PAD_SZ, PAD_SZ, imCurrent.width() - PAD_SZ));
	}

	// A = alpha * A + beta * B
	void ConvolutionPyramid::AddImage(FloatImage& A, const FloatImage& B, float alpha, float beta)
	{
		const int W = A.width();
		const int H = A.height();
		if (W != B.width() || H != B.height())
		{
			LVG_LOG(LVG_LOG_ERROR, "size mis-matched");
			return;
		}

#ifdef CV_SIMD128
		v_float32x4 valpha, vbeta, va, vb;
		valpha = v_setall_f32(alpha);
		vbeta = v_setall_f32(beta);
#endif

		for (int y = 0; y < H; y++)
		{
			float* pA = A.rowPtr(y);
			const float* pB = B.rowPtr(y);
			int x = 0;
#ifdef CV_SIMD128
			for (; x < W - 3; x += 4)
			{
				va = v_load(pA + x);
				vb = v_load(pB + x);
				va = valpha * va + vbeta * vb;
				v_store(pA + x, va);
			}
#endif
			for (; x < W; x++)
				pA[x] = alpha * pA[x] + beta * pB[x];
		} // y 
	}

	// A = alpha * A + (1-alpha)*B
	void ConvolutionPyramid::BlendImage(FloatImage& A, const FloatImage& B, const FloatImage& alpha)
	{
		const int W = A.width();
		const int H = A.height();
		if (W != B.width() || H != B.height())
		{
			LVG_LOG(LVG_LOG_ERROR, "size mis-matched");
			return;
		}

#ifdef CV_SIMD128
		v_float32x4 valpha, va, vb, vone;
		vone = v_setall_f32(1.f);
#endif
		for (int y = 0; y < H; y++)
		{
			float* pA = A.rowPtr(y);
			const float* pB = B.rowPtr(y);
			const float* pAlpha = alpha.rowPtr(y);
			int x = 0;
#ifdef CV_SIMD128
			for (; x < W - 3; x += 4)
			{
				va = v_load(pA + x);
				vb = v_load(pB + x);
				valpha = v_load(pAlpha + x);
				va = valpha * va + (vone - valpha) * vb;
				v_store(pA + x, va);
			}
#endif
			for (; x < W; x++)
				pA[x] = pAlpha[x] * pA[x] + (1 - pAlpha[x]) * pB[x];
		} // y 
	}

	// A = alpha * A * B + beta
	void ConvolutionPyramid::MultImage(FloatImage& A, const FloatImage& B, float alpha, float beta)
	{
		const int W = A.width();
		const int H = A.height();
		if (W != B.width() || H != B.height())
		{
			LVG_LOG(LVG_LOG_ERROR, "size mis-matched");
			return;
		}
#ifdef CV_SIMD128
		v_float32x4 valpha, vbeta, va, vb;
		valpha = v_setall_f32(alpha);
		vbeta = v_setall_f32(beta);
#endif

		for (int y = 0; y < H; y++)
		{
			float* pA = A.rowPtr(y);
			const float* pB = B.rowPtr(y);
			int x = 0;
#ifdef CV_SIMD128
			for (; x < W - 3; x += 4)
			{
				va = v_load(pA + x);
				vb = v_load(pB + x);
				va = valpha * va * vb + vbeta;
				v_store(pA + x, va);
			}
#endif
			for (; x < W; x++)
				pA[x] = alpha * pA[x] * pB[x] + beta;
		} // y 
	}

	// A = alpha * A / B + beta
	void ConvolutionPyramid::DivImage(FloatImage& A, const FloatImage& B, float alpha, float beta)
	{
		const int W = A.width();
		const int H = A.height();
		if (W != B.width() || H != B.height())
		{
			LVG_LOG(LVG_LOG_ERROR, "size mis-matched");
			return;
		}
#ifdef CV_SIMD128
		v_float32x4 valpha, vbeta, va, vb;
		valpha = v_setall_f32(alpha);
		vbeta = v_setall_f32(beta);
#endif

		for (int y = 0; y < H; y++)
		{
			float* pA = A.rowPtr(y);
			const float* pB = B.rowPtr(y);
			int x = 0;
#ifdef CV_SIMD128
			for (; x < W - 3; x += 4)
			{
				va = v_load(pA + x);
				vb = v_load(pB + x);
				va = valpha * va / vb + vbeta;
				v_store(pA + x, va);
			}
#endif
			for (; x < W; x++)
				pA[x] = alpha * pA[x] / pB[x] + beta;
		} // y 
	}

	void ConvolutionPyramid::DownSamplex2(FloatImage& imDst, const FloatImage& imSrc)
	{
		if (imDst.memoryOverlap(imSrc))
		{
			LVG_LOG(LVG_LOG_ERROR, "does not support inplace operation");
			return;
		}

		const int lW = imSrc.width() / 2;
		const int lH = imSrc.height() / 2;

		imDst.create(lW, lH);

		for (int y = 0; y < lH; y++)
		{
			const float* src_y_ptr = imSrc.rowPtr(y * 2);
			float* dst_y_ptr = imDst.rowPtr(y);
			int x = 0;
#ifdef CV_SIMD128
			for (; x < lW - 3; x += 4)
			{
				int s = (x << 1);
				v_float32x4 va(src_y_ptr[s], src_y_ptr[s + 2], src_y_ptr[s + 4], src_y_ptr[s + 6]);
				v_store(dst_y_ptr + x, va);
			}
#endif
			for (; x < lW; x++)
				dst_y_ptr[x] = src_y_ptr[x * 2];
		}// end for y
	}

	void ConvolutionPyramid::VolumeUpscalex2_ZeroHalf(FloatImage& imDst, const FloatImage& imSrc)
	{
		const int sW = imSrc.width();
		const int sH = imSrc.height();
		const int dW = imDst.width();
		const int dH = imDst.height();

		if (dW / 2 != sW || dH / 2 != sH)
		{
			LVG_LOG(LVG_LOG_ERROR, "illegal size");
			return;
		}

		for (int y = 0; y < sH; y++)
		{
			const float* src_y_ptr = imSrc.rowPtr(y);
			float* dst_y_ptr = imDst.rowPtr(y * 2);
			memset(imDst.rowPtr(y * 2 + 1), 0, sizeof(float)*imDst.width()*imDst.channels());
			for (int x = 0; x < sW; x++)
			{
				int x2 = (x << 1);
				dst_y_ptr[x2] = src_y_ptr[x];
				dst_y_ptr[x2 + 1] = 0;
			}// end for x
		}// end for y
	}

	void ConvolutionPyramid::SeparateChannels(std::vector<FloatImage>& imgChannels, const ColorImage& img)
	{
		const int sW = img.width();
		const int sH = img.height();
		imgChannels.resize(img.channels());
		for (int c = 0; c < ColorImage::ChannelNum; c++)
			imgChannels[c].create(sW, sH);
#pragma omp parallel for
		for (int y = 0; y < sH; y++)
		{
			const unsigned char* imgRow = img.rowPtr(y);
			float* cptrs[ColorImage::ChannelNum];
			for (int c = 0; c < ColorImage::ChannelNum; c++)
				cptrs[c] = imgChannels[c].rowPtr(y);
			for (int x = 0; x < sW; x++)
			{
				for (int c = 0; c < ColorImage::ChannelNum; c++)
					cptrs[c][x] = float(imgRow[c]) / float(255.f);
				imgRow += ColorImage::ChannelNum;
			} // x
		} // y
	}

	void ConvolutionPyramid::SeparateChannels(std::vector<FloatImage>& imgChannels, const RgbFloatImage& img)
	{
		const int sW = img.width();
		const int sH = img.height();
		imgChannels.resize(img.channels());
		for (int c = 0; c < ColorImage::ChannelNum; c++)
			imgChannels[c].create(sW, sH);
#pragma omp parallel for
		for (int y = 0; y < sH; y++)
		{
			const float* imgRow = img.rowPtr(y);
			float* cptrs[ColorImage::ChannelNum];
			for (int c = 0; c < ColorImage::ChannelNum; c++)
				cptrs[c] = imgChannels[c].rowPtr(y);
			for (int x = 0; x < sW; x++)
			{
				for (int c = 0; c < ColorImage::ChannelNum; c++)
					cptrs[c][x] = imgRow[c];
				imgRow += ColorImage::ChannelNum;
			} // x
		} // y
	}

	void ConvolutionPyramid::MergeChannels(ColorImage& img, const std::vector<FloatImage>& imgChannels, const MaskImage& mask)
	{
		if (imgChannels.size() != ColorImage::ChannelNum)
		{
			LVG_LOG(LVG_LOG_ERROR, "channel num not matched!");
			return;
		}
		const int sW = img.width();
		const int sH = img.height();
#pragma omp parallel for
		for (int y = 0; y < sH; y++)
		{
			const unsigned char* maskRow = mask.rowPtr(y);
			unsigned char* imgRow = img.rowPtr(y);
			const float* cptrs[ColorImage::ChannelNum];
			for (int c = 0; c < ColorImage::ChannelNum; c++)
				cptrs[c] = imgChannels[c].rowPtr(y);
			for (int x = 0; x < sW; x++)
			{
				if (maskRow[x] > 128)
					for (int c = 0; c < ColorImage::ChannelNum; c++)
						imgRow[c] = (unsigned char)std::max(0.f, std::min(255.f, cptrs[c][x] * 255.f));
				imgRow += ColorImage::ChannelNum;
			} // x
		} // y
	}

	void ConvolutionPyramid::MergeChannels(RgbFloatImage& img, const std::vector<FloatImage>& imgChannels, const MaskImage& mask)
	{
		if (imgChannels.size() != ColorImage::ChannelNum)
		{
			LVG_LOG(LVG_LOG_ERROR, "channel num not matched!");
			return;
		}
		const int sW = img.width();
		const int sH = img.height();
#pragma omp parallel for
		for (int y = 0; y < sH; y++)
		{
			const unsigned char* maskRow = mask.rowPtr(y);
			float* imgRow = img.rowPtr(y);
			const float* cptrs[ColorImage::ChannelNum];
			for (int c = 0; c < ColorImage::ChannelNum; c++)
				cptrs[c] = imgChannels[c].rowPtr(y);
			for (int x = 0; x < sW; x++)
			{
				if (maskRow[x] > 128)
					for (int c = 0; c < ColorImage::ChannelNum; c++)
						imgRow[c] = cptrs[c][x];
				imgRow += ColorImage::ChannelNum;
			} // x
		} // y
	}

	void ConvolutionPyramid::MaskToBoundary(FloatImage& boundary, const MaskImage& mask)
	{
		const int sW = mask.width();
		const int sH = mask.height();
		boundary.create(mask.width(), mask.height());

		for (int y = 0; y < sH; y++)
		{
			const unsigned char* maskRow = mask.rowPtr(y);
			float* bdRow = boundary.rowPtr(y);
			for (int x = 0; x < sW; x++)
				bdRow[x] = float(maskRow[x] > 128);
		} // y

		max_filter2<float, 3>(boundary.data(), sW, sH, boundary.stride());

		for (int y = 0; y < sH; y++)
		{
			const unsigned char* maskRow = mask.rowPtr(y);
			float* bdRow = boundary.rowPtr(y);
			for (int x = 0; x < sW; x++)
				bdRow[x] -= float(maskRow[x] > 128);
		} // y
	}

	void ConvolutionPyramid::MaskToFloat(FloatImage& maskF, const MaskImage& mask)
	{
		const int sW = mask.width();
		const int sH = mask.height();
		maskF.create(mask.width(), mask.height());

		for (int y = 0; y < sH; y++)
		{
			const unsigned char* maskRow = mask.rowPtr(y);
			float* bdRow = maskF.rowPtr(y);
			for (int x = 0; x < sW; x++)
				bdRow[x] = float(maskRow[x] > 128);
		} // y
	}

	// dX = imfilter(I, [1 -1 0])
	// dY = imfilter(I, [1 -1 -1]')
	void ConvolutionPyramid::ComputeGradient(FloatImage& dX, FloatImage& dY, const FloatImage& I)
	{
		const int W = I.width();
		const int H = I.height();
		dX.create(W, H);
		dY.create(W, H);

		for (int y = 1; y < H; y++)
		{
			const float* pI_prev = I.rowPtr(y - 1);
			const float* pI = I.rowPtr(y);
			float* pX = dX.rowPtr(y);
			float* pY = dY.rowPtr(y);
			for (int x = 1; x < W; x++)
			{
				pX[x] = pI[x - 1] - pI[x];
				pY[x] = pI_prev[x] - pI[x];
			}
		}

		for (int x = 0; x < W; x++)
			dY.rowPtr(0)[x] = -I.rowPtr(0)[x];

		for (int y = 0; y < H; y++)
			dX.rowPtr(y)[0] = -I.rowPtr(y)[0];
	}

	// G = imfilter(dx_f, [0 1 -1]) + imfilter(dy_f, [0 1 -1]'); 
	void ConvolutionPyramid::ComputeDivergence(FloatImage& G, const FloatImage& dX, const FloatImage& dY, float alpha)
	{
		const int W = dX.width();
		const int H = dX.height();
		if (W != dY.width() || H != dY.height())
		{
			LVG_LOG(LVG_LOG_ERROR, "size not matched!");
			return;
		}
		G.create(W, H);
		G.create(W, H);

		for (int y = 0; y < H - 1; y++)
		{
			const float* pX = dX.rowPtr(y);
			const float* pY = dY.rowPtr(y);
			const float* pY_1 = dY.rowPtr(y + 1);
			float* pG = G.rowPtr(y);
			for (int x = 0; x < W - 1; x++)
				pG[x] = alpha*(pX[x] - pX[x + 1] + pY[x] - pY_1[x]);
		}

		for (int x = 0; x < W - 1; x++)
			G.rowPtr(H - 1)[x] = alpha * (dX.rowPtr(H - 1)[x] - dX.rowPtr(H - 1)[x + 1] + dY.rowPtr(H - 1)[x]);

		for (int y = 0; y < H - 1; y++)
			G.rowPtr(y)[W - 1] = alpha * (dX.rowPtr(y)[W - 1] + dY.rowPtr(y)[W - 1] - dY.rowPtr(y + 1)[W - 1]);

		G.rowPtr(H - 1)[W - 1] = alpha * (dX.rowPtr(H - 1)[W - 1] + dY.rowPtr(H - 1)[W - 1]);
	}

	float ConvolutionPyramid::ComputeMean(const FloatImage& I)
	{
		const int W = I.width();
		const int H = I.height();
		double sum = 0;
		for (int y = 0; y < H; y++)
		{
			const float* pI = I.rowPtr(y);
			for (int x = 0; x < W; x++)
				sum += pI[x];
		}
		sum /= double(W*H);
		return float(sum);
	}
}