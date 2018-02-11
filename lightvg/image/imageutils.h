#ifndef __LVGIMAGEUTILS_H__
#define __LVGIMAGEUTILS_H__

#include "Image.h"

namespace lvg
{
	// 5x5 Gaussian blur, neon optimized
	void gaussianBlur_5x5(const ByteImage& src, ByteImage& dst, std::vector<int>* workBuffer = nullptr);

	// 5x5 Gaussian blur, neon optimized
	void gaussianBlur_7x7(const ByteImage& src, ByteImage& dst, std::vector<int>* workBuffer = nullptr);

	// fast corner detector, borrowed from OpenCV
	void fastCornerDetect(const ByteImage& src, std::vector<KeyPoint>& keyPoints,
		int threshold, bool bNonmaxSurpress, std::vector<unsigned char>* workBuffer = nullptr);
	void keyPointsFilter_retainBest(std::vector<KeyPoint>& keypoints, int n_points);

	// separable 2d convolution, corresponding to conv2('same') of matlab
	void separableConv2(const FloatImage& src, FloatImage& dst, const float* kernel, int nKernel);

	// max filter, often used for dilation
	void maxFilter(const FloatImage& src, FloatImage& dst, int nKernel);
	void maxFilter(const ByteImage& src, ByteImage& dst, int nKernel);
	void maxFilter(const IntImage& src, IntImage& dst, int nKernel);
	void minFilter(const FloatImage& src, FloatImage& dst, int nKernel);
	void minFilter(const ByteImage& src, ByteImage& dst, int nKernel);
	void minFilter(const IntImage& src, IntImage& dst, int nKernel);

	void boxFilter(const ByteImage& src, ByteImage& dst, int nKernel, ByteImage* workBuffer=nullptr);
	void boxFilter(const RgbImage& src, RgbImage& dst, int nKernel, ByteImage* workBuffer = nullptr);
	void boxFilter(const RgbaImage& src, RgbaImage& dst, int nKernel, ByteImage* workBuffer = nullptr);
	void boxFilter(const FloatImage& src, FloatImage& dst, int nKernel, ByteImage* workBuffer = nullptr);
	void boxFilter(const RgbFloatImage& src, RgbFloatImage& dst, int nKernel, ByteImage* workBuffer = nullptr);
	void boxFilter(const RgbaFloatImage& src, RgbaFloatImage& dst, int nKernel, ByteImage* workBuffer = nullptr);

	// guided image filter of Kaiming, details see the original paper
	// img: input/output image
	// guideImg: the guidence image
	// r: local window radius, similar with the spatial threshold of bilateral
	// eps: similar with the color threshold of bilateral, try 0.04 or 0.01
	// s: subsampling ratio to speed up. try r or r/4, if s < 0, then s=r/4
	void fastGuidedFilter(FloatImage& img, const FloatImage& guideImg, int r, float eps, int s = -1);

	// similar with Matlab's bwdist()
	// for mask > 128, compute the pixel distances to the boundary (mask < 128)
	FloatImage bwdist(const ByteImage& imMask);

	// resizing
	enum ResizeMethod
	{
		ResizeNearest,
		ResizeLinear,
		ResizeLanczos3,
	};
	ByteImage imresize(const ByteImage& src, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	IntImage imresize(const IntImage& src, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	FloatImage imresize(const FloatImage& src, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	RgbImage imresize(const RgbImage& src, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	RgbaImage imresize(const RgbaImage& src, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	RgbFloatImage imresize(const RgbFloatImage& src, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	RgbaFloatImage imresize(const RgbaFloatImage& src, int dstW, int dstH, ResizeMethod m = ResizeLinear);

	void imresize(const ByteImage& src, ByteImage& dst, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	void imresize(const IntImage& src, IntImage& dst, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	void imresize(const FloatImage& src, FloatImage& dst, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	void imresize(const RgbImage& src, RgbImage& dst, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	void imresize(const RgbaImage& src, RgbaImage& dst, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	void imresize(const RgbFloatImage& src, RgbFloatImage& dst, int dstW, int dstH, ResizeMethod m = ResizeLinear);
	void imresize(const RgbaFloatImage& src, RgbaFloatImage& dst, int dstW, int dstH, ResizeMethod m = ResizeLinear);

	// color conversion
	void sRgb2Lab(const RgbImage& imgRgb, RgbFloatImage& imgLab);
	void Lab2sRgb(const RgbFloatImage& imgLab, RgbImage& imgRgb);

	// rgba <-> bgra, neon/sse optimized
	void rgba2bgra(const RgbaImage& src, RgbaImage& dst);

	// transpose
	void imtranspose(const FloatImage& src, FloatImage& dst);
} // lvg

#endif