#pragma once

#include "Image.h"

namespace lvg
{
	// copy src to dst(bl:br, bt:bb), and the border is mirrored from center
	void copyMakeBorderReflect(const ByteImage& src, ByteImage& dst, int bl, int br, int bt, int bb);

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
} // lvg