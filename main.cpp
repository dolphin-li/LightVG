#include <stdio.h>
#include "lightvg/LightVG.hpp"
#include <opencv2/opencv.hpp>

#define DEBUG_GUIDED_FILTER
//#define DEBUG_BOUNDARY_INTERPOLATE
//#define DEBUG_POISSON
//#define DEBUG_RESIZE
//#define DEBUG_CONVERT_COLOR
//#define DEBUG_PATCHMATCH
//#define DEBUG_CONV

int main()
{
#ifdef DEBUG_GUIDED_FILTER
	{
		lvg::ByteImage src;
		lvg::ByteImage workBuf;
		lvg::FloatImage srcf;
		lvg::imread("r.jpg", src);
		imwrite("result_aref.png", src);
		src.convertTo(srcf, 1.f / 255.f);

		lvg::tic();
		lvg::fastGuidedFilter(srcf, srcf, 4, 0.01f, 4);
		lvg::toc();

		srcf *= 255.f;
		imwrite("result.png", srcf);
	}
#endif
#ifdef DEBUG_BOUNDARY_INTERPOLATE
	{
		RgbImage src;
		ByteImage mask;
		imread("shi.png", src);
		imread("shi_mask.png", mask);
		
		mask *= -1;
		mask += 255;
		//lvg::maxFilter(mask, mask, 5);

		tic();
		ConvolutionPyramid conv;
		conv.fillHole(src, mask);
		toc();

		imwrite("result.png", src);
}
#endif

#ifdef DEBUG_POISSON
	{
		RgbImage src, dst;
		ByteImage mask;
		imread("debug.jpg", src);
		tic();
		ConvolutionPyramid conv;
		src = src.zeroPadding(1);
		std::vector<FloatImage> srcChannels;
		conv.SeparateChannels(srcChannels, src);
#pragma omp parallel for
		for (int i = 0; i < int(srcChannels.size()); i++)
		{
			const float meanI = conv.ComputeMean(srcChannels[i]);
			FloatImage dX, dY;
			conv.ComputeGradient(dX, dY, srcChannels[i]);
			conv.solvePoisson(srcChannels[i], dX, dY);
			const float meanG = conv.ComputeMean(srcChannels[i]);
			srcChannels[i] += meanI - meanG;
		}
		mask.create(src.width(), src.height());
		mask.setConstant(255);
		conv.MergeChannels(src, srcChannels, mask);
		dst = src.range(1, src.height() - 1, 1, src.width() - 1);
		toc();
		imwrite("result.png", dst);
}
#endif

#ifdef DEBUG_RESIZE
	{
		RgbImage img;
		ByteImage mask;
		imread("debug.jpg", img);

		tic();
		img = imresize(img, 212, 212, ResizeLanczos3);
		toc();

		imwrite("result.jpg", img);
	}
#endif

#ifdef DEBUG_CONVERT_COLOR
	{
		RgbImage img;
		ByteImage mask;
		RgbFloatImage lab;
		imread("debug.jpg", img);

		tic();
		sRgb2Lab(img, lab);
		Lab2sRgb(lab, img);
		toc();

		imwrite("result.jpg", img);
	}
#endif

#ifdef DEBUG_PATCHMATCH
	RgbImage img;
	ByteImage mask;
	imread("a3.png", img);
	imread("a2.png", mask);
	//maxFilter(mask, mask, 5);
	mask *= -1;
	mask += 255;

	tic();
	PatchMatchCompletion pc;
	pc.completion(img, mask, img);
	toc();

	imwrite("result.jpg", img);
#endif

#ifdef DEBUG_CONV
	FloatImage img, dst;
	img.create(1024, 1024);
	img.setZero();
	dst.create(img.width(), img.height());
	dst.setZero();

	enum {N = 11};
	float kernel[N] = { 0 };

	lvg::tic();
	for (int i = 0; i < 100; i++)
		lvg::separableConv2(img, dst, kernel, N);
	lvg::toc();
#endif

	return 0;
}