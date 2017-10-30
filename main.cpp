#include <stdio.h>
#include "lightvg\LightVG.hpp"
using namespace lvg;

int main()
{	
	RgbImage src, dst;
	ByteImage mask;
	imread("a3.png", src);
	imread("a1.jpg", dst);
	imread("a2.png", mask);
	mask *= -1;
	mask += 255;


	ConvolutionPyramid conv;
#if 1
	tic();
	conv.blendImage(dst, src, mask);
#else
	tic();
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
#endif
	toc();

	imwrite("result.png", dst);
	return 0;
}