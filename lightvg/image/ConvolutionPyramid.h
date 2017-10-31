#pragma once
#include "Image.h"
namespace lvg
{
	class ConvolutionPyramid
	{
	public:
		typedef ByteImage MaskImage;
		typedef RgbImage ColorImage;
		enum {PAD_SZ = 5};
	public:
		ConvolutionPyramid();
		~ConvolutionPyramid();

	public:
		// dst(mask) = src, then blend
		void blendImage(ColorImage& dst, const ColorImage& src, const MaskImage& mask);
		void blendImage(RgbFloatImage& dst, const RgbFloatImage& src, const MaskImage& mask);

		// dst(mask) = interpolate
		void fillHole(ColorImage& srcDst, const MaskImage& mask);
		void fillHole(RgbFloatImage& srcDst, const MaskImage& mask);

		// boundary interpolation
		// assume the given volume has values at the boundary and 0 inside.
		// then this method smoothly interpolate those boundary values inward
		void convolveBoundary(FloatImage& srcDst);

		// solve a poisson equation with Neumann boundary conditions
		// given the gridents dX, dY, solve for image I
		void solvePoisson(FloatImage& I, const FloatImage& dX, const FloatImage& dY);
	public:
		//imDst = imSrc conv Kernel (Kernel is implicitly defined by 
		//	kernel5x5 (h1), kernel3x3 (g)), and kernel5x5up (h2);
		static void PyramidConvolve(FloatImage& srcDst, const float* kernel5x5,
			const float* kernel3x3, const float* kernel5x5up);

		// imDst(p) = imSrc(p*2)
		static void DownSamplex2(FloatImage& imDst, const FloatImage& imSrc);

		// imDst(p*2) = imSrc(p), imDst(p*2+1) = 0;
		// NOTE: imDst should be pre-allocated
		static void VolumeUpscalex2_ZeroHalf(FloatImage& imDst, const FloatImage& imSrc);

		// A = alpha * A + beta * B
		static void AddImage(FloatImage& A, const FloatImage& B, float alpha = 1.f, float beta = 1.f);

		// A = (A - B) * C
		static void SubMultImage(FloatImage& A, const FloatImage& B, const FloatImage& C);

		// A = alpha * A + (1-alpha)*B;
		static void BlendImage(FloatImage& A, const FloatImage& B, const FloatImage& alpha);

		// A = alpha * A * B + beta
		static void MultImage(FloatImage& A, const FloatImage& B, float alpha = 1.f, float beta = 0.f);

		// A = alpha * A / B + beta
		static void DivImage(FloatImage& A, const FloatImage& B, float alpha = 1.f, float beta = 0.f);

		static void MaskToBoundary(FloatImage& boundary, const MaskImage& mask);

		static void MaskToFloat(FloatImage& maskF, const MaskImage& mask);

		static void SeparateChannels(std::vector<FloatImage>& imgChannels, const RgbFloatImage& img);
		static void SeparateChannels(std::vector<FloatImage>& imgChannels, const ColorImage& img);

		static void MergeChannels(RgbFloatImage& img, const std::vector<FloatImage>& imgChannels, const MaskImage& mask);
		static void MergeChannels(ColorImage& img, const std::vector<FloatImage>& imgChannels, const MaskImage& mask);

		// dX = imfilter(I, [1 -1 0])
		// dY = imfilter(I, [1 -1 -1]')
		static void ComputeGradient(FloatImage& dX, FloatImage& dY, const FloatImage& I);

		// G = alpha * imfilter(dx_f, [0 1 -1]) + imfilter(dy_f, [0 1 -1]'); 
		static void ComputeDivergence(FloatImage& G, const FloatImage& dX, const FloatImage& dY, float alpha = 1.f);

		static float ComputeMean(const FloatImage& I);
	};
}