#pragma once

#include "lightvg\image\imageutils.h"
#include <vector>
#include <numeric>


namespace lvg
{
	class CBasicPatchMatchOutputFieldInt;
	class CBasicPatchMatchOutputFieldFloat;

	class PatchMatchCompletion
	{
		typedef Image<CBasicPatchMatchOutputFieldInt, 1> ImageANNInt;
		typedef Image<CBasicPatchMatchOutputFieldFloat, 1> ImageANNFloat;
		typedef ImageANNFloat ImageANN;
		typedef RgbFloatImage ImageLabFloat;
	public:
		PatchMatchCompletion();
		~PatchMatchCompletion();

		void completion(const RgbImage &imSrc, const ByteImage& imMask, RgbImage& imDst, int nPatchSize = 7, int nMaxLevel = 10);

	protected:

		void privateCompletion(const RgbImage &imSrc, const ByteImage& imMask, RgbImage& imDst, int nPatchSize, int nMaxLevel);

		//init approximate nearest neighbor field with random value in given mask
		bool randomANN(const ByteImage& imPatchMask, ImageANN& annField)const;

		//smoothingly fill image holes, perform poisson image on given src image restricted in given mask
		void fillSmooth(RgbImage &imSrc, const ByteImage& imMask)const;

		//given coarse level (L1) ANN field and mask in dense level (L), fill ANN field of level (L) with super-sampling
		void upSample(const ImageANN &annSrcL1, const ByteImage& imPatchMaskL, ImageANN &annDstL, float2 scale)const;

		//perform Iterate() and Average() iteratively
		void pass(int level, ImageLabFloat &imSrcExt, const ByteImage& imMask, const ByteImage& imPatchMask,
			ImageANN &annField, const FloatImage& distField, bool freeze_result)const;

		//call after each average when images are changed
		void updateANNValues(ImageLabFloat &imSrcExt, const ByteImage& imPatchMask, ImageANN &annField)const;

		//perform Patch-Match algorithm
		void iterate(ImageLabFloat &imSrcExt, const ByteImage& imPatchMask, ImageANN &annField, int downDir)const;

		//one-pixel in Iterate()
		void iterateOnePixel(ImageLabFloat &imSrcExt, const ByteImage& imPatchMask, ImageANN &annField,
			int xt, int yt, int ofs)const;

		//same with Average, but the input src image must be boundary extended version
		void averageExt(ImageLabFloat &imSrcExt, const ByteImage& imMask, const ImageANN &annField, const FloatImage& distField)const;

		//one-pixel in Average()
		void averageOnePixelExt(ImageLabFloat &imSrcExt, const ImageANN &annFieldExt,
			const FloatImage& weightsExt, int xt, int yt)const;

		//whether ANN of a given point(xt, yt) can be updated to (xnew, ynew)
		void tryUpdate(const ImageLabFloat& imSrcExt, const ByteImage& imPatchMask, int& xbest, int& ybest, float& dbest,
			int xt, int yt, int xnew, int ynew)const;

		//distance of two patches, src1 and src2 should be the left-top corner.
		static float patchDist(const float3* src1, int stride1, const float3* src2, int stride2, int patchSize, float cutoff = FLT_MAX);

		//build gauss pyramid with given level and scalar
		void buildLanczos3Pyramid();

		Rect computeHoleBoundingBox(const ByteImage& patchMask);

		//call each beginning of Completion
		void clear();
	protected:
		//input image
		RgbImage m_imSrc;

		//input mask
		ByteImage m_imMask;

		//image pyramid
		std::vector<ImageLabFloat> m_srcPyramidLab;
		std::vector<float2> m_pyramidScales;

		//mask pyramid
		std::vector<ByteImage> m_maskPyramid;

		//expanded mask pyramid, if one patch has HOLE pixels, it will be marked as HOLE
		std::vector<ByteImage> m_patchMaskPyramid;

		//computed ANN field
		std::vector<ImageANN> m_annFieldPyramid;

		//computed the nearest distance from one HOLE pixel to any NOT_HOLE pixels
		std::vector<FloatImage> m_distFiledPyramid;

		//HOLE bounding box of each level
		std::vector<Rect> m_patchHoleBoundingBoxes;

		//size of each patch, 2*m_nPatchRadius+1
		int m_nPatchSize;
		int m_nPatchRadius;

		//max level of image pyramid
		int m_nMaxLevel;

		//current pyramid iteration level in Completion().
		int m_nCurrentLevel;

		const static double SCALE_PER_LEVEL;

		//time statistics, for debug
		mutable double m_timeIterate;
		mutable double m_timeAverage;
		mutable double m_timeUpSample;
		mutable double m_timeOthers;
	};//class CBasicPatchMatch

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class CBasicPatchMatchOutputFieldFloat
	{
	private:
		unsigned short m_x, m_y;
		float m_value;
	public:
		CBasicPatchMatchOutputFieldFloat() :m_x(0), m_y(0), m_value(0) {}
		void InitMax() { m_x = 0xffff; m_y = 0xffff; m_value = FLT_MAX; }
		unsigned int X()const { return (unsigned int)m_x; }
		unsigned int Y()const { return (unsigned int)m_y; }
		void SetXY(unsigned int x, unsigned int y) { assert(x < 0xffff && y < 0xffff); m_x = (unsigned short)x; m_y = (unsigned short)y; }
		void GetXY(int& x, int & y)const { x = X(); y = Y(); }
		float Value()const { return m_value; }
		void SetValue(float v) { m_value = v; }
		void SetXYValue(unsigned int x, unsigned int y, float v) { SetXY(x, y); SetValue(v); }
	};

	class CBasicPatchMatchOutputFieldInt
	{
	private:
		unsigned short m_x, m_y;
		int m_value;
	public:
		CBasicPatchMatchOutputFieldInt() :m_x(0), m_y(0), m_value(0) {}
		void InitMax() { m_x = 0xffff; m_y = 0xffff; m_value = 0x7fffffff; }
		unsigned int X()const { return (unsigned int)m_x; }
		unsigned int Y()const { return (unsigned int)m_y; }
		void SetXY(unsigned int x, unsigned int y) { assert(x < 0xffff && y < 0xffff); m_x = (unsigned short)x; m_y = (unsigned short)y; }
		void GetXY(int& x, int & y)const { x = X(); y = Y(); }
		int Value()const { return m_value; }
		void SetValue(int v) { m_value = v; }
		void SetXYValue(unsigned int x, unsigned int y, int v) { SetXY(x, y); SetValue(v); }
	};

}