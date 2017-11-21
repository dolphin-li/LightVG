#include "PatchMatchCompletion.h"
#include "lightvg/common/logger.h"
#include "lightvg/common/timeutils.h"
#include "lightvg/image/ConvolutionPyramid.h"
#include "lightvg/image/opencvdebug.h"

#define INFLATION_RATIO 3	//size of contex around the hole
#define IS_HOLE(val) ((val) > 128)
#define IS_NOT_HOLE(val) (!IS_HOLE(val))
//#define VOTING_ENABLE_MEANSHIFT

namespace lvg
{
	typedef PatchMatchCompletion::SrcVecType SrcVecType;
	typedef PatchMatchCompletion::PixelDifType PixelDifType;

	const double PatchMatchCompletion::SCALE_PER_LEVEL = 0.5;

	PatchMatchCompletion::PatchMatchCompletion()
	{

	}

	PatchMatchCompletion::~PatchMatchCompletion()
	{

	}

	void PatchMatchCompletion::clear()
	{
		m_imSrc.release();
		m_imMask.release();
		m_srcPyramidLab.clear();
		m_maskPyramid.clear();
		m_patchMaskPyramid.clear();
		m_annFieldPyramid.clear();
		m_distFiledPyramid.clear();
		m_patchHoleBoundingBoxes.clear();
		m_nPatchSize = 0;
		m_nPatchRadius = 0;
		m_nMaxLevel = 0;
		m_nCurrentLevel = 0;
		m_nMinIterEachLevel = 0;

		//for debug
		m_timeIterate = 0;
		m_timeAverage = 0;
		m_timeUpSample = 0;
		m_timeOthers = 0;
	}

	void PatchMatchCompletion::completion(const RgbImage &imSrc, const ByteImage& imMask, RgbImage& imDst,
		int minIterEachLevel, int nPatchSize, int nMaxLevel)
	{
		//check input
		if (imSrc.width() != imMask.width() || imSrc.height() != imMask.height())
		{
			LVG_LOG(LVG_LOG_ERROR, "src and mask size not matched!");
			return;
		}

		//clear
		clear();

		//select sub-image
		Rect subRect = computeHoleBoundingBox(imMask);
		if (subRect.empty())
			return;
		else
			subRect = subRect.inflate(3).intersect(imSrc.rect());

		const int inf = INFLATION_RATIO;
		subRect = subRect.inflate(subRect.width * inf, subRect.height * inf).intersect(imMask.rect());
		imDst = imSrc.clone();
		RgbImage imSubDst = imDst.range(subRect);
		RgbImage imSubColor = imSrc.range(subRect);
		ByteImage imSubMask = imMask.range(subRect);

		//inpainting
		privateCompletion(imSubColor, imSubMask, imSubDst, minIterEachLevel, nPatchSize, nMaxLevel);
	}

	void PatchMatchCompletion::privateCompletion(const RgbImage &imSrc, const ByteImage& imMask, RgbImage& imDst,
		int minIterEachLevel, int nPatchSize, int nMaxLevel)
	{
		clear();

		m_imSrc = imSrc;
		m_imMask = imMask;
		m_nPatchSize = nPatchSize;
		m_nMinIterEachLevel = minIterEachLevel;
		m_nPatchRadius = m_nPatchSize / 2;
		m_nPatchSize = m_nPatchRadius * 2 + 1;

		int tmpMinSize = std::min(imSrc.width(), imSrc.height());
		while (tmpMinSize*pow(SCALE_PER_LEVEL, double(nMaxLevel - 1)) < std::max(32, 2 * nPatchSize + 1))
			nMaxLevel--;
		m_nMaxLevel = nMaxLevel;
		srand(0);

		//init image pyramid
		buildLanczos3Pyramid();
		LVG_LOG(LVG_LOG_VERBOSE, "pyramid built");

		//main loop, pyramid down
		int startLevel = m_nMaxLevel - 1;
		for (m_nCurrentLevel = m_nMaxLevel - 1; m_nCurrentLevel >= 0; m_nCurrentLevel--)
		{
			SrcImageType &imgLab = m_srcPyramidLab[m_nCurrentLevel];
			ByteImage &mask = m_maskPyramid[m_nCurrentLevel];
			ByteImage &patchMask = m_patchMaskPyramid[m_nCurrentLevel];
			ImageANN &annField = m_annFieldPyramid[m_nCurrentLevel];
			FloatImage &distField = m_distFiledPyramid[m_nCurrentLevel];

			SrcImageType imgLabExt;

			//update hole values
			if (m_nCurrentLevel == startLevel)
			{
				if (!randomANN(patchMask, annField))
				{
					startLevel--;	//prevent the possibility that there are no known pathes in current level
					continue;
				}
				fillSmooth(imgLab, mask);
				imgLabExt = imgLab.mirrorPadding(m_nPatchRadius);
				pass(m_nCurrentLevel, imgLabExt, mask, patchMask, annField, distField, false);
			}//end if level
			else
			{
				upSample(m_annFieldPyramid[m_nCurrentLevel + 1], patchMask, annField, m_pyramidScales[m_nCurrentLevel + 1]);
				imgLabExt = imgLab.mirrorPadding(m_nPatchRadius);
				updateANNValues(imgLabExt, patchMask, annField);
				averageExt(imgLabExt, mask, annField, distField);
			}//end else

			pass(m_nCurrentLevel, imgLabExt, mask, patchMask, annField, distField, true);
			Rect rt = Rect(m_nPatchRadius, m_nPatchRadius, imgLab.width(), imgLab.height());
			imgLab.copyFrom(imgLabExt.range(rt));
			LVG_LOG(LVG_LOG_VERBOSE, std::string("level processed: ")+std::to_string(m_nCurrentLevel));
		}//end for level

		printf("Time Statistics: Iterate(%f), Average(%f), UpSample(%f), Others(%f)\n", m_timeIterate, m_timeAverage, m_timeUpSample, m_timeOthers);

		//write back result
		Lab2sRgb(m_srcPyramidLab[0], imDst);
	}

	Rect PatchMatchCompletion::computeHoleBoundingBox(const ByteImage& patchMask)
	{
		Rect rc(patchMask.width(), patchMask.height(), 0, 0);
		for (int y = 0; y < patchMask.height(); y++)
		{
			const uchar* pMask = patchMask.rowPtr(y);
			for (int x = 0; x < patchMask.width(); x++)
			{
				if (IS_HOLE(pMask[x]))
				{
					rc.left = std::min(rc.left, x);
					rc.top = std::min(rc.top, y);
					rc.width = std::max(rc.width, x);
					rc.height = std::max(rc.height, y);
				}
			}//end for x
		}//end for y
		rc.width = rc.width - rc.left + 1;
		rc.height = rc.height - rc.top + 1;
		return rc;
	}

	bool PatchMatchCompletion::randomANN(const ByteImage& imPatchMask, ImageANN& annField)const
	{
		gtime_t t1 = gtime_now();

		if (annField.width() != imPatchMask.width() || annField.height() != imPatchMask.height())
			annField.create(imPatchMask.width(), imPatchMask.height());
		const int nHeight = imPatchMask.height();
		const int nWidth = imPatchMask.width();
		std::vector<Point> nonHolePoints;
		for (int y = 0; y < nHeight; y++)
		{
			const uchar* ptr = imPatchMask.rowPtr(y);
			for (int x = 0; x < nWidth; x++)
			{
				if (IS_NOT_HOLE(ptr[x]))
					nonHolePoints.push_back(Point(x, y));
			}
		} // y
		if (nonHolePoints.size() == 0)
			return false;

		for (int y = 0; y < nHeight; y++)
		{
			const uchar* ptr = imPatchMask.rowPtr(y);
			CBasicPatchMatchOutputFieldFloat* pAnn = annField.rowPtr(y);
			for (int x = 0; x < nWidth; x++)
			{
				if (IS_NOT_HOLE(ptr[x]))
					pAnn[x].SetXYValue(x, y, 0);
				else
				{
					int idx = rand() % nonHolePoints.size();
					pAnn[x].SetXYValue(nonHolePoints[idx][0], nonHolePoints[idx][1], FLT_MAX);
				}//end else
			}//end for x
		}//end for y

		m_timeOthers += gtime_seconds(t1, gtime_now());
		return true;
	}

	void PatchMatchCompletion::fillSmooth(SrcImageType &imSrc, const ByteImage& imMask)const
	{
		gtime_t t1 = gtime_now();

		//poisson init
		SrcImageType tSrc = imSrc.mirrorPadding(1);
		ByteImage tMask = imMask.mirrorPadding(1);
		ConvolutionPyramid cpy;
		cpy.fillHole(tSrc, tMask);
		imSrc.copyFrom(tSrc.range(Rect(1, 1, imSrc.width(), imSrc.height())));

		m_timeOthers += gtime_seconds(t1, gtime_now());
	}

	void PatchMatchCompletion::upSample(const ImageANN &annSrc, const ByteImage& imPatchMask, ImageANN &annDst, float2 scale)const
	{
		gtime_t t1 = gtime_now();

		//UpSample The OffSet
		const int nWidth = annDst.width();
		const int nHeight = annDst.height();
		for (int yt = 0; yt < nHeight; yt++)
		{
			for (int xt = 0; xt < nWidth; xt++)
			{
				if (IS_NOT_HOLE(imPatchMask.rowPtr(yt)[xt]))
				{
					annDst.rowPtr(yt)[xt].SetXYValue(xt, yt, 0);
					continue;
				}

				int xs = 0, ys = 0;
				int validSrcXT = std::max(0, std::min(nWidth - 1, int(xt / scale.x() - 0.5f*(1.f - 1.f / scale.x()) + 0.5f)));
				int validSrcYT = std::max(0, std::min(nHeight - 1, int(yt / scale.y() - 0.5f*(1.f - 1.f / scale.y()) + 0.5f)));
				annSrc.rowPtr(validSrcYT)[validSrcXT].GetXY(xs, ys);

				xs = std::max(0, std::min(nWidth - 1, int(xt - scale.x()*(validSrcXT - xs) + 0.5)));
				ys = std::max(0, std::min(nHeight - 1, int(yt - scale.y()*(validSrcYT - ys) + 0.5)));

				if (xs >= 0 && ys >= 0 && IS_NOT_HOLE(imPatchMask.rowPtr(ys)[xs]))
				{
					annDst.rowPtr(yt)[xt].SetXYValue(xs, ys, FLT_MAX);
				}
				else
				{
					while (IS_HOLE(imPatchMask.rowPtr(ys)[xs]))
					{
						xs = rand() % annDst.width();
						ys = rand() % annDst.height();
					}//end while
					annDst.rowPtr(yt)[xt].SetXYValue(xs, ys, FLT_MAX);
				}
			}//end for x
		}//end for y

		m_timeUpSample += gtime_seconds(t1, gtime_now());
	}

	void PatchMatchCompletion::pass(int level, SrcImageType &imSrcExt, const ByteImage& imMask,
		const ByteImage& imPatchMask, ImageANN &annField, const FloatImage& distField, bool freeze_result)const
	{
		const static int nIter = 4;
		const int nPasses = std::max(m_nMinIterEachLevel, 30 - 10 * (m_nMaxLevel - level - 1));

		for (int pass = 0; pass < nPasses; pass++)
		{
			updateANNValues(imSrcExt, imPatchMask, annField);
			for (int k = 0; k < nIter; k++)
				iterate(imSrcExt, imPatchMask, annField, (k % 2 == 0));

			if (freeze_result)
			{
				averageExt(imSrcExt, imMask, annField, distField);
				gtime_t t1 = gtime_now();
				Rect rt(m_nPatchRadius, m_nPatchRadius, imSrcExt.width() - 2 * m_nPatchRadius, imSrcExt.height() - 2 * m_nPatchRadius);
				imSrcExt.range(rt).mirrorPadding(imSrcExt, m_nPatchRadius);
				m_timeOthers += gtime_seconds(t1, gtime_now());
			}
		}//end for pass
	}

	void PatchMatchCompletion::updateANNValues(SrcImageType &imSrcExt, const ByteImage& imPatchMask, ImageANN &annField)const
	{
		Rect rc = m_patchHoleBoundingBoxes[m_nCurrentLevel];
		int ybegin = rc.top, yend = rc.top + rc.height;
		int xbegin = rc.left, xend = rc.left + rc.width;

#pragma omp parallel for
		for (int yt = ybegin; yt < yend; yt++)
		{
			const uchar* pMask = imPatchMask.rowPtr(yt);
			CBasicPatchMatchOutputFieldFloat* pAnn = annField.rowPtr(yt);
			for (int xt = xbegin; xt < xend; xt++)
			{
				if (IS_NOT_HOLE(pMask[xt]))
					continue;
				int xbest = 0, ybest = 0;
				pAnn[xt].GetXY(xbest, ybest);

				//here we use boundary extended image, thus add offset.
				float dbest = patchDist((const SrcVecType*)imSrcExt.rowPtr(yt) + xt, imSrcExt.stride(),
					(const SrcVecType*)imSrcExt.rowPtr(ybest) + xbest, imSrcExt.stride(), m_nPatchSize);

				pAnn[xt].SetValue(dbest);
			}
		} // yt
	}

	void PatchMatchCompletion::iterate(SrcImageType &imSrc, const ByteImage& imPatchMask, ImageANN &annField, int downDir)const
	{
		gtime_t t1 = gtime_now();

		int ofs = -1;
		Rect rc = m_patchHoleBoundingBoxes[m_nCurrentLevel];
		int ybegin = rc.top, yend = rc.top + rc.height;
		int xbegin = rc.left, xend = rc.left + rc.width;

		if (!downDir)
		{
			ofs = 1;
			ybegin = rc.top + rc.height - 1;
			yend = rc.top - 1;
			xbegin = rc.left + rc.width - 1;
			xend = rc.left - 1;
		}

		if (downDir)
		{
#pragma omp parallel for
			for (int yt = ybegin; yt < yend; ++yt)
				for (int xt = xbegin; xt < xend; ++xt)
					iterateOnePixel(imSrc, imPatchMask, annField, xt, yt, ofs);
		}
		else
		{
#pragma omp parallel for
			for (int yt = ybegin; yt > yend; --yt)
				for (int xt = xbegin; xt > xend; --xt)
					iterateOnePixel(imSrc, imPatchMask, annField, xt, yt, ofs);
		}


		m_timeIterate += gtime_seconds(t1, gtime_now());
	}

	void PatchMatchCompletion::iterateOnePixel(SrcImageType &imSrc, const ByteImage& imPatchMask,
		ImageANN &annField, int xt, int yt, int ofs)const
	{
		if (IS_NOT_HOLE(imPatchMask.rowPtr(yt)[xt]))
			return;

		const int nHeight = imSrc.height() - 2 * m_nPatchRadius;
		const int nWidth = imSrc.width() - 2 * m_nPatchRadius;

		int xbest = 0, ybest = 0;
		float dbest = annField.rowPtr(yt)[xt].Value();
		annField.rowPtr(yt)[xt].GetXY(xbest, ybest);

		if ((xt + ofs >= 0 && xt + ofs < nWidth))
		{
			if (IS_HOLE(imPatchMask.rowPtr(yt)[xt + ofs]))
			{
				int xnew = 0, ynew = 0;
				annField.rowPtr(yt)[xt + ofs].GetXY(xnew, ynew);
				xnew -= ofs;
				if (xnew >= 0 && xnew < nWidth)
					tryUpdate(imSrc, imPatchMask, xbest, ybest, dbest, xt, yt, xnew, ynew);
			}
		}
		if ((yt + ofs >= 0 && yt + ofs < nHeight))
		{
			if (IS_HOLE(imPatchMask.rowPtr(yt + ofs)[xt]))
			{
				int xnew = 0, ynew = 0;
				annField.rowPtr(yt + ofs)[xt].GetXY(xnew, ynew);
				ynew -= ofs;
				if (ynew >= 0 && ynew < nHeight)
					tryUpdate(imSrc, imPatchMask, xbest, ybest, dbest, xt, yt, xnew, ynew);
			}
		}

		const int xbest0 = xbest;
		const int ybest0 = ybest;
		for (int rad = std::max(nWidth, nHeight); rad; rad /= 2)
		{
			int xmin = std::max(xbest0 - rad, 0), ymin = std::max(ybest0 - rad, 0);
			int xmax = std::min(xbest0 + rad + 1, nWidth), ymax = std::min(ybest0 + rad + 1, nHeight);
			int xnew = rand() % (xmax - xmin) + xmin;
			int ynew = rand() % (ymax - ymin) + ymin;
			tryUpdate(imSrc, imPatchMask, xbest, ybest, dbest, xt, yt, xnew, ynew);
		}//end while

		annField.rowPtr(yt)[xt].SetXYValue(xbest, ybest, dbest);
	}

	void PatchMatchCompletion::averageExt(SrcImageType &imSrcExt, const ByteImage& imMask,
		const ImageANN &annField, const FloatImage& distField)const
	{
		gtime_t t1 = gtime_now();

		Rect rc = m_patchHoleBoundingBoxes[m_nCurrentLevel];
		int ybegin = rc.top, yend = rc.top + rc.height;
		int xbegin = rc.left, xend = rc.left + rc.width;
		FloatImage imWeightsExt;
		ImageANN annFieldExt;

#ifdef VOTING_ENABLE_MEANSHIFT
		//estimate \sigma
		const static float percent = 0.75f;
		std::vector<float> dists;
		dists.reserve((yend - ybegin)*(xend - xbegin));
		for (int yt = ybegin; yt < yend; ++yt)
		{
			for (int xt = xbegin; xt < xend; ++xt)
			{
				if (IS_HOLE(imMask.rowPtr(yt)[xt]))
					dists.push_back(annField.rowPtr(yt)[xt].Value());
			}//end for xt
		}//end for yt
		const int nPercent = int((dists.size() - 1)*percent + 0.5f);
		std::sort(dists.begin(), dists.end());
		float dt = dists[nPercent];
		if (dt < dists[dists.size() - 1] * 0.1f)
			dt = dists[dists.size() - 1] * 0.1f;
		if (dt < 0.001f)
			dt = 1000.f;
		const float sigma = 1.0f / (2.f*dt);

		//calc weights to parse
		imWeightsExt.create(annField.width() + 2 * m_nPatchRadius, annField.height() + 2 * m_nPatchRadius);
		for (int y = 0; y < annField.height(); y++)
		{
			for (int x = 0; x < annField.width(); x++)
			{
				float s = (float)exp(double(-annField.rowPtr(y)[x].Value()*sigma));
				imWeightsExt.rowPtr(y + m_nPatchRadius)[x + m_nPatchRadius] = s * distField.rowPtr(y)[x];
			}
		}
		Rect rt(m_nPatchRadius, m_nPatchRadius, imWeightsExt.width() - 2 * m_nPatchRadius, imWeightsExt.height() - 2 * m_nPatchRadius);
		imWeightsExt.range(rt).mirrorPadding(imWeightsExt, m_nPatchRadius);
		annField.mirrorPadding(annFieldExt, m_nPatchRadius);
#else
		distField.mirrorPadding(imWeightsExt, m_nPatchRadius);
		annField.mirrorPadding(annFieldExt, m_nPatchRadius);
#endif

		//voting
#pragma omp parallel for
		for (int yt = ybegin; yt < yend; ++yt)
		{
			for (int xt = xbegin; xt < xend; ++xt)
			{
				if (IS_HOLE(imMask.rowPtr(yt)[xt]))
					averageOnePixelExt(imSrcExt, annFieldExt, imWeightsExt, xt, yt);
			}//end for xt
		}//end for yt

		//timer
		m_timeAverage += gtime_seconds(t1, gtime_now());
	}

	void PatchMatchCompletion::averageOnePixelExt(SrcImageType &imSrcExt, const ImageANN &annFieldExt,
		const FloatImage& weightsExt, int xt, int yt)const
	{
		//meanshift: weighted sum
		float3 mean = float3::Constant(0.0f);
		float variance = 0.0f;
		float wSum = 0.0f;
#ifdef VOTING_ENABLE_MEANSHIFT
		std::vector<std::pair<float3, float>> votes;
#endif
		for (int y = yt - m_nPatchRadius; y <= yt + m_nPatchRadius; y++)
		{
			for (int x = xt - m_nPatchRadius; x <= xt + m_nPatchRadius; x++)
			{
				int xs = 0, ys = 0;
				annFieldExt.rowPtr(y + m_nPatchRadius)[x + m_nPatchRadius].GetXY(xs, ys);
				xs += xt - x;
				ys += yt - y;
				float w = weightsExt.rowPtr(y + m_nPatchRadius)[x + m_nPatchRadius];
				SrcVecType bcolor = imSrcExt.pixel(Point(xs + m_nPatchRadius, ys + m_nPatchRadius));
				float3 color(bcolor[0], bcolor[1], bcolor[2]);
				mean += color * w;
				wSum += w;
#ifdef VOTING_ENABLE_MEANSHIFT
				votes.push_back(std::pair<float3, float>(color, w));
#endif
			}//end for dx
		}//end for dy
		mean *= 1.f / wSum;

#ifdef VOTING_ENABLE_MEANSHIFT
		//calculate vairance
		for (size_t i = 0; i < votes.size(); i++)
			variance += (mean - votes[i].first).squaredNorm() * votes[i].second;
		variance /= wSum;

		//mean shift: window width
		for (float width = 3.0f; width > 1.2f; width *= 0.5f)
		{
			float3 T = float3::Constant(0.0f);
			const float kernelsigma = 1.0f / (2.0f*width*variance);
			float kernelW = 0.0f;
			for (size_t i = 0; i < votes.size(); i++)
			{
				std::pair<float3, float>& V = votes[i];
				float dist = (V.first - mean).squaredNorm();
				float kerneldist = (float)exp(double(-dist*kernelsigma));
				if (dist < 0.001f && variance < 0.001f)
					kerneldist = 1.0f;
				T += V.first * V.second * kerneldist;
				kernelW += V.second * kerneldist;
			}//end for i

			if (kernelW > 0.0)
				mean = T / kernelW;
		}//end for width
#endif

		//output pixel
		imSrcExt.pixel(Point(xt + m_nPatchRadius, yt + m_nPatchRadius)) = mean;
	}

	template<int N> float patch_dist(const SrcVecType* src1, int stride1, const SrcVecType* src2, int stride2, float cutoff)
	{
		PixelDifType sum = PixelDifType(0);
		for (int y = 0; y < N; ++y)
		{
			for (int x = 0; x < N; ++x)
			{
				const SrcVecType& pA = src1[x];
				const SrcVecType& pB = src2[x];
				PixelDifType dl = pA[0] - pB[0];
				PixelDifType da = pA[1] - pB[1];
				PixelDifType db = pA[2] - pB[2];
				sum += abs(dl) + abs(da) + abs(db);
			}//end for x
			if ((float)sum >= cutoff)
				return cutoff;
			src1 = (const SrcVecType*)((const byte*)src1 + stride1);
			src2 = (const SrcVecType*)((const byte*)src2 + stride2);
		}
		return (float)sum;
	}

	float PatchMatchCompletion::patchDist(const SrcVecType* src1, int stride1, 
		const SrcVecType* src2, int stride2, int patchSize, float cutoff)
	{
		if (patchSize == 9)
			return patch_dist<9>(src1, stride1, src2, stride2, cutoff);
		if (patchSize == 7)
			return patch_dist<7>(src1, stride1, src2, stride2, cutoff);
		if (patchSize == 5)
			return patch_dist<5>(src1, stride1, src2, stride2, cutoff);
		if (patchSize == 3)
			return patch_dist<3>(src1, stride1, src2, stride2, cutoff);
		if (patchSize == 1)
			return patch_dist<1>(src1, stride1, src2, stride2, cutoff);
		PixelDifType sum = PixelDifType(0);
		for (int y = 0; y <= patchSize; ++y)
		{
			for (int x = 0; x < patchSize; ++x)
			{
				const SrcVecType& pA = src1[x];
				const SrcVecType& pB = src2[x];
				PixelDifType dl = pA[0] - pB[0];
				PixelDifType da = pA[1] - pB[1];
				PixelDifType db = pA[2] - pB[2];
				sum += abs(dl) + abs(da) + abs(db);
			}//end for x
			if ((float)sum >= cutoff)
				return cutoff;
			src1 = (const SrcVecType*)((const byte*)src1 + stride1);
			src2 = (const SrcVecType*)((const byte*)src2 + stride2);
		}
		return (float)sum;
	}

	void PatchMatchCompletion::tryUpdate(const SrcImageType& imSrcExt, const ByteImage& imPatchMask,
		int& xbest, int& ybest, float& dbest, int xt, int yt, int xnew, int ynew)const
	{
		if (IS_HOLE(imPatchMask.rowPtr(ynew)[xnew]))
			return;

		//here imSrc is extended image, thus offset should be added
		float newDist = patchDist((const SrcVecType*)imSrcExt.rowPtr(yt) + xt, imSrcExt.stride(),
			(const SrcVecType*)imSrcExt.rowPtr(ynew) + xnew, imSrcExt.stride(), m_nPatchSize, dbest);
		if (newDist < dbest)
		{
			xbest = xnew;
			ybest = ynew;
			dbest = newDist;
		}
	}

	void PatchMatchCompletion::buildLanczos3Pyramid()
	{
		gtime_t t1 = gtime_now();

		m_srcPyramidLab.resize(m_nMaxLevel);
		m_maskPyramid.resize(m_nMaxLevel);
		m_pyramidScales.resize(m_nMaxLevel);
		m_annFieldPyramid.resize(m_nMaxLevel);
		m_distFiledPyramid.resize(m_nMaxLevel);
		m_patchMaskPyramid.resize(m_nMaxLevel);
		m_patchHoleBoundingBoxes.resize(m_nMaxLevel);

		sRgb2Lab(m_imSrc, m_srcPyramidLab[0]);
		m_maskPyramid[0] = m_imMask;
		m_pyramidScales[0] = float2::Constant(1.f);

		for (int i = 0; i < m_nMaxLevel; i++)
		{
			if (i > 0)
			{
				const int dW = int(m_srcPyramidLab[i - 1].width()*SCALE_PER_LEVEL);
				const int dH = int(m_srcPyramidLab[i - 1].height()*SCALE_PER_LEVEL);
				m_srcPyramidLab[i] = imresize(m_srcPyramidLab[i - 1], dW, dH, ResizeLanczos3);
				m_maskPyramid[i] = imresize(m_maskPyramid[i - 1], dW, dH, ResizeLanczos3);
				m_pyramidScales[i] = float2(float(m_srcPyramidLab[i - 1].width()) / float(m_srcPyramidLab[i].width()),
					float(m_srcPyramidLab[i - 1].height()) / float(m_srcPyramidLab[i].height()));
				for (int y = 0; y < dH; y++)
				{
					uchar* ptr = m_maskPyramid[i].rowPtr(y);
					for (int x = 0; x < dW; x++)
						ptr[x] = ptr[x] < 10 ? 0 : 255;
				}
			} // i>0

			m_annFieldPyramid[i].create(m_srcPyramidLab[i].width(), m_srcPyramidLab[i].height());
			maxFilter(m_maskPyramid[i], m_patchMaskPyramid[i], m_nPatchRadius);
			m_patchHoleBoundingBoxes[i] = computeHoleBoundingBox(m_patchMaskPyramid[i]);

			//calc distance field
			m_distFiledPyramid[i] = bwdist(m_maskPyramid[i]);

			FloatImage& D = m_distFiledPyramid[i];
			const ByteImage& imMask = m_maskPyramid[i];
			for (int y = 0; y < D.height(); y++)
				for (int x = 0; x < D.width(); x++)
					D.rowPtr(y)[x] = 5.0f + 200.0f * (float)pow(1.3, double(-D.rowPtr(y)[x] - 5 * IS_HOLE(imMask.rowPtr(y)[x])));

		}//end for i

		m_timeOthers += gtime_seconds(t1, gtime_now());
	}

} // lvg