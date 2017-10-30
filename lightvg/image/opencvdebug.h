#pragma once

#include "Image.h"
#include <string>

namespace lvg
{
#ifdef LVG_ENABLE_OPENCV_DEBUG
	void imshow(std::string name, ByteImage img);
	void imshow(std::string name, RgbImage img);
	void imshow(std::string name, RgbaImage img);
	void imshow(std::string name, FloatImage img);
	void imshow(std::string name, RgbFloatImage img);
	void imshow(std::string name, RgbaFloatImage img);
	void imread(std::string name, ByteImage& img);
	void imread(std::string name, RgbImage& img);
	void imread(std::string name, RgbaImage& img);
	void imwrite(std::string name, ByteImage img);
	void imwrite(std::string name, RgbImage img);
	void imwrite(std::string name, RgbaImage img);
	void imwrite(std::string name, FloatImage img);
	void imwrite(std::string name, RgbFloatImage img);
	void imwrite(std::string name, RgbaFloatImage img);
	RgbImage imresize_opencv(RgbImage img, int dstW, int dstH);
	void waitKey(int t = 0);
#endif
}