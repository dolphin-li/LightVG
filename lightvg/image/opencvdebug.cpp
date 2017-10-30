#include "opencvdebug.h"

#ifdef LVG_ENABLE_OPENCV_DEBUG
#include <opencv2\opencv.hpp>
#endif

namespace lvg
{
#ifdef LVG_ENABLE_OPENCV_DEBUG
	void imshow(std::string name, ByteImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_8UC1, img.data(), img.stride()).clone();
		cv::imshow(name, mat);
	}
	void imshow(std::string name, RgbImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_8UC3, img.data(), img.stride()).clone();
		cv::cvtColor(mat, mat, CV_RGB2BGR);
		cv::imshow(name, mat);
	}
	void imshow(std::string name, RgbaImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_8UC4, img.data(), img.stride()).clone();
		cv::cvtColor(mat, mat, CV_RGBA2BGRA);
		cv::imshow(name, mat);
	}
	void imshow(std::string name, FloatImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_32FC1, img.data(), img.stride()).clone();
		cv::imshow(name, mat);
	}
	void imshow(std::string name, RgbFloatImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_32FC3, img.data(), img.stride()).clone();
		cv::cvtColor(mat, mat, CV_RGB2BGR);
		cv::imshow(name, mat);
	}
	void imshow(std::string name, RgbaFloatImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_32FC4, img.data(), img.stride()).clone();
		cv::cvtColor(mat, mat, CV_RGBA2BGRA);
		cv::imshow(name, mat);
	}
	void imread(std::string name, ByteImage& img)
	{
		cv::Mat mat = cv::imread(name, 0);
		img.create(mat.cols, mat.rows);
		for (int y = 0; y < img.height(); y++)
		{
			for (int x = 0; x < img.width(); x++)
				img.pixel(Point(x, y))[0] = mat.at<uchar>(y, x);
		}
	}
	void imread(std::string name, RgbImage& img)
	{
		cv::Mat mat = cv::imread(name, 1);
		if (mat.channels() == 4)
			cv::cvtColor(mat, mat, CV_BGRA2RGB);
		else
			cv::cvtColor(mat, mat, CV_BGR2RGB);
		img.create(mat.cols, mat.rows);
		for (int y = 0; y < img.height(); y++)
		{
			for (int x = 0; x < img.width(); x++)
			{
				const cv::Vec3b& c = mat.at<cv::Vec3b>(y, x);
				img.pixel(Point(x, y)) = uchar3(c[0], c[1], c[2]);
			}
		}
	}
	void imread(std::string name, RgbaImage& img)
	{
		cv::Mat mat = cv::imread(name, 1);
		if (mat.channels() == 4)
			cv::cvtColor(mat, mat, CV_BGRA2RGBA);
		else
			cv::cvtColor(mat, mat, CV_BGR2RGBA);
		img.create(mat.cols, mat.rows);
		for (int y = 0; y < img.height(); y++)
		{
			for (int x = 0; x < img.width(); x++)
			{
				const cv::Vec4b& c = mat.at<cv::Vec4b>(y, x);
				img.pixel(Point(x, y)) = uchar4(c[0], c[1], c[2], c[3]);
			}
		}
	}
	void imwrite(std::string name, ByteImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_8UC1, img.data(), img.stride()).clone();
		cv::imwrite(name, mat);
	}
	void imwrite(std::string name, RgbImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_8UC3, img.data(), img.stride()).clone();
		cv::cvtColor(mat, mat, CV_RGB2BGR);
		cv::imwrite(name, mat);
	}
	void imwrite(std::string name, RgbaImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_8UC4, img.data(), img.stride()).clone();
		cv::cvtColor(mat, mat, CV_RGBA2BGRA);
		cv::imwrite(name, mat);
	}
	void imwrite(std::string name, FloatImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_32FC1, img.data(), img.stride()).clone();
		cv::imwrite(name, mat);
	}
	void imwrite(std::string name, RgbFloatImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_32FC3, img.data(), img.stride()).clone();
		cv::cvtColor(mat, mat, CV_RGB2BGR);
		cv::imwrite(name, mat);
	}
	void imwrite(std::string name, RgbaFloatImage img)
	{
		cv::Mat mat = cv::Mat(img.rows(), img.cols(), CV_32FC4, img.data(), img.stride()).clone();
		cv::cvtColor(mat, mat, CV_RGBA2BGRA);
		cv::imwrite(name, mat);
	}
	void waitKey(int t)
	{
		cv::waitKey(t);
	}
#endif
}