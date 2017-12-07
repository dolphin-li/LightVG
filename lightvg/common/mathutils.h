#pragma once

#include "definations.h"
#include <eigen/Dense>
namespace lvg
{
	typedef unsigned char uchar;
	typedef Eigen::Matrix2f Mat2f;
	typedef Eigen::Matrix3f Mat3f;
	typedef Eigen::Matrix4f Mat4f;
	typedef Eigen::Matrix2d Mat2d;
	typedef Eigen::Matrix3d Mat3d;
	typedef Eigen::Matrix4d Mat4d;
	typedef Eigen::Quaternionf QuaternionF;
	typedef Eigen::Quaterniond QuaternionD;
	typedef Eigen::AngleAxisf AngleAxisF;
	typedef Eigen::AngleAxisd AngleAxisD;
	typedef Eigen::Vector2i int2;
	typedef Eigen::Vector3i int3;
	typedef Eigen::Vector4i int4;
	typedef Eigen::Vector2f float2;
	typedef Eigen::Vector3f float3;
	typedef Eigen::Vector4f float4;
	typedef Eigen::Vector2d double2;
	typedef Eigen::Vector3d double3;
	typedef Eigen::Vector4d double4;
	typedef Eigen::Vector2i Point;
	typedef Eigen::Matrix<uchar, 2, 1> uchar2;
	typedef Eigen::Matrix<uchar, 3, 1> uchar3;
	typedef Eigen::Matrix<uchar, 4, 1> uchar4;

	const float MATH_PI = 3.141592653589793f;
	const float MATH_TWO_PI = 2 * MATH_PI;
	const float MATH_HALF_PI = 0.5f * MATH_PI;
	const float MATH_QUARTER_PI = 0.25f * MATH_PI;
	const float MATH_INV_PI = 1.0f / MATH_PI;
	const float MATH_INV_TWO_PI = 1.0f / MATH_TWO_PI;
	const float MATH_DEG_TO_RAD = MATH_PI / 180.0f;
	const float MATH_RAD_TO_DEG = 180.0f / MATH_PI;

	template<class T> inline T clamp(T v, T minV, T maxV)
	{
		return std::min(maxV, std::max(minV, v));
	}

	inline float cross2(float2 a, float2 b)
	{
		return a[0] * b[1] - a[1] * b[0];
	}

	inline float4 float3ToPoint(float3 p)
	{
		return float4(p[0], p[1], p[2], 1.f);
	}

	inline float4 float3ToNormal(float3 p)
	{
		return float4(p[0], p[1], p[2], 0.f);
	}

	// treat float4 as float3 normal vector and cross
	inline float4 crossAsNormal(float4 a, float4 b)
	{
		return float4(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0.f);
	}
	inline float3 crossAsNormal(float3 a, float3 b)
	{
		return float3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
	}

	struct KeyPoint
	{
		float2 pt;
		int octave;
		float angle;
		float response;
		int size;
		KeyPoint() :octave(0), angle(0), response(0), size(0) {}
		KeyPoint(float2 pt_, int octave_, float angle_, float response_, int size_) :
			pt(pt_), octave(octave_), angle(angle_), response(response_), size(size_) {}
	};

	struct Rect
	{
		int left;
		int top;
		int width;
		int height;
		Rect() :left(0), top(0), width(0), height(0) {}
		Rect(int l, int t, int w, int h) :left(l), top(t), width(w), height(h) {}
		bool empty()const
		{
			return width <= 0 || height <= 0;
		}
		Rect inflate(int xl, int xr, int yl, int yr)const
		{
			Rect r;
			r.left = left - xl;
			r.top = top - yl;
			r.width = width + xl + xr;
			r.height = height + yl + yr;
			return r;
		}
		Rect inflate(int v)const
		{
			return inflate(v, v, v, v);
		}
		Rect inflate(int w, int h)const
		{
			return inflate(w, w, h, h);
		}
		Rect intersect(const Rect& other)const
		{
			int b = std::min(top + height, other.top + other.height);
			int r = std::min(left + width, other.left + other.width);
			Rect result;
			result.left = std::max(left, other.left);
			result.top = std::max(top, other.top);
			result.width = std::max(0, r - result.left);
			result.height = std::max(0, b - result.top);
			return result;
		}
	};

	inline Mat3f Rodrigues(float3 r)
	{
		const float theta = r.norm();
		r /= theta;
		const float cosTheta = cos(theta);
		Mat3f S = Mat3f::Zero();
		S(0, 1) = -r.z();	S(1, 0) = -S(0, 1);
		S(0, 2) = r.y();	S(2, 0) = -S(0, 2);
		S(1, 2) = -r.x();	S(2, 1) = -S(1, 2);

		return cosTheta * Mat3f::Identity() + (1 - cosTheta) * r*r.transpose() + sin(theta) * S;
	}
}