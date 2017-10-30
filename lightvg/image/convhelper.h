#pragma once

#include <algorithm>

#ifdef _WIN32
//#define CONV_HELPER_ENABLE_SSE
#endif

#ifdef CONV_HELPER_ENABLE_SSE
#include <xmmintrin.h>      // __m128 data type and SSE functions
#endif

#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
namespace lvg
{
#ifdef CONV_HELPER_ENABLE_SSE
	template<int N> void max_filter_sse(float* dst, const float* src, int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		__m128 s;

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::lowest());
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_max_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst = (float*)(((char*)dst) + dstStride);
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::lowest());
			for (int k = -L; k <= R; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_max_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst = (float*)(((char*)dst) + dstStride);
		}// end for x

		 // the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::lowest());
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_max_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst = (float*)(((char*)dst) + dstStride);
		}
	}

	template<int N> void min_filter_sse(float* dst, const float* src, int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		__m128 s;

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::max());
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_min_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst = (float*)(((char*)dst) + dstStride);
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::max());
			for (int k = -L; k <= R; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_min_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst = (float*)(((char*)dst) + dstStride);
		}// end for x

		 // the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_set_ps1(std::numeric_limits<float>::max());
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				v = _mm_min_ps(v, s);
			}
			_mm_storeu_ps(dst, v);
			dst = (float*)(((char*)dst) + dstStride);
		}
	}

	template<int N> void conv_sse(float* dst, const float* src, const float* kernel, int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		__m128 s, knl;

		// the first few elements that do not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_setzero_ps();
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				knl = _mm_set_ps1(kernel[R - k]);
				v = _mm_add_ps(v, _mm_mul_ps(s, knl));
			}
			_mm_storeu_ps(dst, v);
			dst = (float*)(((char*)dst) + dstStride);
		}

		// middle elements that fullfill the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			__m128 v = _mm_setzero_ps();
			for (int k = -L; k <= R; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				knl = _mm_set_ps1(kernel[R - k]);
				v = _mm_add_ps(v, _mm_mul_ps(s, knl));
			}
			_mm_storeu_ps(dst, v);
			dst = (float*)(((char*)dst) + dstStride);
		}// end for x

		 // the last few elements that do not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			__m128 v = _mm_setzero_ps();
			for (int k = xb; k <= xe; k++)
			{
				s = _mm_loadu_ps(src + (k + x) * 4);
				knl = _mm_set_ps1(kernel[R - k]);
				v = _mm_add_ps(v, _mm_mul_ps(s, knl));
			}
			_mm_storeu_ps(dst, v);
			dst = (float*)(((char*)dst) + dstStride);
		}
	}
#endif

	template<typename T, int N> void max_filter(T* dst, const T* src, int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = std::numeric_limits<T>::lowest();
			for (int k = xb; k <= xe; k++)
			if (src[k + x] > v)
				v = src[k + x];
			*dst = v;
			dst = (T*)(((char*)dst) + dstStride);
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			T v = std::numeric_limits<T>::lowest();
			for (int k = -L; k <= R; k++)
			if (src[k + x] > v)
				v = src[k + x];
			*dst = v;
			dst = (T*)(((char*)dst) + dstStride);
		}// end for x

		// the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = std::numeric_limits<T>::lowest();
			for (int k = xb; k <= xe; k++)
			if (src[k + x] > v)
				v = src[k + x];
			*dst = v;
			dst = (T*)(((char*)dst) + dstStride);
		}
	}

	template<typename T, int N> void min_filter(T* dst, const T* src, int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = std::numeric_limits<T>::max();
			for (int k = xb; k <= xe; k++)
			if (src[k + x] < v)
				v = src[k + x];
			*dst = v;
			dst = (T*)(((char*)dst) + dstStride);
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			T v = std::numeric_limits<T>::max();
			for (int k = -L; k <= R; k++)
			if (src[k + x] < v)
				v = src[k + x];
			*dst = v;
			dst = (T*)(((char*)dst) + dstStride);
		}// end for x

		// the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = std::numeric_limits<T>::max();
			for (int k = xb; k <= xe; k++)
			if (src[k + x] < v)
				v = src[k + x];
			*dst = v;
			dst = (T*)(((char*)dst) + dstStride);
		}
	}

	template<typename T, int N> void max_filter2(T* srcDst, int W, int H, int stride)
	{
		std::vector<T> tmpBuffer(std::max(W, H));
#ifdef CONV_HELPER_ENABLE_SSE
		std::vector<__m128> tmpBuffers_sse;
		if(typeid(T) == typeid(float))
			tmpBuffers_sse.resize(std::max(W, H));
#endif

		// max filtering along x direction
		for (int y = 0; y < H; y++)
		{
			T* dstPtr = (T*)((char*)srcDst + y * stride);
			memcpy(tmpBuffer.data(), dstPtr, W * sizeof(T));
			max_filter<T, N>(dstPtr, tmpBuffer.data(), W, sizeof(T));
		}// end for y
		
		// max filtering along y direction
		int x = 0;
#ifdef CONV_HELPER_ENABLE_SSE
		if (typeid(float) == typeid(T))
		{
			for (; x < W - 3; x += 4)
			{
				char* bytePtr = (char*)(srcDst + x);
				for (int y = 0; y < H; y++, bytePtr += stride)
					tmpBuffers_sse[y] = _mm_loadu_ps((float*)bytePtr);
				max_filter_sse<N>((float*)srcDst + x, (const float*)tmpBuffers_sse.data(), H, stride);
			}
		}
#endif
		for (; x < W; x++)
		{
			char* bytePtr = (char*)(srcDst + x);
			for (int y = 0; y < H; y++, bytePtr += stride)
				tmpBuffer[y] = *(T*)bytePtr;
			max_filter<T, N>(srcDst + x, tmpBuffer.data(), H, stride);
		}// end for x
	}

	template<typename T, int N> void min_filter2(T* srcDst, int W, int H, int stride)
	{
		std::vector<T> tmpBuffer(std::max(W, H));
#ifdef CONV_HELPER_ENABLE_SSE
		std::vector<__m128> tmpBuffers_sse;
		if (typeid(T) == typeid(float))
			tmpBuffers_sse.resize(std::max(W, H));
#endif

		// min filtering along x direction
		for (int y = 0; y < H; y++)
		{
			T* dstPtr = (T*)((char*)srcDst + y * stride);
			memcpy(tmpBuffer.data(), dstPtr, W * sizeof(T));
			min_filter<T, N>(dstPtr, tmpBuffer.data(), W, sizeof(T));
		}// end for y

		// min filtering along y direction
		int x = 0;
#ifdef CONV_HELPER_ENABLE_SSE
		if (typeid(float) == typeid(T))
		{
			for (; x < W - 3; x += 4)
			{
				char* bytePtr = (char*)(srcDst + x);
				for (int y = 0; y < H; y++, bytePtr += stride)
					tmpBuffers_sse[y] = _mm_loadu_ps((float*)bytePtr);
				min_filter_sse<N>((float*)srcDst + x, (const float*)tmpBuffers_sse.data(), H, stride);
			}
		}
#endif
		for (; x < W; x++)
		{
			char* bytePtr = (char*)(srcDst + x);
			for (int y = 0; y < H; y++, bytePtr += stride)
				tmpBuffer[y] = *(T*)bytePtr;
			min_filter<T, N>(srcDst + x, tmpBuffer.data(), H, stride);
		}// end for x
	}

	// 1D conv, the same with matlab conv(..., 'same')
	//	assume:
	//		the stride of src is 1 
	//		the size of src @num
	//		the size of dst @num
	//		kernel size is @N
	template<typename T, int N> void conv(T* dst, const T* src, const T* kernel, int num, int dstStride)
	{
		const static int L = N / 2 - (N % 2 == 0);
		const static int R = N / 2;
		const int head_pos = std::min((int)num, R);
		const int tail_pos = num - R;
		const int tail_head_pos = std::max(head_pos, tail_pos);

		// the first few elements that does not fullfill the conv kernel
		for (int x = 0; x < head_pos; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = 0;
			for (int k = xb; k <= xe; k++)
				v += src[k + x] * kernel[R - k];
			*dst = v;
			dst = (T*)((char*)dst + dstStride);
		}

		// middle elements that fullfills the conv kernel
		for (int x = R; x < tail_pos; x++)
		{
			T v = 0;
			for (int k = -L; k <= R; k++)
				v += src[k + x] * kernel[R - k];
			*dst = v;
			dst = (T*)((char*)dst + dstStride);
		}// end for x

		// the last few elements that does not fullfill the conv kernel
		for (int x = tail_head_pos; x < num; x++)
		{
			const int xb = std::max(-L, -x);
			const int xe = std::min(num - x - 1, R);
			T v = 0;
			for (int k = xb; k <= xe; k++)
				v += src[k + x] * kernel[R - k];
			*dst = v;
			dst = (T*)((char*)dst + dstStride);
		}
	}

	template<typename T, int N> void conv2(T* srcDst, const T* kernel, int W, int H, int stride)
	{
		std::vector<T> tmpBuffer(std::max(W, H));
#ifdef CONV_HELPER_ENABLE_SSE
		std::vector<__m128> tmpBuffers_sse;
		if (typeid(T) == typeid(float))
			tmpBuffers_sse.resize(std::max(W, H));
#endif

		// conv along x direction
		for (int y = 0; y < H; y++)
		{
			T* dstPtr = (T*)((char*)srcDst + y * stride);
			memcpy(tmpBuffer.data(), dstPtr, W * sizeof(T));
			conv<T, N>(dstPtr, tmpBuffer.data(), kernel, W, sizeof(T));
		}// end for y

		// conv along y direction
		int x = 0;
#ifdef CONV_HELPER_ENABLE_SSE
		if (typeid(float) == typeid(T))
		{
			for (; x < W - 3; x += 4)
			{
				char* bytePtr = (char*)(srcDst + x);
				for (int y = 0; y < H; y++, bytePtr += stride)
					tmpBuffers_sse[y] = _mm_loadu_ps((float*)bytePtr);
				conv_sse<N>((float*)srcDst + x, (const float*)tmpBuffers_sse.data(), kernel, H, stride);
			}
		}
#endif
		for (; x < W; x++)
		{
			char* bytePtr = (char*)(srcDst + x);
			for (int y = 0; y < H; y++, bytePtr += stride)
				tmpBuffer[y] = *(T*)bytePtr;
			conv<T, N>(srcDst + x, tmpBuffer.data(), kernel, H, stride);
		}// end for x
	}
}
#pragma pop_macro("max")
#pragma pop_macro("min")