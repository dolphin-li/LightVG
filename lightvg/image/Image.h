#pragma once

#include "lightvg/common/mathutils.h"
#include <vector>
#include <atomic>


namespace lvg
{
	/** **************************************************************************************
	* Image: An aligned, shared-memory image representation. Light-weighted and cross-platform
	*			Part of functions are implemented corresponding to OpenCV
	*	NOTE: it is better to separate declarations with definations with template-instance technique
	*			however, some compilers are too stupid to support this technique.
	* ***************************************************************************************/
	template<typename T, int Channels, int alignBytes = sizeof(T)>
	class Image
	{
	public:
		enum {
			ALIGN_BYTES = alignBytes,
			ChannelNum = Channels,
			ALIGN_ELEMENTS = ALIGN_BYTES/sizeof(T),
		};
		typedef Eigen::Matrix<T, Channels, 1> VecType;
	public:
		Image() :m_data(nullptr), m_dataAlloc(nullptr), m_width(0), m_height(0), m_stride(0), m_refCount(nullptr) {}

		Image(const Image& r)
		{
			if (this == &r)
				return;
			m_width = r.m_width;
			m_height = r.m_height;
			m_stride = r.m_stride;
			m_data = r.m_data;
			m_dataAlloc = r.m_dataAlloc;
			m_refCount = r.m_refCount;
			if (m_refCount)
				std::atomic_fetch_add(m_refCount, 1);
		}

		~Image()
		{
			release();
		}

		// create and fill with zero if not created
		Image& create(int w, int h)
		{
			bool bAlloc = true;
			if (m_refCount) {
				int value = 1;
				bool one = std::atomic_compare_exchange_strong(m_refCount, &value, value);
				bAlloc = (!one) || (w*h != m_width*m_height);
			}
			if (bAlloc)
				release();
			m_width = w;
			m_height = h;
			if (bAlloc) {
				m_refCount = new std::atomic<int>(1);
				m_stride = (w*Channels * sizeof(T) + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
				m_data = (T*)aligned_malloc(h*m_stride);
				m_dataAlloc = m_data;
			}
			return *this;
		}

		Image& create(int w, int h, int stride, T* ptr)
		{
			release();
			m_refCount = nullptr;
			m_width = w;
			m_height = h;
			m_stride = stride;
			m_data = ptr;
			m_dataAlloc = nullptr;
			return *this;
		}

		void release()
		{
			if (m_refCount)
			{
				int oldVal = std::atomic_fetch_sub(m_refCount, 1);
				if (oldVal == 1)
				{
					if (m_dataAlloc)
						aligned_free(m_dataAlloc);
					delete m_refCount;
				}
			}
			m_width = 0;
			m_height = 0;
			m_stride = 0;
			m_data = nullptr;
			m_dataAlloc = nullptr;
			m_refCount = nullptr;
		}

		Image clone()const
		{
			Image im;
			im.create(width(), height());
			for (int r = 0; r < height(); r++)
				memcpy(im.rowPtr(r), rowPtr(r), im.width() * sizeof(T)*Channels);
			return im;
		}

		Image& operator = (const Image& r)
		{
			if (this == &r)
				return *this;
			release();
			m_width = r.m_width;
			m_height = r.m_height;
			m_stride = r.m_stride;
			m_data = r.m_data;
			m_dataAlloc = r.m_dataAlloc;
			m_refCount = r.m_refCount;
			if (m_refCount)
				std::atomic_fetch_add(m_refCount, 1);
			return *this;
		}

#define UNARY_OP_EQ(op)\
		Image& operator op (const VecType& val)\
		{\
			for (int y = 0; y < m_height; y++)\
			{\
				T* pSrcRow = rowPtr(y);\
				for (int x = 0; x < m_width; x++)\
				{\
					for (int c = 0; c < Channels; c++)\
						pSrcRow[c] op val[c];\
					pSrcRow += Channels; \
				}\
			}\
			return *this;\
		}\
		template<class E> Image& operator op (const E& val)\
		{\
			for (int y = 0; y < m_height; y++)\
			{\
				T* pSrcRow = rowPtr(y); \
				for (int x = 0; x < m_width; x++)\
				{\
					for (int c = 0; c < Channels; c++)\
						pSrcRow[c] op val; \
						pSrcRow += Channels; \
				}\
			}\
			return *this; \
		}
		UNARY_OP_EQ(*= );
		UNARY_OP_EQ(/= );
		UNARY_OP_EQ(+= );
		UNARY_OP_EQ(-= );
#undef UNARY_OP_EQ

		Image& swap(Image& r)
		{
			std::swap(m_width, r.m_width);
			std::swap(m_height, r.m_height);
			std::swap(m_stride, r.m_stride);
			std::swap(m_data, r.m_data);
			std::swap(m_dataAlloc, r.m_dataAlloc);
			std::swap(m_refCount, r.m_refCount);
			return *this;
		}

		Image& rgb2bgr()
		{
			if (Channels < 3)
				return *this;
			for (int y = 0; y < m_height; y++)
			{
				T* pSrcRow = rowPtr(y);
				for (int x = 0; x < m_width; x++)
				{
					std::swap(pSrcRow[0], pSrcRow[2]);
					pSrcRow += Channels;
				} // c
			} // r
			return *this;
		}

		Image& setZero()
		{
			for (int y = 0; y < m_height; y++)
				memset(rowPtr(y), 0, m_width*Channels * sizeof(T));
			return *this;
		}

		Image& setConstant(T val)
		{
			const int wc = m_width * Channels;
			for (int y = 0; y < m_height; y++)
			{
				T* pSrcRow = rowPtr(y);
				for (int x = 0; x < wc; x++)
					pSrcRow[x] = val;
			} // r
			return *this;
		}

		Image& setConstant(const VecType& val)
		{
			for (int y = 0; y < m_height; y++)
			{
				T* pSrcRow = rowPtr(y);
				for (int x = 0; x < m_width; x++)
				{
					for (int c = 0; c < Channels; c++)
						pSrcRow[c] = val[c];
					pSrcRow += Channels;
				} // c
			} // r
			return *this;
		}

		Image& copyFrom(const Image& rhs)
		{
			assert(width() == rhs.width() && height() == rhs.height());
			for (int y = 0; y < m_height; y++)
				memcpy(rowPtr(y), rhs.rowPtr(y), m_width * Channels * sizeof(T));
			return *this;
		}

		template<class E>
		void convertTo(Image<E, Channels>& dst, E alpha = E(1), E beta = E(0))const
		{
			if (dst.width() != m_width || dst.height() != m_height)
				dst.create(m_width, m_height);

			for (int y = 0; y < m_height; y++)
			{
				const T* psrc = rowPtr(y);
				E* pdst = dst.rowPtr(y);
				for (int x = 0; x < m_width; x++)
				{
					for (int c = 0; c < Channels; c++)
						pdst[c] = E(psrc[c]) * alpha + beta;
					psrc += Channels;
					pdst += Channels;
				} // x
			} // y
		}

		// padding by 0
		Image zeroPadding(int l, int r, int t, int b)const
		{
			Image dst;		
			zeroPadding(dst, l, r, t, b);
			return dst;
		}

		Image zeroPadding(int sz)const
		{
			return zeroPadding(sz, sz, sz, sz);
		}

		void zeroPadding(Image& dst, int l, int r, int t, int b)const
		{
			if (dst.width() != m_width + l + r || dst.height() != m_height + t + b)
				dst.create(m_width + l + r, m_height + t + b);

			// middel rows
			for (int y = 0; y < m_height; y++)
			{
				T* dst_y = dst.rowPtr(y + t);
				const T* src_y = rowPtr(y);
				memcpy(dst_y + l * Channels, src_y, m_width * Channels * sizeof(T));
				memset(dst_y, 0, l*Channels * sizeof(T));
				memset(dst_y + (m_width + l)*Channels, 0, r*Channels * sizeof(T));
			}// y

			 // top rows
			for (int y = 0; y < t; y++)
			{
				T* dst_y = dst.rowPtr(y);
				memset(dst_y, 0, dst.m_width * Channels * sizeof(T));
			} // y

			  // bottom rows
			for (int y = 0; y < b; y++)
			{
				T* dst_y = dst.rowPtr(y + m_height + t);
				memset(dst_y, 0, dst.m_width * Channels * sizeof(T));
			} // y
		}

		void zeroPadding(Image& dst, int sz)const
		{
			return zeroPadding(dst, sz, sz, sz, sz);
		}

		// padding by boundary values
		Image boundaryPadding(int l, int r, int t, int b)const
		{
			Image dst;
			boundaryPadding(dst, l, r, t, b);
			return dst;
		}

		Image boundaryPadding(int sz)const
		{
			return boundaryPadding(sz, sz, sz, sz);
		}

		void boundaryPadding(Image& dst, int l, int r, int t, int b)const
		{
			if (dst.width() != m_width + l + r || dst.height() != m_height + t + b)
				dst.create(m_width + l + r, m_height + t + b);

			// middel rows
			for (int y = 0; y < m_height; y++)
			{
				T* dst_y = dst.rowPtr(y + t);
				const T* src_y = rowPtr(y);

				// left
				for (int x = 0; x < l; x++)
				{
					for (int c = 0; c < Channels; c++)
						dst_y[c] = src_y[c];
					dst_y += Channels;
				}

				// middel
				memcpy(dst_y, src_y, m_width * Channels * sizeof(T));
				dst_y += m_width * Channels;
				src_y += (m_width - 1) * Channels;

				// right
				for (int x = 0; x < r; x++)
				{
					for (int c = 0; c < Channels; c++)
						dst_y[c] = src_y[c];
					dst_y += Channels;
				}
			}// y

			 // top rows
			for (int y = 0; y < t; y++)
			{
				T* dst_y = dst.rowPtr(y);
				const T* src_y = dst.rowPtr(t);
				memcpy(dst_y, src_y, dst.m_width * Channels * sizeof(T));
			} // y

			  // bottom rows
			for (int y = 0; y < b; y++)
			{
				T* dst_y = dst.rowPtr(y + m_height + t);
				const T* src_y = dst.rowPtr(t + m_height - 1);
				memcpy(dst_y, src_y, dst.m_width * Channels * sizeof(T));
			} // y
		}

		void boundaryPadding(Image& dst, int sz)const
		{
			return boundaryPadding(dst, sz, sz, sz, sz);
		}

		// padding by mirror near-boundary values
		Image mirrorPadding(int l, int r, int t, int b)const
		{
			Image dst;
			mirrorPadding(dst, l, r, t, b);
			return dst;
		}

		Image mirrorPadding(int sz)const
		{
			return mirrorPadding(sz, sz, sz, sz);
		}

		void mirrorPadding(Image& dst, int l, int r, int t, int b)const
		{
			if(dst.width() != m_width + l + r || dst.height() != m_height + t + b)
				dst.create(m_width + l + r, m_height + t + b);

			const int wCT = m_width * Channels * sizeof(T);
			const int lC = l * Channels;
			const int w_lC = (m_width + l) * Channels;
			const int lCT = lC * sizeof(T);
			const int rCT = r * Channels * sizeof(T);

			//center
			for (int y = 0; y < m_height; y++)
				memcpy(dst.rowPtr(y + t) + lC, rowPtr(y), wCT);

			//top
			for (int y = 0; y < t; y++)
				memcpy(dst.rowPtr(y) + lC, dst.rowPtr(2 * t - y) + lC, wCT);

			//bottom
			for (int y = m_height + t; y < dst.m_height; y++)
				memcpy(dst.rowPtr(y) + lC, dst.rowPtr(2 * (m_height + t - 1) - y) + lC, wCT);

			//left
			for (int y = t; y < t + m_height; y++)
			{
				const T* pSrc = dst.rowPtr(y) + 2 * lC;
				T* pDst = dst.rowPtr(y);
				for (int x = 0; x < l; x++)
				{
					for (int c = 0; c < Channels; c++)
						pDst[c] = pSrc[c];
					pDst += Channels;
					pSrc -= Channels;
				} // x
			} // y

			  //right
			for (int y = t; y < t + m_height; y++)
			{
				T* pDst = dst.rowPtr(y) + w_lC;
				const T* pSrc = pDst - Channels;
				for (int x = 0; x < r; x++)
				{
					for (int c = 0; c < Channels; c++)
						pDst[c] = pSrc[c];
					pDst += Channels;
					pSrc -= Channels;
				} // x
			} // y

			  //left top
			for (int y = 0; y < t; y++)
				memcpy(dst.rowPtr(y), dst.rowPtr(2 * t - y), lCT);

			//left bottom
			for (int y = t + m_height; y < dst.m_height; y++)
				memcpy(dst.rowPtr(y), dst.rowPtr(2 * (m_height + t - 1) - y), lCT);

			//right top
			for (int y = 0; y < t; y++)
				memcpy(dst.rowPtr(y) + w_lC, dst.rowPtr(2 * t - y) + w_lC, rCT);

			//right bottom
			for (int y = m_height + t; y < dst.m_height; y++)
				memcpy(dst.rowPtr(y) + w_lC, dst.rowPtr(2 * (m_height + t - 1) - y) + w_lC, rCT);
		}

		void mirrorPadding(Image& dst, int sz)const
		{
			mirrorPadding(dst, sz, sz, sz, sz);
		}

		Image rowRange(int rowBegin, int rowEnd)const
		{
			return range(rowBegin, rowEnd, 0, m_width);
		}

		Image row(int row)const
		{
			return rowRange(row, row + 1);
		}

		Image colRange(int colBegin, int colEnd)const
		{
			return range(0, m_height, colBegin, colEnd);
		}

		Image col(int col)const
		{
			return colRange(col, col + 1);
		}

		Image range(int rowBegin, int rowEnd, int colBegin, int colEnd)const
		{
			Image r;
			r.m_width = colEnd - colBegin;
			r.m_height = rowEnd - rowBegin;
			r.m_stride = m_stride;
			r.m_data = (T*)rowPtr(rowBegin) + colBegin * Channels;
			r.m_dataAlloc = m_dataAlloc;
			r.m_refCount = m_refCount;
			if (m_refCount)
				std::atomic_fetch_add(m_refCount, 1);
			return r;
		}

		Image range(const Rect& rc)const
		{
			return range(rc.top, rc.top + rc.height, rc.left, rc.left + rc.width);
		}

		bool empty()const { return m_data == nullptr; }

		bool memoryOverlap(const Image& r)const {
			return (m_data > r.m_data && m_data < r.m_data + r.m_stride*r.m_height)
				|| (r.m_data > m_data && r.m_data < m_data + m_stride*m_height);
		}

		bool sameWith(const Image& r)const
		{
			return m_data == r.m_data && m_dataAlloc == r.m_dataAlloc
				&& m_width == r.m_width && m_height == r.m_height;
		}

		bool sameShape(const Image& r)const
		{
			return m_width == r.m_width && m_height == r.m_height;
		}

		T* data() { return m_data; }
		const T* data()const { return m_data; }
		T* rowPtr(int row) { return (T*)(((char*)m_data) + row*m_stride); }
		const T* rowPtr(int row)const { return (T*)(((char*)m_data) + row*m_stride); }
		int stride()const { return m_stride; }
		static int channels() { return Channels; }
		int width()const { return m_width; }
		int height()const { return m_height; }
		int cols()const { return m_width; }
		int rows()const { return m_height; }
		Rect rect()const
		{
			return Rect(0, 0, width(), height());
		}
		inline bool contains(const Point& p)const
		{
			return p[0] >= 0 && p[0] < m_width && p[1] >= 0 && p[1] < m_height;
		}
		inline VecType& pixel(const Point& p)
		{
			return *(VecType*)(rowPtr(p[1]) + p[0] * Channels);
		}
		inline const VecType& pixel(const Point& p)const
		{
			return *(VecType*)(rowPtr(p[1]) + p[0] * Channels);
		}
	protected:
		static void *aligned_malloc(int size) {
			void *mem = malloc(size + ALIGN_BYTES + sizeof(void*));
			void **ptr = (void**)((uintptr_t)((char*)mem + ALIGN_BYTES + sizeof(void*)) & ~(ALIGN_BYTES - 1));
			ptr[-1] = mem;
			return ptr;
		}

		static void aligned_free(void *ptr) {
			free(((void**)ptr)[-1]);
		}
	private:
		T* m_data;
		T* m_dataAlloc;
		int m_width;
		int m_height;
		int m_stride;
		std::atomic<int>* m_refCount;
	};

	typedef Image<uchar, 1, 4> ByteImage;
	typedef Image<uchar, 3, 4> RgbImage;
	typedef Image<uchar, 4, 4> RgbaImage;
	typedef Image<int, 1> IntImage;
	typedef Image<float, 1> FloatImage;
	typedef Image<float, 3> RgbFloatImage;
	typedef Image<float, 4> RgbaFloatImage;
}