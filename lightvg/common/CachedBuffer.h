#pragma once
#include "mempool.h"
namespace lvg
{
	template <class T> class CachedBuffer
	{
	public:
		CachedBuffer() : m_data(0), m_size(0) {}
		CachedBuffer(size_t size) :CachedBuffer() { create(size); }
		~CachedBuffer() { clear(); }

		void create(size_t size)
		{
			if (m_data)
				clear();
			m_size = size;
			m_data = (T*)MemPool::allocate(size * sizeof(T));
		}

		void resize(size_t size)
		{
			T* data_old = m_data;
			m_data = (T*)MemPool::allocate(size * sizeof(T));
			if (data_old)
			{
				memcpy(m_data, data_old, std::min(size, m_size) * sizeof(T));
				MemPool::deallocate((char*)m_data);
			}
			m_size = size;
		}

		void clear()
		{
			MemPool::deallocate((char*)m_data);
			m_data = nullptr;
		}

		T& operator[](size_t i)
		{
			return m_data[i];
		}

		const T& operator[](size_t i)const
		{
			return m_data[i];
		}

		size_t size()const 
		{ 
			return m_size; 
		}

		T* data()
		{
			return m_data;
		}

		const T* data()const
		{
			return m_data;
		}
	private:
		T* m_data;
		size_t m_size;
	};
}
