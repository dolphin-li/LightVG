#pragma once

#include "definations.h"
#include <map>
#include <mutex>
namespace lvg
{
	// MemPool: a simple allocator for caching allocation requests
	class MemPool
	{
	public:
		enum {ALIGN_BYTES = 4};
	public:
		MemPool() {}
		~MemPool();

		// allocate aligned memory from the pool
		static char *allocate(std::ptrdiff_t num_bytes);

		// deallocate from the pool
		static void deallocate(char* ptr);

		// manually free all memories reserved by the pool
		static void free_all();
	private:
		static void *aligned_malloc(size_t size, int alignBytes);
		static void aligned_free(void *ptr);
	private:
		typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
		typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;
		static free_blocks_type      free_blocks;
		static allocated_blocks_type allocated_blocks;
		static std::mutex s_mutex;
	};
} // lvg