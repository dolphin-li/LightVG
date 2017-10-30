#pragma once

#include "definations.h"
#include <map>
#include <mutex>
namespace lvg
{
	// cached_allocator: a simple allocator for caching allocation requests
	class MemPool
	{
	public:
	public:
		MemPool() {}
		~MemPool();
		static char *allocate(std::ptrdiff_t num_bytes, int alignBytes);
		static void deallocate(char* ptr);
	protected:
		static void free_all();
		static void *aligned_calloc(size_t size, int alignBytes);
		static void aligned_free(void *ptr);
	private:
		typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
		typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;
		static free_blocks_type      free_blocks;
		static allocated_blocks_type allocated_blocks;
		static std::mutex s_mutex;
	};
} // lvg