#pragma once

#include "definations.h"
#include "logger.h"
#include <stack>

namespace lvg
{
#if defined(LVG_OS_WIN)
	typedef LARGE_INTEGER gtime_t;
#elif defined(LVG_OS_MAC)
	typedef uint64_t gtime_t;
#elif defined(LVG_OS_LNX)
	typedef struct timeval gtime_t;
#endif


	// get current time
	static inline gtime_t gtime_now(void)
	{
#if defined(LVG_OS_WIN)
		gtime_t time;
		QueryPerformanceCounter(&time);
		return time;
#elif defined(LVG_OS_MAC)
		return mach_absolute_time();
#elif defined(LVG_OS_LNX)
		gtime_t time;
		gettimeofday(&time, NULL);
		return time;
#endif
	}

	// absolute difference between two times (in seconds)
	static inline double gtime_seconds(gtime_t start, gtime_t end)
	{
#if defined(LVG_OS_WIN)
		if (start.QuadPart > end.QuadPart) {
			gtime_t temp = end;
			end = start;
			start = temp;
		}
		gtime_t system_freq;
		QueryPerformanceFrequency(&system_freq);
		return (double)(end.QuadPart - start.QuadPart) / system_freq.QuadPart;
#elif defined(LVG_OS_MAC)
		if (start > end) {
			uint64_t temp = start;
			start = end;
			end = temp;
		}
		// calculate platform timing epoch
		static mach_timebase_info_data_t info;
		mach_timebase_info(&info);
		double nano = (double)info.numer / (double)info.denom;
		return (end - start) * nano * 1e-9;
#elif defined(LVG_OS_LNX)
		struct timeval elapsed;
		timersub(&start, &end, &elapsed);
		long sec = elapsed.tv_sec;
		long usec = elapsed.tv_usec;
		double t = sec + usec * 1e-6;
		return t >= 0 ? t : -t;
#endif
	}

	// matlab style functions for convinient timing calculation
	static std::stack<gtime_t> __lvg_static_tic_recorder_;
	static inline void tic() 
	{
		__lvg_static_tic_recorder_.push(gtime_now());
	}
	static inline double toc(const char* label = "TimeCost", bool isInfoShow = true) 
	{
		double sc = 0;
		if (!__lvg_static_tic_recorder_.empty())
		{
			sc = gtime_seconds(__lvg_static_tic_recorder_.top(), gtime_now());
			__lvg_static_tic_recorder_.pop();
			if (isInfoShow) 
			{
				logging(LogLevel::LVG_LOG_VERBOSE, label, std::to_string(sc).c_str());
			}
		}
		return sc;
	}
} // lvg

#ifdef LVG_DEBUG_TIMING

#define LVG_TIC() lvg::tic()
#define LVG_TOC(a, b) lvg::toc(a, b)

#define TIME_ACCUMULATE_INIT(num_frames_to_average) 	\
const static int lvg_debug_frames = num_frames_to_average;\
static int lvg_debug_cnt = 0;\
static float lvg_debug_sum[100] = { 0 }

#define TIME_ACCUMULATE_BEGIN(id) \
LVG_TIC()

#define TIME_ACCUMULATE_END(id) \
lvg_debug_sum[id] +=LVG_TOC("", false)

#define TIME_ACCUMULATE_RELEASE(callBackFunction) \
if (++lvg_debug_cnt == lvg_debug_frames)\
{\
	for (int i = 0; i < 100; i++)\
		lvg_debug_sum[i] /= lvg_debug_cnt;\
	callBackFunction;\
	memset(lvg_debug_sum, 0, sizeof(float) * 100); \
	lvg_debug_cnt = 0;\
}

#define TIME_ACCUMULATE_PRINT(msg, id)\
LVG_LOG(LogLevel::LVG_LOG_DEBUG, (msg + std::string(", ") + std::to_string(lvg_debug_sum[id])).c_str())

#else
#define LVG_TIC() 0
#define LVG_TOC(a, b) 0
#define TIME_ACCUMULATE_INIT(n) 0
#define TIME_ACCUMULATE_BEGIN(id) 0
#define TIME_ACCUMULATE_END(id) 0
#define TIME_ACCUMULATE_RELEASE(function) 0
#define TIME_ACCUMULATE_PRINT(msg, id) 0
#endif