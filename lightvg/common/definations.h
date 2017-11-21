#pragma once

// determine which OS
#if defined(_WIN32)
#define LVG_OS_WIN
#elif defined(__APPLE__) && defined(__MACH__)
#define LVG_OS_MAC
#else
#define LVG_OS_LNX
#endif

// define internal datatype
#if defined(LVG_OS_WIN)
#include <windows.h>
#undef min
#undef max
#elif defined(LVG_OS_MAC)
// http://developer.apple.com/qa/qa2004/qa1398.html
#include <mach/mach_time.h>
#elif defined(LVG_OS_LNX)
#include <sys/time.h>
#endif

#if defined(__ARM_NEON)
#define LVG_ENABLE_NEON
#include <arm_neon.h>
#include "intrin_neon.h"
#endif

#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifdef _WIN32
#define LVG_ENABLE_SSE
#include <xmmintrin.h>      // __m128 data type and SSE functions
#endif

// turn on this macro if you need opencv for debugging
#define LVG_ENABLE_OPENCV_DEBUG

// it seems android ndk does not support to_string()
#ifdef __ANDROID__
namespace std
{
	template <typename T>
	std::string to_string(T value)
	{
		std::ostringstream os;
		os << value;
		return os.str();
	}
}
#endif
