#ifndef __LVGDEFINATIONS_H__
#define __LVGDEFINATIONS_H__

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
#include <unistd.h>
#elif defined(LVG_OS_LNX)
#include <sys/time.h>
#include <unistd.h>
#endif

#include <string>
#include <iostream>
#include <sstream>

#if (defined(__ARM_NEON__) || (defined(__ANDROID__) && defined(__ARM_ARCH_7A__)) || defined(__aarch64__))
#define LVG_ENABLE_NEON
#include "intrin_neon.h"
#endif

#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifdef _WIN32
#define LVG_ENABLE_SSE
#include "intrin_sse.h"
#endif

// turn on this macro if you need opencv for debugging
#define LVG_ENABLE_OPENCV_DEBUG

#endif
