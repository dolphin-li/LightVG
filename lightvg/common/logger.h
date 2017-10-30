#pragma once

#include <stdio.h>
#include <string>
#include "definations.h"

namespace lvg
{
	enum LogLevel
	{
		LVG_LOG_VERBOSE,
		LVG_LOG_DEBUG,
		LVG_LOG_INFO,
		LVG_LOG_WARN,
		LVG_LOG_ERROR,
		LVG_LOG_FATAL,
	};

#ifdef __ANDROID__
	inline void logging(LogLevel level, const char* tag, const char* msg)
	{
		android_LogPriority pri = ANDROID_LOG_INFO;
		switch (level)
		{
		default:
			break;
		case LVG_LOG_VERBOSE:
			pri = ANDROID_LOG_VERBOSE;
			break;
		case LVG_LOG_DEBUG:
			pri = ANDROID_LOG_DEBUG;
			break;
		case LVG_LOG_INFO:
			pri = ANDROID_LOG_INFO;
			break;
		case LVG_LOG_WARN:
			pri = ANDROID_LOG_WARN;
			break;
		case LVG_LOG_ERROR:
			pri = ANDROID_LOG_ERROR;
			break;
		case LVG_LOG_FATAL:
			pri = ANDROID_LOG_FATAL;
			break;
		}
		__android_log_write(pri, tag, msg);
	}

#else

	inline void logging(LogLevel level, const std::string& tag, const std::string& msg)
	{
		printf("[%s]: %s\n", tag.c_str(), msg.c_str());
	}
#endif

#define FILE_AND_LINE ((std::string(__FILE__)+"["+std::to_string(__LINE__)+"]").c_str())
#define LVG_LOG(level, msg) logging(level, FILE_AND_LINE, msg)
} // lvg