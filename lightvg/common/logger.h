#ifndef __LVGLOGGER_H__
#define __LVGLOGGER_H__

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
	inline void logging(LogLevel level, const char* tag, const char* format, ...)
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
   		va_list arg;
   		va_start (arg, format);
		__android_log_vprint(pri, tag, format, arg);
   		va_end (arg);
	}

#else

	inline void logging(LogLevel level, const char* tag, const char* format, ...)
	{		
		char levelC = '\0';
		switch (level)
		{
		default:
			break;
		case LVG_LOG_VERBOSE:
			levelC = 'V';
			break;
		case LVG_LOG_DEBUG:
			levelC = 'D';
			break;
		case LVG_LOG_INFO:
			levelC = 'I';
			break;
		case LVG_LOG_WARN:
			levelC = 'W';
			break;
		case LVG_LOG_ERROR:
			levelC = 'E';
			break;
		case LVG_LOG_FATAL:
			levelC = 'F';
			break;
		}
		printf("%c[%s]: ", levelC, tag);
   		va_list arg;
   		va_start (arg, format);
		vfprintf(stdout, format, arg);
   		va_end (arg);
		printf("\n");
	}
#endif

#define LVG_STRINGIZE(x)  LVG_STRINGIZE2(x)
#define LVG_STRINGIZE2(x) #x
#define LVG_LINE_STRING  LVG_STRINGIZE(__LINE__)
#define FILE_AND_LINE (__FILE__ " " LVG_LINE_STRING)
#define LVG_LOG(level, msg) logging(level, FILE_AND_LINE, "%s", msg)
#define LVG_LOGV(...) logging(lvg::LVG_LOG_VERBOSE, FILE_AND_LINE, __VA_ARGS__)
#define LVG_LOGD(...) logging(lvg::LVG_LOG_DEBUG, FILE_AND_LINE, __VA_ARGS__)
#define LVG_LOGI(...) logging(lvg::LVG_LOG_INFO, FILE_AND_LINE, __VA_ARGS__)
#define LVG_LOGW(...) logging(lvg::LVG_LOG_WARN, FILE_AND_LINE, __VA_ARGS__)
#define LVG_LOGE(...) logging(lvg::LVG_LOG_ERROR, FILE_AND_LINE, __VA_ARGS__)
#define LVG_LOGF(...) logging(lvg::LVG_LOG_FATAL, FILE_AND_LINE, __VA_ARGS__)
} // lvg

#endif