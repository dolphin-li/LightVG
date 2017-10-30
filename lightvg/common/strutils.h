#pragma once

#include "definations.h"
#include <string>
#include <fstream>
namespace lvg
{
	inline std::string fullfile(std::string path, std::string name)
	{
		if (path == "")
			return name;
		if (path.back() != '/' && path.back() != '\\')
			path.append("/");
		if (path != "" && name.size())
		{
			if (name[0] == '/' || name[0] == '\\')
				name = name.substr(1, name.size()-1);
		}
		return path + name;
	}

	inline void fileparts(std::string fullfile, std::string& path, std::string& name, std::string& ext)
	{
		size_t pos = fullfile.find_last_of('/');
		if (pos >= fullfile.size())
			pos = fullfile.find_last_of('\\');
		if (pos >= fullfile.size())
		{
			path = "";
		}
		else
		{
			path = fullfile.substr(0, pos + 1);
			fullfile = fullfile.substr(pos + 1, fullfile.size());
		}

		size_t pos1 = fullfile.find_last_of('.');
		if (pos1 >= fullfile.size())
		{
			name = fullfile;
		}
		else
		{
			name = fullfile.substr(0, pos1);
			ext = fullfile.substr(pos1, fullfile.size());
		}
	}

	inline bool file_exist(const char* path)
	{
		std::ifstream file(path);
		if (file.fail())
			return false;
		else
		{
			file.close();
			return true;
		}
	}

	// sample input:
	//	buffer = "label: abc"
	// sample output:
	//	buffer = "abc"
	//	return "label"
	inline std::string getLineLabel(std::string& buffer, char sep = ':')
	{
		std::string s;
		size_t pos = buffer.find_first_of(sep);
		if (pos < buffer.size())
		{
			s = buffer.substr(0, pos);	
			pos++;
			for (; pos < buffer.size();)
			{
				if (buffer[pos] == ' ')
					pos++;
				else
					break;
			}
			buffer = buffer.substr(pos); // ignore a space after :
		}
		return s;
	}
}