#ifndef ZIP_UTILS_H
#define ZIP_UTILS_H

#include <string>

bool isZipArchive(const std::string &path);
std::string extractZipToCache(const std::string &zipPath);

#endif
