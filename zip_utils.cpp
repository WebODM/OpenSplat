#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstring>
#include <miniz.h>
#include "zip_utils.hpp"

namespace fs = std::filesystem;

static FILE *openRead(const fs::path &p){
#ifdef _WIN32
    return _wfopen(p.c_str(), L"rb");
#else
    return fopen(p.c_str(), "rb");
#endif
}

bool isZipArchive(const std::string &path){
    fs::path p(path);
    if (!fs::is_regular_file(p)) return false;
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".zip";
}

static std::string crc32Hex(const fs::path &p){
    FILE *f = openRead(p);
    if (!f) throw std::runtime_error("Cannot open " + p.string());
    std::vector<unsigned char> buf(1 << 20);
    mz_ulong crc = MZ_CRC32_INIT;
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), f)) > 0){
        crc = mz_crc32(crc, buf.data(), n);
    }
    fclose(f);
    std::ostringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << crc;
    return ss.str();
}

static bool safeEntryPath(const std::string &name, fs::path &rel){
    std::string s = name;
    std::replace(s.begin(), s.end(), '\\', '/');
    fs::path p = fs::path(s).lexically_normal();
    if (p.empty() || p.is_absolute() || p.has_root_name()) return false;
    for (const auto &c : p){
        if (c == "..") return false;
    }
    rel = p;
    return true;
}

static size_t writeCallback(void *opaque, mz_uint64 ofs, const void *buf, size_t n){
    std::ofstream *out = static_cast<std::ofstream *>(opaque);
    out->write(static_cast<const char *>(buf), n);
    return out->good() ? n : 0;
}

static void extractAll(const fs::path &zipPath, const fs::path &destDir){
    FILE *f = openRead(zipPath);
    if (!f) throw std::runtime_error("Cannot open " + zipPath.string());
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));
    if (!mz_zip_reader_init_cfile(&zip, f, fs::file_size(zipPath), 0)){
        fclose(f);
        throw std::runtime_error("Invalid zip archive: " + zipPath.string());
    }
    try{
        mz_uint count = mz_zip_reader_get_num_files(&zip);
        for (mz_uint i = 0; i < count; i++){
            mz_zip_archive_file_stat st;
            if (!mz_zip_reader_file_stat(&zip, i, &st)){
                throw std::runtime_error("Cannot read zip entry in " + zipPath.string());
            }
            std::string name = st.m_filename;
            if (name.empty()) continue;
            fs::path rel;
            if (!safeEntryPath(name, rel)) continue;
            fs::path out = destDir / rel;
            if (mz_zip_reader_is_file_a_directory(&zip, i) || name.back() == '/' || name.back() == '\\'){
                fs::create_directories(out);
                continue;
            }
            if (out.has_parent_path()) fs::create_directories(out.parent_path());
            std::ofstream os(out, std::ios::binary);
            if (!os || !mz_zip_reader_extract_to_callback(&zip, i, writeCallback, &os, 0)){
                throw std::runtime_error("Failed to extract " + rel.string());
            }
        }
    }catch(...){
        mz_zip_reader_end(&zip);
        fclose(f);
        throw;
    }
    mz_zip_reader_end(&zip);
    fclose(f);
}

static fs::path descendSingleDir(fs::path dir){
    for (int depth = 0; depth < 4; depth++){
        fs::path only;
        int count = 0;
        for (const auto &e : fs::directory_iterator(dir)){
            std::string name = e.path().filename().string();
            if (name == "__MACOSX" || name == ".DS_Store") continue;
            only = e.path();
            if (++count > 1) break;
        }
        if (count == 1 && fs::is_directory(only)) dir = only;
        else break;
    }
    return dir;
}

std::string extractZipToCache(const std::string &zipPath){
    fs::path zp = fs::absolute(zipPath);
    fs::path dest = fs::temp_directory_path() / ("opensplat-" + crc32Hex(zp));
    if (!fs::exists(dest)){
        fs::path staging = dest;
        staging += ".partial";
        if (fs::exists(staging)) fs::remove_all(staging);
        std::cout << "Extracting " << zp.string() << " to " << dest.string() << std::endl;
        fs::create_directories(staging);
        extractAll(zp, staging);
        std::error_code ec;
        fs::rename(staging, dest, ec);
        if (ec){
            if (fs::exists(dest)) fs::remove_all(staging);
            else throw std::runtime_error("Cannot finalize extraction to " + dest.string());
        }
    }else{
        std::cout << "Using cached extraction: " << dest.string() << std::endl;
    }
    return descendSingleDir(dest).string();
}
