#include <filesystem>

#include "point_io.hpp"

namespace fs = std::filesystem;

std::string getVertexLine(std::ifstream &reader) {
    std::string line;

    // Skip comments
    do {
        std::getline(reader, line);

        if (line.find("element") == 0)
            return line;
        else if (line.find("comment") == 0)
            continue;
        else if (line.find("obj_info") == 0)
            continue;
        else
            throw std::runtime_error("Invalid PLY file");
    } while (true);
}

size_t getVertexCount(const std::string &line) {

    // Split line into tokens
    std::vector<std::string> tokens;

    std::istringstream iss(line);
    std::string token;
    while (std::getline(iss, token, ' '))
        tokens.push_back(token);

    if (tokens.size() < 3)
        throw std::runtime_error("Invalid PLY file");

    if (tokens[0] != "element" && tokens[1] != "vertex")
        throw std::runtime_error("Invalid PLY file");

    return std::stoi(tokens[2]);
}

PointSet *readPointSet(const std::string &filename) {
    PointSet *r;
    const fs::path p(filename);
    if (p.extension().string() == ".ply") r = fastPlyReadPointSet(filename);
    else if (p.extension().string() == ".bin") r = colmapReadPointSet(filename);
    else r = pdalReadPointSet(filename);

    return r;
}

PointSet *fastPlyReadPointSet(const std::string &filename) {
    std::ifstream reader(filename, std::ios::binary);
    if (!reader.is_open())
        throw std::runtime_error("Cannot open file " + filename);

    auto *r = new PointSet();

    std::string line;
    std::getline(reader, line);
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    if (line != "ply")
        throw std::runtime_error("Invalid PLY file (header does not start with ply)");

    std::getline(reader, line);
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    // We are reading an ascii ply
    bool ascii = line == "format ascii 1.0";

    const auto vertexLine = getVertexLine(reader);
    const auto count = getVertexCount(vertexLine);

    std::cout << "Reading " << count << " points" << std::endl;

    checkHeader(reader, "x");
    checkHeader(reader, "y");
    checkHeader(reader, "z");

    int c = 0;
    bool hasViews = false;
    bool hasNormals = false;
    bool hasColors = false;

    size_t redIdx = 0, greenIdx = 1, blueIdx = 2;

    std::getline(reader, line);
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    while (line != "end_header") {
        if (hasHeader(line, "nx") || hasHeader(line, "normal_x") || hasHeader(line, "normalx")) hasNormals = true;
        if (hasHeader(line, "red")) {
            hasColors = true;
            redIdx = c;
        }
        if (hasHeader(line, "green")) {
            hasColors = true;
            greenIdx = c;
        }
        if (hasHeader(line, "blue")) {
            hasColors = true;
            blueIdx = c;
        }
        if (hasHeader(line, "views")) hasViews = true;

        if (c++ > 100) break;
        std::getline(reader, line);
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
    }

    size_t colorIdxMin = std::min<size_t>(std::min<size_t>(redIdx, greenIdx), blueIdx);
    redIdx -= colorIdxMin;
    greenIdx -= colorIdxMin;
    blueIdx -= colorIdxMin;
    if (redIdx + greenIdx + blueIdx != 3) throw std::runtime_error("red/green/blue properties need to be contiguous");

    r->points.resize(count);
    if (hasNormals) r->normals.resize(count);
    if (hasColors) r->colors.resize(count);
    if (hasViews) r->views.resize(count);

    // if (hasNormals) std::cout << "N";
    // if (hasColors) std::cout << "C";
    // if (hasViews) std::cout << "V";
    // std::cout << std::endl;

    // Read points
    if (ascii) {
        uint16_t buf;

        for (size_t i = 0; i < count; i++) {
            reader >> r->points[i][0]
                >> r->points[i][1]
                >> r->points[i][2];
            if (hasNormals) {
                reader >> r->normals[i][0]
                    >> r->normals[i][1]
                    >> r->normals[i][2];
            }
            if (hasColors) {
                reader >> buf;
                r->colors[i][redIdx] = static_cast<uint8_t>(buf);
                reader >> buf;
                r->colors[i][greenIdx] = static_cast<uint8_t>(buf);
                reader >> buf;
                r->colors[i][blueIdx] = static_cast<uint8_t>(buf);
            }
            if (hasViews) {
                reader >> buf;
                r->views[i] = static_cast<uint8_t>(buf);
            }
        }
    }
    else {

        // Read points
        uint8_t color[3];

        for (size_t i = 0; i < count; i++) {
            reader.read(reinterpret_cast<char *>(r->points[i].data()), sizeof(float) * 3);

            if (hasNormals) {
                reader.read(reinterpret_cast<char *>(r->normals[i].data()), sizeof(float) * 3);
            }

            if (hasColors) {
                reader.read(reinterpret_cast<char *>(&color), sizeof(uint8_t) * 3);
                r->colors[i][redIdx] = color[0];
                r->colors[i][greenIdx] = color[1];
                r->colors[i][blueIdx] = color[2];
            }

            if (hasViews) {
                reader.read(reinterpret_cast<char *>(&r->views[i]), sizeof(uint8_t));
            }

        }
    }

    // std::vector<size_t> classes(255, 0);
    // for (size_t idx = 0; idx < count; idx++) {
    //     std::cout << r->points[idx][0] << " ";
    //     std::cout << r->points[idx][1] << " ";
    //     std::cout << r->points[idx][2] << " ";

    //     std::cout << std::to_string(r->colors[idx][0]) << " ";
    //     std::cout << std::to_string(r->colors[idx][1]) << " ";
    //     std::cout << std::to_string(r->colors[idx][2]) << " ";

    //     std::cout << std::endl;

    //     if (idx > 9) exit(1);
    // }

    // for (size_t i = 0; i < classes.size(); i++){
    //     std::cout << i << ": " << classes[i] << std::endl;
    // }
    // exit(1);

    reader.close();

    return r;
}

PointSet *pdalReadPointSet(const std::string &filename) {
    #ifdef WITH_PDAL
    pdal::StageFactory factory;
    const std::string driver = pdal::StageFactory::inferReaderDriver(filename);
    if (driver.empty()) {
        throw std::runtime_error("Can't infer point cloud reader from " + filename);
    }

    auto *r = new PointSet();
    pdal::Stage *s = factory.createStage(driver);
    pdal::Options opts;
    opts.add("filename", filename);
    s->setOptions(opts);

    auto *table = new pdal::PointTable();

    std::cout << "Reading points from " << filename << std::endl;

    s->prepare(*table);
    const pdal::PointViewSet pvSet = s->execute(*table);

    r->pointView = *pvSet.begin();
    const pdal::PointViewPtr pView = r->pointView;

    if (pView->empty()) {
        throw std::runtime_error("No points could be fetched");
    }

    std::cout << "Number of points: " << pView->size() << std::endl;

    const size_t count = pView->size();
    const pdal::PointLayoutPtr layout(table->layout());

    r->points.resize(count);
    bool hasColors = false;
    bool largeColors = false;

    if (layout->hasDim(pdal::Dimension::Id::Green)) {
        r->colors.resize(count);
        hasColors = true;
        for (pdal::PointId idx = 0; idx < count; ++idx) {
            if (pView->getFieldAs<uint16_t>(pdal::Dimension::Id::Green, idx) > 255) {
                largeColors = true;
                break;
            }
        }
    }

    for (pdal::PointId idx = 0; idx < count; ++idx) {
        auto p = pView->point(idx);
        r->points[idx][0] = p.getFieldAs<float>(pdal::Dimension::Id::X);
        r->points[idx][1] = p.getFieldAs<float>(pdal::Dimension::Id::Y);
        r->points[idx][2] = p.getFieldAs<float>(pdal::Dimension::Id::Z);

        if (hasColors) {
            if (largeColors) {
                r->colors[idx][0] = static_cast<uint8_t>((p.getFieldAs<double>(pdal::Dimension::Id::Red) / 65535.0) * 255.0);
                r->colors[idx][1] = static_cast<uint8_t>((p.getFieldAs<double>(pdal::Dimension::Id::Green) / 65535.0) * 255.0);
                r->colors[idx][2] = static_cast<uint8_t>((p.getFieldAs<double>(pdal::Dimension::Id::Blue) / 65535.0) * 255.0);
            }
            else {
                r->colors[idx][0] = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Red);
                r->colors[idx][1] = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Green);
                r->colors[idx][2] = p.getFieldAs<uint8_t>(pdal::Dimension::Id::Blue);
            }
        }

    }

    // std::vector<std::size_t> classes (255, 0);
    // for (size_t idx = 0; idx < count; idx++) {
        // std::cout << r->points[idx][0] << " ";
        // std::cout << r->points[idx][1] << " ";
        // std::cout << r->points[idx][2] << " ";

        // std::cout << std::to_string(r->colors[idx][0]) << " ";
        // std::cout << std::to_string(r->colors[idx][1]) << " ";
        // std::cout << std::to_string(r->colors[idx][2]) << " ";

        // std::cout << std::endl;

        // if (idx > 9) exit(1);

    // }

    // for (size_t i = 0; i < classes.size(); i++){
    //     std::cout << i << ": " << classes[i] << std::endl;
    // }
    // exit(1);

    return r;
    #else
    fs::path p(filename);
    throw std::runtime_error("Unsupported file extension " + p.extension().string() + ", build program with PDAL support for additional file types support.");
    #endif
}

PointSet *colmapReadPointSet(const std::string &filename){
    
    std::ifstream reader(filename, std::ios::binary);
    if (!reader.is_open()) throw std::runtime_error("Cannot open " + filename);

    auto *r = new PointSet();
    size_t numPoints = readBinary<uint64_t>(reader);
    std::cout << "Reading " << numPoints << " points" << std::endl;

    r->points.resize(numPoints);
    r->colors.resize(numPoints);

    for (size_t i = 0; i < numPoints; i++){
        readBinary<uint64_t>(reader); // point ID

        r->points[i][0] = readBinary<double>(reader);
        r->points[i][1] = readBinary<double>(reader);
        r->points[i][2] = readBinary<double>(reader);
        r->colors[i][0] = readBinary<uint8_t>(reader);
        r->colors[i][1] = readBinary<uint8_t>(reader);
        r->colors[i][2] = readBinary<uint8_t>(reader);

        readBinary<double>(reader); // error
        size_t trackLen = readBinary<uint64_t>(reader);
        for (size_t j = 0; j < trackLen; j++){
            readBinary<uint32_t>(reader); // imageId
            readBinary<uint32_t>(reader); // point2D Idx
        }
    }

    reader.close();

    return r;
}

void checkHeader(std::ifstream &reader, const std::string &prop) {
    std::string line;
    std::getline(reader, line);
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

    if (line.substr(line.length() - prop.length(), prop.length()) != prop) {
        throw std::runtime_error("Invalid PLY file (expected 'property * " + prop + "', but found '" + line + "')");
    }
}

bool hasHeader(const std::string &line, const std::string &prop) {
    //std::cout << line << " -> " << prop << " : " << line.substr(line.length() - prop.length(), prop.length()) << std::endl;
    return line.substr(0, 8) == "property" && line.substr(line.length() - prop.length(), prop.length()) == prop;
}
