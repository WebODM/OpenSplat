#ifndef UNDISTORT_H
#define UNDISTORT_H

#include <opencv2/core/core.hpp>

struct UndistortParams{
    float srcFx, srcFy, srcCx, srcCy;
    int srcW, srcH;
    float dstFx, dstFy, dstCx, dstCy;
    int dstW, dstH;
    float k1, k2, k3, k4, k5, k6, p1, p2;
};

UndistortParams computeUndistortParams(float fx, float fy, float cx, float cy,
                                       int width, int height,
                                       float k1, float k2, float k3,
                                       float k4, float k5, float k6,
                                       float p1, float p2,
                                       float blankPixels = 0.0f);
void buildUndistortMaps(const UndistortParams &p, cv::Mat &mapx, cv::Mat &mapy);

#endif
