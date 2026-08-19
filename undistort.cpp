#include <cmath>
#include <algorithm>
#include <limits>
#include "undistort.hpp"

static void distortPoint(const UndistortParams &p, float x, float y, float &dx, float &dy){
    float r2 = x * x + y * y;
    float num = 1.0f + r2 * (p.k1 + r2 * (p.k2 + r2 * p.k3));
    float den = 1.0f + r2 * (p.k4 + r2 * (p.k5 + r2 * p.k6));
    float radial = num / den;
    dx = x * radial + 2.0f * p.p1 * x * y + p.p2 * (r2 + 2.0f * x * x);
    dy = y * radial + p.p1 * (r2 + 2.0f * y * y) + 2.0f * p.p2 * x * y;
}


// Newton iteration, port of COLMAP IterativeUndistortion
static bool undistortPoint(const UndistortParams &p, float dx, float dy, float &ux, float &uy){
    const int kNumIterations = 100;
    const double kMinStepSquaredNorm = 1e-10;
    const double kRelStepRadius = 0.1;
    const double kStepRadius = 0.1;

    const double x0 = dx, y0 = dy;
    double x = x0, y = y0;
    bool converged = false;

    for (int it = 0; it < kNumIterations; it++){
        const double r2 = x * x + y * y;
        const double num = 1.0 + r2 * (p.k1 + r2 * (p.k2 + r2 * p.k3));
        const double den = 1.0 + r2 * (p.k4 + r2 * (p.k5 + r2 * p.k6));
        if (std::fabs(den) < 1e-12) break;
        const double radial = num / den;
        // d(radial)/d(r2) by the quotient rule
        const double dNum = p.k1 + r2 * (2.0 * p.k2 + 3.0 * p.k3 * r2);
        const double dDen = p.k4 + r2 * (2.0 * p.k5 + 3.0 * p.k6 * r2);
        const double dRadial = (dNum * den - num * dDen) / (den * den);

        const double fx = x * radial + 2.0 * p.p1 * x * y + p.p2 * (r2 + 2.0 * x * x) - x0;
        const double fy = y * radial + p.p1 * (r2 + 2.0 * y * y) + 2.0 * p.p2 * x * y - y0;
        const double j00 = radial + 2.0 * x * x * dRadial + 2.0 * p.p1 * y + 6.0 * p.p2 * x;
        const double j01 = 2.0 * x * y * dRadial + 2.0 * p.p1 * x + 2.0 * p.p2 * y;
        const double j10 = j01;
        const double j11 = radial + 2.0 * y * y * dRadial + 6.0 * p.p1 * y + 2.0 * p.p2 * x;
        const double det = j00 * j11 - j01 * j10;
        if (std::fabs(det) < 1e-12) break;

        double sx = (fx * j11 - fy * j01) / det;
        double sy = (fy * j00 - fx * j10) / det;

        // Trust region: |step| <= max(|x| * kRelStepRadius, kStepRadius)
        const double radiusSqr = (std::max)(r2 * kRelStepRadius * kRelStepRadius,
                                            kStepRadius * kStepRadius);
        const double stepSqr = sx * sx + sy * sy;
        if (stepSqr > radiusSqr){
            const double s = std::sqrt(radiusSqr / stepSqr);
            sx *= s;
            sy *= s;
        }
        x -= sx;
        y -= sy;
        if (sx * sx + sy * sy < kMinStepSquaredNorm){
            converged = true;
            break;
        }
    }

    ux = static_cast<float>(x);
    uy = static_cast<float>(y);
    return converged && std::isfinite(x) && std::isfinite(y);
}

UndistortParams computeUndistortParams(float fx, float fy, float cx, float cy,
                                       int width, int height,
                                       float k1, float k2, float k3,
                                       float k4, float k5, float k6,
                                       float p1, float p2,
                                       float blankPixels){
    UndistortParams p;
    p.srcFx = fx; p.srcFy = fy; p.srcCx = cx; p.srcCy = cy;
    p.srcW = width; p.srcH = height;
    p.dstFx = fx; p.dstFy = fy; p.dstCx = cx; p.dstCy = cy;
    p.dstW = width; p.dstH = height;
    p.k1 = k1; p.k2 = k2; p.k3 = k3; p.k4 = k4; p.k5 = k5; p.k6 = k6; p.p1 = p1; p.p2 = p2;

    if (k1 == 0.0f && k2 == 0.0f && k3 == 0.0f && k4 == 0.0f && k5 == 0.0f && k6 == 0.0f &&
        p1 == 0.0f && p2 == 0.0f) return p;

    const float inf = std::numeric_limits<float>::max();
    float leftMinX = inf, leftMaxX = -inf, rightMinX = inf, rightMaxX = -inf;
    float topMinY = inf, topMaxY = -inf, bottomMinY = inf, bottomMaxY = -inf;

    // Undistort a source pixel center (corner-origin convention) and reproject
    auto trace = [&](float px, float py, float &ox, float &oy){
        float ux, uy;
        if (!undistortPoint(p, (px - cx) / fx, (py - cy) / fy, ux, uy)) return false;
        ox = fx * ux + cx;
        oy = fy * uy + cy;
        return true;
    };

    for (int y = 0; y < height; y++){
        float ox, oy;
        if (trace(0.5f, y + 0.5f, ox, oy)){
            leftMinX = (std::min)(leftMinX, ox);
            leftMaxX = (std::max)(leftMaxX, ox);
        }
        if (trace(width - 0.5f, y + 0.5f, ox, oy)){
            rightMinX = (std::min)(rightMinX, ox);
            rightMaxX = (std::max)(rightMaxX, ox);
        }
    }
    for (int x = 0; x < width; x++){
        float ox, oy;
        if (trace(x + 0.5f, 0.5f, ox, oy)){
            topMinY = (std::min)(topMinY, oy);
            topMaxY = (std::max)(topMaxY, oy);
        }
        if (trace(x + 0.5f, height - 0.5f, ox, oy)){
            bottomMinY = (std::min)(bottomMinY, oy);
            bottomMaxY = (std::max)(bottomMaxY, oy);
        }
    }

    // If a whole border failed to solve there is nothing sane to rescale to
    if (leftMinX == inf || rightMaxX == -inf || topMinY == inf || bottomMaxY == -inf ||
        leftMaxX == -inf || rightMinX == inf || topMaxY == -inf || bottomMinY == inf){
        return p;
    }

    // Scale such that the undistorted image contains all source pixels (min)
    // or no blank pixels (max)
    float minScaleX = (std::min)(cx / (cx - leftMinX), (width - 0.5f - cx) / (rightMaxX - cx));
    float minScaleY = (std::min)(cy / (cy - topMinY), (height - 0.5f - cy) / (bottomMaxY - cy));
    float maxScaleX = (std::max)(cx / (cx - leftMaxX), (width - 0.5f - cx) / (rightMinX - cx));
    float maxScaleY = (std::max)(cy / (cy - topMaxY), (height - 0.5f - cy) / (bottomMinY - cy));

    float scaleX = 1.0f / (minScaleX * blankPixels + maxScaleX * (1.0f - blankPixels));
    float scaleY = 1.0f / (minScaleY * blankPixels + maxScaleY * (1.0f - blankPixels));
    scaleX = std::clamp(scaleX, 0.2f, 2.0f);
    scaleY = std::clamp(scaleY, 0.2f, 2.0f);

    p.dstW = (std::max)(1, static_cast<int>(scaleX * width));
    p.dstH = (std::max)(1, static_cast<int>(scaleY * height));
    p.dstCx = cx * static_cast<float>(p.dstW) / static_cast<float>(width);
    p.dstCy = cy * static_cast<float>(p.dstH) / static_cast<float>(height);
    return p;
}

void buildUndistortMaps(const UndistortParams &p, cv::Mat &mapx, cv::Mat &mapy){
    mapx.create(p.dstH, p.dstW, CV_32FC1);
    mapy.create(p.dstH, p.dstW, CV_32FC1);
    for (int oy = 0; oy < p.dstH; oy++){
        float *rx = mapx.ptr<float>(oy);
        float *ry = mapy.ptr<float>(oy);
        for (int ox = 0; ox < p.dstW; ox++){
            float nx = (ox + 0.5f - p.dstCx) / p.dstFx;
            float ny = (oy + 0.5f - p.dstCy) / p.dstFy;
            float dnx, dny;
            distortPoint(p, nx, ny, dnx, dny);
            rx[ox] = dnx * p.srcFx + p.srcCx - 0.5f;
            ry[ox] = dny * p.srcFy + p.srcCy - 0.5f;
        }
    }
}
