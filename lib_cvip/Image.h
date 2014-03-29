#ifndef IMAGE_H
#define IMAGE_H

#include "Definitions.h"

namespace cvip
{
    struct RecRes {
        std::string name;
        float conf;
    };

    double factorial(int n);
    cv::Mat rgb2gray(const cv::Mat& I);

    cv::Mat orthogonalization(const cv::Mat &mat);
  
    // rotate a point with respect to another point
    cv::Point2f rotateWrt(const cv::Point& src, double oX, double oY, double angle);

    // rotate a point with respect to (0,0) point
    cv::Point2f rotate(const cv::Point& src, double angle);

    cv::Mat rotate(const cv::Mat& src, double angle);

    enum DetectorType {
        FACE, EYE, MOUTH, MID_OF_EYES, NOSE,
                           LEC, REC,   // left eye center, right eye center
                           LELC, LERC, // left  eye left corner, left eye right corner
                           RELC, RERC, // right  " ..
                           MLC, MRC,   // mouth left corner, mouth right corner
                           NL, NR,     // nose left, mouth right
                           NC,         // nose center
                           MC          // mouth center
    };

    // Rectangle class
    class Rect
    {
    public:
        int x1, y1, x2, y2; // Original bounds (inclusive)
        int width, height;

        Rect(int _x1, int _y1, uint _width, uint _height)
            : x1(_x1), y1(_y1), x2(_x1 + _width-1), y2(_y1 + _height-1), width(_width), height(_height)
        {}

        // default constructor
        Rect() : x1(0), y1(0), x2(0), y2(0), width(0), height(0) {}

        // Finds the area of overlap between 2 rectangles
        static uint intersect(const Rect &r1, const Rect &r2);

        // Determines whether the two rectangles match
        static bool match(const Rect &r1, const Rect &r2);

        bool isLegal() const { return width > 0 && height > 0 && x1 >= 0 && x2 > 0 && y1 >= 0 && y2 > 0; }

        bool in(const cv::Size& sz) const { return (x2 < sz.width) && (y2 < sz.height); }

        // shift rectangle
        void shift(int x, int y) { shiftX(x); shiftY(y); }

        // negative shifts left, positive shifts right
        void shiftX (int x) { x1 += x; x2 += x; }

        // negative shifts left, positive shifts right
        void shiftY (int y) { y1 += y; y2 += y; }

        // get opencv style Rectangle
        cv::Rect toCvStyle() const { return cv::Rect(x1, y1, width, height); }

        // Destructor
        virtual ~Rect() {}

        // extend the rect, add padding to all sizes as much as the image limits allow
        Rect extended(const cv::Size& sz, double maxPadRate = 0.20) const;

        Rect rescale(const cv::Size& sz, double maxPadRate = 0.20) const;

        // get the center point of the Rectangle
        cv::Point2f centerPoint() const { return cv::Point2f(cvip::round((x1+x2)/2.), cvip::round((y1+y2)/2.)); }
    };

    class UnscaledRect : public Rect
    {
        double actScale;

    public:
        UnscaledRect(int _x1, int _y1, uint _width, uint _height, double _actScale) :
            Rect(_x1, _y1, _width, _height), actScale(_actScale) {}

        Rect scaled() { return Rect(x1*actScale, y1*actScale, width*actScale, height*actScale); }
    };

    // Coordinates container, fetch all features such as coordsContainer["leftEye"] = cv::Rect;
    typedef std::map<std::string, cvip::Rect > CoordsContainer;
    typedef std::map<std::string, cv::Mat > ZernikeContainer;

    class DetectionRect : public Rect
    {
    public:
        uint pose, angle;
        double scale, confidence;

        DetectionRect(int _x1, int _y1, uint _width, uint _height, uint _pose, uint _angle, double _scale = 1.0, double _confidence = 1)
            : Rect(_x1, _y1, _width, _height),
            pose(_pose), angle(_angle), scale(_scale), confidence(_confidence) {}

        // Converts the rectangles coordinates to the original images coordinate system
        void applyScale();

        DetectionRect shrink(const cv::Size &sz, double newScale) const;

        // extend the rect, add padding to all sizes as much as the image limits allow
        DetectionRect extended(const cv::Size& sz, double maxPadRate = 0.20) const;
    };

    // Combines multiple detections
    void combine_detections(const std::vector<DetectionRect> &detections, std::vector<DetectionRect> &resultVector, uint minDetects = 2);

    // Combines multiple detections
    void _combine_detections(const std::vector<DetectionRect> &detections, std::vector<DetectionRect> &resultVector, uint minDetects, bool isMin);

    // Get strongest group of uncombined detections
    DetectionRect getStrongestCombination(const std::vector<DetectionRect> &detections);

    // Get strongest group of uncombined detections
    DetectionRect getStrongestCombinationAndErase(std::vector<DetectionRect> &detections);

    //eliminates some candidates by a simple heuristic
    std::vector<DetectionRect> eliminateCandidates(const std::vector<DetectionRect> &detections, int numOfSelected);

    class TrackingRect : public DetectionRect
    {
    public:
        const uint id;

        TrackingRect(uint _x1, uint _y1, uint _width, uint _height, uint _id, uint _pose, uint _angle, double _scale = 1.0, double _confidence = 1)
            : DetectionRect(_x1, _y1, _width, _height, _pose, _angle, _scale, _confidence), id(_id) {}
    };

    // Wrapper class for Mat
    class Image
    {
    public:
        double scale;
        uint width, height;
        cv::Mat I;		// Original image

        // Constructor
        Image(cv::Mat& Im, double _scale=1.0);

        static cv::Mat doubleToUchar(const cv::Mat& input);

        static std::vector<std::string> readStringVectorFromFile(std::string& filePath);

        static std::vector<double> readFileToVector(const std::string& FileName);

        static void writeVectorToFile(const std::vector<double>& doubles, const std::string& FileName);

        // Converts a BGR image to a grayscale image
        static void rgb2gray(const cv::Mat &I, cv::Mat &gI);

        // subsample a vector (row vector)
        static cv::Mat getSubsampled(const cv::Mat& input, int vectorSampling);

        static cv::Mat rgb2gray(const cv::Mat& I);

        static double dist(const cv::Point2f& p1, const cv::Point2f& p2);

        static void illumNorm(const cv::Mat &Input, cv::Mat &Output);

        // lbp transform of image
        static cv::Mat lbp(const cv::Mat& I, const uint d=2);

        // internal function for TPLBP transform
        static bool f(double d, double tau=0.01) { return d>=tau; }

        // internal function for TPLBP transform
        static double d(const cv::Mat& im, int iSrc, int jSrc, int iDst, int jDst, uint w);

        // three patch lbp of an image
        static cv::Mat tplbp(const cv::Mat& im, uint r=2, uint w=3);

        // three patch lbp of a single pixel
        static uchar tplbp(const cv::Mat& im, int ip, int jp, uint r, uint w, uint S=8, uint alpha=5);

        // extract dct features from an image
        static cv::Mat dct(const cv::Mat& _im);

        // Create a histogram descriptor from coded image (such as MCT or LBP)
        static cv::Mat histDescriptor(const cv::Mat& codeI, const uint m=3, const uint n=3);

        // convert image to mat file / read mat from file
        static void writeToFile(const cv::Mat& Data, std::string FileName);
        static cv::Mat readFromFile(const std::string& FileName, int rows, int cols);

        static void writeHistsToFile(const std::vector<cv::Mat>& hists, const std::string& filePath);
        static std::vector<cv::Mat> readHistsFromFile(const std::string& filePath, size_t binCnt);

        static bool exists(std::string filename);

        static std::vector<int> randomPerm(int n);

        // Vectorization
        static cv::Mat vectorize(const cv::Mat &I);

        static void zNormalize(cv::Mat& input);

        //Histogram matching
        static cv::Mat histogramMatch(const cv::Mat &Image, const cv::Mat &Hist);

        //convert cv::mat to uchar
        static uchar* convertCvMat2Uchar(const cv::Mat Image);
    };

    class HaarImage : public Image
    {
    public:
        cv::Mat II;
        cv::Mat sII;

        HaarImage(cv::Mat& Im, double _scale = 1);

        // Returns the integral image of the given grayscale image
        cv::Mat integral_image(const cv::Mat &I);

        static std::vector<HaarImage*> createScaleSpace(const cv::Mat &frame, uint W_SIZE, double SCALE_START, double SCALE_MAX, double SCALE_STEP);

        // Returns the sum of all pixels in the given rectangle according to the offset of the current window
        double sum(const Rect *rect, uint offsetX = 0, uint offsetY = 0) const;
    };


    //
    class Window
    {
    public:
        uint offsetX, offsetY, wSize, N, W_SIZE;
        double mean, std;

        // Constructors
        Window(uint _W_SIZE);

        // Sets the coordinates of a scaled and shifted window
        void setScanWindow(const HaarImage *Im, uint i, uint j);
    };
}
#endif
