#include "Image.h"
#include <queue>

using namespace cvip;

double cvip::factorial(int n) {
    double result;
    
    for (int i=0; i<=n; i++) {
        if (i == 0)
            result = 1;
        else
            result = result * i;
    }
    
    return result;
}


std::vector<std::string> Image::readStringVectorFromFile(std::string& filePath)
{
    std::vector<std::string> file;
    std::string line;
    file.clear();

    std::ifstream infile (filePath.c_str(), std::ios_base::in);
    while (getline(infile, line, '\n')) {
        file.push_back (line);
    }

    return file;
}


//function to read MAT from File
std::vector<double> Image::readFileToVector(const std::string& FileName) {

    std::vector<double> vals;

    std::ifstream DataFile(FileName.c_str());

    if (!DataFile.is_open()) {
        std::cerr << "Cannot open file: '" << FileName << "'" << std::endl;
        exit(-1);
    }

    double tmp(0);
    while (DataFile >> std::setprecision(16) >> tmp)
    {
        //DataFile >> std::setprecision(16) >> tmp;
        vals.push_back(tmp);
    }

    DataFile.close();

    return vals;
}


cv::Mat Image::doubleToUchar(const cv::Mat& input)
{
    cv::Mat output(input.rows, input.cols, CV_64FC1, cv::Scalar::all(0));

    double mn, mx;
    cv::minMaxLoc(input, &mn, &mx);

    output = input-mn;
    cv::minMaxLoc(output, &mn, &mx);
    output = (255/(mx+1e-16))*(output);

    output.convertTo(output, CV_8U);

    return output;
}

//function to read MAT from File
void Image::writeVectorToFile(const std::vector<double>& doubles, const std::string& fileName) {

    std::ofstream of(fileName.c_str());

    for (size_t i=0; i<doubles.size(); ++i) {
        of << doubles[i] << std::endl;
    }

    of.close();

    return;
}


cv::Mat cvip::rgb2gray(const cv::Mat& I) {
    if (I.channels() > 1) {
        cv::Mat I2(I.rows, I.cols, CV_8U);
        for (int i=0; i<I2.rows; i++) {
            for (int j=0; j<I2.cols; j++) {
                I2.at<uchar>(i,j) = cvip::round(I.at<cv::Vec3b>(i,j)[2]*0.2989 + I.at<cv::Vec3b>(i,j)[1]*0.5870 + I.at<cv::Vec3b>(i,j)[0]*0.1140);
            }
        }
        return I2;
    }
    return I;
}


cv::Mat cvip::orthogonalization(const cv::Mat &mat) {
    
    cv::Mat q = cv::Mat::zeros(mat.rows, mat.cols, CV_64FC1);
    cv::Mat r = cv::Mat::zeros(mat.cols, mat.cols, CV_64FC1);
    
    for (int k=0; k < mat.cols; k++) {
        
        r.at<double>(k,k) = cv::norm(mat.col(k).clone());
        
        if (r.at<double>(k,k) == 0.0) {
            break;
        }
        
        cv::Mat q_i = q.col(k);
        ((cv::Mat)(mat.col(k) / r.at<double>(k,k))).copyTo(q_i);
        
        for (int j=k+1; j<mat.cols; j++) {
            
            r.at<double>(k,j) = q.col(k).dot(mat.col(j));
            
            cv::Mat mat_i = mat.col(j);
            ((cv::Mat)(mat.col(j) - r.at<double>(k,j) * q.col(k))).copyTo(mat_i);
        }
    }
    
    return q;
}



double Image::dist(const cv::Point2f& p1, const cv::Point2f& p2)
{
    return sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2));
}



// METHOD DEFINITONS FOR THE "RECT" CLASS

// Finds the area of overlap between 2 rectangles
uint Rect::intersect(const Rect &r1, const Rect &r2)
{
    if ((r1.x1 <= r2.x2) &&
            (r1.x2 >= r2.x1) &&
            (r1.y1 <= r2.y2) &&
            (r1.y2 >= r2.y1)   )
        return (min<uint>(r1.x2,r2.x2)-max<uint>(r1.x1,r2.x1)+1)*(min<uint>(r1.y2,r2.y2)-max<uint>(r1.y1,r2.y1)+1);
    else
        return 0;
}

// rotate a point wrt to another point, angle in radians
cv::Point2f cvip::rotateWrt(const cv::Point &src, double oX, double oY, double angle)
{
    angle *= cvip::PI/180;
    
    return cv::Point2f(cos(angle)*(src.x-oX) - sin(angle)*(src.y-oY)+oX,
                       sin(angle)*(src.x-oX) + cos(angle)*(src.y-oY)+oY);
}

// rotate a point wrt to another point, angle in radians
cv::Point2f cvip::rotate(const cv::Point &src, double angle)
{
    angle *= cvip::PI/180;
    
    return cv::Point2f(cos(angle)*(src.x) - sin(angle)*(src.y),
                       sin(angle)*(src.x) + cos(angle)*(src.y));
}

// rotate image, angle in degrees
cv::Mat cvip::rotate(const cv::Mat &src, double angle)
{
    using namespace cv;
    Mat rotMat = getRotationMatrix2D(Point2f(src.cols/2.0,src.rows/2.0), angle, 1);
    Mat rotated(src.rows, src.cols, CV_8U, Scalar::all(0));
    warpAffine(src, rotated, rotMat, src.size());
    
    return rotated;
}


void cvip::combine_detections(const std::vector<DetectionRect> &detections, std::vector<DetectionRect> &resultVector, uint minDetects)
{
    std::vector<DetectionRect> tmpDetections;
    cvip::_combine_detections(detections, tmpDetections, minDetects, true);
    cvip::_combine_detections(tmpDetections, resultVector, 0, false);
}

// Combines multiple detections
void cvip::_combine_detections(const std::vector<DetectionRect> &detections, std::vector<DetectionRect> &resultVector, uint minDetects, bool isMin)
{
    std::vector< std::vector<uint> > groups;	// groups[i][j] = index (in the detection vector) of the jth rectangle belonging to the ith disjoint set
    std::vector<uint> intGroups;			// Will store the indexes of groups that have overlapping rectangles with the rectangle in question
    
    // If there are no detections, return an empty vector
    if (detections.size() == 0) return;
    
    // Create the first group and add it the first rectangle
    std::vector<uint> v;		// Dummy vector for storing the vectors before adding to the group
    v.push_back(0);
    groups.push_back(v);
    
    for (uint i=1; i<detections.size(); i++) // For every other rectangle except the first one
    {
        intGroups.clear();
        for (uint j=0; j<groups.size(); j++)	// Look at each group
        {
            for (uint k=0; k<groups[j].size(); k++)	// Look at all rectangles in the current group
            {
                // If there is a rectangle in the current group that overlaps with the current rectangle
                // add that group to the overlapping groups vector
                uint area = Rect::intersect(detections[i],detections[groups[j][k]]);
                if (area > 0)
                {
                    // Check if the amount of overlap is greater than a fixed amount
                    // If it isn't, then don't count this as an intersection
                    double ratio1 = (double)area/(detections[i].width*detections[i].height);
                    double ratio2 = (double)area/(detections[groups[j][k]].width*detections[groups[j][k]].height);
                    
                    if (isMin)
                    {
                        if (min<double>(ratio1,ratio2) > 0.70f)
                        {
                            intGroups.push_back(j);
                            break;
                        }
                    } else
                    {
                        if (max<double>(ratio1,ratio2) > 0.75f)
                        {
                            intGroups.push_back(j);
                            break;
                        }
                    }
                }
            }
        }
        // If one or more groups overlap with the current rectangle, combine the groups
        // and add the current rectangle to the resulting group
        uint cnt = intGroups.size();
        if (cnt > 0)
        {
            groups[intGroups[0]].push_back(i); // Add the current rectangle to the first overlapping group
            for (uint j=cnt-1; j>0; j--)
            {
                // Copy the rectangles in the jth group to the first group
                for (uint k=0; k<groups[intGroups[j]].size(); k++)
                    groups[intGroups[0]].push_back(groups[intGroups[j]][k]);
                // Delete the jth group
                groups.erase(groups.begin( ) + intGroups[j]);
            }
        }
        else // If the current rectangle doesn't belong to an existing group, create a new group and add the rectangle to it
        {
            v.clear();
            v.push_back(i);
            groups.push_back(v);
        }
    }
    
    // Combine the detections to form a single detection for each group
    for (uint i=0; i<groups.size(); i++)
    {
        uint x1 = 0, y1 = 0, x2 = 0, y2 = 0, j;
        
        uint gSize = groups[i].size();
        
        if (gSize < minDetects) continue;
        
        int poses[3] = {0,0,0};
        int angles[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
        
        for (j=0; j<gSize; j++)
        {
            poses[detections[groups[i][j]].pose]++;
            angles[detections[groups[i][j]].angle/30]++;
        }
        
        int amax = angles[0], amaxInd = 0;
        for (j=1; j<12; j++)
        {
            if (angles[j] > amax)
            {
                amax = angles[j];
                amaxInd = j;
            }
        }
        
        int pmax = poses[0], pmaxInd = 0;
        for (j=1; j<3; j++)
        {
            if (poses[j] > pmax)
            {
                pmax = poses[j];
                pmaxInd = j;
            }
        }
        
        if (amax < minDetects) continue;
        
        for (j=0; j<gSize; j++)
        {
            x1 += detections[groups[i][j]].x1;
            y1 += detections[groups[i][j]].y1;
            x2 += detections[groups[i][j]].x2;
            y2 += detections[groups[i][j]].y2;
        }
        x1 /= j;
        y1 /= j;
        x2 /= j;
        y2 /= j;
        
        DetectionRect r(x1,y1,x2-x1+1,y2-y1+1,pmaxInd,amaxInd*30,detections[groups[i][j-1]].scale);
        //#pragma omp critical
        resultVector.push_back(r);
    }
}


/**
 * Take multiple detections and return only the strongest one
 *
 * @param vector<DetectionRect> detections - Unc detectiosn
 * @return DetectionRect - the combination of the detection group with the most detections
 */
cvip::DetectionRect cvip::getStrongestCombination(const std::vector<DetectionRect> &detections)
{
    if (detections.size() == 0) {
        return cvip::DetectionRect(0,0,0,0,0,0);
    }
    std::vector<DetectionRect> resultVector;
    std::vector< std::vector<uint> > groups;	// groups[i][j] = index (in the detection vector) of the jth rectangle belonging to the ith disjoint set
    std::vector<uint> intGroups;			// Will store the indexes of groups that have overlapping rectangles with the rectangle in question

    // Create the first group and add it the first rectangle
    std::vector<uint> v;		// Dummy vector for storing the vectors before adding to the group
    v.push_back(0);
    groups.push_back(v);

    for (uint i=1; i<detections.size(); i++) // For every other rectangle except the first one
    {
        intGroups.clear();
        for (uint j=0; j<groups.size(); j++)	// Look at each group
        {
            for (uint k=0; k<groups[j].size(); k++)	// Look at all rectangles in the current group
            {
                // If there is a rectangle in the current group that overlaps with the current rectangle
                // add that group to the overlapping groups vector
                uint area = Rect::intersect(detections[i],detections[groups[j][k]]);
                if (area > 0)
                {
                    // Check if the amount of overlap is greater than a fixed amount
                    // If it isn't, then don't count this as an intersection
                    double ratio1 = (double)area/(detections[i].width*detections[i].height);
                    double ratio2 = (double)area/(detections[groups[j][k]].width*detections[groups[j][k]].height);
                    if (min<double>(ratio1,ratio2) > 0.65)
                    {
                        intGroups.push_back(j);
                        break;
                    }
                }
            }
        }
        // If one or more groups overlap with the current rectangle, combine the groups
        // and add the current rectangle to the resulting group
        uint cnt = intGroups.size();
        if (cnt > 0)
        {
            groups[intGroups[0]].push_back(i); // Add the current rectangle to the first overlapping group
            for (uint j=cnt-1; j>0; j--)
            {
                // Copy the rectangles in the jth group to the first group
                for (uint k=0; k<groups[intGroups[j]].size(); k++)
                    groups[intGroups[0]].push_back(groups[intGroups[j]][k]);
                // Delete the jth group
                groups.erase(groups.begin( ) + intGroups[j]);
            }
        }
        else // If the current rectangle doesn't belong to an existing group, create a new group and add the rectangle to it
        {
            v.clear();
            v.push_back(i);
            groups.push_back(v);
        }
    }

    int largestGroupIdx = -1;
    uint largestGroupSize = 0;

    // Combine the detections to form a single detection for each group
    for (uint i=0; i<groups.size(); i++)
    {
        uint x1 = 0, y1 = 0, x2 = 0, y2 = 0, j;

        uint gSize = groups[i].size();

        int poses[3] = {0,0,0};
        int angles[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

        for (j=0; j<gSize; j++)
        {
            poses[detections[groups[i][j]].pose]++;
            angles[detections[groups[i][j]].angle/30]++;
        }

        int amax = angles[0], amaxInd = 0;
        for (j=1; j<12; j++)
        {
            if (angles[j] > amax)
            {
                amax = angles[j];
                amaxInd = j;
            }
        }

        int pmax = poses[0], pmaxInd = 0;
        for (j=1; j<3; j++)
        {
            if (poses[j] > pmax)
            {
                pmax = poses[j];
                pmaxInd = j;
            }
        }

        for (j=0; j<gSize; j++)
        {
            x1 += detections[groups[i][j]].x1;
            y1 += detections[groups[i][j]].y1;
            x2 += detections[groups[i][j]].x2;
            y2 += detections[groups[i][j]].y2;
        }
        x1 /= j;
        y1 /= j;
        x2 /= j;
        y2 /= j;

        DetectionRect r(x1,y1,x2-x1+1,y2-y1+1,pmaxInd,amaxInd*30,detections[groups[i][j-1]].scale);
        //#pragma omp critical

        if (largestGroupSize < gSize)
        {
            largestGroupSize = gSize;
            if (largestGroupIdx != -1)
                resultVector.erase(resultVector.begin()+largestGroupIdx);
            largestGroupIdx = i;
            resultVector.push_back(r);
        }
    }

    if (resultVector.size()>0)
        return resultVector[0];

    return DetectionRect(0,0,0,0,0,0,0);
}


/**
 * Take multiple detections and return only the strongest one
 *
 * @param vector<DetectionRect> detections - Unc detectiosn
 * @return DetectionRect - the combination of the detection group with the most detections
 */
cvip::DetectionRect cvip::getStrongestCombinationAndErase(std::vector<DetectionRect> &detections)
{
    if (detections.size() == 0) {
        return cvip::DetectionRect(0,0,0,0,0,0);
    }
    std::vector<DetectionRect> resultVector;
    std::vector< std::vector<uint> > groups;	// groups[i][j] = index (in the detection vector) of the jth rectangle belonging to the ith disjoint set
    std::vector<uint> intGroups;			// Will store the indexes of groups that have overlapping rectangles with the rectangle in question

    // Create the first group and add it the first rectangle
    std::vector<uint> v;		// Dummy vector for storing the vectors before adding to the group
    v.push_back(0);
    groups.push_back(v);

    std::priority_queue<uint> keysToErase;

    for (uint i=1; i<detections.size(); i++) // For every other rectangle except the first one
    {
        intGroups.clear();
        for (uint j=0; j<groups.size(); j++)	// Look at each group
        {
            for (uint k=0; k<groups[j].size(); k++)	// Look at all rectangles in the current group
            {
                // If there is a rectangle in the current group that overlaps with the current rectangle
                // add that group to the overlapping groups vector
                uint area = Rect::intersect(detections[i],detections[groups[j][k]]);
                if (area > 0)
                {
                    // Check if the amount of overlap is greater than a fixed amount
                    // If it isn't, then don't count this as an intersection
                    double ratio1 = (double)area/(detections[i].width*detections[i].height);
                    double ratio2 = (double)area/(detections[groups[j][k]].width*detections[groups[j][k]].height);
                    if (min<double>(ratio1,ratio2) > 0.65)
                    {
                        intGroups.push_back(j);
                        break;
                    }
                }
            }
        }
        // If one or more groups overlap with the current rectangle, combine the groups
        // and add the current rectangle to the resulting group
        uint cnt = intGroups.size();
        if (cnt > 0)
        {
            groups[intGroups[0]].push_back(i); // Add the current rectangle to the first overlapping group
            for (uint j=cnt-1; j>0; j--)
            {
                // Copy the rectangles in the jth group to the first group
                for (uint k=0; k<groups[intGroups[j]].size(); k++)
                    groups[intGroups[0]].push_back(groups[intGroups[j]][k]);
                // Delete the jth group
                groups.erase(groups.begin( ) + intGroups[j]);
            }
        }
        else // If the current rectangle doesn't belong to an existing group, create a new group and add the rectangle to it
        {
            v.clear();
            v.push_back(i);
            groups.push_back(v);
        }
    }

    int largestGroupIdx = -1;
    uint largestGroupSize = 0;

    // Combine the detections to form a single detection for each group
    for (uint i=0; i<groups.size(); i++)
    {
        uint x1 = 0, y1 = 0, x2 = 0, y2 = 0, j;

        uint gSize = groups[i].size();

        int poses[3] = {0,0,0};
        int angles[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

        for (j=0; j<gSize; j++)
        {
            poses[detections[groups[i][j]].pose]++;
            angles[detections[groups[i][j]].angle/30]++;
        }

        int amax = angles[0], amaxInd = 0;
        for (j=1; j<12; j++)
        {
            if (angles[j] > amax)
            {
                amax = angles[j];
                amaxInd = j;
            }
        }

        int pmax = poses[0], pmaxInd = 0;
        for (j=1; j<3; j++)
        {
            if (poses[j] > pmax)
            {
                pmax = poses[j];
                pmaxInd = j;
            }
        }

        for (j=0; j<gSize; j++)
        {
            x1 += detections[groups[i][j]].x1;
            y1 += detections[groups[i][j]].y1;
            x2 += detections[groups[i][j]].x2;
            y2 += detections[groups[i][j]].y2;
        }
        x1 /= j;
        y1 /= j;
        x2 /= j;
        y2 /= j;

        DetectionRect r(x1,y1,x2-x1+1,y2-y1+1,pmaxInd,amaxInd*30,detections[groups[i][j-1]].scale);
        //#pragma omp critical

        if (largestGroupSize < gSize)
        {
            largestGroupSize = gSize;
            if (largestGroupIdx != -1)
                resultVector.erase(resultVector.begin()+largestGroupIdx);
            largestGroupIdx = i;
            resultVector.push_back(r);
        }
    }

    if (resultVector.size()>0) {

        // remove elements of this group
        if (largestGroupSize>0) {
            for (uint i=0; i<groups[largestGroupIdx].size(); ++i) {
                keysToErase.push(groups[largestGroupIdx][i]);
            }

            while (keysToErase.size()) {
                detections.erase(keysToErase.top()+detections.begin());
                keysToErase.pop();
            }
        }
        return resultVector[0];
    }

    return DetectionRect(0,0,0,0,0,0,0);
}


//eliminates some candidates by a simple heuristic
std::vector<DetectionRect> cvip::eliminateCandidates(const std::vector<DetectionRect> &detections, int numOfSelected)
{
    cv::Mat distances(detections.size(), detections.size(), CV_32F);
    cv::Mat means(1, detections.size(), CV_32F);
    cv::Mat ordered;
    
    std::vector<DetectionRect> selected;
    
    for (uint k=0; k<detections.size(); ++k) {
        for (uint l=0; l<detections.size(); ++l) {
            float one = std::pow(detections[k].centerPoint().x - detections[l].centerPoint().x, 2);
            float two = std::pow(detections[k].centerPoint().y - detections[l].centerPoint().y, 2);
            distances.at<float>(k,l) = std::sqrt(one + two);
        }
        means.at<float>(k) = *(cv::mean(distances.row(k)).val);
    }
    
    cv::sortIdx(means, ordered, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
    
    for (uint k=0; k<numOfSelected; ++k) {
        int l = ordered.at<int>(k);
        
        selected.push_back(detections[l]);
    }
    
    return selected;
}

/**
 * Subsample an input vector, assuming that the input vector is a row vector
 *
 * @param input - mat row vector
 * @return mat - subsampled vector
 */
cv::Mat Image::getSubsampled(const cv::Mat &input, int vectorSampling)
{
    cv::Mat output(1, std::floor((double)input.cols/vectorSampling), input.type());
    
    // subsample
    for (uint jj=0; jj<input.cols; jj+=vectorSampling)
    {
        double mean=0;
        for (uint ii=0; ii<vectorSampling; ++ii)
            mean += input.at<double>(0, jj+ii);
        output.at<double>(0, std::floor((double)jj/vectorSampling)) = mean/vectorSampling;
    }
    
    return output;
}

// Determines whether the two rectangles match
bool Rect::match(const Rect &r1, const Rect &r2)
{
    uint area = intersect(r1,r2);
    if (area > 0)
    {
        // Check if the amount of overlap is greater than a fixed amount
        // If it is, then they match, else they don't
        double ratio1 = (double)area/(r1.width*r1.height);
        double ratio2 = (double)area/(r2.width*r2.height);
        
        if (min<double>(ratio1,ratio2) < 0.4)
            return false;
        
        // Check the distances between their centers
        // If it is shorter than the half of the width of the smaller rectangle, than they match, else they don't
        uint minw = min<uint>(r1.width,r2.width);
        double dist = std::sqrt(std::pow((double)r1.x1 + r1.width/2.0f - r2.x1 - r2.width/2.0f, 2)+std::pow((double)r1.y1 + r1.height/2.0f - r2.y1 - r2.height/2.0f, 2));
        if (dist < minw/2.5f)
            return true;
    }
    return false;
}

// extend the rect, add padding to all sizes as much as the image limits allow
// usually needed when rotating image parts, to avoid black regions
Rect Rect::extended(const cv::Size& sz, double maxPadRate) const
{
    maxPadRate /= 2;
    Rect d(*this);
    
    int maxL = x1;
    int maxR = sz.width-x2-1;
    int maxU = y1;
    int maxB = sz.height-y2-1;
    
    using std::min;
    
    // calculate the minimum rate that this box can extand
    int padding = min(maxL, min(maxR, min(maxU, maxB)));
    if (padding > d.width*maxPadRate) padding = d.width*maxPadRate;
    
    d.x1 -= padding;
    d.y1 -= padding;
    d.x2 += padding;
    d.y2 += padding;
    //uint minEdge = std::min<uint>(d.x2-d.x1, d.y2-d.y1);
    d.width = d.x2-d.x1+1;
    d.height= d.y2-d.y1+1;
    
    return d;
}

Rect Rect::rescale(const cv::Size &sz, double newScale) const
{
    if (newScale > 1.0) {
        return extended(sz, newScale-1.0);
    }

    uint newWidth = newScale*width;
    uint newHeight = newScale*height;

    uint newX1 = ((x1+x2)/2.) - newWidth/2.;
    uint newY1 = ((y1+y2)/2.) - newHeight/2.;

    return Rect(newX1, newY1, newWidth, newHeight);
}

DetectionRect DetectionRect::shrink(const cv::Size &sz, double newScale) const
{
    uint newWidth = newScale*width;
    uint newHeight = newScale*height;

    uint newX1 = ((x1+x2)/2.) - newWidth/2.;
    uint newY1 = ((y1+y2)/2.) - newHeight/2.;

    return DetectionRect(newX1, newY1, newWidth, newHeight, this->pose, this->angle, this->scale*newScale, this->confidence);
}

void DetectionRect::applyScale()
{
    x1 = round(x1*scale);
    y1 = round(y1*scale);
    
    width = round(width*scale);
    height = round(height*scale);
    x2 = x1 + width - 1;
    y2 = y1 + height - 1;
}

// extend the rect, add padding to all sizes as much as the image limits allow
// usually needed when rotating image parts, to avoid black regions
DetectionRect DetectionRect::extended(const cv::Size& sz, double maxPadRate) const
{
    DetectionRect d(*this);
    
    int maxL = x1;
    int maxR = sz.width-x2-1;
    int maxU = y1;
    int maxB = sz.height-y2-1;
    
    using std::min;
    
    // calculate the minimum rate that this box can extand
    int padding = min(maxL, min(maxR, min(maxU, maxB)));
    if (padding > d.width*maxPadRate) padding = d.width*maxPadRate;
    
    d.x1 -= padding;
    d.y1 -= padding;
    d.x2 += padding;
    d.y2 += padding;
    uint minEdge = std::min<uint>(d.x2-d.x1, d.y2-d.y1);
    d.width = minEdge;
    d.height= minEdge;
    d.scale = d.width*scale/width;
    
    return d;
}

// METHOD DEFINITONS FOR THE "IMAGE" CLASS
Image::Image(cv::Mat& Im, double _scale)
    :I(Im.rows, Im.cols, CV_8U)
{
    if (Im.channels() > 1)
        cv::cvtColor(Im, I, CV_RGB2GRAY);
    else
        I = Im;
    
    scale = _scale;
    width = Im.cols;
    height = Im.rows;
}

void Image::rgb2gray(const cv::Mat &I, cv::Mat &gI)
{
    for (int i=0; i<gI.rows; i++)
        for (int j=0; j<gI.cols; j++)
            gI.at<uchar>(i,j) = round(I.at<cv::Vec3b>(i,j)[2]*0.2989f + I.at<cv::Vec3b>(i,j)[1]*0.5870f + I.at<cv::Vec3b>(i,j)[0]*0.1140f);
}

cv::Mat Image::rgb2gray(const cv::Mat& I) {
    if (I.channels() > 1) {
        cv::Mat I2(I.rows, I.cols, CV_8U);
        for (int i=0; i<I2.rows; i++) {
            for (int j=0; j<I2.cols; j++) {
                I2.at<uchar>(i,j) = round(I.at<cv::Vec3b>(i,j)[2]*0.2989 + I.at<cv::Vec3b>(i,j)[1]*0.5870 + I.at<cv::Vec3b>(i,j)[0]*0.1140);
            }
        }
        return I2;
    }
    return I;
}


void Image::illumNorm(const cv::Mat &Input, cv::Mat &Output)
{
    float gamma = 0.2;
    float sigma0 = 1.0;
    float sigma1 = 2.0;
    float alfa = 0.1;
    float tau = 10;
    
    //Gamma correction
    cv::Mat faceGamma(Input.rows, Input.cols, CV_64F);
    cv::pow(Input, gamma, faceGamma);
    
    //Image::writeToFile(faceGamma, "faceGamma.mat");
    
    //DoG filtering
    cv::Mat faceGauss1(Input.rows, Input.cols, CV_64F);
    cv::Mat faceGauss2(Input.rows, Input.cols, CV_64F);
    cv::Mat faceDoG(Input.rows, Input.cols, CV_64F);
    
    cv::Mat Kernel1 = cv::getGaussianKernel(7,sigma0); //narrow filter (7)
    cv::sepFilter2D(faceGamma, faceGauss1, -1, Kernel1, Kernel1);
    cv::Mat Kernel2 = cv::getGaussianKernel(13,sigma1); //wide filter (13)
    cv::sepFilter2D(faceGamma, faceGauss2, -1, Kernel2, Kernel2);
    
    faceDoG = faceGauss1 - faceGauss2;
    
    //Image::writeToFile(faceDoG, "faceDoG.mat");
    
    //Global contrast normalization
    cv::Mat faceAbs(Input.rows, Input.cols, CV_64F);
    cv::pow(cv::abs(faceDoG), alfa, faceAbs);
    
    faceAbs = faceDoG / std::pow(cv::mean(faceAbs).val[0], 1.0/alfa);
    
    cv::Mat faceNorm(Input.rows, Input.cols, CV_64F);
    cv::min(cv::abs(faceAbs), tau, faceNorm);
    cv::pow(faceNorm, alfa, faceNorm);
    faceNorm = faceAbs / std::pow(cv::mean(faceNorm).val[0], 1.0/alfa);
    
    //Image::writeToFile(faceNorm, "faceNorm.mat");
    
    for (int i=0; i<faceNorm.rows; i++)
        for (int j=0; j<faceNorm.cols; j++)
            faceNorm.at<double>(i,j) = tau * std::tanh(faceNorm.at<double>(i,j) / tau);
    
    Output = faceNorm;//faceNorm.convertTo(Output, CV_8U, 255.0/(2.0*tau),127.5);
    
    //Image::writeToFile(Output, "faceTanh.mat");
}

/**
 * Get LBP code of input image.
 * Takes CV_8U images! Will not work with CV_64F etc.
 *
 * @param Mat in - input image: single channel and CV_8U
 * @param uint d - the distance of LBP code
 *
 * @return Mat - LBP coded image
 */
cv::Mat Image::lbp(const cv::Mat& in, const uint d)
{
    cv::Mat out = cv::Mat::zeros(in.rows, in.cols, CV_8U);
    for (uint j=d; j<in.rows-d; ++j) {
        for (uint i=d; i<in.cols-d; ++i)
        {
            uchar val = 0;
            val |= (uchar)(in.at<uchar>(j-d,i-d) > in.at<uchar>(j,i)) << 7;
            val |= (uchar)(in.at<uchar>(j-d,i) > in.at<uchar>(j,i)) << 6;
            val |= (uchar)(in.at<uchar>(j-d,i+d) > in.at<uchar>(j,i)) << 5;
            val |= (uchar)(in.at<uchar>(j,d-d) > in.at<uchar>(j,i)) << 4;
            val |= (uchar)(in.at<uchar>(j,d+d) > in.at<uchar>(j,i)) << 3;
            val |= (uchar)(in.at<uchar>(j+d,i-d) > in.at<uchar>(j,i)) << 2;
            val |= (uchar)(in.at<uchar>(j+d,i) > in.at<uchar>(j,i)) << 1;
            val |= (uchar)(in.at<uchar>(j+d,i+d) > in.at<uchar>(j,i));
            
            out.at<uchar>(j,i) = val;
        }
    }
    
    return out;
}

/**
 * Take a coded image (such as LBP or MCT) and return histogram output
 *
 * @param Mat codeI - transformed input image (must be single channel!) calcHist function requires cv_8u or cv_32F
 * @param uint m - divide image to m pieces in X direction
 * @param uint n - divide image to n pieces in Y direction
 *
 * @return Mat - return concatenated partial histograms (type: CV_64F)
 */
cv::Mat Image::histDescriptor(const cv::Mat& codeI, const uint m, const uint n)
{
    uint stepWidth = std::floor((double)codeI.cols/m);
    uint stepHeight = std::floor((double)codeI.rows/n);
    std::vector<cv::Mat> hists(m*n, cv::Mat());
    
    // calculate local histograms: divide image to an mxn grid
    for (uint j=0; j<n; ++j) {
        uint y1 = j*stepHeight;
        
        for (uint i=0; i<m; ++i) {
            uint x1 = i*stepWidth;
            
            // histogram of this part
            hists[j*m+i] = cv::Mat::zeros(256,1,CV_64F);
            
            for (uint jj=y1; jj<y1+stepHeight; ++jj)
                
                for (uint ii=x1; ii<x1+stepWidth; ++ii)
                    hists[j*m+i].at<double>(codeI.at<uchar>(jj,ii),0) = hists[j*m+i].at<double>(codeI.at<uchar>(jj,ii),0)+1.;
        }
    }
    
    // now create and fill the descriptor
    cv::Mat out(1,m*n*hists[0].rows,CV_64F, cv::Scalar::all(0));
    
    for (uint i=0; i<m*n; ++i)
        for (int k=0; k<hists[0].rows; ++k)
        {
            // a temp matrix is needed for min function
            cv::Mat temp(hists[i].rows, hists[i].cols, CV_64F);
            
            cv::normalize(hists[i], hists[i]);
            cv::min(hists[i], 0.2, temp);
            cv::normalize(temp, temp);
            
            out.at<double>(0,i*hists[0].rows+k) = temp.at<double>(k,0);
        }
    
    return out;
}



/**
 * Internal function for TPLBP transformation
 * Operates on CV_64F images!
 *
 * @param Mat im - input image CV_64F
 * @param uint iSrc, jSrc - coordinates of the first pixel
 * @param uint iDst, jDst - coordinates of the second pixel
 *
 * @return double - the distance bw two patches
 */
double Image::d(const cv::Mat& im, int iSrc, int jSrc, int iDst, int jDst, uint w)
{
    double val=0;
    
    for (int j=-std::floor(w/2.); j<std::ceil(w/2.); ++j)
        for (int i=-std::floor(w/2.); i<std::ceil(w/2.); ++i)
            val += std::pow(im.at<double>(jSrc+j, iSrc+i)-im.at<double>(jDst+j,iDst+i),2);
    
    return std::sqrt(val);
}

/**
 * TPLBP (three patch lbp) transform of a single pixel
 *
 * @param Mat im - input image
 * @param int ip, jp - pixel coordinates of the point to convert to TPLBP
 * @param uint r, w, S, alpha - parameters of TPLBP transform
 *
 * @return uchar - TPLBP transform of the pixel
 */
uchar Image::tplbp(const cv::Mat& im, int ip, int jp, uint r, uint w, uint S, uint alpha)
{
    // output value
    uchar val = 0;
    
    for (uint s=0; s<S; ++s)
    {
        // bildiride interpolasyon yerine n.neighbour kullaniyorlar
        int i1 = ip+cvip::round(r*cos(315*(s+4)*cvip::PI/180.)); // 315 derece = 360-45
        int j1 = jp+cvip::round(r*sin(315*(s+4)*cvip::PI/180.)); // yonumuz saat yonunun tersi
        
        // descriptoru hesaplarken, iki adim sonraki LBP yamasini da hesaba katiyoruz (s+2)
        int i2 = ip+cvip::round(r*cos(315*(s+alpha+4)*cvip::PI/180.));
        int j2 = jp+cvip::round(r*sin(315*(s+alpha+4)*cvip::PI/180.));
        
        val |= f(d(im, i1, j1, ip, jp, w)-d(im, i2, j2, ip, jp, w)) << s;
    }
    
    return val;
}

/**
 * TPLBP (three patch lbp) transform of a whole image
 *
 * @param Mat im - input image
 * @param uint r, w - parameters of TPLBP transform
 *
 * @return Mat - TPLBP transform of an image
 */
cv::Mat Image::tplbp(const cv::Mat& im, uint r, uint w)
{
    cv::Mat scratchIm, out(im.rows, im.cols, CV_8U, cv::Scalar::all(0));
    
    // we need double image for TPLBP computation
    im.convertTo(scratchIm, CV_64F);
    
    for (uint j=r+w/2; j<scratchIm.rows-r-w/2; ++j)
        for (uint i=r+w/2; i<scratchIm.cols-r-w/2; ++i)
            out.at<uchar>(j,i) = tplbp(scratchIm, i, j, r, w);
    
    return out;
}


/**
 * Extract dct features from a given image
 *
 * @param Mat im - input image, must be GRAYSCALE and CV_64F!!!
 * @return Mat - dct feature vector, see Ekenel and Stiefelhagen, 2006, CVPR(w)
 */
cv::Mat Image::dct(const cv::Mat& _im)
{
    // we are working with 64x64 images
    cv::Mat im;
    cv::resize(_im, im, cv::Size(64,64), 0, 0, cv::INTER_LANCZOS4);
    
    std::vector<cv::Point> pts;
    pts.reserve(5);
    
    // the points where features are extracted from, see Ekenel and Stiefelhagen, 2006, CVPR(w)
    pts.push_back(cv::Point(1,0));
    pts.push_back(cv::Point(0,1));
    pts.push_back(cv::Point(0,2));
    pts.push_back(cv::Point(1,1));
    pts.push_back(cv::Point(2,0));
    
    // output vector, create and fill immediately
    cv::Mat out = cv::Mat::zeros(8*8*pts.size(), 1, CV_64F);
    
    for (uint j=0; j<8; ++j) {
        for (uint i=0; i<8; ++i) {
            for (uint p=0; p<pts.size(); ++p)
            {
                double sum=0;
                for (uint y=0; y<8; ++y)
                    for (uint x=0; x<8; ++x)
                        sum += im.at<double>(j*8+y, i*8+x)*cos((2*x+1)*pts[p].x*cvip::PI/(2*8))*cos((2*y+1)*pts[p].y*cvip::PI/(2*8));
                
                out.at<double>(i*40+j*5+p) = sqrt((pts[p].x==0 ? 1./8 : 2./8))*sqrt((pts[p].y==0 ? 1./8 : 2./8))*sum;
            }
        }
    }
    
    // output L2-norm
    return out/cv::norm(out);
}

//function to write MAT into File
void Image::writeToFile(const cv::Mat& Data, std::string FileName) {

    std::ofstream DataFile;
    DataFile.open(FileName.c_str());
    for (int i = 0; i < Data.rows; i++) {
        for (int j = 0; j < Data.cols; j++) {
            if (Data.type() == CV_64F)
                DataFile << std::setprecision(16) << Data.at<double>(i,j) << " ";
            else if (Data.type() == CV_32F)
                DataFile << std::setprecision(16) << Data.at<float>(i,j) << " ";
            else if (Data.type() == CV_32S)
                DataFile << Data.at<int>(i,j) << " ";
            else
                DataFile << (int) Data.at<uchar>(i,j) << " ";
        }
        DataFile << std::endl;
    }
    DataFile.close();
}

//function to write MAT into File
void Image::writeHistsToFile(const std::vector<cv::Mat>& hists, const std::string& filePath) {


    std::ofstream file(filePath.c_str());

    if (!file.is_open()) {
        std::cerr << "Can't open file to write hists: " << filePath << std::endl;
        exit(-1);
    }

    for (size_t j=0; j<hists.size(); ++j)
        for (size_t i=0; i<hists[j].cols; ++i)
            file << std::setprecision(16) <<  hists[j].at<double>(0,i) << std::endl;

    file.close();

    return;
}

//function to write MAT into File
std::vector<cv::Mat> Image::readHistsFromFile(const std::string& filePath, size_t binCnt) {

    std::ifstream file(filePath.c_str());

    if (!file.is_open()) {
        std::cerr << "Can't open file to read hists: " << filePath << std::endl;
        exit(-1);
    }

    std::vector<cv::Mat> hists;

    while (true) {
        if(file.eof())
            break;

        cv::Mat hist(1, binCnt, CV_64F, cv::Scalar::all(0));

        size_t i=0;
        for (i=0; i<binCnt; ++i) {
            if(file.eof())
                break;
            file >> std::setprecision(16) >> hist.at<double>(0,i);
        }

        if (i == binCnt)
            hists.push_back(hist);
    }

    file.close();

    return hists;
}


//function to read MAT from File
cv::Mat Image::readFromFile(const std::string& FileName, int rows, int cols) {
    
    cv::Mat Data = cv::Mat(rows, cols, CV_64FC1);
    std::ifstream DataFile(FileName.c_str());
    if (DataFile.is_open()) {
        for (uint i = 0; i < Data.rows; i++) {
            for (uint j = 0; j < Data.cols; j++) {
                DataFile >> std::setprecision(16) >> Data.at<double>(i,j);
            }
        }
        DataFile.close();
    }
    else {
        std::cerr << "Cannot open file: '" << FileName << "'" << std::endl;
        exit(-1);
    }
    return Data;
}


//function to vectorize the image matrix
cv::Mat Image::vectorize(const cv::Mat &I) {
    
    int ih = I.rows;
    int iw = I.cols;
    
    cv::Mat outputImage(1, iw*ih, CV_64FC1);
    
    for (int reci = 0; reci < iw; reci++) {
        for (int recj = 0; recj < ih; recj++) {
            outputImage.at<double>(reci*ih+recj) = I.at<double>(recj,reci);
        }
    }
    
    return outputImage;
}


bool Image::exists(std::string filename) {
    FILE* fp = fopen(filename.c_str(), "r");
    if (fp) {
        fclose(fp);
        return true;
    } else {
        return false;
    }
}


std::vector<int> Image::randomPerm(int n) {
    std::vector<int> result;
    
    for(int i = 1; i <= n; i++)
        result.push_back(i);
    
    for(int i = 0; i < n-1; i++) {
        int c = rand() / (RAND_MAX/(n-i) + 1);
        int t = result[i]; result[i] = result[i+c]; result[i+c] = t;
    }
    
    return result;
}



void Image::zNormalize(cv::Mat& input)
{
    //normalization
    cv::Scalar mean;
    cv::Scalar std;

    cv::meanStdDev(input, mean, std);
    for (int reci = 0; reci < input.rows; reci++) {
        for (int recj = 0; recj < input.cols; recj++) {
            input.at<double>(reci,recj) = (input.at<double>(reci,recj) - (mean.val[0]+1e-10)) /  (std.val[0]+1e-10);
        }
    }
}

//function to perform histogram matching
cv::Mat Image::histogramMatch(const cv::Mat &Image, const cv::Mat &destCum) {
    //calculate the source histogram
    cv::Mat srcHist = cv::Mat::zeros(1, 256, CV_64FC1);
    for (int i=0; i<Image.rows; ++i) {
        for (int j=0; j<Image.cols; ++j) {
            int bin = Image.at<uchar>(i,j);
            srcHist.at<double>(0,bin) += 1;
        }
    }
    
    //normalize histogram
    double minVal, maxVal;
    cv::minMaxLoc(srcHist, &minVal, &maxVal);
    cv::Mat tempMat(srcHist.rows, srcHist.cols, CV_64FC1, cv::Scalar::all(1.0/maxVal));
    cv::multiply(srcHist, tempMat, srcHist);
    
    //calculate the source cummulative distribution function
    cv::Mat srcCum = cv::Mat::zeros(1, 256, CV_64FC1);
    srcCum.at<double>(0,0) = srcHist.at<double>(0,0);
    for (int i=1; i<256; ++i) {
        srcCum.at<double>(0,i) = srcCum.at<double>(0,i-1) + srcHist.at<double>(0,i);
    }
    
    //normalize cummulative distribution function
    cv::minMaxLoc(srcCum, &minVal, &maxVal);
    tempMat = cv::Mat(srcCum.rows, srcCum.cols, CV_64FC1, cv::Scalar::all(1.0/maxVal));
    cv::multiply(srcCum, tempMat, srcCum);
    
    cv::Mat lut = cv::Mat::zeros(1, 256, CV_8UC1);
    
    uchar last = 0;
    for(int j=0; j<srcCum.cols; j++) {
        double F1j = srcCum.at<double>(0,j);
        
        for(uchar k=last; k<destCum.cols; k++) {
            double F2k = destCum.at<double>(0,k);
            if(F2k > F1j - 1e-6) {
                lut.at<uchar>(0,j) = k;
                last = k;
                break;
            }
        }
    }
    
    cv::Mat outputImage(Image.rows, Image.cols, Image.type());
    cv::LUT(Image, lut, outputImage);
    
    return outputImage;
}


//convert cv::mat to uchar
uchar* Image::convertCvMat2Uchar(const cv::Mat Image) {
    int imageWidth = Image.cols;
    int imageHeight = Image.rows;
    
    uchar* imageData = new unsigned char[4*imageWidth*imageHeight];
    
    int imPixels = imageWidth * imageHeight;
    uchar* imSrc = (uchar*)(Image.data);
    uchar* imSrcEnd = imSrc + (3*imPixels);
    uchar* imDdest = imageData;
    
    do {
        memcpy(imDdest, imSrc, 3);
        imDdest += 4;
        imSrc += 3;
    } while(imSrc < imSrcEnd);
    
    return imageData;
}




std::vector<HaarImage*> HaarImage::createScaleSpace(const cv::Mat &frame, uint W_SIZE, double SCALE_START, double SCALE_MAX, double SCALE_STEP)
{
    uint minSize = min<uint>(frame.rows,frame.cols);	// Smallest dimension of the input image
    double scale = SCALE_START;						// Starting scale
    double maxScale = (double)minSize/W_SIZE;				// Maximum scale possible for this image
    if (maxScale > SCALE_MAX) maxScale = SCALE_MAX;
    
    // Create the scale space
    std::vector<HaarImage*> images;
    cv::Mat resized = cv::Mat(round(frame.rows/scale),round(frame.cols/scale),frame.type());
    cv::resize(frame, resized, resized.size(), 0, 0, cv::INTER_CUBIC);
    HaarImage *Im = new HaarImage(resized,scale);
    images.push_back(Im);
    while (scale*SCALE_STEP <= maxScale) // For each scale
    {
        scale *= SCALE_STEP;
        cv::Mat resized = cv::Mat(round(frame.rows/scale),round(frame.cols/scale),Im->I.type());
        cv::resize(Im->I, resized, resized.size(), 0, 0, cv::INTER_CUBIC);
        Im = new HaarImage(resized,scale);
        images.push_back(Im);
    }
    return images;
}

// Returns the integral image of the given grayscale image
cv::Mat HaarImage::integral_image(const cv::Mat &I)
{
    uint width = I.cols + 1;
    uint height = I.rows + 1;
    
    cv::Mat II = cv::Mat::zeros(height, width, CV_64FC1);
    
    for (uint i=1; i<width; i++)
        for (uint j=1; j<height; j++)
            II.at<double>(j,i) = I.at<double>(j-1,i-1) + II.at<double>(j,i-1) + II.at<double>(j-1,i) - II.at<double>(j-1,i-1);
    return II;
}

// Returns the sum of all pixels in the given rectangle according to the offset of the current window
double HaarImage::sum(const Rect *rect, uint offsetX, uint offsetY) const
{
    // Compute the min and max x-y coordinates of the scaled and shifted rectangle
    // Add 1 to all coordinates because the integral image has one zero row and column
    uint tx1 = offsetX + rect->x1 + 1;
    uint tx2 = tx1 + rect->width - 1;  // No need to add 1 because it's already added to tx1
    uint ty1 = offsetY + rect->y1 + 1;
    uint ty2 = ty1 + rect->height - 1; // No need to add 1 because it's already added to ty1
    
    // Compute and return the sum of the pixels
    return (II.at<double>(ty2,tx2) - II.at<double>(ty2,tx1-1) - II.at<double>(ty1-1,tx2) + II.at<double>(ty1-1,tx1-1));
}

HaarImage::HaarImage(cv::Mat &Im, double _scale)
    : Image(Im, _scale)
{
    Im.convertTo(Im, CV_64FC1);
    II = integral_image(Im);
    sII = integral_image(Im.mul(Im));
}


// METHOD DEFINITONS FOR THE "WINDOW" CLASS

// Constructor
Window::Window(uint _W_SIZE)
{
    wSize = _W_SIZE;//round(W_SIZE*scale);
    N = wSize*wSize; // Number of pixels in the window
}

// Sets the coordinates of a scaled and shifted window
void Window::setScanWindow(const HaarImage *Im, uint i, uint j)
{
    // Compute the min and max x-y coordinates of the window according to the given parameters
    // Store the results in member variables
    uint x1 = offsetX = i;
    uint y1 = offsetY = j;
    
    uint x2 = x1 + wSize - 1;
    uint y2 = y1 + wSize - 1;
    
    // Compute the mean and standard deviation of the current window using the integral images of the input image
    // and its square. Note that x1,x2,y1,y2 don't include the zero padding of the integral image and therefore
    // 1 is added each time they're used below, in contrast to the usage in the sum() function above.
    mean = (Im->II.at<double>(y2+1,x2+1) - Im->II.at<double>(y2+1,x1) - Im->II.at<double>(y1,x2+1) + Im->II.at<double>(y1,x1))/N;
    std = std::sqrt(fabs((double)mean*mean - (Im->sII.at<double>(y2+1,x2+1) - Im->sII.at<double>(y2+1,x1) - Im->sII.at<double>(y1,x2+1) + Im->sII.at<double>(y1,x1))/N));
}
