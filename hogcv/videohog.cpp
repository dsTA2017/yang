
#include <bitset>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    
    
    
     /*
    Mat src1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    HOGDescriptor *desc1=new HOGDescriptor();
    vector<float> w1;
    double start1 = getTickCount();
    desc1->compute(src1,w1,cvSize(22222,22222),cvSize(0,0));
    
    //cout<< w.size()<<endl;
    
    
    
    
    
    double end1 = getTickCount();
    double t1= getTickFrequency();
    cout<< (end1 - start1)/t1<<endl;
            
            
                
    */
    
    
    
    
    
    //Mat src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    UMat src ;
    imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE).copyTo(src);
    
    //Mat ab = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    
    //imwrite("/Users/yang50309/Desktop/opencvthreshold/opencvthreshold/gray.jpg",ab);
    
    
    HOGDescriptor *desc=new HOGDescriptor();
    vector<float> w;
    float start = getTickCount();
    desc->compute(src,w,cvSize(222222,222222),cvSize(0,0));
    
    //cout<< w.size()<<endl;
    
   
    
    
    
    float end = getTickCount();
    float t= getTickFrequency();
    cout<< "cl"<<(end - start)/t<<endl;
    
    
    
    
    
    //HOGDescriptor *desc=new HOGDescriptor();
    //vector<float> w;
        VideoCapture video("/Users/yang50309/Desktop/opencvthreshold/opencvthreshold/1440p_1min.mp4");
        if (!video.isOpened()){
            return -1;
        }
    
        namedWindow("video demo", CV_WINDOW_AUTOSIZE);
        Mat videoFrame;
        Mat grayImage;
        UMat a;
        int i = 0;
        float start2 = getTickCount();
        while(true){
            video >> videoFrame;
            if( videoFrame.empty()){
                break;
            }
            
            cvtColor(videoFrame, grayImage, CV_RGB2GRAY);
            grayImage.copyTo(a);
            desc->compute(a,w,cvSize(50,50),cvSize(0,0));
            i++;
            
        }
    cout<<i<<endl;
    float end2 = getTickCount();
    float t2= getTickFrequency();
    cout<< "cl"<<(end2 - start2)/t2<<endl;
    
    cout<< w.size();
    
    
    
    return 0;
}
