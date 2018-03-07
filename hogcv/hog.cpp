
#include <bitset>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    
    
    
  /*   
    Mat src1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    HOGDescriptor *desc1=new HOGDescriptor();//cvSize(64,128),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
    vector<float> w1;
    double start1 = getTickCount();
    desc1->compute(src1,w1,cvSize(22222,22222),cvSize(0,0));
    
    cout<< w1.size()<<endl;

    
    
    for(int i=150;i<160;i++)
       cout<< w1[i]; 
    
    
    double end1 = getTickCount();
    double t1= getTickFrequency();
    //cout<< (end1 - start1)/t1<<endl;
            
            
                
    
    
    
    
    
    //Mat src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    
    //Mat ab = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    
    //imwrite("/Users/yang50309/Desktop/opencvthreshold/opencvthreshold/gray.jpg",ab);
    
    
    
    //cout<< w.size()<<endl;
    
   
    
    
    
    */
    
    
   
    
    HOGDescriptor *desc=new HOGDescriptor();
    vector<float> w;
        VideoCapture video("car.mp4");
        if (!video.isOpened()){
            return -1;
        }
    
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
            desc->compute(grayImage,w,cvSize(64,128),cvSize(0,0));
            i++;
            
        }
    cout<<i<<endl;
    float end2 = getTickCount();
    float t2= getTickFrequency();
    cout<< "cl"<<(end2 - start2)/t2<<endl;
    cout<<w.size()<<endl;    
    for (int i=10000; i<10010;i++)
      cout<< w[i]<<" ";
    
    
    
    return 0;
}
