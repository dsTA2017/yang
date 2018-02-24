
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    
    
    /*
    Mat dst1 = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
    Mat img;
    copyMakeBorder(dst1, img, 1, 1, 1, 1, BORDER_CONSTANT, 0);
    
    for(int height=1; height<img.rows-1; height++){
        for(int width=1; width<img.cols-1;width++){
          
           int x = (-1)*img.at<uchar>(width-1, height-1)+img.at<uchar>(width+1, height-1)
                -2*img.at<uchar>(width-1, height)+2*img.at<uchar>(width+1, height)
            -img.at<uchar>(width-1, height+1)+img.at<uchar>(width+1, height+1);
            
            int y = (-1)*img.at<uchar>(width-1, height-1)+img.at<uchar>(width-1, height+1)
            -2*img.at<uchar>(width, height-1)+2*img.at<uchar>(width, height+1)
            -img.at<uchar>(width+1, height-1)+img.at<uchar>(width+1, height+1);
            
            dst1.at<uchar>(width-1, height-1)=abs(x)+abs(y);
          
        
     
            
            
            
            
            
            
            
                
                
            }
        }
    
    
    imwrite("lena_sobel.jpg", dst1);
    
    
    */
    
    
    VideoCapture video("car.mp4");
    if (!video.isOpened()){
        return -1;
    }
    
    
    
   
    Mat videoFrame;
    
    Mat grayImage;
    
    int i = 0;
    float start2 = getTickCount();
    while(true){
        video >> videoFrame;
        
        if( videoFrame.empty()){
            break;
        }
        
        
        
        cvtColor(videoFrame, grayImage, CV_RGB2GRAY);
        
     
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        
        
        Sobel(grayImage, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
        convertScaleAbs(grad_x, abs_grad_x);
        Sobel(grayImage, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
        convertScaleAbs(grad_y, abs_grad_y);
        
        Mat dst1, dst2;
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1);
        //imshow("video demo", dst1);
        //waitKey(33);

        
        i++;
        
        //if(i == 1)
            //break;
         
     
    }
    cout<<i<<endl;
    float end2 = getTickCount();
    float t2= getTickFrequency();
    cout<< "cl"<<(end2 - start2)/t2<<endl;
    
    
    
    
    
    /*
    UMat videoFrame;
    UMat grayImage;
    
    int i = 0;
    float start2 = getTickCount();
    while(true){
        video >> videoFrame;
        if( videoFrame.empty()){
            break;
        }
        
        
        
        cvtColor(videoFrame, grayImage, CV_RGB2GRAY);
        UMat grad_x, grad_y;
        UMat abs_grad_x, abs_grad_y;
        
        
        Sobel(grayImage, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
        convertScaleAbs(grad_x, abs_grad_x);
        Sobel(grayImage, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
        convertScaleAbs(grad_y, abs_grad_y);
        
        UMat dst1, dst2;
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1);
        //imshow("video demo", dst1);
        //waitKey(33);
        i++;
        
        //if(i == 1)
            //break;
    }
    cout<<i<<endl;
    float end2 = getTickCount();
    float t2= getTickFrequency();
    cout<< "cl"<<(end2 - start2)/t2<<endl;
    
    
    
    */
    
    
    
    
    
    /*
    UMat src ;
    imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE).copyTo(src);
    
    UMat grad_x, grad_y;
    UMat abs_grad_x, abs_grad_y;
    
    float start = getTickCount();
    Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs(grad_y, abs_grad_y);
    
    UMat dst1, dst2;
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1);
    //float end = getTickCount();
    //float t= getTickFrequency();
    //cout<< (end - start)/t<<endl;
    
     */
    
     //imwrite("cvSobel_1", dst1);
    
    
    return 0;
}

