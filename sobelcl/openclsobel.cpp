//
//  main.cpp
//  openc++
//
//  Created by 楊植翰 on 2017/10/30.
//  Copyright © 2017年 楊植翰. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#ifdef __APPLE__
#include "cl.hpp"
#else
//#include <OpenCL/opencl.h>

#include "cl.hpp"
#endif

#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;




cl_program load_program(cl_context context, const char* filename)
{
    std::ifstream in(filename, std::ios_base::binary);
    if(!in.good()) {
        return 0;
    }
    
    // get file length
    in.seekg(0, std::ios_base::end);
    size_t length = in.tellg();
    in.seekg(0, std::ios_base::beg);
    
    // read program source
    std::vector<char> data(length + 1);
    in.read(&data[0], length);
    data[length] = 0;
    
    // create and build program
    const char* source = &data[0];
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
    if(program == 0) {
        return 0;
    }
    
    if(clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
        return 0;
    }
    
    return program;
}


int main(int argc,char** argv)
{
    
    //float start2 = getTickCount();
    cl_int err;
    cl_uint num;
    int width,height;
    err = clGetPlatformIDs(0, 0, &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platforms\n";
        return 0;
    }
    
    std::vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platform ID\n";
        return 0;
    }
    
    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
    cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
    if(context == 0) {
        std::cerr << "Can't create OpenCL context\n";
        return 0;
    }
    
    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);
    
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    std::string devname;
    devname.resize(cb);
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    std::cout << "Device: " << devname.c_str() << "\n";
    
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
    if(queue == 0) {
        std::cerr << "Can't create command queue\n";
        clReleaseContext(context);
        return 0;
    }
    
    cl_program program = load_program(context, "sobel.cl");  ///Users/yang50309/Desktop/openc++/openc++/
    if(program == 0) {
        std::cerr << "Can't load or build program\n";}
   
    cl_kernel threshold = clCreateKernel(program, "sobel", 0);
    if(threshold == 0) {
        std::cerr << "Can't load kernel\n";}
    
    
    
    
    
    
    /*
    Mat img = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
    Mat dst1;
    if (!img.data) {
               std::cout << "fail to open the file:" << std::endl;
                 return -1;
             }
    
    */
    
    
    
    
    
    VideoCapture video("car.mp4");
    if (!video.isOpened()){
        return -1;
    }
    
    //namedWindow("video demo", CV_WINDOW_AUTOSIZE);
    
    
    Mat videoFrame;
    Mat grayImage;
    Mat dst1;
    int i = 1;
    video >> videoFrame;
    cvtColor(videoFrame, grayImage, CV_RGB2GRAY);
    copyMakeBorder(grayImage, dst1, 1, 1, 1, 1, BORDER_CONSTANT, 0);
    width = dst1.cols;
    height = dst1.rows;
    std::cout << "picture width: " << width << ", height: " << height << std::endl;
    unsigned char *bufInput = NULL, *bufOutput = NULL;
    if (NULL == (bufInput = (unsigned char *)malloc(width * height * sizeof(unsigned char)))) {
        std::cerr << "Failed to malloc buffer for input image. " << std::endl;
        return -1;
    }
    
    
    if (NULL == (bufOutput = (unsigned char *)malloc((width-2) * (height-2) * sizeof(unsigned char)))) {
        std::cerr << "Failed to malloc buffer for output image. " << std::endl;
        return -1;
    }
    
    
    
    
    float start2 = getTickCount();
    while(true){
        
        
        //memcpy(bufInput, dst1.data, width * height * sizeof(unsigned char));
        //memset(bufOutput, 0x0, (width-2) * (height-2) * sizeof(unsigned char));
        
        
        cl_mem cl_origin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, width * height * sizeof(unsigned char), dst1.data, NULL);
        cl_mem cl_threshold = clCreateBuffer(context, CL_MEM_WRITE_ONLY| CL_MEM_USE_HOST_PTR, (width-2) * (height-2) * sizeof(unsigned char), bufOutput, NULL);
        
        
        if(cl_origin == 0 || cl_threshold == 0){ cerr << "Can't create OpenCL buffer\n";}
        
        
        
        
        
        
        int w = width - 2;
        
        clSetKernelArg(threshold, 0, sizeof(cl_mem), &cl_origin);
        clSetKernelArg(threshold, 1, sizeof(cl_mem), &cl_threshold);
        clSetKernelArg(threshold, 2, sizeof(int), &w);
        
        
        
        size_t work_size = (width-2)*(height-2);
        
        err = clEnqueueNDRangeKernel(queue, threshold, 1, 0, &work_size, 0, 0, 0, 0);
        
        /*
        if(err == CL_SUCCESS) {
            err = clEnqueueReadBuffer(queue, cl_threshold, CL_TRUE, 0, sizeof(unsigned char) * (width-2)*(height-2), bufOutput, 0, 0, 0);
        }
        
         */
       
        
        
        //memcpy(grayImage.data, bufOutput, (width-2) * (height-2) * sizeof(unsigned char));
        //imshow("video demo", videoFrame);
        //waitKey(10);
        
        
        video >> videoFrame;
        if( videoFrame.empty()){
            break;
        }
        cvtColor(videoFrame, grayImage, CV_RGB2GRAY);
        copyMakeBorder(grayImage, dst1, 1, 1, 1, 1, BORDER_CONSTANT, 0);
        width = dst1.cols;
        height = dst1.rows;
        i++;
        
         
        clReleaseMemObject(cl_origin);
        clReleaseMemObject(cl_threshold);
        if(i==22222222)
            break;
    }
    
    cout<<i<<endl;
    float end2 = getTickCount();
    float t2= getTickFrequency();
    cout<< "cl:"<<(end2 - start2)/t2<<" second"<<endl;
    //memcpy(img.data, bufOutput, (width-2) * (height-2) * sizeof(unsigned char));
    
   
    
    
    
    
    
    clReleaseKernel(threshold);
    clReleaseProgram(program);
    
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}

