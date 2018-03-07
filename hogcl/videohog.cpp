//
//  main.cpp
//  openc++
//
//  Created by 楊植翰 on 2018/2/15.
//  Copyright © 2018年 楊植翰. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include "cl.hpp"
#include "opencv2/core.hpp"
#include <opencv2/opencv.hpp>
#define NTHREADS 256
#define CELL_WIDTH 8
#define CELL_HEIGHT 8
#define CELLS_PER_BLOCK_X 2
#define CELLS_PER_BLOCK_Y 2
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


static int power_2up(unsigned int n)
{
    for(unsigned int i = 1; i<=1024; i<<=1)
        if(n < i)
            return i;
    return -1; // Input is too big
}

static size_t getBlockHistogramSize(Size block_size, Size cell_size, int nbins)
{
    Size cells_per_block = Size(block_size.width / cell_size.width,
        block_size.height / cell_size.height);
    return (size_t)(nbins * cells_per_block.area());
}


static int numPartsWithin(int size, int part_size, int stride)
{
    return (size - part_size + stride) / stride;
}

static Size numPartsWithin(cv::Size size, cv::Size part_size,
                                                cv::Size stride)
{
    return Size(numPartsWithin(size.width, part_size.width, stride.width),
        numPartsWithin(size.height, part_size.height, stride.height));
}



int main(int argc,char** argv)
{
    
    float start2 = getTickCount();





    cl_int err;
    cl_uint num;

    cl_event eventlist[4];
    
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
    
    cl_program program = load_program(context, "hog.cl");  ///Users/yang50309/Desktop/openc++/openc++/
    if(program == 0) {
        std::cerr << "Can't load or build program\n";}
   
    
    cl_kernel k1 = clCreateKernel(program, "compute_gradients_8UC1_kernel", 0);
    if(k1 == 0) {
        std::cerr << "Can't load kernel\n";}
    

    cl_kernel k2 = clCreateKernel(program, "compute_hists_lut_kernel", 0);
    if(k2 == 0) {
        std::cerr << "Can't load kernel\n";}
    cl_kernel k3 = clCreateKernel(program, "normalize_hists_36_kernel", 0);
    if(k3 == 0) {
        std::cerr << "Can't load kernel\n";}

    cl_kernel k4 = clCreateKernel(program, "extract_descrs_by_cols_kernel", 0);
    if(k4 == 0) {
            std::cerr << "Can't load kernel\n";}

    
    
    /*
    
    Mat img = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
   
    if (!img.data) {
               std::cout << "fail to open the file:" << std::endl;
                 return -1;
             }
    
    
    */



      VideoCapture video("car.mp4");
    if (!video.isOpened()){
        return -1;
    }
    
    Mat videoFrame;
    Mat img;
    Mat dst1;
    int i = 1;
    video >> videoFrame;
    cvtColor(videoFrame, img, CV_RGB2GRAY);



    

//----------------------------------------------------------------------------
//function parameter

    Size win_stride = Size(64,128);
    vector<float> _descriptors;
    //int descr_format; 
    Size blockSize = Size(16,16);
    Size cellSize = Size(8,8);
    int nbins = 9;
    Size blockStride = Size(8,8);
    Size winSize = Size(64,128);
    float sigma = 4 ;
    bool gammaCorrection=1;
    float L2HysThreshold = 0.2; 
    bool signedGradient=0;
    
//----------------------------------------------------------------------------  
// initialize variable
    Size imgSize = img.size();
    Size effect_size = imgSize;

    int height = imgSize.height;
    int width  = imgSize.width;

    Mat grad(imgSize, CV_32FC2);
    /*
    float *grad = NULL;
    if (NULL == (grad = (float *)malloc(width * height * sizeof(float)*2))) {
        std::cerr << "Failed to malloc buffer for grad. " << std::endl;
        return -1;
    }
    */
    //int qangle_type = ocl::Device::getDefault().isIntel() ? CV_32SC2 : CV_8UC2;
    Mat qangle(imgSize, CV_8UC2);


    /*
    unsigned char *qangle = NULL;
    if (NULL == (qangle = (unsigned char *)malloc(width * height * sizeof(unsigned char)*2))) {
        std::cerr << "Failed to malloc buffer for qangle. " << std::endl;
        return -1;
    }
    */

    const size_t block_hist_size = getBlockHistogramSize(blockSize, cellSize, nbins);
    const Size blocks_per_img = numPartsWithin(imgSize, blockSize, blockStride);
    Mat block_hists(1, block_hist_size * blocks_per_img.area() + 256, CV_32F);

    //Size wins_per_img = numPartsWithin(imgSize, winSize, win_stride);
    //UMat labels(1, wins_per_img.area(), CV_8U);

    float scale = 1.f / (2.f * sigma * sigma);
    Mat gaussian_lut(1, 512, CV_32FC1);
    int idx = 0;
    for(int i=-8; i<8; i++)
        for(int j=-8; j<8; j++)
            gaussian_lut.at<float>(idx++) = std::exp(-(j * j + i * i) * scale);
    for(int i=-8; i<8; i++)
        for(int j=-8; j<8; j++)
            gaussian_lut.at<float>(idx++) = (8.f - fabs(j + 0.5f)) * (8.f - fabs(i + 0.5f)) / 64.f;
    
    
    //copyMakeBorder(grayImage, dst1, 1, 1, 1, 1, BORDER_CONSTANT, 0);
    //width = dst1.cols;
    //height = dst1.rows;
    //std::cout << "picture width: " << width << ", height: " << height << std::endl;
    /*
    unsigned char *bufInput = NULL, *bufOutput = NULL;
    if (NULL == (bufInput = (unsigned char *)malloc(width * height * sizeof(unsigned char)))) {
        std::cerr << "Failed to malloc buffer for input image. " << std::endl;
        return -1;
    }
    
    
    if (NULL == (bufOutput = (unsigned char *)malloc((width-2) * (height-2) * sizeof(unsigned char)))) {
        std::cerr << "Failed to malloc buffer for output image. " << std::endl;
        return -1;
    }
    
    */
  
    
//----------------------------------------------------------------------------
// Gradients computation
    

    while(true){
    
    

    float angleScale = signedGradient ? (float)(nbins/(2.0*CV_PI)) : (float)(nbins/CV_PI);

        
    


    size_t localThreads[3] = { NTHREADS, 1, 1 };
    size_t globalThreads[3] = { NTHREADS*((size_t)width/NTHREADS + 1 ), (size_t)height, 1 };
    char correctGamma = (gammaCorrection) ? 1 : 0;
    int grad_quadstep = (int)grad.step >> 3;
    int qangle_elem_size = CV_ELEM_SIZE1(qangle.type());
    int qangle_step = (int)qangle.step / (2 * qangle_elem_size);
    int img_step = (int)img.step1();
    cl_mem cl_img = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, width * height * sizeof(unsigned char), img.data, NULL);
    cl_mem cl_grad = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, width * height * sizeof(float) * 2, grad.data, NULL);
    cl_mem cl_qangle = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, width * height * sizeof(unsigned char) * 2, qangle.data, NULL);

    

    clSetKernelArg(k1, 0, sizeof(int), &height);
    clSetKernelArg(k1, 1, sizeof(int), &width);
    clSetKernelArg(k1, 2, sizeof(int), &img_step);
    clSetKernelArg(k1, 3, sizeof(int), &grad_quadstep);
    clSetKernelArg(k1, 4, sizeof(int), &qangle_step);
    clSetKernelArg(k1, 5, sizeof(cl_mem), &cl_img);
    clSetKernelArg(k1, 6, sizeof(cl_mem), &cl_grad);
    clSetKernelArg(k1, 7, sizeof(cl_mem), &cl_qangle);
    clSetKernelArg(k1, 8, sizeof(float), &angleScale);
    clSetKernelArg(k1, 9, sizeof(char), &correctGamma);
    clSetKernelArg(k1, 10, sizeof(int), &nbins);




    err = clEnqueueNDRangeKernel(queue, k1, 2, 0, globalThreads, localThreads, 0, 0, &eventlist[0]);
    
    /*
    if(err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, cl_threshold, CL_TRUE, 0, sizeof(unsigned char) * (width-2)*(height-2), bufOutput, 0, 0, 0);
    }
    
     */
   
    
    
    

   
    
    
        

   
//----------------------------------------------------------------------------
// histogram computation

    
    int block_stride_x = blockStride.width;
    int block_stride_y = blockStride.height;

    int img_block_width = (width - CELLS_PER_BLOCK_X * CELL_WIDTH + block_stride_x)/block_stride_x;
    int img_block_height = (height - CELLS_PER_BLOCK_Y * CELL_HEIGHT + block_stride_y)/block_stride_y;
    int blocks_total = img_block_width * img_block_height;

    int qangle_elem_size_1 = CV_ELEM_SIZE1(qangle.type());
    int grad_quadstep_1 = (int)grad.step >> 2;
    int qangle_step_1 = (int)qangle.step / qangle_elem_size_1;

    int blocks_in_group = 4;
    size_t localThreads_2[3] = { (size_t)blocks_in_group * 24, 2, 1 };
    size_t globalThreads_2[3] = {((img_block_width * img_block_height + blocks_in_group - 1)/blocks_in_group) * localThreads_2[0], 2, 1 };

    int hists_size = (nbins * CELLS_PER_BLOCK_X * CELLS_PER_BLOCK_Y * 12) * sizeof(float);
    int final_hists_size = (nbins * CELLS_PER_BLOCK_X * CELLS_PER_BLOCK_Y) * sizeof(float);

    int smem = (hists_size + final_hists_size) * blocks_in_group;
    int block_hist_size_int = (int)block_hist_size;
    cl_mem cl_gaussian_lut = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * 512, gaussian_lut.data, NULL);
    cl_mem cl_block_hists = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * (block_hist_size * blocks_per_img.area() + 256), block_hists.data, NULL);


    err = clSetKernelArg(k2, 0, sizeof(int), &block_stride_x);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 1, sizeof(int), &block_stride_y);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 2, sizeof(int), &nbins);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 3, sizeof(int), &block_hist_size_int);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 4, sizeof(int), &img_block_width);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 5, sizeof(int), &blocks_in_group);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 6, sizeof(int), &blocks_total);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 7, sizeof(int), &grad_quadstep_1);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 8, sizeof(int), &qangle_step_1);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 9, sizeof(cl_mem), &cl_grad);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 10, sizeof(cl_mem), &cl_qangle);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 11, sizeof(cl_mem), &cl_gaussian_lut);
    //cout << err <<endl;
    err = clSetKernelArg(k2, 12, sizeof(cl_mem), &cl_block_hists);
    //cout << err <<endl;
    clSetKernelArg(k2, 13, (size_t)smem , NULL);
    //cout << err <<endl;
    err = clEnqueueNDRangeKernel(queue, k2, 2, 0, globalThreads_2, localThreads_2, 1, &eventlist[0], &eventlist[1]);
    //cout << err <<endl;

    
    
    
    

//----------------------------------------------------------------------------
//normalize histogram 


    
    int norblock_hist_size = nbins * CELLS_PER_BLOCK_X * CELLS_PER_BLOCK_Y;
    
    int nthreads;
    size_t globalThreads3[3] = { 1, 1, 1  };
    size_t localThreads3[3] = { 1, 1, 1  };
    
    

    

    

  
    
    int norblocks_in_group = NTHREADS / norblock_hist_size;
    nthreads = norblocks_in_group * norblock_hist_size;
    int num_groups = (img_block_width * img_block_height + norblocks_in_group - 1)/norblocks_in_group;
    globalThreads3[0] = nthreads * num_groups;
    localThreads3[0] = nthreads;
    
    
    
   
    err = clSetKernelArg(k3, 0, sizeof(cl_mem), &cl_block_hists);
    //cout << err <<endl;
    err = clSetKernelArg(k3, 1, sizeof(float), &L2HysThreshold);
    //cout << err <<endl;
    err = clSetKernelArg(k3, 2, sizeof(float) * nthreads, NULL);
    //cout << err <<endl;

    err = clEnqueueNDRangeKernel(queue, k3, 2, 0, globalThreads3, localThreads3, 1,&eventlist[1] , &eventlist[2]);

    //cout << err <<endl;
   
    

//----------------------------------------------------------------------------
//extract feature


    Size blocks_per_win = numPartsWithin(winSize, blockSize, blockStride);
    Size wins_per_img = numPartsWithin(effect_size, winSize, win_stride);

    int descr_size = blocks_per_win.area()*(int)block_hist_size;
    int descr_width = (int)block_hist_size*blocks_per_win.width;
    Mat descriptors(wins_per_img.area(), blocks_per_win.area() * block_hist_size, CV_32F);
    cl_mem cl_descriptors = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR , sizeof(float) * wins_per_img.area() * blocks_per_win.area() * block_hist_size, descriptors.data, NULL);


   

   

    int win_stride_x = win_stride.width;
    int win_stride_y = win_stride.height;
    int win_width = winSize.width;
    int win_height = winSize.height;

    int win_block_stride_x = win_stride_x / block_stride_x;
    int win_block_stride_y = win_stride_y / block_stride_y;
    int img_win_width = (width - win_width + win_stride_x) / win_stride_x;
    int img_win_height = (height - win_height + win_stride_y) / win_stride_y;
    

    int descriptors_quadstep = (int)descriptors.step >> 2;
    int nblocks_win_x = blocks_per_win.width;
    int nblocks_win_y = blocks_per_win.height;


    size_t globalThreads4[3] = { (size_t)img_win_width * NTHREADS, (size_t)img_win_height, 1 };
    size_t localThreads4[3] = { NTHREADS, 1, 1 };


    clSetKernelArg(k4, 0, sizeof(int), &block_hist_size_int);
    clSetKernelArg(k4, 1, sizeof(int), &descriptors_quadstep);
    clSetKernelArg(k4, 2, sizeof(int), &descr_size);
    clSetKernelArg(k4, 3, sizeof(int), &nblocks_win_x);
    clSetKernelArg(k4, 4, sizeof(int), &nblocks_win_y);
    clSetKernelArg(k4, 5, sizeof(int), &img_block_width);
    clSetKernelArg(k4, 6, sizeof(int), &win_block_stride_x);
    clSetKernelArg(k4, 7, sizeof(int), &win_block_stride_y);
    clSetKernelArg(k4, 8, sizeof(cl_mem), &cl_block_hists);
    clSetKernelArg(k4, 9, sizeof(cl_mem), &cl_descriptors);


    err = clEnqueueNDRangeKernel(queue, k4, 2, 0, globalThreads4, localThreads4, 1, &eventlist[2], 0);





    video >> videoFrame;
        if( videoFrame.empty()){
            break;
        }
    cvtColor(videoFrame, img, CV_RGB2GRAY);

    i++;


    
    
    clFinish(queue);
    
    
    
    descriptors.reshape(1, (int)descriptors.total()).copyTo(_descriptors);
    
    /*
    for( int i=150; i<160;i++)
        cout << _descriptors[i] <<" ";
    cout <<endl<< _descriptors.size()<<endl;
    
*/

//----------------------------------------------------------------------------
//release object

    
    clReleaseMemObject(cl_img);
    clReleaseMemObject(cl_grad);
    clReleaseMemObject(cl_qangle);
    clReleaseMemObject(cl_gaussian_lut);
    clReleaseMemObject(cl_block_hists);
    clReleaseMemObject(cl_descriptors);

    }
    
   for( int h=10000; h<10010;h++)
      cout<<_descriptors[h]<<" ";
    cout<<_descriptors.size();
    cout<<i<<endl;
    float end2 = getTickCount();
    float t2= getTickFrequency();
    cout<< "cl: "<<(end2 - start2)/t2<<" second"<<endl;

   /*
    clReleaseKernel(k1);
    clReleaseKernel(k2);
    clReleaseKernel(k3);
    clReleaseKernel(k4);
    clReleaseProgram(program);
    
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    */
    return 0;



     /*
    cout<<i<<endl;
    float end2 = getTickCount();
    float t2= getTickFrequency();
    cout<< "cl:"<<(end2 - start2)/t2<<" second"<<endl;
    //memcpy(img.data, bufOutput, (width-2) * (height-2) * sizeof(unsigned char));
    */
    

}








