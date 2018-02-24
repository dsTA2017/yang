__kernel void sobel(__global const unsigned char* a,  __global unsigned char* result, int width)
{
int idx = get_global_id(0);
    int g=2*(idx/width);
    
    idx+=g;
    int x = (-1)*a[idx]+a[idx+2]-2*a[idx+width+2]+2*a[idx+width+4]-a[idx+2*(width+2)]+a[idx+2*(width+2)+2];
    int y = a[idx]+2*a[idx+1]+a[idx+2]-a[idx+2*(width+2)]-2*a[idx+2*(width+2)+1]-a[idx+2*(width+2)+2];
    
    result[idx-g] = abs(x)+abs(y);

}
