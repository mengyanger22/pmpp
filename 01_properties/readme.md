# GPU硬件资源查询
RTX3050官方文档：  https://docs.nvidia.com/cuda/archive/12.9.0/ampere-tuning-guide/index.html  

cudaDeviceProp包含了gpu的硬件信息，这是一个结构体：   
struct __device_builtin__ cudaDeviceProp  
cudaGetDeviceProperties(&prop, dev);  查询gpu的硬件参数并放在prop这个指针里面  
prop是cudaDeviceProperties的一个实例，用一个该类型的指针传入cudaGetDeviceProperties函数里面，dev是int类型表示gpu的编号（有多个gpu的时候）
## GPU的型号和计算能力
prop.name:  gpu的型号，如RTX3050  
prop.major, prop.minor  计算能力  
## Global memory 和 constant memory
prop.totalGlobalMem：  global memory的大小，字节为单位  
prop.totalConstMem:   constant memory的大小，字节为单位  
prop.memoryBusWidth:  gpu显存总线宽度，表示一次能传输的数据量，单位是位  
prop.memoryClockRate: gpu显存时钟频率，表示每秒钟有多少次高低电压跃迁，在一次时钟频率里面有Data Rate Factor次数据传输，可能电压从低到高、从高到底都会传输数据  

显存带宽 = memoryBusWidth / 8 * memoryClockRate * dataRateFactor (单位是字节)  
表示从gpu到显存，显存到gpu每秒能传输的数据量是多少，通常首其他硬件限制用不满  
gpu到显存传输数据和显存到gpu传输数据可能同时发生，比如gpu->显存是100GB/s，显存到gpu是50GB/s，那么当前总线总共的利用起来的带宽就是150GB/s。  
显存到gpu：x = A[i]    gpu到显存：A[i] = x

