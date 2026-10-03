# CPU硬件资源查询
cudaDeviceProp包含了gpu的硬件信息，这是一个结构体：
struct __device_builtin__ cudaDeviceProp
cudaGetDeviceProperties(&prop, dev);
prop是cudaDeviceProperties的一个实例，用一个该类型的指针传入cudaGetDeviceProperties函数里面，dev是int类型表示gpu的编号（有多个gpu的时候）
## GPU的型号和计算能力
