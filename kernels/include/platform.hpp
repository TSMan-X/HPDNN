#include <cuda_runtime.hpp>

/* Device (TODO) 
 * 1. Provide API, called by memory structure and queue structure
 ** 1) alloc memory of device,
 ** 2) create queue,
 * 2. Default info as parameters of memory, queue and primitive config
 ** 1) get max thread num, 
 ** 2) get max block num, 
 ** 3) get device name,
 ** 4) releated with platform etc.. */

namespace device {

enum {
	gpu,
	cpu,
	invalid
};

class deviceInfo {
public:
	deviceInfo() = default;
};


class deviceGPU {
};

class deviceCPU {
};

}
