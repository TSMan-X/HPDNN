#include <vector>

/* Memory (TODO)
 * Memory Type
 * 1. Device Memory
 ** 1) pure device memory
 ** 2) usm memory
 ** 3) other type memory
 * 2. Host Memory
 * Memory operation
 * 1. Memory Copy
 * 2. Memory Set
 * 3. ROI area
 * 4. other operations advanced would add this namespace or create new namespace *Tensor* 
 */
namespace memory {

class memoryDesc {
public:
	memoryDesc() = default;
public:
	dataType dt_;
	dims dims_;
	memLayout tag_;
	
	int eleSize;
	dims stride_;
};


class memoryHost {
};

class memoryDevice {
}

} //memory
