#include <vector>
#include <unordered_map>

// set number of dimensions 
#define MAX_DIMENSIONS 8

// len of dimension 
using dim = uint32;

// array of dimensions
using dims = dim[MAX_DIMENSIONS];

enum class opType {
	opGemm = 1,
	opReorder = 2,
	opSoftmax = 3,
	opInvalid = default,
};

enum class memLayout {
	//2D
	HW = 1,
	WH = 2,
	//3D
	CHW = 3,
	CWH = 4,
	HWC = 5,
	WHC = 6,
	//4D
	NCHW = 7,
	NHWC = 8,
	INVALID,
};

enum class dataType {
	int8,
	int16,
	int32,
	uint8,
	uint16,
	uint32,
	fp8,
	fp16,
	fp32,
	INVALID,
};

// base paramconfig, when parameters your operation needed is out of the three parameters, you can inheriate this class
// this class aims at uniforming API
// config for input, output  and weight 
// you need to set memory layout type 
class paramConfig {
public:
	paramConfig() = default;
public:
	dims dim_;
	memLayout mem_tag_;
	dataType dt_;
}

// base primitive description, when create new operation, you need to inheriate this class and overwrite virtual function
// this class aims at uniforming API
class primitiveDesc {
public:
	primitiveDesc() = default;
	primitiveDesc(opType type): type_(type) {
	}
	// set config value
	virtual bool setInputConfig(paramConfig config) = 0;
	virtual bool setOutputConfig(paramConfig config) = 0;
	virtual bool setWeightConfig(paramConfig config) = 0;
	
	// config all param
	virtual void initConfig(paramConfig in, paramConfig out, paramConfig wei) = 0;
	
	// when config changed, update config
	virtual void updateConfig() = 0;

	// judge whether config is valid for current operation
	virtual bool isValid() { return true; }

private:
	opType type_;
	paramConfig input_config_;
	paramConfig output_config_;
	paramConfig weight_config_;
};
