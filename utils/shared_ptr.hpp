#include <atomic>

struct ptrCount {
public:
	void incCount() { count++; }
	void decCount() { count--; }
	int curCount() { return count;}

private:
	std::atomic<int> count = 0;
};

template<class T>
class sharedPtr {
public:
	//constructor : default
	sharedPtr() = default;

	// accept raw pointers
	sharedPtr(T* ptr) : data_(ptr) {
		count_ = new ptrCount;
		count_->incCount();
	}

	sharedPtr(const sharedPtr& rhs) {
		isInvasive();	
		// cur == nullptr
		data_ = rhs.getData();
		count_ = rhs.getCount();
		if (count_ != nullptr)
			count_->incCount();
	}

	sharedPtr& operator=(const sharedPtr& rhs) {
		if (this == rhs) return rhs;
		isInvasive();
		data_ = rhs.getData();
		count_ = rhs.getCount();
		if (count_ != nullptr)
			count_->incCount();	
		return *this;
	}

	~sharedPtr() {
		count_->decCount();
		if (count_->curCount() == 0) {
			delete data_;
			delete count_;
		}
	}

	T& operator*() const { return *data_; }
	T* operator->() const { return data_; }
	operator bool() const { return data_; }
	
	T* getData() const { return data_;}
	ptrCount* getCount() const { return count_;}

private:
	void isInvasive() {
		if (data_ != nullptr) {
			count_->decCount();
			if (count_->curCount() == 0) {
				delete data_;
				delete count_;
				data_ = nullptr;
				count_ = nullptr;
			}
		}		
	}
	T* data_ = nullptr;
	ptrCount* count_ = nullptr;
};

template<typename T>
using sPtr = sharedPtr<T>;
