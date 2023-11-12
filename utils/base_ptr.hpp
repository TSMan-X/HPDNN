#include<iostream> 

template<class T>
struct defaultDeleter {
  void operator()(T* ptr) const {
    delete ptr;
  }
}
