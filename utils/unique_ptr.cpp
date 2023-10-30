#include <iostream>
#include "base_ptr.hpp"

// TODO: C++23 support
// TODO: non-member function support
// TODO: array [] type support
// This version is after c++17
template<class T, class Deleter=defaultDelate<T> >
class uiquePtr {
public:
  constexpr uniquePtr() noexcept {}
  constexpr uniquePtr( std::nullptr_t ptr ) noexcept {
    data_ = ptr;
  }
  uniquePtr( const uniquePtr& ) = delete;

  ~uniquePtr() {
    getDeleter(get());
  }

  uniquePtr& operator=( const unique_ptr& ) = delete;

  T* release() noexcept {
    T* ptr = get();
    data_ = nullptr;
    return ptr;
  }
  void swap( uniquePtr& other ) noexcept {
    std::swap(*get(), *(other.get()));
    std::swap(getDeleter(), other.getDeleter());
  }

  T* get() const noexcept { 
    return data_;
  }
  Deleter & getDeleter() noexcept {
    return deleter;
  }
  const Deleter & getDeleter() const noexcept {
    return deleter;
  }
  explicit operator bool() const noexcept { 
    return get() != nullptr; 
  }

  T& operator*() const noexcept { 
    return *get(); 
  }
  T* operator->() const noexcept {
    return get(); 
  }

private:
  T* data_ = nullptr;
  Deleter deleter;
};
