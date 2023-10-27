#include <iostream>
#include <vector>
#include "shared_ptr.hpp"

int main() {
	sPtr<int> a = new int(10);
	std::cout << *a << std::endl;	
}
