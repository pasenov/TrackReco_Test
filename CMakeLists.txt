cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Read_pt_Cpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(Read_pt_Cpp Read_pt_Cpp.cpp)
target_link_libraries(Read_pt_Cpp "${TORCH_LIBRARIES}")
set_property(TARGET Read_pt_Cpp PROPERTY CXX_STANDARD 17)
