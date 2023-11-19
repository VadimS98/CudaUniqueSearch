cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(MyCudaApp LANGUAGES CXX CUDA)

set(SOURCES
    kernel.cu
)

add_executable(MyCudaApp ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_CUDA_STANDARD 12)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Thrust REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE ${THRUST_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE ${THRUST_LIBRARY})
