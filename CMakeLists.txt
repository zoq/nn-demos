cmake_minimum_required (VERSION 2.8)
project(nn-demos)

# Include modules in the CMake directory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# Set the CFLAGS and CXXFLAGS.
add_definitions("-Wno-deprecated-declarations")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
SET(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ -O0 ${CMAKE_CXX_FLAGS}")

# Find the libraries we need to compile against.
set(Boost_ADDITIONAL_VERSIONS
  "1.49.0" "1.50.0" "1.51.0" "1.52.0" "1.53.0" "1.54.0" "1.55.0")
find_package(Boost 1.49 COMPONENTS system thread REQUIRED)
find_library(FCGI_LIB fcgi)
find_library(FCGI++_LIB fcgi++)
find_package(OpenCV REQUIRED)
find_package(Armadillo 3.6.0 REQUIRED)
find_package(Mlpack REQUIRED)

# Include directories for the dependencies.
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})

# Set source file path.
set(mnist_source
    http.hpp
    parser.hpp
    graphics.hpp
    mnist.cpp
)

# Define the executable and link against the libraries we need to build the
# source.
add_executable(mnist ${mnist_source})
target_link_libraries(mnist ${Boost_LIBRARIES} ${FCGI_LIB} ${FCGI++_LIB}
                            ${ARMADILLO_LIBRARIES} ${MLPACK_LIBRARY}
                            ${OpenCV_LIBS})
