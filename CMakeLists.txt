cmake_minimum_required (VERSION 2.8)
project(nn-demos)

add_definitions("-Wno-deprecated-declarations")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")

set(Boost_ADDITIONAL_VERSIONS
  "1.49.0" "1.50.0" "1.51.0" "1.52.0" "1.53.0" "1.54.0" "1.55.0")
find_package(Boost 1.49
    COMPONENTS
      system
    REQUIRED
)
find_library(FCGI_LIB fcgi)
find_library(FCGI++_LIB fcgi++)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})

set(mnist_source
    mnist.cpp
)

add_executable(mnist ${mnist_source})
target_link_libraries(mnist ${Boost_LIBRARIES} ${FCGI_LIB} ${FCGI++_LIB})

