cmake_minimum_required(VERSION 3.4.1)
project(OPT)

set(CMAKE_CXX_STANDARD 17)

include(CheckCXXCompilerFlag)
if(NOT MSVC)
    check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
    set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 17)
else()
    check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
    set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 17)
endif()


set(LINK_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/libs/amd64)

MESSAGE(LINK_DIRS: ${LINK_DIRS})
set(UTF8PROC_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/utf8proc)
set(NLOHMANN_JSON_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TOKENZIER_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/tokenizers-cpp)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}
        ${UTF8PROC_INCLUDE_PATH}
        ${NLOHMANN_JSON_INCLUDE_PATH})

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB SOURCES ${SOURCE_DIR}/*.cpp "${SOURCE_DIR}/*.c")



MESSAGE(CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR})

#if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
add_executable(opt ${SOURCES})

# 设置动态库导入库路径
set(UTF8PROC_LIBRARY "${LINK_DIRS}/utf8proc.lib")
MESSAGE(UTF8PROC_LIBRARY: ${UTF8PROC_LIBRARY})

add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

target_include_directories(opt PRIVATE ${TOKENZIER_CPP_PATH}/include)

# 链接静态库到目标
target_link_libraries(opt PUBLIC ${UTF8PROC_LIBRARY})
target_link_libraries(opt PUBLIC tokenizers_cpp)
#endif()

