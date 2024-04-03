cmake_minimum_required(VERSION 3.4.1)
project(OPT)

set(CMAKE_CXX_STANDARD 17)
set(UTF8PROC_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/utf8proc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${UTF8PROC_INCLUDE_PATH})



set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB SOURCES ${SOURCE_DIR}/*.cpp "${SOURCE_DIR}/*.c")

#if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_executable(opt ${SOURCES})
    set(UTF8PROC_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/arm64-v8a/libutf8proc.so)

    #UTF8PROC
    MESSAGE(UTF8PROC_LIBRARY: ${UTF8PROC_LIBRARY})
    target_link_libraries(opt ${UTF8PROC_LIBRARY})


#endif()