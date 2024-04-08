# the minimum version of CMake.
cmake_minimum_required(VERSION 3.4.1)
project(OPT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
set(LINK_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/libs/x86_64_linux)
set(UTF8PROC_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/headers/utf8proc)
set(NLOHMANN_JSON_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/headers)
set(PYTORCH_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/headers/pytorch)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/headers
        ${PYTORCH_INCLUDE_PATH}/include
        ${PYTORCH_INCLUDE_PATH}/include/torch/csrc/api/include
        ${UTF8PROC_INCLUDE_PATH}
        ${NLOHMANN_JSON_INCLUDE_PATH}
)

message(${PYTORCH_INCLUDE_PATH}/include)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB SOURCES "${SOURCE_DIR}/*.cpp" "${SOURCE_DIR}/*.c")



MESSAGE(CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR})

#if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
add_executable(opt ${SOURCES}
        opt_infer_engine.cpp
)

# utf8proc libs
set(UTF8PROC_LIBRARY "${LINK_DIRS}/libutf8proc.so")
# torch libs
set(C10_LIBRARY "${LINK_DIRS}/libc10.so")
set(PYTORCH_0_LIBRARY "${LINK_DIRS}/libtorch.so")
set(TORCH_GLOBAL_DEPS_LIBRARY "${LINK_DIRS}/libtorch_global_deps.so")
set(PYTORCH_LIBRARY "${LINK_DIRS}/libtorch_cpu.so")

# set(BOOST_REGEX_LIBRARY "${LINK_DIRS}/libboost_regex.so")

message(PYTORCH_LIBRARY "####PYTORCH_LIBRARY####: ${PYTORCH_LIBRARY}")
message(C10_LIBRARY "####C10_LIBRARY####: ${C10_LIBRARY}")
message(PYTORCH_0_LIBRARY "####PYTORCH_0_LIBRARY####: ${PYTORCH_0_LIBRARY}")
message(TORCH_GLOBAL_DEPS_LIBRARY "####TORCH_GLOBAL_DEPS_LIBRARY####: ${TORCH_GLOBAL_DEPS_LIBRARY}")
MESSAGE(UTF8PROC_LIBRARY:"####UTF8PROC_LIBRARY#### ${UTF8PROC_LIBRARY}")


# pytorch libs
target_link_libraries(opt PUBLIC ${C10_LIBRARY})
target_link_libraries(opt PUBLIC ${PYTORCH_0_LIBRARY})
target_link_libraries(opt PUBLIC ${TORCH_GLOBAL_DEPS_LIBRARY})
target_link_libraries(opt PUBLIC ${PYTORCH_LIBRARY})
# utf8proc libs
target_link_libraries(opt PUBLIC ${UTF8PROC_LIBRARY})

#endif()

