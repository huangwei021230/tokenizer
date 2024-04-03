cmake_minimum_required(VERSION 3.1)
project(GPT2)

set(CMAKE_CXX_STANDARD 17)

add_executable(GPT2 tokenization.cpp tokenization.h)
