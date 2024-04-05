#!/bin/bash

# 进入到项目目录
cd ./build

# 运行CMake配置项目
cmake ..

# 使用CMake构建项目
cmake --build .

# 运行可执行文件
./opt
