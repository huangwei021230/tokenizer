//
// Created by PC on 2024/4/3.
//
#pragma once

#include "tokenization.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

const std::wstring stripChar = L" \t\n\r\f\v";
using Vocab = std:unordered_map<std::wstring, size_t>;
using InvVocab = std::unordered_map<size_t , std::wstring>;
