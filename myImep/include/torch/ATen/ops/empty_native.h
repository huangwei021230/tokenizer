#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor empty_names(at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt);
TORCH_API at::Tensor & empty_names_out(at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out);
TORCH_API at::Tensor & empty_out(at::IntArrayRef size, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out);
TORCH_API at::Tensor empty_cpu(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt);
TORCH_API at::Tensor empty_cuda(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt);
TORCH_API at::Tensor empty_sparse(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt);
TORCH_API at::Tensor empty_sparse_compressed(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt);
TORCH_API at::Tensor empty_meta_symint(c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt);
TORCH_API at::Tensor empty_mkldnn(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt);
TORCH_API at::Tensor empty_unknown_quantized(at::IntArrayRef size, ::std::optional<at::ScalarType> dtype={}, ::std::optional<at::Layout> layout={}, ::std::optional<at::Device> device={}, ::std::optional<bool> pin_memory={}, ::std::optional<at::MemoryFormat> memory_format=::std::nullopt);
} // namespace native
} // namespace at
