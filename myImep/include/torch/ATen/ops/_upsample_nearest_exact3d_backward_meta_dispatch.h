#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace meta {

TORCH_API at::Tensor _upsample_nearest_exact3d_backward(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, ::std::optional<double> scales_d=::std::nullopt, ::std::optional<double> scales_h=::std::nullopt, ::std::optional<double> scales_w=::std::nullopt);
TORCH_API at::Tensor _upsample_nearest_exact3d_backward_symint(const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_d=::std::nullopt, ::std::optional<double> scales_h=::std::nullopt, ::std::optional<double> scales_w=::std::nullopt);
TORCH_API at::Tensor & _upsample_nearest_exact3d_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, ::std::optional<double> scales_d=::std::nullopt, ::std::optional<double> scales_h=::std::nullopt, ::std::optional<double> scales_w=::std::nullopt);
TORCH_API at::Tensor & _upsample_nearest_exact3d_backward_outf(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input);
TORCH_API at::Tensor & _upsample_nearest_exact3d_backward_symint_out(at::Tensor & grad_input, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_d=::std::nullopt, ::std::optional<double> scales_h=::std::nullopt, ::std::optional<double> scales_w=::std::nullopt);
TORCH_API at::Tensor & _upsample_nearest_exact3d_backward_symint_outf(const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input);

} // namespace meta
} // namespace at
