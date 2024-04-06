#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/linalg_norm_ops.h>

namespace at {


// aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
inline at::Tensor linalg_norm(const at::Tensor & self, const ::std::optional<at::Scalar> & ord=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, bool keepdim=false, ::std::optional<at::ScalarType> dtype=::std::nullopt) {
    return at::_ops::linalg_norm::call(self, ord, dim, keepdim, dtype);
}

// aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
inline at::Tensor linalg_norm(const at::Tensor & self, c10::string_view ord, at::OptionalIntArrayRef dim=::std::nullopt, bool keepdim=false, ::std::optional<at::ScalarType> dtype=::std::nullopt) {
    return at::_ops::linalg_norm_ord_str::call(self, ord, dim, keepdim, dtype);
}

// aten::linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linalg_norm_out(at::Tensor & out, const at::Tensor & self, const ::std::optional<at::Scalar> & ord=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, bool keepdim=false, ::std::optional<at::ScalarType> dtype=::std::nullopt) {
    return at::_ops::linalg_norm_out::call(self, ord, dim, keepdim, dtype, out);
}
// aten::linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linalg_norm_outf(const at::Tensor & self, const ::std::optional<at::Scalar> & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
    return at::_ops::linalg_norm_out::call(self, ord, dim, keepdim, dtype, out);
}

// aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linalg_norm_out(at::Tensor & out, const at::Tensor & self, c10::string_view ord, at::OptionalIntArrayRef dim=::std::nullopt, bool keepdim=false, ::std::optional<at::ScalarType> dtype=::std::nullopt) {
    return at::_ops::linalg_norm_ord_str_out::call(self, ord, dim, keepdim, dtype, out);
}
// aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & linalg_norm_outf(const at::Tensor & self, c10::string_view ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
    return at::_ops::linalg_norm_ord_str_out::call(self, ord, dim, keepdim, dtype, out);
}

}
