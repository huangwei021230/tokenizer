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



#include <ATen/ops/fft_rfftn_ops.h>

namespace at {


// aten::fft_rfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
inline at::Tensor fft_rfftn(const at::Tensor & self, at::OptionalIntArrayRef s=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_rfftn::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor fft_rfftn(const at::Tensor & self, at::OptionalIntArrayRef s=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_rfftn::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm);
  }
}

// aten::fft_rfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
inline at::Tensor fft_rfftn_symint(const at::Tensor & self, at::OptionalSymIntArrayRef s=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_rfftn::call(self, s, dim, norm);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor fft_rfftn(const at::Tensor & self, at::OptionalSymIntArrayRef s=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_rfftn::call(self, s, dim, norm);
  }
}

// aten::fft_rfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & fft_rfftn_out(at::Tensor & out, const at::Tensor & self, at::OptionalIntArrayRef s=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_rfftn_out::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & fft_rfftn_out(at::Tensor & out, const at::Tensor & self, at::OptionalIntArrayRef s=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_rfftn_out::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm, out);
  }
}

// aten::fft_rfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & fft_rfftn_outf(const at::Tensor & self, at::OptionalIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
    return at::_ops::fft_rfftn_out::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & fft_rfftn_outf(const at::Tensor & self, at::OptionalIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
    return at::_ops::fft_rfftn_out::call(self, s.has_value() ? ::std::make_optional(c10::fromIntArrayRefSlow(*s)) : ::std::nullopt, dim, norm, out);
  }
}

// aten::fft_rfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & fft_rfftn_symint_out(at::Tensor & out, const at::Tensor & self, at::OptionalSymIntArrayRef s=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_rfftn_out::call(self, s, dim, norm, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & fft_rfftn_out(at::Tensor & out, const at::Tensor & self, at::OptionalSymIntArrayRef s=::std::nullopt, at::OptionalIntArrayRef dim=::std::nullopt, ::std::optional<c10::string_view> norm=::std::nullopt) {
    return at::_ops::fft_rfftn_out::call(self, s, dim, norm, out);
  }
}

// aten::fft_rfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & fft_rfftn_symint_outf(const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
    return at::_ops::fft_rfftn_out::call(self, s, dim, norm, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & fft_rfftn_outf(const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
    return at::_ops::fft_rfftn_out::call(self, s, dim, norm, out);
  }
}

}
