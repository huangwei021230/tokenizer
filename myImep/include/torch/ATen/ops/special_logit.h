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



#include <ATen/ops/special_logit_ops.h>

namespace at {


// aten::special_logit(Tensor self, float? eps=None) -> Tensor
inline at::Tensor special_logit(const at::Tensor & self, ::std::optional<double> eps=::std::nullopt) {
    return at::_ops::special_logit::call(self, eps);
}

// aten::special_logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & special_logit_out(at::Tensor & out, const at::Tensor & self, ::std::optional<double> eps=::std::nullopt) {
    return at::_ops::special_logit_out::call(self, eps, out);
}
// aten::special_logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & special_logit_outf(const at::Tensor & self, ::std::optional<double> eps, at::Tensor & out) {
    return at::_ops::special_logit_out::call(self, eps, out);
}

}
