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



#include <ATen/ops/index_put_ops.h>

namespace at {


// aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
inline at::Tensor & index_put_(at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate=false) {
    return at::_ops::index_put_::call(self, indices, values, accumulate);
}

// aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
inline at::Tensor index_put(const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate=false) {
    return at::_ops::index_put::call(self, indices, values, accumulate);
}

// aten::index_put.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & index_put_out(at::Tensor & out, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate=false) {
    return at::_ops::index_put_out::call(self, indices, values, accumulate, out);
}
// aten::index_put.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & index_put_outf(const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, at::Tensor & out) {
    return at::_ops::index_put_out::call(self, indices, values, accumulate, out);
}

}
