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



#include <ATen/ops/_sample_dirichlet_ops.h>

namespace at {


// aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor
inline at::Tensor _sample_dirichlet(const at::Tensor & self, ::std::optional<at::Generator> generator=::std::nullopt) {
    return at::_ops::_sample_dirichlet::call(self, generator);
}

// aten::_sample_dirichlet.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _sample_dirichlet_out(at::Tensor & out, const at::Tensor & self, ::std::optional<at::Generator> generator=::std::nullopt) {
    return at::_ops::_sample_dirichlet_out::call(self, generator, out);
}
// aten::_sample_dirichlet.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _sample_dirichlet_outf(const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
    return at::_ops::_sample_dirichlet_out::call(self, generator, out);
}

}
