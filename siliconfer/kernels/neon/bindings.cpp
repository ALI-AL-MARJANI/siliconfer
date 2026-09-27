// bindings.cpp — pybind11 bindings for the siliconfer NEON kernel.
//
// Exposed functions:
//   q4_gemv_sym(W, scales, x, group_size) → y
//   q4_gemv_scalar(W, scales, x, group_size) → y  (reference)
//   q4_gemm_sym(W, scales, X, group_size) → Y
//   q4_gemv_asym(W, scales, zeros, x, group_size) → y
//   q4_gemm_asym(W, scales, zeros, X, group_size) → Y
//   q4_dequant(W, scales, zeros | None, group_size) → float32 [out, in]
//   set_num_threads(n) / get_num_threads()
//   neon_available() → bool
//
// Weight packing is done in Python (siliconfer/kernels/neon/__init__.py).
// The GIL is released around every kernel call.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "q4_gemv.h"

namespace py = pybind11;
using pyf32 = py::array_t<float,   py::array::c_style | py::array::forcecast>;
using pyu8  = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// Shape validation
// ---------------------------------------------------------------------------
struct Dims { int out_f, in_f, n_groups; };

static Dims check_weights(const pyu8& W, const pyf32& scales, const pyf32* zeros, int group_size) {
    if (W.ndim() != 2)      throw std::runtime_error("W must be 2-D [out, in/2]");
    if (scales.ndim() != 2) throw std::runtime_error("scales must be 2-D [out, n_groups]");
    if (group_size <= 0 || group_size % 2 != 0)
        throw std::runtime_error("group_size must be a positive even number");
    Dims d;
    d.out_f = (int)W.shape(0);
    d.in_f  = (int)W.shape(1) * 2;
    if (d.in_f % group_size != 0)
        throw std::runtime_error("in_features must be divisible by group_size");
    d.n_groups = d.in_f / group_size;
    if (scales.shape(0) != d.out_f || scales.shape(1) != d.n_groups)
        throw std::runtime_error("scales shape mismatch");
    if (zeros && (zeros->ndim() != 2 || zeros->shape(0) != d.out_f || zeros->shape(1) != d.n_groups))
        throw std::runtime_error("zeros shape mismatch");
    return d;
}

static void check_x(const pyf32& x, const Dims& d) {
    if (x.ndim() != 1)          throw std::runtime_error("x must be 1-D [in_f]");
    if (x.shape(0) != d.in_f)   throw std::runtime_error("x length mismatch");
}

static int check_X(const pyf32& X, const Dims& d) {
    if (X.ndim() != 2)          throw std::runtime_error("X must be 2-D [T, in_f]");
    if (X.shape(1) != d.in_f)   throw std::runtime_error("X/W in_features mismatch");
    return (int)X.shape(0);
}

// ---------------------------------------------------------------------------
// GEMV
// ---------------------------------------------------------------------------
pyf32 q4_gemv_sym_bind(const pyu8& W, const pyf32& scales, const pyf32& x, int group_size) {
    Dims d = check_weights(W, scales, nullptr, group_size);
    check_x(x, d);
    auto y = py::array_t<float>(std::vector<py::ssize_t>{d.out_f});
    const uint8_t* pW = W.data(); const float* ps = scales.data(); const float* px = x.data();
    float* py_ = y.mutable_data();
    {
        py::gil_scoped_release nogil;
        q4_gemv_sym_neon(pW, ps, px, py_, d.out_f, d.in_f, group_size);
    }
    return y;
}

pyf32 q4_gemv_scalar_bind(const pyu8& W, const pyf32& scales, const pyf32& x, int group_size) {
    Dims d = check_weights(W, scales, nullptr, group_size);
    check_x(x, d);
    auto y = py::array_t<float>(std::vector<py::ssize_t>{d.out_f});
    q4_gemv_sym_scalar(W.data(), scales.data(), x.data(), y.mutable_data(),
                       d.out_f, d.in_f, group_size);
    return y;
}

pyf32 q4_gemv_asym_bind(const pyu8& W, const pyf32& scales, const pyf32& zeros,
                         const pyf32& x, int group_size) {
    Dims d = check_weights(W, scales, &zeros, group_size);
    check_x(x, d);
    auto y = py::array_t<float>(std::vector<py::ssize_t>{d.out_f});
    const uint8_t* pW = W.data(); const float* ps = scales.data(); const float* pz = zeros.data();
    const float* px = x.data(); float* py_ = y.mutable_data();
    {
        py::gil_scoped_release nogil;
        q4_gemv_asym_neon(pW, ps, pz, px, py_, d.out_f, d.in_f, group_size);
    }
    return y;
}

// ---------------------------------------------------------------------------
// GEMM — X[T, in_f] → Y[T, out_f]
// ---------------------------------------------------------------------------
pyf32 q4_gemm_sym_bind(const pyu8& W, const pyf32& scales, const pyf32& X, int group_size) {
    Dims d = check_weights(W, scales, nullptr, group_size);
    int T = check_X(X, d);
    auto Y = py::array_t<float>({T, d.out_f});
    const uint8_t* pW = W.data(); const float* ps = scales.data(); const float* pX = X.data();
    float* pY = Y.mutable_data();
    {
        py::gil_scoped_release nogil;
        q4_gemm_sym_neon(pW, ps, pX, pY, d.out_f, d.in_f, T, group_size);
    }
    return Y;
}

pyf32 q4_gemm_asym_bind(const pyu8& W, const pyf32& scales, const pyf32& zeros,
                         const pyf32& X, int group_size) {
    Dims d = check_weights(W, scales, &zeros, group_size);
    int T = check_X(X, d);
    auto Y = py::array_t<float>({T, d.out_f});
    const uint8_t* pW = W.data(); const float* ps = scales.data(); const float* pz = zeros.data();
    const float* pX = X.data(); float* pY = Y.mutable_data();
    {
        py::gil_scoped_release nogil;
        q4_gemm_asym_neon(pW, ps, pz, pX, pY, d.out_f, d.in_f, T, group_size);
    }
    return Y;
}

// ---------------------------------------------------------------------------
// Dequantize the whole matrix to float32 (testing / inspection)
// ---------------------------------------------------------------------------
pyf32 q4_dequant_bind(const pyu8& W, const pyf32& scales, py::object zeros, int group_size) {
    bool asym = !zeros.is_none();
    pyf32 z = asym ? zeros.cast<pyf32>() : pyf32();
    Dims d = check_weights(W, scales, asym ? &z : nullptr, group_size);
    auto out = py::array_t<float>({d.out_f, d.in_f});
    q4_dequant_rows(W.data(), scales.data(), asym ? z.data() : nullptr, out.mutable_data(),
                    0, d.out_f, d.in_f, group_size);
    return out;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(siliconfer_neon, m) {
    m.doc() = "siliconfer NEON 4-bit matmul kernels";

    m.def("q4_gemv_sym",    &q4_gemv_sym_bind,
          "q4 GEMV (symmetric int4), y = W_q4 x",
          py::arg("W"), py::arg("scales"), py::arg("x"), py::arg("group_size") = 128);

    m.def("q4_gemv_scalar", &q4_gemv_scalar_bind,
          "Scalar reference q4 GEMV (correctness check)",
          py::arg("W"), py::arg("scales"), py::arg("x"), py::arg("group_size") = 128);

    m.def("q4_gemv_asym",   &q4_gemv_asym_bind,
          "q4 GEMV (asymmetric int4), y = (W_q4 - zero) * scale @ x",
          py::arg("W"), py::arg("scales"), py::arg("zeros"), py::arg("x"),
          py::arg("group_size") = 128);

    m.def("q4_gemm_sym",    &q4_gemm_sym_bind,
          "q4 GEMM (symmetric int4), Y = X @ W_q4.T  (prefill)",
          py::arg("W"), py::arg("scales"), py::arg("X"), py::arg("group_size") = 128);

    m.def("q4_gemm_asym",   &q4_gemm_asym_bind,
          "q4 GEMM (asymmetric int4), Y = X @ (W_q4 - zero).T * scale  (prefill)",
          py::arg("W"), py::arg("scales"), py::arg("zeros"), py::arg("X"),
          py::arg("group_size") = 128);

    m.def("q4_dequant",     &q4_dequant_bind,
          "Dequantize packed int4 weights to float32 [out, in]; zeros=None for symmetric",
          py::arg("W"), py::arg("scales"), py::arg("zeros") = py::none(),
          py::arg("group_size") = 128);

    m.def("set_num_threads", &q4_set_num_threads, "Threads used by GEMV / dequantization",
          py::arg("n"));
    m.def("get_num_threads", &q4_get_num_threads);

#ifdef __ARM_NEON
    m.def("neon_available", []() { return true; });
#else
    m.def("neon_available", []() { return false; });
#endif
}
