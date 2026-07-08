// bindings.cpp — pybind11 bindings for the siliconfer NEON kernel.
//
// Exposed functions:
//   q4_gemv_sym(W, scales, x, group_size) → y
//   q4_gemv_scalar(W, scales, x, group_size) → y  (reference)
//   q4_gemm_sym(W, scales, X, group_size) → Y
//   q4_gemv_asym(W, scales, zeros, x, group_size) → y
//   q4_gemm_asym(W, scales, zeros, X, group_size) → Y
//   neon_available() → bool
//
// Weight packing is done in Python (siliconfer/kernels/neon/__init__.py).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "q4_gemv.h"

namespace py = pybind11;
using pyf32 = py::array_t<float,   py::array::c_style | py::array::forcecast>;
using pyu8  = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// Helper: validate shapes and extract pointers
// ---------------------------------------------------------------------------
static void check_gemv(const pyu8& W, const pyf32& scales, const pyf32& x,
                        int& out_f, int& in_f, int& n_groups, int group_size) {
    auto bW = W.request(), bs = scales.request(), bx = x.request();
    if (bW.ndim != 2)  throw std::runtime_error("W must be 2-D [out, in/2]");
    if (bs.ndim != 2)  throw std::runtime_error("scales must be 2-D [out, n_groups]");
    if (bx.ndim != 1)  throw std::runtime_error("x must be 1-D [in_f]");
    out_f    = (int)bW.shape[0];
    int in_h = (int)bW.shape[1];   // in_f / 2
    in_f     = in_h * 2;
    n_groups = in_f / group_size;
    if (bs.shape[0] != out_f || bs.shape[1] != n_groups)
        throw std::runtime_error("scales shape mismatch");
    if (bx.shape[0] != in_f)
        throw std::runtime_error("x length mismatch");
    if (in_f % group_size != 0)
        throw std::runtime_error("in_features must be divisible by group_size");
}

// ---------------------------------------------------------------------------
// q4_gemv_sym
// ---------------------------------------------------------------------------
pyf32 q4_gemv_sym_bind(const pyu8& W, const pyf32& scales, const pyf32& x, int group_size) {
    int out_f, in_f, n_groups;
    check_gemv(W, scales, x, out_f, in_f, n_groups, group_size);

    auto y = py::array_t<float>(std::vector<py::ssize_t>{out_f});
    q4_gemv_sym_neon(
        (const uint8_t*)W.data(), scales.data(), x.data(),
        (float*)y.mutable_data(),
        out_f, in_f, group_size
    );
    return y;
}

// ---------------------------------------------------------------------------
// q4_gemv_scalar (reference, for correctness checks)
// ---------------------------------------------------------------------------
pyf32 q4_gemv_scalar_bind(const pyu8& W, const pyf32& scales, const pyf32& x, int group_size) {
    int out_f, in_f, n_groups;
    check_gemv(W, scales, x, out_f, in_f, n_groups, group_size);

    auto y = py::array_t<float>(std::vector<py::ssize_t>{out_f});
    q4_gemv_sym_scalar(
        (const uint8_t*)W.data(), scales.data(), x.data(),
        (float*)y.mutable_data(),
        out_f, in_f, group_size
    );
    return y;
}

// ---------------------------------------------------------------------------
// q4_gemv_asym
// ---------------------------------------------------------------------------
pyf32 q4_gemv_asym_bind(const pyu8& W, const pyf32& scales, const pyf32& zeros,
                         const pyf32& x, int group_size) {
    int out_f, in_f, n_groups;
    check_gemv(W, scales, x, out_f, in_f, n_groups, group_size);

    auto bz = zeros.request();
    if (bz.ndim != 2 || bz.shape[0] != out_f || bz.shape[1] != n_groups)
        throw std::runtime_error("zeros shape mismatch");

    auto y = py::array_t<float>(std::vector<py::ssize_t>{out_f});
    q4_gemv_asym_neon(
        (const uint8_t*)W.data(), scales.data(), zeros.data(), x.data(),
        (float*)y.mutable_data(),
        out_f, in_f, group_size
    );
    return y;
}

// ---------------------------------------------------------------------------
// q4_gemm_sym — X[T, in_f] → Y[T, out_f]
// ---------------------------------------------------------------------------
pyf32 q4_gemm_sym_bind(const pyu8& W, const pyf32& scales, const pyf32& X, int group_size) {
    auto bW = W.request(), bs = scales.request(), bX = X.request();
    if (bW.ndim != 2) throw std::runtime_error("W must be 2-D");
    if (bX.ndim != 2) throw std::runtime_error("X must be 2-D [T, in_f]");

    int out_f  = (int)bW.shape[0];
    int in_f   = (int)bW.shape[1] * 2;
    int T      = (int)bX.shape[0];
    int n_grp  = in_f / group_size;

    if ((int)bX.shape[1] != in_f)  throw std::runtime_error("X/W in_features mismatch");
    if (bs.shape[0] != out_f || bs.shape[1] != n_grp)
        throw std::runtime_error("scales shape mismatch");

    auto Y = py::array_t<float>({T, out_f});
    q4_gemm_sym_neon(
        (const uint8_t*)W.data(), scales.data(), X.data(),
        (float*)Y.mutable_data(),
        out_f, in_f, T, group_size
    );
    return Y;
}

// ---------------------------------------------------------------------------
// q4_gemm_asym — X[T, in_f] → Y[T, out_f], asymmetric
// ---------------------------------------------------------------------------
pyf32 q4_gemm_asym_bind(const pyu8& W, const pyf32& scales, const pyf32& zeros,
                         const pyf32& X, int group_size) {
    auto bW = W.request(), bs = scales.request(), bX = X.request();
    if (bW.ndim != 2) throw std::runtime_error("W must be 2-D");
    if (bX.ndim != 2) throw std::runtime_error("X must be 2-D [T, in_f]");

    int out_f  = (int)bW.shape[0];
    int in_f   = (int)bW.shape[1] * 2;
    int T      = (int)bX.shape[0];
    int n_grp  = in_f / group_size;

    if ((int)bX.shape[1] != in_f)  throw std::runtime_error("X/W in_features mismatch");
    if (bs.shape[0] != out_f || bs.shape[1] != n_grp)
        throw std::runtime_error("scales shape mismatch");

    auto bz = zeros.request();
    if (bz.ndim != 2 || bz.shape[0] != out_f || bz.shape[1] != n_grp)
        throw std::runtime_error("zeros shape mismatch");

    auto Y = py::array_t<float>({T, out_f});
    q4_gemm_asym_neon(
        (const uint8_t*)W.data(), scales.data(), zeros.data(), X.data(),
        (float*)Y.mutable_data(),
        out_f, in_f, T, group_size
    );
    return Y;
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

#ifdef __ARM_NEON
    m.def("neon_available", []() { return true; });
#else
    m.def("neon_available", []() { return false; });
#endif
}
