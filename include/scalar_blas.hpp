#pragma once
/*
 * scalar_blas.hpp — Scalar-typed rocBLAS / rocSOLVER dispatch.
 *
 * One place that maps a call to the s* (float) or d* (double) variant based
 * on the compile-time Scalar type (types.hpp). This makes the "same type flows
 * end to end" design compile: when -DSCALAR_DOUBLE flips Scalar to double, every
 * BLAS/solver call below switches to its d* counterpart with no call-site churn.
 *
 * Wrappers keep the exact rocBLAS argument order so call sites read identically
 * to the raw rocblas_* call, minus the s/d prefix. Selection is `if constexpr`
 * on sizeof(Scalar) — zero runtime cost, dead branch discarded at compile time.
 */

#include "types.hpp"
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

namespace sblas {

static constexpr bool is_double = (sizeof(Scalar) == 8);

/* ── GEMM ─────────────────────────────────────────────────────────────── */
inline rocblas_status gemm(
    rocblas_handle h, rocblas_operation opA, rocblas_operation opB,
    int m, int n, int k, const Scalar* alpha,
    const Scalar* A, int lda, const Scalar* B, int ldb,
    const Scalar* beta, Scalar* C, int ldc)
{
    if constexpr (is_double)
        return rocblas_dgemm(h, opA, opB, m, n, k,
            (const double*)alpha, (const double*)A, lda, (const double*)B, ldb,
            (const double*)beta, (double*)C, ldc);
    else
        return rocblas_sgemm(h, opA, opB, m, n, k,
            (const float*)alpha, (const float*)A, lda, (const float*)B, ldb,
            (const float*)beta, (float*)C, ldc);
}

/* ── GEMM (strided batched) ───────────────────────────────────────────── */
inline rocblas_status gemm_strided_batched(
    rocblas_handle h, rocblas_operation opA, rocblas_operation opB,
    int m, int n, int k, const Scalar* alpha,
    const Scalar* A, int lda, rocblas_stride sA,
    const Scalar* B, int ldb, rocblas_stride sB,
    const Scalar* beta, Scalar* C, int ldc, rocblas_stride sC, int batch)
{
    if constexpr (is_double)
        return rocblas_dgemm_strided_batched(h, opA, opB, m, n, k,
            (const double*)alpha, (const double*)A, lda, sA,
            (const double*)B, ldb, sB,
            (const double*)beta, (double*)C, ldc, sC, batch);
    else
        return rocblas_sgemm_strided_batched(h, opA, opB, m, n, k,
            (const float*)alpha, (const float*)A, lda, sA,
            (const float*)B, ldb, sB,
            (const float*)beta, (float*)C, ldc, sC, batch);
}

/* ── GEMV ─────────────────────────────────────────────────────────────── */
inline rocblas_status gemv(
    rocblas_handle h, rocblas_operation op, int m, int n,
    const Scalar* alpha, const Scalar* A, int lda,
    const Scalar* x, int incx, const Scalar* beta, Scalar* y, int incy)
{
    if constexpr (is_double)
        return rocblas_dgemv(h, op, m, n, (const double*)alpha,
            (const double*)A, lda, (const double*)x, incx,
            (const double*)beta, (double*)y, incy);
    else
        return rocblas_sgemv(h, op, m, n, (const float*)alpha,
            (const float*)A, lda, (const float*)x, incx,
            (const float*)beta, (float*)y, incy);
}

/* ── DOT ──────────────────────────────────────────────────────────────── */
inline rocblas_status dot(
    rocblas_handle h, int n, const Scalar* x, int incx,
    const Scalar* y, int incy, Scalar* result)
{
    if constexpr (is_double)
        return rocblas_ddot(h, n, (const double*)x, incx,
            (const double*)y, incy, (double*)result);
    else
        return rocblas_sdot(h, n, (const float*)x, incx,
            (const float*)y, incy, (float*)result);
}

/* ── TRSM ─────────────────────────────────────────────────────────────── */
inline rocblas_status trsm(
    rocblas_handle h, rocblas_side side, rocblas_fill uplo,
    rocblas_operation trans, rocblas_diagonal diag, int m, int n,
    const Scalar* alpha, const Scalar* A, int lda, Scalar* B, int ldb)
{
    if constexpr (is_double)
        return rocblas_dtrsm(h, side, uplo, trans, diag, m, n,
            (const double*)alpha, (const double*)A, lda, (double*)B, ldb);
    else
        return rocblas_strsm(h, side, uplo, trans, diag, m, n,
            (const float*)alpha, (const float*)A, lda, (float*)B, ldb);
}

/* ── GEQRF ────────────────────────────────────────────────────────────── */
inline rocblas_status geqrf(
    rocblas_handle h, int m, int n, Scalar* A, int lda, Scalar* ipiv)
{
    if constexpr (is_double)
        return rocsolver_dgeqrf(h, m, n, (double*)A, lda, (double*)ipiv);
    else
        return rocsolver_sgeqrf(h, m, n, (float*)A, lda, (float*)ipiv);
}

/* ── ORGQR ────────────────────────────────────────────────────────────── */
inline rocblas_status orgqr(
    rocblas_handle h, int m, int n, int k, Scalar* A, int lda, Scalar* ipiv)
{
    if constexpr (is_double)
        return rocsolver_dorgqr(h, m, n, k, (double*)A, lda, (double*)ipiv);
    else
        return rocsolver_sorgqr(h, m, n, k, (float*)A, lda, (float*)ipiv);
}

/* ── POTRF ────────────────────────────────────────────────────────────── */
inline rocblas_status potrf(
    rocblas_handle h, rocblas_fill uplo, int n, Scalar* A, int lda, int* info)
{
    if constexpr (is_double)
        return rocsolver_dpotrf(h, uplo, n, (double*)A, lda, info);
    else
        return rocsolver_spotrf(h, uplo, n, (float*)A, lda, info);
}

/* ── GESVD ────────────────────────────────────────────────────────────── */
inline rocblas_status gesvd(
    rocblas_handle h, rocblas_svect lsv, rocblas_svect rsv, int m, int n,
    Scalar* A, int lda, Scalar* S, Scalar* U, int ldu, Scalar* V, int ldv,
    Scalar* E, rocblas_workmode fast, int* info)
{
    if constexpr (is_double)
        return rocsolver_dgesvd(h, lsv, rsv, m, n, (double*)A, lda,
            (double*)S, (double*)U, ldu, (double*)V, ldv, (double*)E, fast, info);
    else
        return rocsolver_sgesvd(h, lsv, rsv, m, n, (float*)A, lda,
            (float*)S, (float*)U, ldu, (float*)V, ldv, (float*)E, fast, info);
}

} // namespace sblas
