#pragma once
#include "Matrix.h"
#include <queue>

namespace nsm {

/******************************************************************************************/

/// Compute A(:, x) B
template <bool TransposeA=false, class T, class V, class I=Identity>
Col<T> sparse_matvec(SpMat<T> const &A, V const &B, I const &x={}) {
    Col<real> O(TransposeA ? A.n_cols : A.n_rows, la::fill::zeros);
    T *Op = O.memptr();
    T const *Bp;
    if constexpr(std::is_same_v<V, Col<T>>) Bp = B.memptr();
    else Bp = B.colmem;
    if (TransposeA) {
        izip(O, [&](auto j, auto &o) {
            la::uword const b = A.col_ptrs[x(j)], e = A.col_ptrs[x(j)+1];
            zip(view(A.row_indices+b, A.row_indices+e), view(A.values+b, A.values+e), [&](auto i, T t) {o += t * Bp[i];});
        });
    } else {
        for (la::uword j : range(B.n_rows)) {
            la::uword const b = A.col_ptrs[x(j)], e = A.col_ptrs[x(j)+1];
            zip(view(A.row_indices+b, A.row_indices+e), view(A.values+b, A.values+e), [&, z=Bp[j]](auto i, T t) {Op[i] += t * z;});
        }
    }
    return O;
}

/// Compute A(:, x) B
template <class T, class M, class I=Identity>
Mat<T> sparse_matmul(SpMat<T> const &A, M const &B, I const &x={}) {
    if (B.n_cols == 0) return B;
    if (B.n_cols == 1) return sparse_matvec(A, B.unsafe_col(0), x);
    Mat<T> const Bt = B.t();
    Mat<T> O(B.n_cols, A.n_rows, la::fill::zeros);
    T *Op = O.memptr();
    T const *Bp = Bt.memptr();
    for (la::uword j : range(B.n_rows)) {
        la::uword const b = A.col_ptrs[x(j)], e = A.col_ptrs[x(j)+1];
        zip(view(A.row_indices+b, A.row_indices+e), view(A.values+b, A.values+e), [&](auto i, auto t) {
            zip(ptr_view(Op + O.n_rows * i, O.n_rows), ptr_view(Bp + Bt.n_rows * j, Bt.n_rows), [t](auto &o, auto const &b) {o += t * b;});
        });
    }
    return O.t();
}

/******************************************************************************************/

template <bool Upper, class M>
auto sparse_column_subset(M const &A, la::uword i) {
    la::uword b = A.col_ptrs[i], e = A.col_ptrs[i+1], m = *binary_search(range(b, e), i, [&](auto j) {return A.row_indices[j];});
    auto const diag = (m < e && A.row_indices[m] == i) ? A.values[m] : 0;
    if (Upper) b = m+1;
    else e = m;
    return std::make_tuple(view(A.row_indices + b, A.row_indices + e), 
                          view(A.values + b, A.values + e), diag);
}

static_assert(is_sparse<SpMat<real>>);
static_assert(!is_sparse<Mat<real>>);

template <bool Upper, class X, class S, class V, NSM_IF(is_sparse<S>)>
void column_vdot(X &x, S const &A, V const &b, uint i) {
    auto const [rows, values, diag] = sparse_column_subset<Upper>(A, i);
    if (diag) {
        typename V::elem_type t = -b(i);
        zip(rows, values, [&](auto r, auto v) {t += x[r] * v;});
        x(i) = -t / diag;
    } else x(i) = 0;
}
// template <bool Upper, class S, class M, class M2, NSM_IF(!is_sparse<S>)>
// void column_mdot(M &X, S const &A, M2 const &B, uint i) {
//     X.row(i) = (B.row(i) - A.col(i).t() * X) / A(i, i);
// }

template <bool Upper, class S, class M, class M2, NSM_IF(is_sparse<S>)>
void column_mdot(M &X, S const &A, M2 const &B, uint i) {
    auto const [rows, values, diag] = sparse_column_subset<Upper>(A, i);
    if (diag) {
        X.row(i) = -B.row(i);
        zip(rows, values, [&](auto r, auto v) {X.row(i) += X.row(r) * v;});
        X.row(i) /= -diag;
    } else X.row(i).zeros();
}

/******************************************************************************************/

// A should be given transposed. Complexity = O(m) for m nonzero elements
template <class M, class T>
Col<T> lower_solve(M const &A, Col<T> const &b) {
    Col<T> x(b.n_rows, la::fill::none);
    for (auto i : range(b.n_rows)) column_vdot<false>(x, A, b, i);
    return x;
}

template <class M, class T>
Mat<T> lower_solve(M const &A, Mat<T> const &B) {
    if (B.n_cols == 1) return lower_solve(A, B.unsafe_col(0));
    Mat<T> X(B.n_rows, B.n_cols, la::fill::none);
    for (auto i : range(B.n_rows)) column_mdot<false>(X, A, B, i);
    return X;
}

template <class M, class M2>
auto lower_solve_transpose(M const &A, M2 const &B) {
    Mat<typename M2::elem_type> X(B.n_cols, B.n_rows, la::fill::none);
    for (auto i : range(B.n_cols)) column_mdot<false>(X, A, B.t(), i);
    return X;
}

// A should be given transposed. . Complexity = O(m) for m nonzero elements
template <class M, class T>
Col<T> upper_solve(M const &A, Col<T> const &b) {
    Col<T> x(b.n_rows, la::fill::none);
    for (auto i : ~range(b.n_rows)) column_vdot<true>(x, A, b, i);
    return x;
}

template <class M, class T>
Mat<T> upper_solve(M const &A, Mat<T> const &B) {
    if (B.n_cols == 1) return upper_solve(A, B.unsafe_col(0));
    Mat<T> X(B.n_rows, B.n_cols, la::fill::none);
    for (auto i : ~range(B.n_rows)) column_mdot<true>(X, A, B, i);
    return X;
}

/******************************************************************************************/

}