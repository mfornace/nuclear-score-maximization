#pragma once
/**
    Engines.h: adaptors for different operator implementations (dense, sparse, factorized, ...)
*/

#include "Matrix.h"
#include <thread>

namespace nsm {

/******************************************************************************************/

// The following classes are designed to provide a consistent "operator inference" for a PSD matrix
// In some cases, the PSD matrix K may be constructed ahead of time (sparse or dense).
// In others, it is more efficient to hold it in factorized form (K = A A^T, A sparse or dense)
// Across these use-cases, the algorithms we develop use the interface exemplified below
// rather than accessing the matrix in direct subscript notation

// Swap two rows of K (and columns also, since it's symmetric)
// void pivot(la::uword a, la::uword b)

// Return the diagonal of K, needed for deterministic algorithms
// Col<T> diagonal() const;
// Return a particular index on the diagonal
// T diagonal(uint i) const;

// Return the diagonal of K^2, needed for deterministic algorithms
// Col<T> square_diagonal() const;

// Return the dense representation of K, for debugging/reference purposes
// Mat<T> dense() const;

// Number of columns in the square root factorization of K used below in sqrt()
// auto m() const;
// Number of columns in K
// auto n() const;

// Return K * v (matvec) for column or matrix v. If v is not length n, treat v as if padded with zeros to make it length n.
// template <class V> Mat<T> full(V const &v);

// Return A * v (matvec) for column or matrix v. There is no requirement on A except that K = A A^T and A.n_cols = m()
// template <class V> Mat<T> sqrt(V const &v) const {return A * v;}

/******************************************************************************************/

// Operator formed as A A^T from given dense matrix A
template <class T>
struct DenseSqrtMatrix {
    Mat<T> A;

    DenseSqrtMatrix(Mat<T> a) : A(std::move(a)) {}

    void pivot(la::uword a, la::uword b) {A.swap_rows(a, b);}

    Col<T> diagonal() const {return la::sum(la::square(A), 1);}
    T diagonal(uint i) const {return la::accu(la::square(A.row(i)));}
    Col<T> square_diagonal() const {return la::sum(la::square(A * la::chol(A.t() * A).t()), 1);}
    Mat<T> dense() const {return A * A.t();}

    auto m() const {return A.n_cols;}
    auto n() const {return A.n_rows;}

    template <class V>
    Mat<T> full(V const &v) const {return A * A.head_rows(v.n_rows).t() * v;}

    template <class V>
    Mat<T> sqrt(V const &v) const {return A * v;}
};

/******************************************************************************************/

// Operator formed as A A^T from given sparse matrix A
template <class T>
struct SparseSqrtMatrix {
    SpMat<T> A, At;
    la::uvec x;
    bool cholesky = false;

    SparseSqrtMatrix(SpMat<T> const &a) : A(a), At(a.t()), x(vmap<la::uvec>(range(n()))) {}

    void pivot(la::uword a, la::uword b) {x.swap_rows(a, b);}

    Col<T> diagonal() const {return Col<T>(la::sum(la::square(At), 0).t())(x);}
    T diagonal(uint i) const {return la::accu(la::square(At.col(x(i))));}
    Col<T> square_diagonal() const {
        if (cholesky) {
            return Col<T>(la::sum(la::square(la::chol(Mat<T>(At * At.t()), "lower").t() * At), 0).t())(x);
        } else {
            Col<T> const o(la::sum(la::square(A * At), 0).t());
            NSM_REQUIRE(std::size(o), ==, n(), "diagonal size error");
            return o(x);
        }
    }
    Mat<T> dense() const {return Mat<T>(At.t() * At).rows(x).eval().cols(x);}

    auto m() const {return At.n_rows;}
    auto n() const {return At.n_cols;}

    template <class V>
    Mat<T> full(V const &v) const {return sparse_matmul(A, sparse_matmul(At, v, x)).rows(x);}

    template <class V>
    Mat<T> sqrt(V const &v) const {return sparse_matmul(A, v).rows(x);}
};

/******************************************************************************************/

// Operator formed as K from given dense matrix K
template <class T>
struct DenseMatrix {
    Mat<T> K;

    DenseMatrix(Mat<T> k) : K(std::move(k)) {}
    Col<T> diagonal() const {return K.diag();}
    T diagonal(uint i) const {return K(i, i);}
    Col<T> square_diagonal() const {return la::sum(la::square(K), 1);}
    auto const &dense() const {return K;}
    
    void pivot(la::uword a, la::uword b) {
        K.swap_rows(a, b);
        K.swap_cols(a, b);
    }

    auto n() const {return K.n_rows;}

    template <class V>
    Mat<T> full(V const &v) const {
        if (v.n_rows == K.n_cols) return K * v;
        else return K.head_cols(v.n_rows) * v;
    }
};

/******************************************************************************************/

// Operator formed as K from given sparse matrix K
template <class T>
struct SparseMatrix {
    SpMat<T> K;
    la::uvec x;
    SparseMatrix(SpMat<T> k) : K(std::move(k)), x(vmap<la::uvec>(range(n()))) {}

    Col<T> diagonal() const {return Col<T>(K.diag())(x);}
    T diagonal(uint i) const {return K(x(i), x(i));}
    Col<T> square_diagonal() const {return Col<T>(la::sum(la::square(K), 1))(x);}
    Mat<T> dense() const {return Mat<T>(K)(x, x);}
    
    void pivot(la::uword a, la::uword b) {x.swap_rows(a, b);}
    auto n() const {return K.n_rows;}

    template <class V>
    Mat<T> full(V const &v) const {return (K.cols(x.head(v.n_rows)) * v).eval().rows(x);}
};

/******************************************************************************************/

// Operator formed as K from given dense matrix K but also possessing factorization C C^T = K
template <class T>
struct FactorizedMatrix : DenseMatrix<T> {
    using base_type = DenseMatrix<T>;
    using base_type::K;
    Mat<T> C;

    FactorizedMatrix(Mat<T> k, Mat<T> c) : base_type(std::move(k)), C(std::move(c)) {}

    void pivot(la::uword a, la::uword b) {
        base_type::pivot(a, b);
        C.swap_rows(a, b);
    }

    auto m() const {return C.n_cols;}

    template <class V>
    Mat<T> sqrt(V const &v) const {return C * v;}
};

/******************************************************************************************/

// Operator formed as K from given sparse matrix K but also possessing factorization C C^T = K
template <class T>
struct SparseFactorizedMatrix : SparseMatrix<T> {
    using base_type = SparseMatrix<T>;
    using base_type::x;
    SpMat<T> C;

    SparseFactorizedMatrix(SpMat<T> k, SpMat<T> c) : base_type(std::move(k)), C(std::move(c)) {}

    // full() is applied in the original ordered basis, so it has to be pivoted appropriately
    // sqrt() is applied only to an unordered basis, so pivoting only on its first dimensions
    template <class V>
    Mat<T> sqrt(V const &v) const {return (C * v).eval().rows(x);}

    auto m() const {return C.n_cols;}
};

/******************************************************************************************/

// Random iid Gaussian matrix
template <class T>
Mat<T> gaussians(la::uword n, la::uword z) {
    // if (z == n) return la::eye(n, n) * std::sqrt(T(n)); // For debugging
    return Mat<T>(n, z, la::fill::randn);
}

/******************************************************************************************/

// Algorithm 5: Matrix-free selection algorithm, forms base of other algorithms.
template <class T>
struct RandomizedSelect {
    Mat<T> U, S;
    la::uvec index;
    
    RandomizedSelect() = default;
    RandomizedSelect(la::uword n, la::uword k) :
          U(k, n, la::fill::zeros), // upper triangular factor, ends up as Cholesky factor of K^-1
          S(n, k, la::fill::none), // some stored columns needed for K^2 diagonal update
          index(vmap<la::uvec>(range(n))) {} // indices represented

    void pivot(la::uword a, la::uword b) {
        U.swap_cols(a, b);
        S.swap_rows(a, b);
        index.swap_rows(a, b);
    }

    auto n() const {return S.n_rows;}

    template <class M>
    void augment(M const &K, la::uword i) {
        cspan a(0, i), A(0, i+1), B(i+1, n());
        U(i, i) = 1;
        U.col(i) /= std::sqrt(K.diagonal(i) - (i ? sq(la::norm(S(i, a))) : 0));
        S.col(i) = -K.full(U(A, i));
        U(A, B) += U(A, i) * S(B, i).t();
    }

    template <class M>
    auto randomized_scores(M const &K, la::uword i, la::uword z) const {
        Mat<T> const N = K.full(gaussians<T>(n(), z));
        Mat<T> const D = K.sqrt(gaussians<T>(K.m(), z));
        Col<T> scores, denominator;
        if (i) {
            cspan A(0, i), B(i, n());
            denominator = la::sum(la::square(D.rows(B) + S(B, A) * U(A, A).t() * D.rows(A)), 1);
            scores = la::sum(la::square(N.rows(B) + S(B, A) * U(A, A).t() * N.rows(A)), 1)
                   / denominator;
        } else {
            denominator = la::sum(la::square(D), 1);
            scores = la::sum(la::square(N), 1) / denominator;
        }
        return std::make_pair(std::move(scores), std::move(denominator));
    }

    Col<T> reference_scores(Mat<T> K, la::uword i) {
        cspan A(0, i);
        if (i) K -= K.cols(A) * la::inv_sympd(K(A, A)) * K.rows(A);
        Col<T> scores = la::sum(K % K, 1) / K.diag();
        return scores(span(i, n()));
    }

    T reference_objective(Mat<T> const &K, la::uword i) {
        cspan A(0, i);
        return i ? la::accu((K.cols(A) * la::inv_sympd(K(A, A)) * K.rows(A)).eval().diag()) : 0;
    }

    T objective(la::uword i) const {
        if (i == 0) return 0;
        return la::accu(la::square(S.head_cols(i)));
    }
};

/******************************************************************************************/

// Algorithm 3: Exact selection algorithm which maintains numerator and denominator at all times
template <class T>
struct ExactSelect : RandomizedSelect<T> {
    using base_type = RandomizedSelect<T>;
    using base_type::S;
    Col<T> d, t, w, f;

    auto n() const {return std::size(d);}
    
    template <class M>
    ExactSelect(M const &K, real threshold, la::uword k) 
        : base_type(K.n(), k), 
          d(K.diagonal()), // denominator in the gain expression, diagonal of K^(i), where K^(i) is K - K[:,:i] @ inv(K[:i,:i]) @ K[:i,:]
          t(threshold * d),
          w(K.square_diagonal()) {} // numerator in the gain expression, diagonal of (K^(i))^2

    template <class M>
    void augment(M const &K, la::uword i) {
        base_type::augment(K, i);
        cspan A(0, i+1), B(i+1, n());
        d(B) -= la::square(S(B, i));
        auto const &s = S.col(i);
        f = K.full(s) - S.cols(A) * S.cols(A).t() * s;
        w -= s % (2 * f + la::dot(s, s) * s);
    }

    Col<T> denominator(uint i) const {return d(span(i, n()));}

    void pivot(la::uword a, la::uword b) {
        base_type::pivot(a, b);
        d.swap_rows(a, b);
        w.swap_rows(a, b);
    }

    auto scores(la::uword i) const {
        cspan B(i, n());
        Col<T> s = w(B) / d(B), den = d(B);
        for (auto i : la::find(d(B) < t(B)).eval()) s(i) = 0;
        return std::make_pair(std::move(s), std::move(den));
    }
};

/******************************************************************************************/

// Algorithm 4: Exact Laplacian selection algorithm which maintains numerator and denominator at all times
template <class T>
struct ExactLaplacianSelect : ExactSelect<T> {
    using base_type = ExactSelect<T>;
    using base_type::S; using base_type::d; using base_type::w; 
    using base_type::U; using base_type::n; using base_type::f;
    Col<T> h; // = stationary eigenvector (norm 1)
    Col<T> y; // = (I - L^+ X (X^T L^+ X)^-1 X^T) h
    Col<T> c; // something for numerator
    T g = 0; // = h^T X (X^T L^+ X)^-1 X^T h

    template <class M>
    ExactLaplacianSelect(M const &K, Col<T> h0, la::uword k) :
        base_type(K, T(0), k), h(h0), y(h0), c(n(), la::fill::zeros) {}
    
    template <class M>
    void augment(M const &K, la::uword i) {
        base_type::augment(K, i);
        cspan A(0, i+1);
        T const t = la::dot(U(A, i), h(A));
        c += t * f - la::dot(S.col(i), y) * S.col(i);
        y += t * S.col(i);
        g += sq(t);
    }

    void pivot(la::uword a, la::uword b) {
        base_type::pivot(a, b);
        h.swap_rows(a, b);
        y.swap_rows(a, b);
        c.swap_rows(a, b);
    }

    Col<T> denominator(la::uword i) const {
        if (i == 0) return la::square(h); // yes seems right
        cspan B(i, n());
        return d(B) + 1 / g * la::square(y(B)); // verified
    }

    auto scores(la::uword i) const {
        Col<T> scores, den;
        if (i == 0) {
            den = la::square(h);
            scores = -base_type::d / den;
        } else {
            cspan B(i, n());
            den = d(B) + 1 / g * la::square(y(B));
            scores = (w(B) + 2 / g * y(B) % c(B) + la::dot(y, y) / sq(g) * la::square(y(B))) / den;
        }
        return std::make_pair(std::move(scores), std::move(den));
    }

    T reference_objective(Mat<T> const &L, la::uword i) const {
        if (i == 0) return 0;
        cspan B(i, n());
        return la::accu(la::inv_sympd(L + h * h.t()).eval().diag()) - 1 - la::accu(la::inv_sympd(L(B, B)).eval().diag());
    }

    T objective(la::uword i) const {
        return i == 0 ? 0 :  base_type::objective(i) - la::dot(y, y) / g;
    }

    Col<T> reference_scores(Mat<T> const &K, la::uword i) const {
        if (i == 0) return -K.diag() / la::square(h);
        Mat<T> const I = la::eye(n(), n()), X = I.head_cols(i);
        Mat<T> const M = X * la::inv_sympd(X.t() * K * X) * X.t();
        T const r = la::as_scalar(h.t() * M * h);
        Mat<T> const P = I - K * M;
        Mat<T> const Q = P * (r * I - h * h.t() * M);
        Col<T> const s = (P * (Q * K * K * Q.t() + h * h.t()) * P.t()).eval().diag()
                       / (P * (sq(r) * K + r * h * h.t()) * P.t()).eval().diag();
        return s.tail(n() - i);
    }

    template <class M>
    auto randomized_scores(M const &K, la::uword i, la::uword z) const {
        Mat<T> const D = K.sqrt(gaussians<T>(K.m(), z));
        Col<T> scores, denominator;
        if (i) {
            Mat<T> const N = gaussians<T>(n(), z);
            cspan A(0, i), B(i, n());
            denominator = (la::mean(la::square(D.rows(B) + S(B, A) * U(A, A).t() * D.rows(A)), 1) + 1 / g * la::square(y(B)));
            scores = (la::mean(la::square((K.full(N).rows(B) - S(B, A) * S.cols(A).t() * N - 1 / g * y(B) * (h - y).t() * N)), 1) + 1 / sq(g) * la::square(y(B))) 
                   / denominator;
        } else {
            denominator = la::square(h);
            scores = -la::mean(la::square(D), 1) / denominator;
        }
        return std::make_pair(std::move(scores), std::move(denominator));
    }
};

/******************************************************************************************/

// Algorithm 6: Matrix-free Laplacian selection algorithm
template <class T>
struct RandomizedLaplacianSelect : RandomizedSelect<T> {
    using base_type = RandomizedSelect<T>;
    using base_type::S; using base_type::U; using base_type::n;
    Col<T> h, y;
    T g = 0;

    RandomizedLaplacianSelect(Col<T> h0, la::uword k) :
        base_type(h0.n_rows, k), h(std::move(h0)), y(h) {}
    
    template <class M>
    void augment(M const &K, la::uword i) {
        base_type::augment(K, i);
        cspan A(0, i+1);
        T const t = la::dot(U(A, i), h(A));
        y += S.col(i) * t;
        g += sq(t);
    }

    T objective(la::uword i) const {
        return i == 0 ? 0 :  base_type::objective(i) - la::dot(y, y) / g;
    }

    void pivot(la::uword a, la::uword b) {
        base_type::pivot(a, b);
        h.swap_rows(a, b);
        y.swap_rows(a, b);
    }

    template <class M>
    Col<T> denominator(M const &K, la::uword i, la::uword z) const {
        if (i == 0) return la::square(h);
        cspan A(0, i), B(i, n());
        Mat<T> const D = K.sqrt(gaussians<T>(n(), z));
        return la::mean(la::square(D.rows(B) + S(B, A) * U(A, A).t() * D.rows(A)), 1) + 1 / g * la::square(y(B));
    }

    template <class M>
    auto randomized_scores(M const &K, la::uword i, la::uword z) const {
        Mat<T> const D = K.sqrt(gaussians<T>(n(), z));
        Col<T> scores, denominator;
        if (i) {
            Mat<T> const N = gaussians<T>(n(), z);
            cspan A(0, i), B(i, n());
            denominator = la::mean(la::square(D.rows(B) + S(B, A) * U(A, A).t() * D.rows(A)), 1) + 1 / g * la::square(y(B));
            scores = (la::mean(la::square((K.full(N).rows(B) - S(B, A) * S.cols(A).t() * N - 1 / g * y(B) * (h - y).t() * N)), 1) + 1 / sq(g) * la::square(y(B))) 
                     / denominator;
        } else {
            denominator = la::square(h);
            scores = -la::mean(la::square(D), 1) / denominator;
        }
        return std::make_pair(std::move(scores), std::move(denominator));
    }
};

/******************************************************************************************/

}
