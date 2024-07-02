#pragma once
/**
    Preconditioning.h: routines to incorporate rchol preconditioner, perform Chebyshev interpolation, and perform subspace iteration
*/

#include "Cholesky.h"
#include "Matrix.h"
#include "Execution.h"

#include <sparse.hpp>
#include <rchol/rchol.hpp>
#include <util/util.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/minimum_degree_ordering.hpp>

namespace nsm {

/******************************************************************************************/

// Conversion to rchol from armadillo
inline SparseCSR to_csr(SpMat<real> const &A) {
    return {vec<std::size_t>(A.col_ptrs, A.col_ptrs+A.n_cols+1), 
            vec<std::size_t>(A.row_indices, A.row_indices + A.n_nonzero), 
            vec<real>(A.values, A.values + A.n_nonzero)};
}

// Conversion from rchol to armadillo
inline SpMat<real> to_spmat(SparseCSR const &A) {
    la::uword const n = A.size(), nnz = A.nnz();
    return SpMat<real>(vmap<la::uvec>(ptr_view(A.colIdx, nnz)), vmap<la::uvec>(ptr_view(A.rowPtr, n+1)), vmap<la::Col<double>>(ptr_view(A.val, nnz)), n, n);
}

/******************************************************************************************/

// Generate randomized Cholesky factor given sparse matrix A and random number generator
inline SpMat<real> rchol(rchol_rng &gen, SpMat<real> const &A) {
    SparseCSR G, Ar = to_csr(A);
    rchol(gen, Ar, G);
    return to_spmat(G);
}

/******************************************************************************************/

// Conjugate gradient function which includes preconditioner and stopping condition inputs
template <class O, class M, class B, class P, class C>
std::size_t conjugate_gradient(O &&x, M const &A, B const &b, P &&preconditioner, C &&condition) {
    using T = typename std::decay_t<decltype(x.eval())>::elem_type;
    Col<T> r = b - A(x), z = preconditioner(r), p = z, Ap;
    T rz = la::dot(r, z);
    for (std::size_t t = 0; !condition(r, t); ++t) {
        Ap = A(p);
        T const alpha = rz / la::dot(p, Ap);
        x += alpha * p;
        r = b - A(x);
        z = preconditioner(r);
        T const rz0 = std::exchange(rz, la::dot(r, z));
        T const beta = rz / rz0;
        p = beta * p + z;
    }
    return 0;
}

/******************************************************************************************/

// Facilities to make Boost graph from a sparse matrix
using SimpleDirectedGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>;

template <class T>
auto directed_adjacency_list(SpMat<T> const &A) {
    SimpleDirectedGraph G(A.n_rows);
    for (auto it : iterators(A)) if (it.row() != it.col()) boost::add_edge(it.row(), it.col(), G);
    return G;
}

/******************************************************************************************/

// Minimum degree ordering algorithm (can be slow for large graphs)
// Returns:
//      perm: map(old index -> new index)
//      inverse_perm: map(new index -> old index)
inline auto minimum_degree_ordering(SimpleDirectedGraph G, int delta=0) {
    auto const n = boost::num_vertices(G);
    vec<int> inverse_perm(n, 0);
    vec<int> perm(n, 0);
    vec<int> supernode_sizes(n, 1); // init has to be 1
    auto const id = boost::get(boost::vertex_index, G);

    vec<int> degree(n, 0);

    boost::minimum_degree_ordering(G,
        make_iterator_property_map(&degree[0], id, degree[0]), &inverse_perm[0], &perm[0],
        make_iterator_property_map(&supernode_sizes[0], id, supernode_sizes[0]), delta, id);

    return std::make_pair(std::move(perm), std::move(inverse_perm));
}

// Return new sparse matrix given map from old index to new index
template <class T, class V>
SpMat<T> permute_spmat(SpMat<T> const &A, V const &map) {
    la::umat locations(2, A.n_nonzero, la::fill::none);
    Col<T> values(A.n_nonzero, la::fill::none);
    la::uword n = 0;
    for (auto it : iterators(A)) {
        locations(0, n) = map(it.row());
        locations(1, n) = map(it.col());
        values(n++) = *it;
    }
    return SpMat<T>(std::move(locations), std::move(values), A.n_rows, A.n_cols);
}

/******************************************************************************************/

// Preconditioner iterative options
struct PreconditionedOptions {
    real tolerance = 1e-8;
    la::uword iters = 1000;
};

// Convergence failure for iterative methods
template <class T>
struct ConvergenceFailure : std::runtime_error {
    vec<T> history;
    Col<T> residual, h;
    T tolerance;

    ConvergenceFailure(std::string s, vec<T> hist, Col<T> r, Col<T> h, T t) 
        : std::runtime_error(std::move(s)), history(std::move(hist)), residual(std::move(r)), h(std::move(h)), tolerance(t) {}
};

// Operator using an approximately factorized Laplacian
template <class T>
struct PreconditionedMatrix {
    Executor executor;
    std::function<void(std::size_t, vec<T>)> callback;
    SpMat<T> A, U, L; // A: the Laplacian, U and L: inverses of triangular preconditioning factors
    Col<T> h; // stationary eigenvector of the Laplacian -- not the preconditioned matrix!!
    Col<T> v; // stationary eigenvector of preconditioned inverse
    la::uvec o;
    T tolerance;
    la::uword iters;

    PreconditionedMatrix() = default;

    PreconditionedMatrix(Executor ex, SpMat<T> l, SpMat<T> u, Col<T> h0, PreconditionedOptions const &ops) 
        : executor(std::move(ex)), A(std::move(l)), U(std::move(u)), L(U.t()), h(std::move(h0)), o(vmap<la::uvec>(range(n()))),
          tolerance(ops.tolerance), iters(ops.iters) {
            if (U(U.n_rows-1, U.n_rows-1) == 0) {
                v.zeros(U.n_rows);
                v.back() = 1;
            } else {
                v = lower_solve(U, h);
                v /= la::norm(v);
            }
        }

    Mat<T> preconditioned_inverse() const {return la::pinv(Mat<T>(L)) * A * la::pinv(Mat<T>(U));}
    Mat<T> dense() const {return la::pinv(Mat<T>(A));}

    template <class X>
    void solve(X &&x, Col<T> &b) const {
        b -= la::dot(b, h) * h;
        vec<T> history;
        auto steps = conjugate_gradient(x, [&](auto const &x) -> Col<T> {
            return sparse_matvec(A, x);
        }, b, [&](auto const &e) -> Col<T> {
            Col<T> const fix_e = e - la::dot(e, h) * h;
            Col<T> o = upper_solve(L, lower_solve(U, fix_e));
            o -= la::dot(o, h) * h;
            return o;
        }, [&, tol=tolerance * la::norm(b), iters=iters](auto const &r, auto t) {
            // print("-- solve CG", t, tol, la::norm(r));
            history.emplace_back(la::norm(r));
            if (t == iters) throw ConvergenceFailure("PreconditionedMatrix::solve(): CG failed to converge", history, r, h, tol);
            return history.back() < tol;
        });
        if (callback) callback(steps, std::move(history));
    }

    T diagonal(uint i) const {
        Col<T> x(n(), la::fill::zeros), b(n(), la::fill::zeros);
        b(o(i)) = 1;
        solve(x.col(0), b);
        return x(o(i));
    }
    
    void pivot(la::uword a, la::uword b) {o.swap_rows(a, b);}
    auto n() const {return A.n_rows;}

    template <class V>
    Mat<T> full(V const &v) const {
        Mat<T> X(n(), v.n_cols, la::fill::zeros);
        executor.map(range(v.n_cols), [&](auto j) {
            Col<T> b(n(), la::fill::zeros);
            b(o.head(v.n_rows)) = v.col(j);
            solve(X.col(j), b);
        });
        return std::move(X).rows(o);
    }
};

/******************************************************************************************/

struct AlwaysFalse {
    template <class ...Ts>
    constexpr bool operator()(Ts const &...) const {return false;}
};

// Simple power method implementation
template <class T, class M, class F=AlwaysFalse>
T power_method(M const &K, la::uword n, la::uword iters, F &&predicate={}) {
    Col<T> x(n, la::fill::randn), y;
    T estimate0 = 0, estimate = 0;
    for (auto t : range(iters)) {
        x /= la::norm(x);
        K(y, x);
        estimate = la::dot(y, x);
        if (predicate(t, y, estimate0, estimate)) break;
        estimate0 = estimate;
        x.swap(y);
    }
    return estimate;
}

/******************************************************************************************/

// Simple subspace method options and method
struct SimultaneousIterationOptions {
    la::uword iters, n = 0, k, kmax = 0;
    la::uword max_k() const {return kmax ? kmax : std::numeric_limits<la::uword>::max();}
};

template <class T, class M, class F>
auto psd_simultaneous_iteration(Executor const &ex, M const &K, F &&predicate, SimultaneousIterationOptions const &ops) {
    la::uword const batch = ex.n_workers();
    la::uword const k = min(ops.max_k(), min(ops.n, std::ceil(real(min(ops.n, ops.k)) / batch) * batch)); // Number of columns to actually use
    Mat<T> X(ops.n, k, la::fill::randn), Q, R;
    Col<T> e;
    T tr = 0;
    for (auto t : range(ops.iters)) {
        NSM_ASSERT(la::qr_econ(Q, R, X), "QR decomposition failed in psd_simultaneous_iteration()");
        NSM_REQUIRE(X.n_rows, ==, Q.n_rows, "QR rows do not match");
        NSM_REQUIRE(X.n_cols, ==, Q.n_cols, "QR cols do not match");
        ex.map(range(k), [&](auto i) {K(X.unsafe_col(i), Q.unsafe_col(i));});
        T const tr2 = la::accu(la::square(X));
        if (predicate(t, tr, tr2)) break;
        tr = tr2;
    }
    NSM_ASSERT(la::eig_sym(e, X.t() * X), "Eigendecomposition failed in psd_simultaneous_iteration()", X.t() * X);
    e = la::sqrt(e.clamp(0, std::numeric_limits<T>::infinity()));
    sort(e, [](auto const &x, auto const &y) {return std::abs(x) > std::abs(y);});
    return std::make_pair(std::move(e), std::move(Q));
}

template <class T, class M, class F>
auto svd_simultaneous_iteration(Executor const &ex, M const &A, F &&predicate, SimultaneousIterationOptions ops) {
    ops.kmax = min(min(ops.max_k(), A.n_rows), A.n_cols);
    ops.n = A.n_rows;
    auto p = psd_simultaneous_iteration<T>(ex, [&](auto &&y, auto const &x) {y = A * A.t() * x;}, predicate, ops);
    p.first = la::sqrt(p.first);
    return p;
}

/******************************************************************************************/

// Returns Chebyshev roots in ascending order
template <class T>
Col<T> chebyshev_points(T a, T b, uint n) {
    return vmap<Col<T>>(range(n), [f=T(M_PI) / (2 * n), a, b](uint k) {
        return (1 - std::cos((2 * k + 1) * f)) / 2 * (b - a) + a;
    });
}

/******************************************************************************************/

// Chebyshev interpolation options
struct InterpolationOptions {
    real chebyshev_tolerance = 1e-8;
    uint power_iters = 100;
    real multiplier = 2;
    real power_tolerance = 1e-10;
};

/******************************************************************************************/

// Chebyshev interpolator object
template <class T>
struct InvSqrtInterpolation {
    T a, b;
    Col<T> cs; // Chebyshev coefficients

    InvSqrtInterpolation(T a, T b, uint pts) : a(a), b(b), cs(pts, la::fill::none) {
        Col<real> const xs = chebyshev_points<real>(-1, 1, pts);
        Col<real> const fs = la::pow((xs + 1) / 2 * (b - a) + a, -0.5);
        Col<real> t0(std::size(xs), la::fill::ones), t1 = std::sqrt(2.0) * xs, t2 = 2 * xs % t1 - std::sqrt(2.0);
        for (auto i : indices(xs)) {
            cs(i) = la::dot(fs, t0) / std::size(xs);
            t0.swap(t1);
            t1.swap(t2);
            t2 = 2 * xs % t1 - t0;
        }
    }

    template <class F>
    Mat<T> operator()(Mat<T> B, F &&f) const { 
        auto &X0 = B; // B plays the role of X0
        Mat<real> X1 = std::sqrt(2.0) * 2 / (b - a) * f(X0) + -std::sqrt(2.0) * (2 / (b - a) * a + 1) * X0;
        Mat<real> X2 = 4 / (b - a) * f(X1) - (4 / (b - a) * a + 2) * X1 - std::sqrt(2.0) * X0;
        Mat<real> O(B.n_rows, B.n_cols, la::fill::zeros);
        for (auto i : indices(cs)) {
            O += cs(i) * X0;
            X0.swap(X1);
            X1.swap(X2);
            if (i+2 < std::size(cs)) X2 = 4 / (b - a) * f(X1) + -(2 + 4 / (b - a) * a) * X1 - X0;
        }
        return O;
    }
};

/******************************************************************************************/

// Function to decide number of Chebyshev nodes
inline uint chebyshev_isqrt_points(real k, real e) {
    if (k == 1) return 1;
    NSM_REQUIRE(k, >, 1, "invalid number of nodes in chebyshev_isqrt_points()");
    NSM_REQUIRE(e, >, 0, "invalid epsilon in chebyshev_isqrt_points()");
    uint n = std::ceil(
          (1 - 2 * std::log(e) + 2 * std::log(std::sqrt(k) - 1))
        / (4 * std::log(std::sqrt(k) + 1) - 2 * std::log(k - 1)));

    auto const rng = range(1, 2 * n);
    uint m = *std::lower_bound(rng.begin(), rng.end(), e, [k](uint n, real e) {
        real exact = (std::pow(2, n + 2) * n * (std::sqrt(k) - 1)) 
            / ((4 * n - 1 - std::sqrt(k)) * std::pow(((2 * n - 1) * sq(1 + std::sqrt(k))) / (n * (k - 1)), n));
        return exact > e;
    });
    return m;
}

/******************************************************************************************/

// Function to create inverse square root interpolation including subtraction of stationary mode
template <class T>
InvSqrtInterpolation<T> stationary_isqrt_interpolation(Executor const &ex, PreconditionedMatrix<T> const &p, InterpolationOptions const &ops) {
    T const min = 1 / psd_simultaneous_iteration<T>(ex, [&](auto &&x, Col<T> b) { // solve x = (U^-1 A L^-1)^-1 b via CG
        b -= p.v * la::dot(b, p.v);
        Col<T> const Lb = upper_solve(p.L, b);
        x = b * la::as_scalar(Lb.t() * p.A * Lb);
        conjugate_gradient(x, [&](auto const &b) {
            Col<T> x = lower_solve(p.U, sparse_matvec(p.A, upper_solve(p.L, b)));
            x -= la::dot(x, p.v) * p.v;
            return x;
        }, b, Identity(), [tol=p.tolerance * la::norm(b), n=p.iters, &b, history=vec<T>()](auto const &r, auto t) mutable {
            // print("    -- isqrt CG:", t, la::norm(r), tol);
            history.emplace_back(la::norm(r));
            if (t == n) throw ConvergenceFailure("CG did not converge in inverse sqrt computation", history, r, {}, tol);
            return history.back() < tol;
        });
    }, [&ops](auto t, T b, T e) {
        print("-- inverse power method", t, e, e/b - 1);
        return std::abs(e/b - 1) < ops.power_tolerance;
    }, {.iters=ops.power_iters, .n=p.n(), .k=1}).first(0);
    T const max = psd_simultaneous_iteration<T>(ex, [&](auto &&x, auto const &b) {
        x = lower_solve(p.U, sparse_matvec(p.A, upper_solve(p.L, b)));
        x -= la::dot(x, p.v) * p.v;
    }, [&ops](auto t, T b, T e) {
        // print("-- forward power method", t, e(0), e(0)/b(0) - 1);
        return std::abs(e/b - 1) < ops.power_tolerance;
    }, {.iters=ops.power_iters, .n=p.n(), .k=1}).first(0);
    uint const npts = chebyshev_isqrt_points(max / min * sq(ops.multiplier), ops.chebyshev_tolerance);
    return InvSqrtInterpolation<T>(min / ops.multiplier, max * ops.multiplier, npts);
}

/******************************************************************************************/

// Operator extending PreconditionedMatrix to apply square roots using Chebyshev interpolation
template <class T>
struct PreconditionedSqrt : PreconditionedMatrix<T> {
    using base_type = PreconditionedMatrix<T>;
    InvSqrtInterpolation<T> interpolation;
    
    PreconditionedSqrt() = default;

    PreconditionedSqrt(Executor ex, SpMat<T> l, SpMat<T> u, Col<T> h, PreconditionedOptions const &o, InterpolationOptions const &s) : 
        base_type(std::move(ex), std::move(l), std::move(u), std::move(h), o), interpolation(stationary_isqrt_interpolation(base_type::executor, *this, s)) {}

    Mat<T> sqrt(Mat<T> const &B) const {
        if (B.n_cols > 1 && base_type::executor.n_workers() > 1) {
            Mat<T> O(B.n_rows, B.n_cols, la::fill::none);
            base_type::executor.map(range(B.n_cols), [&](auto j) {O.col(j) = sqrt(B.col(j));});
            return O;
        }
        auto const &p = static_cast<base_type const &>(*this);
        NSM_REQUIRE(B.n_rows, ==, std::size(p.v), "size mismatch in PreconditionedSqrt()");
        Mat<T> C = interpolation(B, [&](auto const &B) {
            Mat<T> O = lower_solve(p.U, sparse_matmul(p.A, upper_solve(p.L, B))); // = p.preconditioned_inverse() * B 
            O -= p.v * p.v.t() * O; // needed to prevent numerical precision from blowing up the stationary mode
            return O;
        });
        C = upper_solve(p.L, C);
        C -= p.h * p.h.t() * C; // correct again for stationary mode
        return std::move(C).rows(p.o);
    }
};

/******************************************************************************************/

}