#pragma once
#include "Engines.h"

namespace nsm {

/******************************************************************************************/

enum class Selector {direct, greedy, random, uniform, ordered};

// Randomized selection out of any matrix, Laplacian inverse or otherwise
template <class E, class M>
std::pair<Mat<real>, la::uvec> randomized_select(E &&e, M &&K, Selector s, uint k, uint z, int first=-1) {
    Mat<real> gains(2, k, la::fill::zeros);
    real objective = 0;
    for (auto i : range(k)) {
        auto const [g, d] = e.randomized_scores(K, i, z);
        la::uvec const f = la::find_finite(g);
        if (f.empty()) break;
        la::uword c;
        switch (s) {
            case Selector::direct: {c = g(f).index_max(); break;}
            case Selector::greedy: {c = d(f).index_max(); break;}
            case Selector::random: {c = discrete_distribution(d(f).eval())(StaticRNG); break;}
            case Selector::uniform: {c = random_range(0, len(f)); break;}
            case Selector::ordered: {c = 0; break;}
        }
        c = f(c);
        if (first >= 0 && i == 0) c = first;
        gains(1, i) = g(c);
        e.pivot(i, i+c);
        K.pivot(i, i+c);
        e.augment(K, i);
        real const obj = e.objective(i+1);
        gains(0, i) = obj - std::exchange(objective, obj);
        // if (i % 10 == 0) // print("-- selection gain", enum_to_string(s), i, gains(0, i), gains(1, i));
    }
    return {std::move(gains), e.index.head(k)};
}

/******************************************************************************************/

// Randomized selection out of any matrix, Laplacian inverse or otherwise
template <class E, class M>
std::pair<Col<real>, la::uvec> exact_select(E &&e, M &&K, Selector s, uint k) {
    Col<real> gains(k, la::fill::zeros);
    la::uvec idx(k, la::fill::none);
    la::uword i = 0;
    for (auto const t : range(k)) {
        auto const [g, d] = e.scores(i);
        la::uvec const f = la::find_finite(g);
        if (f.empty()) break;
        la::uword c;
        switch (s) {
            case Selector::direct: {c = g(f).index_max(); break;}
            case Selector::greedy: {c = d(f).index_max(); break;}
            case Selector::random: {c = discrete_distribution(d(f).eval())(StaticRNG); break;}
            case Selector::uniform: {c = random_range(0, len(f)); break;}
            case Selector::ordered: {c = 0; break;}
        }
        c = f(c);
        idx(t) = e.index(i+c);
        gains(t) = g(c);
        e.pivot(i, i+c);
        K.pivot(i, i+c);
        e.augment(K, i);
        ++i;
    }
    return {std::move(gains), std::move(idx)};
}

template <class Matrix>
auto condition_number(Matrix const &M, uint b=0, uint e=0) {
    auto v = la::eig_sym(M).eval();
    sort(v, less_abs);
    return std::abs(v(v.n_rows - 1 - e)) / std::abs(v(b));
}

template <class T>
SpMat<T> renormalized_rchol(rchol_rng &gen, SpMat<T> L, Col<T> const &h) {
    for (auto it : iterators(L)) *it *= h(it.col()) * h(it.row());
    SpMat<T> U = rchol(gen, L).t();
    for (auto it : iterators(U)) *it /= h(it.col());
    return U;
}

la::uword necessary_power_iterations(la::uword n, real pfail, real epsilon) {
    la::uword const k = std::ceil(0.5 + (-std::log(pfail) + std::log(0.824) + std::log(n) / 2) / epsilon);
    auto const r = range(1, 2 * k);
    return *std::lower_bound(r.begin(), r.end(), pfail, [&](la::uword k, real p) {
        auto o = std::min(0.824, 0.354 / std::sqrt(epsilon * (k - 1))) * std::sqrt(n) * std::pow(1 - epsilon, k - 0.5);
        return o > p;
    });
}

PROTOTYPE("cs/cholesky-orth") = [](Context ct) {
    Mat<real> A(100000, 100, la::fill::randu);
    Mat<real> C = la::chol(A.t() * A).t();
    Mat<real> X = la::solve(la::trimatl(C.t()), A.t()).t();
    // print(X.t() * X);
};

struct EvaluationOptions {
    InterpolationOptions interpolation;
    PreconditionedOptions preconditioner;
    real eigenvalue_norm_tolerance = 1e-8;
    uint eigenvalue_iters = 100;
    bool fix_first, do_exact;
};

template <class T, class M>
Mat<T> stationary_inverse(M const &L, Col<T> const &h) {
    T const tr = la::accu(L.diag()) / len(h);
    return la::inv_sympd(L + tr * h * h.t()) - 1 / tr * h * h.t();
}

template <class T>
void evaluate_selection(Local const &exec, SpMat<T> L, Col<T> h, uint reps, uint seed, uint k, uint z, EvaluationOptions const &ops, std::function<void(json const &)> &&callback) {
    print("Finding minimum degree ordering");
    auto [perm, inverse_perm] = minimum_degree_ordering(directed_adjacency_list(L), 0); 
    print("Permuting matrix");
    L = permute_spmat(L, vmap<la::uvec>(perm));
    h = h(vmap<la::uvec>(inverse_perm));
    print("Running renormalized rchol");
    rchol_rng gen;
    gen.seed(seed + 1);
    SpMat<T> const R = renormalized_rchol<T>(gen, L, h);
    NSM_REQUIRE(Col<T>(R.diag()).head(R.n_rows-1).min(), >, 0);
    print("Last rchol element", R(R.n_rows - 1, R.n_rows - 1));
    print("Number of renormalized rchol nonzeros =", R.n_nonzero);

    print("Running incomplete Cholesky");
    SpMat<T> const C = incomplete_cholesky(L).t();
    NSM_REQUIRE(C.diag().min(), >, 0);
    print("Number of incomplete Cholesky nonzeros =", C.n_nonzero);

    la::arma_rng::set_seed(seed + 2);
    print("Setting up square root interpolation");
    auto const K = PreconditionedSqrt<T>(exec, L, R, h, ops.preconditioner, ops.interpolation);
    print("Chebyshev limits =", K.interpolation.a, K.interpolation.b);
    print("Deduced condition number =", K.interpolation.b / K.interpolation.a / sq(ops.interpolation.multiplier));

    auto const predicate = [tol=ops.eigenvalue_norm_tolerance](auto t, T tr0, T tr) {
        print("-- power method eigenvalues", t, abs(tr/tr0 - 1), t); 
        return abs(tr - tr0) < tol * tr0;
    };

    print("Finding inverse eigenvalues via power method");
    auto const leigs = psd_simultaneous_iteration<real>(exec, [&](auto &&y, auto const &x) {y = L * x;}, predicate, {.iters=ops.eigenvalue_iters, .n=L.n_rows, .k=k}).first;
    print("Deduced eigenvalues =", json(leigs));

    print("Finding optimal eigenvalues via power method");
    auto const eigs = psd_simultaneous_iteration<real>(exec, [&](auto &&y, auto const &x) {y = K.full(x);}, predicate, {.iters=ops.eigenvalue_iters, .n=L.n_rows, .k=k}).first;
    print("Deduced eigenvalues =", json(eigs));
    print("Deduced partial trace =", la::accu(eigs.head(k)));
    
    print("Running selection algorithms");
    json entry = {
        {"options", ops}, {"minimum", K.interpolation.a}, {"maximum", K.interpolation.b}, 
        {"eigenvalues", eigs}, {"inverse_eigenvalues", leigs},
        {"k", k}, {"n", L.n_rows}, {"nnz", L.n_nonzero}, {"nnz_rchol", R.n_nonzero}, {"nnz_ic", C.n_nonzero}, {"ncheb", len(K.interpolation.cs)}
    };
    int first = -1;
    for (auto rep : range(reps)) {
        for (auto s : {Selector::direct, Selector::greedy, Selector::random, Selector::uniform}) {
            callback(entry);
            la::arma_rng::set_seed(seed+rep+3);
            auto &stuff = entry["results"].emplace_back();
            stuff["method"] = enum_to_string(s);
            print(rep, stuff.at("method"));
            stuff["time"] = time_it([&, inverse_perm=inverse_perm] {
                auto K2 = K;
                std::size_t n = 0, iters = 0;
                K2.callback = [&](la::uword its, Ignore) {++n; iters += its;};
                auto const [gains, choices] = randomized_select(RandomizedLaplacianSelect<T>(h, k), K2, s, k, z, first);
                if (ops.fix_first && first == -1) first = choices(0);
                stuff["exact_gains"] = Col<real>(gains.row(0).t());
                stuff["random_gains"] = Col<real>(gains.row(1).t());
                stuff["choices"] = vmap(choices, [&, x=inverse_perm](auto i) {return x[i];});
                stuff["solves"] = n;
                stuff["iters"] = iters;
                print_lns(gains);
                print("Achieved objective =", la::accu(gains.row(0)));
            });
        }
    }
    if (ops.do_exact) {
        print("-- starting inversion");
        auto const K0 = DenseMatrix<real>(stationary_inverse(L, h));
        ExactLaplacianSelect<T> const engine(K0, h, k);
        for (auto s : {Selector::direct, Selector::greedy, Selector::random, Selector::uniform})
            for (auto r : range(s == Selector::random || s == Selector::uniform ? reps : 1)) {
                print("-- doing exact version");
                auto const [gains, choices] = exact_select(copy(engine), copy(K0), s, k);
                entry["results"].emplace_back() = {
                    {"exact_gains", gains}, {"method", string(enum_to_string(s)) + "-exact"},
                    {"choices", vmap(choices, [&, x=inverse_perm](auto i) {return x[i];})}
                };
            }
    }
    callback(entry);
}

/******************************************************************************************/

template <class T>
T sparse_dot(SpMat<T> const &A, Col<T> const &x, uint col) {
    T t = 0;
    auto b = A.begin_col(col), e = A.end_col(col);
    for (; b != e; ++b) t += x[b.row()] * *b;
    return t;
}

template <class T>
Col<T> sparse_dot(SpMat<T> const &A, Mat<T> const &X, uint col) {
    Col<T> t(X.n_cols, la::fill::zeros);
    auto b = A.begin_col(col), e = A.end_col(col);
    for (; b != e; ++b) t += X.row(b.row()).t() * *b;
    return t;
}

/******************************************************************************************/

}
