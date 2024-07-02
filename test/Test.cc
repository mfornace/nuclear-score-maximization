#include <random>
#define ARMA_USE_HDF5 1
#include <nuclear-score-maximization/Preconditioning.h>
#include <nuclear-score-maximization/Selection.h>
#include <nuclear-score-maximization/Cholesky.h>
#include <nuclear-score-maximization/Serialize.h>
#include "Test.h"

#include <nlohmann/json.hpp>
#include <magic_enum.hpp>
#include <fmt/format.h>

namespace nsm {

/******************************************************************************************/

template <class T>
auto enum_to_string(T const &t) {return magic_enum::enum_name(t);}

template <class T>
T enum_from_string(std::string_view name) {
    auto out = magic_enum::enum_cast<T>(name);
    NSM_ASSERT(out, "invalid name for enum", name, magic_enum::enum_type_name<T>(), magic_enum::enum_names<T>());
    return *out;
}

/******************************************************************************************/

struct EvaluationOptions {
    InterpolationOptions interpolation;
    PreconditionedOptions preconditioner;
    real eigenvalue_norm_tolerance = 1e-8;
    uint eigenvalue_iters = 100;
    bool fix_first, do_exact;
};

/******************************************************************************************/

// Evaluation routine for inverse Laplacian column selection
template <class T>
void evaluate_selection(Executor const &exec, SpMat<T> L, Col<T> h, uint reps, uint seed, uint k, uint z, EvaluationOptions const &ops, std::function<void(json const &)> &&callback) {
    print("Finding minimum degree ordering");
    auto [perm, inverse_perm] = minimum_degree_ordering(directed_adjacency_list(L), 0); 
    print("Permuting matrix");
    L = permute_spmat(L, vmap<la::uvec>(perm));
    h = h(vmap<la::uvec>(inverse_perm));
    print("Running renormalized rchol");
    rchol_rng gen;
    gen.seed(seed + 1);
    SpMat<T> const R = renormalized_rchol<T>(gen, L, h);
    NSM_REQUIRE(Col<T>(R.diag()).head(R.n_rows-1).min(), >, 0, "negative diagonal entry in R");
    print("Last rchol element", R(R.n_rows - 1, R.n_rows - 1));
    print("Number of renormalized rchol nonzeros =", R.n_nonzero);

    print("Running incomplete Cholesky");
    SpMat<T> const C = incomplete_cholesky(L).t();
    NSM_REQUIRE(C.diag().min(), >, 0, "negative diagonal entry in C");
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
        {"k", k}, {"n", L.n_rows}, {"nnz", L.n_nonzero}, {"nnz_rchol", R.n_nonzero}, 
        {"nnz_ic", C.n_nonzero}, {"ncheb", std::size(K.interpolation.cs)}
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
                auto const [gains, choices] = matrix_free_selection(RandomizedLaplacianSelect<T>(h, k), K2, s, k, z, first);
                if (ops.fix_first && first == -1) first = choices(0);
                stuff["exact_gains"] = Col<real>(gains.row(0).t());
                stuff["random_gains"] = Col<real>(gains.row(1).t());
                stuff["choices"] = vmap(choices, [&, x=inverse_perm](auto i) {return x[i];});
                stuff["solves"] = n;
                stuff["iters"] = iters;
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
                auto const [gains, choices] = deterministic_selection(copy(engine), copy(K0), s, k);
                entry["results"].emplace_back() = {
                    {"exact_gains", gains}, {"method", std::string(enum_to_string(s)) + "-exact"},
                    {"choices", vmap(choices, [&, x=inverse_perm](auto i) {return x[i];})}
                };
            }
    }
    callback(entry);
}

/******************************************************************************************/

// Test deterministic algorithms vs reference algorithms
template <class T, class M>
void test_exact(Context ct, M K, uint m) {
    auto chol = ExactSelect<T>(K, 1e-8, m);
    T obj0 = 0;
    for (auto t : range(m)) {
        auto const [g, d] = chol.scores(t);
        auto const g0 = chol.reference_scores(K.dense(), t);
        ct(HERE).all(ct.within_log(1e-8), g, g0);
        
        auto c = t + g.index_max();
        chol.pivot(t, c);
        K.pivot(t, c);
        
        chol.augment(K, t);
        auto const obj = chol.reference_objective(K.dense(), t+1);
        ct(HERE).within_log(g.max(), obj - obj0, 1e-8);
        auto const obj2 = chol.objective(t+1);
        ct(HERE).within_log(obj, obj2, 1e-8);
        obj0 = obj;
    }
}

/******************************************************************************************/

UNIT_TEST("cs/exact") = [](Context ct) {
    using T = real;
    uint const k = 10;
    Mat<T> const K = la::random_spd<T>(20); 
    test_exact<T>(ct, DenseMatrix<T>(K), k);
    test_exact<T>(ct, SparseMatrix<T>(SpMat<T>(K)), k);
    
    Mat<T> const A(20, 11, la::fill::randn);
    test_exact<T>(ct, DenseSqrtMatrix<T>(A), k);
    test_exact<T>(ct, SparseSqrtMatrix<T>(SpMat<T>(A)), k);
};

/******************************************************************************************/

template <class T, class M>
void test_random(Context ct, M K, uint m) {
    auto chol = ExactSelect<T>(K, 1e-8, m);
    auto rand = RandomizedSelect<T>(K.n(), m);

    for (auto i : range(m)) {
        auto const g = rand.randomized_scores(K, i, 100000).first;
        auto const [g0, d0] = chol.scores(i);
        ct(HERE).all(ct.within_log(1e-1), g, g0);
        auto c = i + g0.index_max();

        rand.pivot(i, c);
        chol.pivot(i, c);
        K.pivot(i, c);

        rand.augment(K, i);
        chol.augment(K, i);
    }
}

// Test randomized algorithms vs deterministic algorithms
UNIT_TEST("cs/random") = [](Context ct) {
    using T = real;
    uint const m = 10;
    Mat<T> const K = la::random_spd<T>(20), C = la::chol(K).t();
    test_random<T>(ct, FactorizedMatrix<T>(K, C), m);
    test_random<T>(ct, SparseFactorizedMatrix<T>(SpMat<T>(K), SpMat<T>(C)), m);
};

/******************************************************************************************/

template <class T>
Mat<T> random_laplacian(la::uword n) {
    Mat<T> L(n, n, la::fill::randu);
    L = L * L.t();
    L = la::diagmat(la::sum(L, 1)) - L;
    return L;
}

template <class T>
Mat<T> stationary_cholesky(Mat<T> K) {
    auto const n = K.n_rows;
    K(n-1, n-1) += la::accu(K.diag()) / K.n_rows;
    Mat<real> C = la::chol(K, "lower");
    C(n-1, n-1) = 0;
    return C;
}

/******************************************************************************************/

// Test of deterministic Laplacian column selection vs reference versions
UNIT_TEST("cs/laplacian/exact") = [](Context ct) {
    using T = real;
    uint const k = 10, n = 20;
    Mat<T> L = random_laplacian<T>(n);
    Col<T> h = la::randu(n); h /= la::norm(h);
    L /= h * h.t();
    Mat<T> const K0 = la::inv_sympd(L + h * h.t()) - h * h.t(), C = stationary_cholesky(K0);
    auto K = FactorizedMatrix<T>(K0, C);
    
    auto chol = ExactLaplacianSelect<T>(K, h, k);
    auto rand = RandomizedLaplacianSelect<T>(h, k);
    T obj0 = 0;
    for (auto i : range(k)) {
        print(obj0);
        auto const [g, d] = chol.scores(i);
        auto const g0 = chol.reference_scores(K.dense(), i);
        auto const gr = chol.randomized_scores(K, i, 100000).first;
        auto const [gR, dR] = rand.randomized_scores(K, i, 100000);
        ct(HERE).all(ct.within_log(1e-8), g, g0);
        ct(HERE).all(ct.within_log(1e-1), g, gr);
        ct(HERE).all(ct.within_log(1e-1), g, gR);

        auto const d0 = chol.denominator(i);
        ct(HERE).all(ct.within_log(1e-1), d0, dR);
        
        auto c = i + g.index_max();
        chol.pivot(i, c);
        rand.pivot(i, c);
        K.pivot(i, c);
        L.swap_rows(i, c);
        L.swap_cols(c, i);
        
        chol.augment(K, i);
        rand.augment(K, i);
        auto const obj = chol.reference_objective(L, i+1);
        ct(HERE).within_log(g.max(), obj - obj0, 1e-8);
        auto const obj2 = chol.objective(i+1);
        ct(HERE).within_log(obj, obj2, 1e-8);
        obj0 = obj;
    }
};

/******************************************************************************************/

// Test of inverse square root Chebyshev interpolation for matrices
UNIT_TEST("cs/isqrt") = [](Context ct, uint seed, uint n, uint c) {
    using T = real;
    la::arma_rng::set_seed(seed);
    Mat<T> A = la::randn(n, n);
    A = A * A.t();
    Col<real> eigs = la::eig_sym(A);
    auto const interp = InvSqrtInterpolation<T>(eigs.min()/1.1, eigs.max()*1.1, c);
    print(interp.a, interp.b);
    Mat<T> B = la::eye(n, n);
    print(interp(B, [&](auto const &B) {return A * B;}));
    print(la::pinv(la::real(la::sqrtmat(A))));
};

/******************************************************************************************/

// Test of inverse square root Chebyshev interpolation for matrices including stationary mode
UNIT_TEST("cs/isqrt/stationary") = [](Context ct, uint seed, uint n, uint c) {
    using T = real;
    la::arma_rng::set_seed(seed);
    Col<T> h = la::randu(n); h /= la::norm(h);
    Mat<T> A = random_laplacian<real>(n); A /= h * h.t();
    A += la::accu(A.diag()) / A.n_rows * h * h.t();
    Col<real> eigs = la::eig_sym(A);
    print(eigs(0), eigs.back(), eigs.t());
    auto const interp = InvSqrtInterpolation<T>(eigs(0)/1.1, eigs.back()*1.1, c);
    print(interp.a, interp.b);
    Mat<T> const P = la::eye(n, n);
    print(interp(P, [&](auto const &B) -> Mat<T> {return A * B;}));
    print(la::real(la::sqrtmat(la::pinv(A))));
    print(h.t() * A * h);
};

/******************************************************************************************/

UNIT_TEST("cs/isqrt/tolerance") = [](Context ct, real kappa, real epsilon) {
    print(chebyshev_isqrt_points(kappa, epsilon));
};

/******************************************************************************************/

// Test of rchol factorization
UNIT_TEST("cs/rchol") = [](Context ct, uint seed, uint n, uint k) {
    using T = real;
    la::arma_rng::set_seed(seed);
    Col<T> h = la::randu(n); h /= la::norm(h);
    auto L = SpMat<T>(random_laplacian<T>(n) / (h * h.t()));

    auto [perm, inverse_perm] = minimum_degree_ordering(directed_adjacency_list(L), 0); 
    L = permute_spmat(L, vmap<la::uvec>(perm));
    h = h(vmap<la::uvec>(inverse_perm));

    rchol_rng gen;
    gen.seed(seed);
    SpMat<real> const U = rchol(gen, L).t();
    // SpMat<real> const U(stationary_cholesky(Mat<T>(L)).t());
    Mat<real> const I = lower_solve(U, lower_solve(U, Mat<real>(L)).t().eval());
    print("preconditioned matrix", L.n_rows, la::eig_sym(Mat<real>(L)).t(), la::eig_sym(I).t());

    auto K = PreconditionedMatrix<T>(Executor(), L, U, h, {.tolerance=1e-8, .iters=1000});
    Col<T> b = la::randu(n);
    Col<T> exact = la::solve(Mat<T>(L), b);
    Col<T> iter = K.full(b);
    print((L * h).t(), (L * exact - b).t(), (L * iter - b).t());

    Mat<T> const P = la::eye(n, n) - h * h.t();
    b = P * b;
    print(P * I * b);
    print(P * lower_solve(U.eval(), Col<real>(L * upper_solve(U.t().eval(), b))));

    auto pre = PreconditionedSqrt<T>(Executor(), L, U, h, {.tolerance=1e-8, .iters=1000}, {.chebyshev_tolerance=1e-8, .power_iters=100, .multiplier=2});
    Mat<T> const B = la::eye(n, n);
    Mat<T> const C = pre.sqrt(B);

    print(pre.dense());
    print(C * C.t());
};

UNIT_TEST("cs/rchol-select") = [](Context ct, uint seed, uint n, uint k, uint z) {
    using T = real;
    la::arma_rng::set_seed(seed);
    Col<T> h = la::randu(n); h /= la::norm(h);
    auto L = SpMat<T>(random_laplacian<T>(n) / (h * h.t()));

    auto [perm, inverse_perm] = minimum_degree_ordering(directed_adjacency_list(L), 0); 
    L = permute_spmat(L, vmap<la::uvec>(perm));
    h = h(vmap<la::uvec>(inverse_perm));
    rchol_rng gen;
    gen.seed(seed);
    SpMat<T> const U = rchol(gen, L).t();

    Mat<T> K0 = la::inv_sympd(L + h * h.t()) - h * h.t(), C0 = stationary_cholesky(K0);
    auto Ke = FactorizedMatrix<T>(K0, C0);
    auto Kp = PreconditionedSqrt<T>(Executor(), L, U, h, {.tolerance=1e-8, .iters=1000}, {.chebyshev_tolerance=1e-8, .power_iters=100, .multiplier=2});
    
    auto chol = ExactLaplacianSelect<T>(Ke, h, k);
    auto rand = RandomizedLaplacianSelect<T>(h, k);

    for (auto i : range(k)) {
        auto const g = rand.randomized_scores(Kp, i, z).first;
        auto const [g0, d0] = chol.scores(i);
        auto const ge = chol.reference_scores(K0, i);
        ct(HERE).all(ct.within_log(1e-7), g0, ge);
        ct(HERE).all(ct.within_log(1e-1), g, g0);
        std::cout << (g / g0 - 1).t();
        
        auto const c = i + g.index_max();
        
        rand.pivot(i, c);
        Kp.pivot(i, c);
        chol.pivot(i, c);
        Ke.pivot(i, c);
        K0.swap_rows(i, c);
        K0.swap_cols(i, c);

        rand.augment(Kp, i);
        chol.augment(Ke, i);
    }
};

template <class T, class M>
T lanczos_method(M const &K, la::uword n, la::uword iters) {
    Mat<T> X(n, iters+1);
    X.col(0).randn();
    T o = 0;
    for (auto t : range(iters)) {
        X.col(t) /= la::norm(X.col(t));
        K(X.col(t+1), X.col(t));
        o = la::abs(la::eig_pair(X.head_cols(t+1).t() * X.cols(1, t+1), X.head_cols(t+1).t() * X.head_cols(t+1))).max();
    }
    return o;
}

UNIT_TEST("cs/power-method") = [](Context ct) {
    using T = real;
    uint n = 20;
    Mat<T> A(n, n, la::fill::randn);
    A = A * A.t();
    T const eig = la::eig_sym(A).max();
    print(eig, "\n");
    uint t1 = 0, t2 = 0;
    power_method<T>([&](auto &y, auto const &x) {
        y = A * x;
        print(t1++, eig - la::dot(y, x));
    }, n, 100);
    print();
    lanczos_method<T>([&](auto &&y, auto const &x) {
        y = A * x;
        print(t2++, eig - la::dot(y, x));
    }, n, 100);
};


/******************************************************************************************/

// Test of discrete weighted distribution
UNIT_TEST("cs/discrete_distribution") = [](Context ct) {
    Col<real> s = {0.1, 0.7, 0.9, 0.01, 0.1};
    Col<real> c = 0 * s;
    DefaultRNG rng;
    for (auto i : range(100000)) c(discrete_distribution(s(la::find(s)).eval())(rng)) += 1;
    c /= la::accu(c);
    print(c.t());
    print(s.t() / la::accu(s));
};

/******************************************************************************************/

// Benchmark codes for Laplacian systems
UNIT_TEST("cs/rchol/compare") = [](Context ct, uint seed, uint n, uint k, uint z) {
    using T = real;
    la::arma_rng::set_seed(seed);
    Col<T> h = la::randu(n); h /= la::norm(h);
    auto L = SpMat<T>(random_laplacian<T>(n) / (h * h.t()));

    auto [perm, inverse_perm] = minimum_degree_ordering(directed_adjacency_list(L), 0); 
    L = permute_spmat(L, vmap<la::uvec>(perm));
    h = h(vmap<la::uvec>(inverse_perm));
    rchol_rng gen;
    gen.seed(seed);
    SpMat<T> const U = rchol(gen, L).t();

    auto const K = PreconditionedSqrt<T>(Executor(), L, U, h, {.tolerance=1e-8, .iters=1000}, {.chebyshev_tolerance=1e-8, .power_iters=100, .multiplier=2});
    print(K.interpolation.a, K.interpolation.b);
    auto chol = RandomizedLaplacianSelect<T>(h, k);
    DefaultRNG rng;
    auto const [gains, choices] = matrix_free_selection(&rng, copy(chol), copy(K), Selector::direct, k, z);
    print(gains);
};

/******************************************************************************************/

UNIT_TEST("cs/index_max") = [](Context ct) {
    Col<real> x(10, la::fill::ones);
    ct(HERE).equal(x.index_max(), 0);
    x(0) = 0;
    ct(HERE).equal(x.index_max(), 1);
    x(0) = std::numeric_limits<real>::quiet_NaN();
    ct(HERE).equal(x.index_max(), 1);
};

/******************************************************************************************/

UNIT_TEST("cs/exact/compare") = [](Context ct, uint seed, uint n, uint k) {
    using T = real;
    la::arma_rng::set_seed(seed);
    Col<T> h = la::randu(n); h /= la::norm(h);
    auto const L = SpMat<T>(random_laplacian<T>(n) / (h * h.t()));
    Mat<T> const K0 = la::inv_sympd(L + h * h.t()) - h * h.t();

    auto K = DenseMatrix<T>(K0);
    auto chol = ExactLaplacianSelect<T>(K, h, k);
    DefaultRNG rng;
    auto const [gains, choices] = deterministic_selection(&rng, chol, K, Selector::direct, k);
};

/******************************************************************************************/

UNIT_TEST("cs/suitesparse/compare") = [](Context ct, bool order, std::string path, std::string file, uint k) {
    auto K = json::parse(std::ifstream(fmt::format("{}/{}.json", path, file))).get<SpMat<real>>();
    json entry;
    print(K.n_rows, K.n_nonzero, K.is_symmetric());
    entry["svds"] = Col<real>(la::sort(la::svds(K, k), "descend"));
    if (order) K = K.t() * K;
    else K = K * K.t();
    la::uvec nz = la::find(K.diag());
    K = K.cols(nz).t().eval().cols(nz);
    NSM_ASSERT(K.is_symmetric(), "K should be symmetric");
    print(order, std::size(nz));
    entry["trace"] = la::accu(K.diag());
    entry["eigenvalues"] = Col<real>(la::sort(la::eigs_sym(K, k), "descend"));
    DefaultRNG rng;
    for (auto s : {Selector::direct, Selector::uniform, Selector::greedy, Selector::random}) {
        print("    --", enum_to_string(s));
        auto &e = entry[enum_to_string(s)];
        auto Ks = SparseMatrix<real>(K);
        auto chol = ExactSelect<real>(Ks, 1e-8, k);
        std::tie(e["gains"], e["choices"]) = deterministic_selection(&rng, chol, Ks, s, k);
    }
    std::ofstream(fmt::format("{}/{}-out-{}.json", path, file, int(order))) << entry;
};

UNIT_TEST("cs/suitesparse/compare-all") = [](Context ct, std::string path, uint k) {
    for (auto s : {"bayer01", "bcsstk36", "c-67b", "c-69", "cbuckle", "crankseg_2", "ct20stif", "g7jac200sc", "venkat01", "bcircuit"}) {
        print(s);
        ct.call("proto/cs/suitesparse/compare", false, path, s, k);
        ct.call("proto/cs/suitesparse/compare", true, path, s, k);
    }
};

// Main benchmark code for CUR factorization examples
json suitesparse_svd(Executor const &exec, SpMat<real> const &A, uint k, uint z, uint reps) {
    print(A.n_rows, A.n_cols, A.n_nonzero, A.is_symmetric(), k, z, reps);

    Col<real> const svds = svd_simultaneous_iteration<real>(exec, A, [](auto t, auto const &b, auto const &e) {
        print("-- calculating svds", t, abs(e/b - 1));
        return abs(e/b - 1) < 1e-10;
    }, {.iters=10000, .k=k}).first.head(k);
    // Col<real> const svds = la::sort(la::svds(A, k), "descend");
    print("-- calculated svds", json(svds));
    k = min(k, std::size(svds));

    json entry{
        {"svds", svds}, {"k", k}, {"reps", reps}, {"trace", la::accu(la::square(A))},
        {"rows", A.n_rows}, {"cols", A.n_cols}, {"nnz", A.n_nonzero}, {"symmetric", A.is_symmetric()}
    };

    for (auto s : {Selector::direct, Selector::uniform, Selector::greedy, Selector::random}) 
        for (auto t : range(s == Selector::random || s == Selector::uniform || z ? reps : 1))
            entry["results"].emplace_back()["method"] = enum_to_string(s);
    print("-- evaluating", std::size(entry.at("results")));

    thread_local DefaultRNG rng;
    exec.map(entry.at("results"), [&](json &e) {
        print("-- running", e.at("method"));
        auto const s = enum_from_string<Selector>(e.at("method").get<std::string>());
        auto const res = exec.map(range(2), [=, &A](bool b) -> std::tuple<RandomizedSelect<real>, Col<real>, Col<real>, la::uvec> {
            auto Ks = SparseSqrtMatrix<real>(b ? A.t().eval() : A);
            if (z) {
                auto chol = RandomizedSelect<real>(Ks.n(), k);
                auto [gains, choices] = matrix_free_selection(&rng, chol, Ks, s, k, z);
                return move_as_tuple(chol, gains.row(0).t(), gains.row(1).t(), choices);
            } else {
                auto chol = ExactSelect<real>(Ks, 1e-8, k);
                auto [gains, choices] = deterministic_selection(&rng, chol, Ks, s, k);
                return move_as_tuple(chol, gains, gains, choices);
            }
        });
        la::uvec const rows = fourth_of(res[0]), cols = fourth_of(res[1]);
        e.update({{"rows", rows}, {"row_gains", second_of(res[0])}, {"randomized_row_gains", third_of(res[0])},
                  {"cols", cols}, {"col_gains", second_of(res[1])}, {"randomized_col_gains", third_of(res[1])}});
        SpMat<real> const Arc = A.cols(first_of(res[1]).index.head(k)).t().eval().cols(first_of(res[0]).index).t();
        auto const &Sr = first_of(res[0]).S;
        auto const &Uc = first_of(res[1]).U;
        print("-- benchmarking", e.at("method"));
        e["values"] = vmap(range(1, k+1), [&](auto i) {
            return la::accu(la::square(Sr.head_cols(i).t() * Arc.head_cols(i) * Uc(span(0, i), span(0, i))));
        });

        print("-- done", e.at("method"), e.at("values"));
    });
    return entry;
}

UNIT_TEST("cs/suitesparse/check-svd") = [](Context ct) {
    SpMat<real> A(Mat<real>(20, 6, la::fill::randn));
    print(suitesparse_svd(Executor(0), A, 5, 0, 2));
};

UNIT_TEST("cs/suitesparse/compare-svd") = [](Context ct, std::string path, uint threads, uint k, uint z, uint reps) {
    json out;
    for (std::string s : {"bayer01", "bcsstk36", "c-67b", "c-69", "cbuckle", "crankseg_2", "ct20stif", "g7jac200sc", "venkat01"}) {
        print(s);
        auto const A = json::parse(std::ifstream(fmt::format("{}/{}.json", path, s))).get<SpMat<real>>();
        out.emplace_back(suitesparse_svd(Executor(threads), A, s == "cbuckle" ? 5000 : k, z, reps))["file"] = s;
        std::ofstream("suitesparse-results.json") << out;
    }
};

/******************************************************************************************/

UNIT_TEST("cs/subspace-svd") = [](Context ct) {
    Mat<real> A(10, 5, la::fill::randn), Us, Vs;
    Col<real> s;
    uint k = 3;

    NSM_ASSERT(la::svd(Us, s, Vs, A), "SVD failed");
    Executor exec(0);
    auto [s2, W] = svd_simultaneous_iteration<real>(exec, A, AlwaysFalse(), {.iters=100, .k=k});

    ct(HERE).all(ct.within_log(1e-4), s.head(k), s2.head(k));
    // print(la::abs(la::sum(W % (A * V), 0)));
};

/******************************************************************************************/

UNIT_TEST("cs/toy-examples/compare") = [](Context ct, std::string file, uint threads, uint k, uint m, real threshold) {
    Mat<real> const K = [&]{
        Mat<real> K;
        NSM_ASSERT(K.load(file + ".h5"), "Failed to load h5 file");
        print(K.n_rows, K.is_symmetric());
        return K;
    }();
    json entry;
    entry["trace"] = la::accu(K.diag());
    entry["eigenvalues"] = Col<real>(la::sort(la::eig_sym(K), "descend"));
    for (auto s : {Selector::direct, Selector::uniform, Selector::greedy, Selector::random}) {
        for (auto t : range(s == Selector::direct || s == Selector::greedy ? 1 : m))
            entry["results"].emplace_back()["method"] = enum_to_string(s);
    }
    print("-- starting evaluation");
    thread_local DefaultRNG rng;
    Executor(threads).map(entry["results"], [&](json &e) {
        auto const s = enum_from_string<Selector>(e.at("method").get<std::string>());
        print("-- evaluating", e.at("method"));
        auto Kd = DenseMatrix<real>(K);
        auto chol = ExactSelect<real>(Kd, threshold, k);
        std::tie(e["gains"], e["choices"]) = deterministic_selection(&rng, chol, Kd, s, k);
    });

    std::ofstream(file + "-out.json") << entry;
};

/******************************************************************************************/

UNIT_TEST("cs/toy-examples/check") = [](Context ct, std::string file, uint k) {
    Mat<real> K;
    NSM_ASSERT(K.load(file + ".h5"), "Failed to load h5 file");
    print(K.n_rows, K.is_symmetric());
    auto Kd = DenseMatrix<real>(K);
    auto chol = ExactSelect<real>(Kd, 1e-8, k);
    DefaultRNG rng;
    auto const [gains, choices] = deterministic_selection(&rng, chol, Kd, Selector::uniform, k);
    print(gains);
};

/******************************************************************************************/

template <class T>
SpMat<T> pathological_block_matrix(la::uword nd, la::uword nb, la::uword b, T df, T bf) {
    la::uword const nz = nd + nb * sq(b);
    la::umat locs(2, nz);
    la::uword l = 0, k = 0;
    for (auto i : range(nd)) {
        locs(1, l) = locs(0, l) = i;
        ++k;
        ++l;
    }
    for (auto i : range(nb)) {
        for (auto bi : range(b)) for (auto bj : range(b)) {
            locs(0, l) = k + bi;
            locs(1, l) = k + bj;
            ++l;
        }
        k += b;
    }
    NSM_REQUIRE(l, ==, nz, "incorrect number of entries");
    NSM_REQUIRE(k, ==, nd + nb * b, "incorrect block dimensions");
    Col<T> values(nz);
    values.fill(bf);
    values.head(nd).fill(df);
    return SpMat<T>(locs, values);
}

UNIT_TEST("cs/pathological/compare") = [](Context ct, std::string file, uint reps, uint k, uint m, uint nb, uint b, real bf) {
    SpMat<real> K = pathological_block_matrix<real>(m, nb, b, 1, bf);
    if (K.n_rows < 50) print(Mat<real>(K));
    json entry;
    entry["trace"] = la::accu(K.diag());
    entry["eigenvalues"] = Col<real>(la::sort(la::eigs_sym(K, k), "descend"));
    DefaultRNG rng;
    for (auto rep : range(reps)) for (auto s : {Selector::direct, Selector::uniform, Selector::greedy, Selector::random}) {
        auto &e = entry["results"].emplace_back();
        e["method"] = enum_to_string(s);
        auto Kd = SparseMatrix<real>(K);
        auto chol = ExactSelect<real>(Kd, 1e-8, k);
        std::tie(e["gains"], e["choices"]) = deterministic_selection(&rng, chol, Kd, s, k);
    }
    std::ofstream(file) << entry;
};

/******************************************************************************************/

template <class T>
auto pathological_laplacian_0(la::uword n, T e) {
    Mat<T> L(n, n, la::fill::zeros);
    Col<T> h(n, la::fill::ones);
    h(0) = e;
    h /= la::norm(h);
    L.row(0).ones();
    L.col(0).ones();
    L = la::diagmat(la::sum(L)) - L;
    L /= h * h.t();
    return std::make_pair(std::move(L), std::move(h));
}

template <class T>
auto pathological_laplacian_1(la::uword n, T epsilon, T delta) {
    Mat<T> L(n, n);
    L.fill(-1);
    L.row(0).zeros();
    L.col(0).zeros();
    uint m = n - 2;
    L(0, 0) = (m / (1 + epsilon) - delta) * epsilon;
    for (auto i : range(1, n)) L(i, i) = m + epsilon;
    Col<T> h(n, la::fill::ones);
    h /= la::norm(h);
    return std::make_pair(L, h);
}

UNIT_TEST("cs/pathological/laplacian") = [](Context ct, std::string file, uint reps, uint n, uint k, real e) {
    auto const [L, h] = pathological_laplacian_0<real>(n, e);
    Mat<real> const K0 = la::inv_sympd(L + h * h.t()) - h * h.t();
    
    json entry;
    entry["trace"] = la::accu(K0.diag());
    entry["eigenvalues"] = Col<real>(la::sort(la::eig_sym(K0), "descend"));
    DefaultRNG rng;
    for (auto rep : range(reps)) for (auto s : {Selector::direct, Selector::uniform, Selector::greedy, Selector::random}) {
        auto &e = entry["results"].emplace_back();
        e["method"] = enum_to_string(s);
        print(rep, e.at("method"));
        auto K = DenseMatrix<real>(K0);
        auto chol = ExactLaplacianSelect<real>(K, h, k);
        std::tie(e["gains"], e["choices"]) = deterministic_selection(&rng, chol, K, s, k);
    }
    std::ofstream(file) << entry;
};

/******************************************************************************************/

UNIT_TEST("cs/parallel-subspace") = [](Context ct) {
    uint n = 100, k = 10;
    Mat<real> K = la::randn(n, n);
    K = K * K.t();
    Executor ex(10);
    Col<real> e = psd_simultaneous_iteration<real>(ex, [&](auto &&Y, auto const &X) {Y = K.t() * X;}, [](auto t, auto const &b, auto const &e) {
        print(t, abs(e/b - 1));
        return abs(e/b - 1) < 1e-10;
    }, {.iters=100, .n=n, .k=k}).first;
    ct(HERE).all(ct.within_log(1e-6), e, la::reverse(la::eig_sym(K).tail(k)).eval());
    // print(la::eigs_sym(SpMat<real>(K), k).t());
};

/******************************************************************************************/

UNIT_TEST("cs/power-iterations") = [](Context ct) {
    print(necessary_power_iterations(1e6, 1e-6, 0.1));
    print(necessary_power_iterations(1e6, 1e-10, 0.1));
};

/******************************************************************************************/

template <class T>
Mat<T> pathological_kernel(uint nd, uint nb, uint sb, T d, T b) {
    uint N = nd + sb * nb;
    Mat<T> M(N, N, la::fill::zeros);
    M.diag().fill(d);
    for (auto i : range(nb)) M(span(i * sb, (i+1) * sb), span(i * sb, (i+1) * sb)).fill(b);
    return M;
}

UNIT_TEST("cs/pathological") = [](Context ct) {
    print(pathological_kernel<real>(10, 4, 2, 1, 2));
};

/******************************************************************************************/

}