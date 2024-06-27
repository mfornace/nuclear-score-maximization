#include <random>
#define ARMA_USE_HDF5 1
#include <nuclear-score-maximization/Preconditioning.h>
#include <nuclear-score-maximization/Selection.h>
#include <nuclear-score-maximization/Cholesky.h>
#include "Test.h"

namespace nsm {

/******************************************************************************************/

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

RELEASE_TEST("cs/exact") = [](Context ct) {
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

RELEASE_TEST("cs/random") = [](Context ct) {
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

RELEASE_TEST("cs/laplacian/exact") = [](Context ct) {
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

PROTOTYPE("cs/isqrt") = [](Context ct, uint seed, uint n, uint c) {
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

PROTOTYPE("cs/isqrt/stationary") = [](Context ct, uint seed, uint n, uint c) {
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

PROTOTYPE("cs/isqrt/tolerance") = [](Context ct, real kappa, real epsilon) {
    print(chebyshev_isqrt_points(kappa, epsilon));
};

/******************************************************************************************/

PROTOTYPE("cs/rchol") = [](Context ct, uint seed, uint n, uint k) {
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

    auto K = PreconditionedMatrix<T>(Local(), L, U, h, {.tolerance=1e-8, .iters=1000});
    Col<T> b = la::randu(n);
    Col<T> exact = la::solve(Mat<T>(L), b);
    Col<T> iter = K.full(b);
    print((L * h).t(), (L * exact - b).t(), (L * iter - b).t());

    Mat<T> const P = la::eye(n, n) - h * h.t();
    b = P * b;
    print(P * I * b);
    print(P * lower_solve(U.eval(), L * upper_solve(U.t().eval(), b)));

    auto pre = PreconditionedSqrt<T>(Local(), L, U, h, {.tolerance=1e-8, .iters=1000}, {.chebyshev_tolerance=1e-8, .power_iters=100, .multiplier=2});
    Mat<T> const B = la::eye(n, n);
    Mat<T> const C = pre.sqrt(B);

    print(pre.dense());
    print(C * C.t());
};

PROTOTYPE("cs/rchol-select") = [](Context ct, uint seed, uint n, uint k, uint z) {
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
    auto Kp = PreconditionedSqrt<T>(Local(), L, U, h, {.tolerance=1e-8, .iters=1000}, {.chebyshev_tolerance=1e-8, .power_iters=100, .multiplier=2});
    
    auto chol = ExactLaplacianSelect<T>(Ke, h, k);
    auto rand = RandomizedLaplacianSelect<T>(h, k);

    for (auto i : range(k)) {
        auto const g = rand.randomized_scores(Kp, i, z).first;
        auto const [g0, d0] = chol.scores(i);
        auto const ge = chol.reference_scores(K0, i);
        ct(HERE).all(ct.within_log(1e-7), g0, ge);
        ct(HERE).all(ct.within_log(1e-1), g, g0);
        io::out() << (g / g0 - 1).t();
        
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

PROTOTYPE("cs/power-method") = [](Context ct) {
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

PROTOTYPE("cs/discrete_distribution") = [](Context ct) {
    Col<real> s = {0.1, 0.7, 0.9, 0.01, 0.1};
    Col<real> c = 0 * s;
    for (auto i : range(100000)) c(discrete_distribution(s(la::find(s)).eval())(StaticRNG)) += 1;
    c /= la::accu(c);
    print(c.t());
    print(s.t() / la::accu(s));
};


/******************************************************************************************/

PROTOTYPE("cs/rchol/compare") = [](Context ct, uint seed, uint n, uint k, uint z) {
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

    auto const K = PreconditionedSqrt<T>(Local(), L, U, h, {.tolerance=1e-8, .iters=1000}, {.chebyshev_tolerance=1e-8, .power_iters=100, .multiplier=2});
    print(K.interpolation.a, K.interpolation.b);
    auto chol = RandomizedLaplacianSelect<T>(h, k);
    auto const [gains, choices] = randomized_select(copy(chol), copy(K), Selector::direct, k, z);
    print(gains);
};

/******************************************************************************************/

    // print("factorizing");
    // la::spsolve_factoriser SF;
    // la::superlu_opts ops;
    // ops.permutation=la::superlu_opts::NATURAL;
    // ops.symmetric = true;
    // bool status = SF.factorise(L, ops);
    // print(status);



/******************************************************************************************/

PROTOTYPE("cs/index_max") = [](Context ct) {
    Col<real> x(10, la::fill::ones);
    ct(HERE).equal(x.index_max(), 0);
    x(0) = 0;
    ct(HERE).equal(x.index_max(), 1);
    x(0) = std::numeric_limits<real>::quiet_NaN();
    ct(HERE).equal(x.index_max(), 1);
};

/******************************************************************************************/

PROTOTYPE("cs/exact/compare") = [](Context ct, uint seed, uint n, uint k) {
    using T = real;
    la::arma_rng::set_seed(seed);
    Col<T> h = la::randu(n); h /= la::norm(h);
    auto const L = SpMat<T>(random_laplacian<T>(n) / (h * h.t()));
    Mat<T> const K0 = la::inv_sympd(L + h * h.t()) - h * h.t();

    auto K = DenseMatrix<T>(K0);
    auto chol = ExactLaplacianSelect<T>(K, h, k);
    auto const [gains, choices] = exact_select(chol, K, Selector::direct, k);
};

/******************************************************************************************/

PROTOTYPE("cs/suitesparse/compare") = [](Context ct, bool order, string path, string file, uint k) {
    auto K = json::parse(std::ifstream(fmt::format("{}/{}.json", path, file))).get<SpMat<real>>();
    json entry;
    print(la::shape(K), K.n_nonzero, K.is_symmetric());
    entry["svds"] = Col<real>(la::sort(la::svds(K, k), "descend"));
    if (order) K = K.t() * K;
    else K = K * K.t();
    la::uvec nz = la::find(K.diag());
    K = K.cols(nz).t().eval().cols(nz);
    NSM_ASSERT(K.is_symmetric());
    print(order, len(nz));
    entry["trace"] = la::accu(K.diag());
    entry["eigenvalues"] = Col<real>(la::sort(la::eigs_sym(K, k), "descend"));
    for (auto s : {Selector::direct, Selector::uniform, Selector::greedy, Selector::random}) {
        print("    --", enum_to_string(s));
        auto &e = entry[enum_to_string(s)];
        auto Ks = SparseMatrix<real>(K);
        auto chol = ExactSelect<real>(Ks, 1e-8, k);
        std::tie(e["gains"], e["choices"]) = exact_select(chol, Ks, s, k);
    }
    std::ofstream(fmt::format("{}/{}-out-{}.json", path, file, int(order))) << entry;
};

PROTOTYPE("cs/suitesparse/compare-all") = [](Context ct, string path, uint k) {
    for (auto s : {"bayer01", "bcsstk36", "c-67b", "c-69", "cbuckle", "crankseg_2", "ct20stif", "g7jac200sc", "venkat01", "bcircuit"}) {
        print(s);
        ct.call("proto/cs/suitesparse/compare", false, path, s, k);
        ct.call("proto/cs/suitesparse/compare", true, path, s, k);
    }
};

json suitesparse_svd(Local const &exec, SpMat<real> const &A, uint k, uint z, uint reps) {
    print(la::shape(A), A.n_nonzero, A.is_symmetric(), k, z, reps);

    Col<real> const svds = svd_simultaneous_iteration<real>(exec, A, [](auto t, auto const &b, auto const &e) {
        print("-- calculating svds", t, abs(e/b - 1));
        return abs(e/b - 1) < 1e-10;
    }, {.iters=10000, .k=k}).first.head(k);
    // Col<real> const svds = la::sort(la::svds(A, k), "descend");
    print("-- calculated svds", json(svds));
    k = min(k, len(svds));

    json entry{
        {"svds", svds}, {"k", k}, {"reps", reps}, {"trace", la::accu(la::square(A))},
        {"rows", A.n_rows}, {"cols", A.n_cols}, {"nnz", A.n_nonzero}, {"symmetric", A.is_symmetric()}
    };

    for (auto s : {Selector::direct, Selector::uniform, Selector::greedy, Selector::random}) 
        for (auto t : range(s == Selector::random || s == Selector::uniform || z ? reps : 1))
            entry["results"].emplace_back()["method"] = enum_to_string(s);
    print("-- evaluating", len(entry.at("results")));

    exec.map(entry.at("results"), [&](json &e) {
        print("-- running", e.at("method"));
        auto const s = enum_from_string<Selector>(e.at("method").get<string>());
        auto const res = exec.map(range(2), [=, &A](bool b) -> std::tuple<RandomizedSelect<real>, Col<real>, Col<real>, la::uvec> {
            auto Ks = SparseSqrtMatrix<real>(b ? A.t().eval() : A);
            if (z) {
                auto chol = RandomizedSelect<real>(Ks.n(), k);
                auto [gains, choices] = randomized_select(chol, Ks, s, k, z);
                return move_as_tuple(chol, gains.row(0).t(), gains.row(1).t(), choices);
            } else {
                auto chol = ExactSelect<real>(Ks, 1e-8, k);
                auto [gains, choices] = exact_select(chol, Ks, s, k);
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

PROTOTYPE("cs/suitesparse/check-svd") = [](Context ct) {
    SpMat<real> A(Mat<real>(20, 6, la::fill::randn));
    print(suitesparse_svd(Local(0), A, 5, 0, 2));
};

PROTOTYPE("cs/suitesparse/compare-svd") = [](Context ct, string path, uint threads, uint k, uint z, uint reps) {
    json out;
    for (string s : {"bayer01", "bcsstk36", "c-67b", "c-69", "cbuckle", "crankseg_2", "ct20stif", "g7jac200sc", "venkat01", "bcircuit"}) {
        print(s);
        auto const A = json::parse(std::ifstream(fmt::format("{}/{}.json", path, s))).get<SpMat<real>>();
        out.emplace_back(suitesparse_svd(Local(threads), A, s == "cbuckle" ? 5000 : k, z, reps))["file"] = s;
        std::ofstream("suitesparse-results.json") << out;
    }
};

/******************************************************************************************/

RELEASE_TEST("cs/subspace-svd") = [](Context ct) {
    Mat<real> A(10, 5, la::fill::randn), Us, Vs;
    Col<real> s;
    uint k = 3;

    NSM_ASSERT(la::svd(Us, s, Vs, A));
    Local exec(0);
    auto [s2, W] = svd_simultaneous_iteration<real>(exec, A, AlwaysFalse(), {.iters=100, .k=k});

    ct(HERE).all(ct.within_log(1e-4), s.head(k), s2.head(k));
    // print(la::abs(la::sum(W % (A * V), 0)));
};

/******************************************************************************************/

PROTOTYPE("cs/toy-examples/compare") = [](Context ct, string file, uint threads, uint k, uint m, real threshold) {
    Mat<real> const K = [&]{
        Mat<real> K;
        NSM_ASSERT(K.load(file + ".h5"));
        print(la::shape(K), K.is_symmetric());
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
    Local(threads).map(entry["results"], [&](json &e) {
        auto const s = enum_from_string<Selector>(e.at("method").get<string>());
        print("-- evaluating", e.at("method"));
        auto Kd = DenseMatrix<real>(K);
        auto chol = ExactSelect<real>(Kd, threshold, k);
        std::tie(e["gains"], e["choices"]) = exact_select(chol, Kd, s, k);
    });

    std::ofstream(file + "-out.json") << entry;
};

/******************************************************************************************/

PROTOTYPE("cs/toy-examples/check") = [](Context ct, string file, uint k) {
    Mat<real> K;
    NSM_ASSERT(K.load(file + ".h5"));
    print(la::shape(K), K.is_symmetric());
    auto Kd = DenseMatrix<real>(K);
    auto chol = ExactSelect<real>(Kd, 1e-8, k);
    auto const [gains, choices] = exact_select(chol, Kd, Selector::uniform, k);
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
    NSM_REQUIRE(l, ==, nz);
    NSM_REQUIRE(k, ==, nd + nb * b);
    Col<T> values(nz);
    values.fill(bf);
    values.head(nd).fill(df);
    return SpMat<T>(locs, values);
}

PROTOTYPE("cs/pathological/compare") = [](Context ct, string file, uint reps, uint k, uint m, uint nb, uint b, real bf) {
    SpMat<real> K = pathological_block_matrix<real>(m, nb, b, 1, bf);
    if (K.n_rows < 50) print(Mat<real>(K));
    json entry;
    entry["trace"] = la::accu(K.diag());
    entry["eigenvalues"] = Col<real>(la::sort(la::eigs_sym(K, k), "descend"));
    for (auto rep : range(reps)) for (auto s : {Selector::direct, Selector::uniform, Selector::greedy, Selector::random}) {
        auto &e = entry["results"].emplace_back();
        e["method"] = enum_to_string(s) | echo;
        auto Kd = SparseMatrix<real>(K);
        auto chol = ExactSelect<real>(Kd, 1e-8, k);
        std::tie(e["gains"], e["choices"]) = exact_select(chol, Kd, s, k);
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
// 1000, 100, 20, 0.9999
PROTOTYPE("cs/pathological/laplacian") = [](Context ct, string file, uint reps, uint n, uint k, real e) {
    auto const [L, h] = pathological_laplacian_0<real>(n, e);
    Mat<real> const K0 = la::inv_sympd(L + h * h.t()) - h * h.t();
    if (L.n_rows < 50) print_lns(h.t(), (K0.diag() / la::square(h)).t(), L, K0);
    
    json entry;
    entry["trace"] = la::accu(K0.diag());
    entry["eigenvalues"] = Col<real>(la::sort(la::eig_sym(K0), "descend"));

    for (auto rep : range(reps)) for (auto s : {Selector::direct, Selector::uniform, Selector::greedy, Selector::random}) {
        auto &e = entry["results"].emplace_back();
        e["method"] = enum_to_string(s);
        print(rep, e.at("method"));
        auto K = DenseMatrix<real>(K0);
        auto chol = ExactLaplacianSelect<real>(K, h, k);
        std::tie(e["gains"], e["choices"]) = exact_select(chol, K, s, k);
    }
    std::ofstream(file) << entry;
};

/******************************************************************************************/

RELEASE_TEST("cs/parallel-subspace") = [](Context ct) {
    uint n = 100, k = 10;
    Mat<real> K = la::randn(n, n);
    K = K * K.t();
    Local ex(10);
    Col<real> e = psd_simultaneous_iteration<real>(ex, [&](auto &&Y, auto const &X) {Y = K.t() * X;}, [](auto t, auto const &b, auto const &e) {
        print(t, abs(e/b - 1));
        return abs(e/b - 1) < 1e-10;
    }, {.iters=100, .n=n, .k=k}).first;
    ct(HERE).all(ct.within_log(1e-6), e, la::reverse(la::eig_sym(K).tail(k)).eval());
    // print(la::eigs_sym(SpMat<real>(K), k).t());
};

/******************************************************************************************/

PROTOTYPE("cs/power-iterations") = [](Context ct) {
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

PROTOTYPE("cs/pathological") = [](Context ct) {
    print(pathological_kernel<real>(10, 4, 2, 1, 2));
};

/******************************************************************************************/

}