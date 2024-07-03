#pragma once
/**
Selection.h: Top level routines for column selection
*/

#include "Engines.h"

namespace nsm {

/******************************************************************************************/

enum class Selector {
    nuclear_max, // Select columns via greedy nuclear norm maximization
    diagonal_max, // Select columns via diagonal / determinant maximization
    diagonal_sample, // Select columns via RPCholesky / randomly pivoted QR
    uniform_sample, // Select columns via uniform sampling without replacement
    in_order // Select columns in order: 0, 1, 2, ...
};

// Matrix-free selection out of any matrix, Laplacian inverse or otherwise
// - rng: random number generator (like std::mt19337). Must not be null if a random selection method is used
// - e: engine. One of the "Select" types in Engines.h which updates the intermediate information during selection
// - K: operator that column selection is being performed on. Should be one of the operator types in Engines.h
// - s: Selector enum to choose which method is used
// - k: number of columns to choose
// - z: number of random vectors to use for matrix-free approach
// - first: optional index that should be >= 0 if first index is manually chosen (uncommon)
template <class RNG, class E, class M>
std::pair<Mat<real>, la::uvec> matrix_free_selection(RNG *rng, E &&e, M &&K, Selector s, uint k, uint z, int first=-1) {
    Mat<real> gains(2, k, la::fill::zeros);
    real objective = 0;
    for (auto i : range(k)) {
        auto const [g, d] = e.randomized_scores(K, i, z);
        la::uvec const f = la::find_finite(g);
        if (f.empty()) break;
        la::uword c;
        switch (s) {
            case Selector::nuclear_max: {c = g(f).index_max(); break;}
            case Selector::diagonal_max: {c = d(f).index_max(); break;}
            case Selector::diagonal_sample: {c = discrete_distribution(d(f).eval())(*rng); break;}
            case Selector::uniform_sample: {c = random_range(*rng, 0, std::size(f)); break;}
            case Selector::in_order: {c = 0; break;}
        }
        c = f(c);
        if (first >= 0 && i == 0) c = first;
        gains(1, i) = g(c);
        e.pivot(i, i+c);
        K.pivot(i, i+c);
        e.augment(K, i);
        real const obj = e.objective(i+1);
        gains(0, i) = obj - std::exchange(objective, obj);
    }
    return {std::move(gains), e.index.head(k)};
}

/******************************************************************************************/

// Deterministic selection out of any matrix, Laplacian inverse or otherwise
// - rng: random number generator (like std::mt19337). Must not be null if a random selection method is used
// - e: engine. One of the "Select" types in Engines.h which updates the intermediate information during selection
// - K: operator that column selection is being performed on. Should be one of the operator types in Engines.h
// - s: Selector enum to choose which method is used
// - k: number of columns to choose
template <class RNG, class E, class M>
std::pair<Col<real>, la::uvec> deterministic_selection(RNG *rng, E &&e, M &&K, Selector s, uint k) {
    Col<real> gains(k, la::fill::zeros);
    la::uvec idx(k, la::fill::none);
    la::uword i = 0;
    for (auto const t : range(k)) {
        auto const [g, d] = e.scores(i);
        la::uvec const f = la::find_finite(g);
        if (f.empty()) break;
        la::uword c;
        switch (s) {
            case Selector::nuclear_max: {c = g(f).index_max(); break;}
            case Selector::diagonal_max: {c = d(f).index_max(); break;}
            case Selector::diagonal_sample: {c = discrete_distribution(d(f).eval())(*rng); break;}
            case Selector::uniform_sample: {c = random_range(*rng, 0, std::size(f)); break;}
            case Selector::in_order: {c = 0; break;}
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

/******************************************************************************************/

// Helper functions, generally for testing/prototyping

template <class Matrix>
auto condition_number(Matrix const &M, uint b=0, uint e=0) {
    auto v = la::eig_sym(M).eval();
    sort(v, [](auto a, auto b) {return std::abs(a) < std::abs(b);});
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

template <class T, class M>
Mat<T> stationary_inverse(M const &L, Col<T> const &h) {
    T const tr = la::accu(L.diag()) / std::size(h);
    return la::inv_sympd(L + tr * h * h.t()) - 1 / tr * h * h.t();
}

/******************************************************************************************/

}
