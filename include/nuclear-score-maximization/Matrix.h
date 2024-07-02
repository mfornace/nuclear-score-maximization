#pragma once
/**
    Matrix.h: Typedefs and wrapping facilities for armadillo matrix library
*/

#include "Common.h"
#include <armadillo>

/******************************************************************************************/

// Wrapped armadillo namespace and a few helper types and definitions

namespace nsm::la {
using namespace arma;

NSM_DETECT(has_eval, decltype(std::declval<T>().eval()));

/// Evaluate expression template if possible
template <class T, NSM_IF(traits::has_eval<T &&>)>
decltype(auto) eval(T &&t) {return std::forward<T>(t).eval();}

template <class T, NSM_IF(!traits::has_eval<T &&>)>
decltype(auto) eval(T &&t) {return std::forward<T>(t);}

/// Type from evaluating an expression template
template <class T>
using eval_result = no_qual<decltype(eval(std::declval<if_t<std::is_array<T>::value, std::decay_t<T>, T>>()))>;

template <class T>
static constexpr bool is_sparse = arma::is_arma_sparse_type<eval_result<T>>::value;

static_assert(is_sparse<SpMat<real>>);
static_assert(!is_sparse<Mat<real>>);

template <class T>
Mat<T> random_spd(uword n) {
    Mat<T> A(n, n, fill::randn);
    return A * A.t();
}

}

/******************************************************************************************/

// Exported types from armadillo
namespace nsm {

using arma::Col;
using arma::Mat;
using arma::SpMat;

// Half-open span type to use with armadillo
struct span {
    la::uword b, e;
    span(la::uword b, la::uword e) : b(b), e(e) {}

    operator la::span() const {return la::span(b, e-1);}
};

using cspan = span const;

// Overload len() so it works with armadillo types
template <class T>
struct Len<Col<T>> {constexpr auto operator()(Col<T> const &t) const {return t.n_rows;}};

}

/******************************************************************************************/