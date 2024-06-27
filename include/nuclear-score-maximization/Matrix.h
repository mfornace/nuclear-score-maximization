#pragma once
#include <armadillo>
#include <ranges>

#define NSM_IF(...) typename std::enable_if<__VA_ARGS__, bool>::type = 0

#define NSM_ASSERT(cond, ...) {if (!cond) throw ::nsm::Error::from_location(__FILE__, __LINE__, __VA_ARGS__);}
#define NSM_REQUIRE(x, op, y, ...) NSM_ASSERT(x op y, x, y, __VA_ARGS__)

namespace nsm {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;

    template <class ...Ts>
    static Error from_location(std::string_view file, int line, Ts &&...ts) {
        std::stringstream ss;
        ss << file << ":" << line << " ";
        ((ss << ts), ...);
        return Error(ss.str());
    }
};

namespace la = arma;
using arma::Mat;
using arma::SpMat;
using arma::Col;
using real = double;

template <class T>
using vec = std::vector<T>;

struct Identity {
    template <class T>
    T &&operator()(T &&t) const {return static_cast<T &&>(t);}
};

template <class O, class V, class F=Identity>
O vmap(V &&v, F &&f={}) {return std::forward<V>(v) |  std::views::transform(f) | std::ranges::to<O>;}

template <template <class...> class O, class V, class F=Identity>
auto vmap(V &&v, F &&f={}) {return vmap<O<decltype(f(*v.begin()))>>(std::forward<V>(v), std::forward<F>(f));}

template <class V, class F=Identity>
auto vmap(V &&v, F &&f={}) {return vmap<std::vector>(std::forward<V>(v), std::forward<F>(f));}

template <class T>
auto range(T t) {return std::ranges::iota(t);}

template <class B, class E>
auto range(B b, E e) {return std::ranges::iota(b, e);}

template <class P>
auto ptr_view(P p, std::size_t n) {return std::ranges::subrange(p, p+n);}


/// Detect if a given type is well-formed
#define NSM_DETECT(NAME, expr) \
namespace detail { \
    template <class T, class=void> struct NAME##_t : std::false_type {}; \
    template <class T> struct NAME##_t<T, std::void_t<expr>> : std::true_type {}; \
} \
namespace traits {template <class T> static constexpr bool NAME = detail::NAME##_t<T>::value;}

template <class T>
using no_qual = std::remove_cv_t<std::remove_reference_t<T>>;

NSM_DETECT(has_eval, decltype(std::declval<T>().eval()));

/// Evaluate expression template if possible
template <class T, NSM_IF(traits::has_eval<T &&>)>
decltype(auto) eval(T &&t) {return std::forward<T>(t).eval();}

template <class T, NSM_IF(!traits::has_eval<T &&>)>
decltype(auto) eval(T &&t) {return std::forward<T>(t);}

template <bool B, class T, class U>
using if_t = std::conditional_t<B, T, U>;

/// Type from evaluating an expression template
template <class T>
using eval_result = no_qual<decltype(eval(std::declval<if_t<std::is_array<T>::value, std::decay_t<T>, T>>()))>;




template <class T>
static constexpr bool is_sparse = arma::is_arma_sparse_type<eval_result<T>>::value;

}