#pragma once
/**
Common.h: common typedefs, type traits, and routines
 */

#include <__atomic/aliases.h>
#define BOOST_PP_VARIADICS 1
#include <iterator>
#include <sstream>
#include <random>
#include <iostream>
#include <nlohmann/json.hpp>

#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>

/******************************************************************************************/

// Conditional compilation macro

#define NSM_IF(...) typename std::enable_if<__VA_ARGS__, bool>::type = 0

/******************************************************************************************/

// Assertion macros

#define NSM_ITEM(R, DATA, N, X) BOOST_PP_COMMA() BOOST_PP_IF(N, ::nsm::json_keypair(BOOST_PP_STRINGIZE(X), (X)), X)

#define NSM_ASSERT(cond, ...) {if (!(cond)) throw ::nsm::Error::from_location(__FILE__, __LINE__ \
    BOOST_PP_SEQ_FOR_EACH_I(NSM_ITEM, , BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) );}

#define NSM_REQUIRE(x, op, y, ...) NSM_ASSERT((x op y), __VA_ARGS__, x, y)

/******************************************************************************************/

namespace nsm {

// JSON type used for serialization
using json = nlohmann::json;
using json_keypair = std::pair<std::string_view, nlohmann::json>;

// Error class used by all library functions
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;

    template <class ...Ts>
    static Error from_location(std::string_view file, int line, std::string_view msg, Ts ...ts) {
        std::stringstream ss;
        ss << file << ":" << line << ": " << msg;
        json j;
        ((j[ts.first] = std::move(ts.second)), ...);
        if (sizeof...(Ts)) ss << " " << j;
        return Error(ss.str());
    }
};

// Default floating point type
using real = double;
using uint = std::uint_fast32_t;

// Default vector type
template <class T>
using vec = std::vector<T>;

/******************************************************************************************/

// Identity functor
struct Identity {
    template <class T>
    T &&operator()(T &&t) const {return static_cast<T &&>(t);}
};

// Type used for ignored arguments
struct Ignore {
    template <class ...Ts>
    constexpr Ignore(Ts const &...) {}
};

// Simple copy function
template <class T>
constexpr auto copy(T &&t) {return t;}

/******************************************************************************************/

// Length overloading to handle std and arma types
template <class T, class SFINAE=void>
struct Len {constexpr auto operator()(T const &t) const {return std::size(t);}};

// General length function
template <class T>
auto len(T const &t) {return Len<T>()(t);}

// Function map yielding a vector-like output
template <class O, class V, class F=Identity>
O vmap(V &&v, F &&f={}) {
    O  o(len(v));
    std::transform(std::begin(v), std::end(v), std::begin(o), std::forward<F>(f));
    return o;
}

template <template <class...> class O, class V, class F=Identity>
auto vmap(V &&v, F &&f={}) {return vmap<O<decltype(f(*v.begin()))>>(std::forward<V>(v), std::forward<F>(f));}

template <class V, class F=Identity>
auto vmap(V &&v, F &&f={}) {return vmap<std::vector>(std::forward<V>(v), std::forward<F>(f));}

/******************************************************************************************/

template <bool B, class T, class U>
using if_t = std::conditional_t<B, T, U>;

/// Detect if a given type is well-formed
#define NSM_DETECT(NAME, expr) \
namespace detail { \
    template <class T, class=void> struct NAME##_t : std::false_type {}; \
    template <class T> struct NAME##_t<T, std::void_t<expr>> : std::true_type {}; \
} \
namespace traits {template <class T> static constexpr bool NAME = detail::NAME##_t<T>::value;}

/// Detect if a given type is well-formed
#define NSM_DETECT_2(NAME, expr) \
namespace detail { \
    template <class T, class U, class=void> struct NAME##_t : std::false_type {}; \
    template <class T, class U> struct NAME##_t<T, U, std::void_t<expr>> : std::true_type {}; \
} \
namespace traits {template <class ...Ts> static constexpr bool NAME = detail::NAME##_t<Ts...>::value;}

// Detection of some common operations
NSM_DETECT_2(can_minus, decltype(std::declval<T>() - std::declval<U>()));
NSM_DETECT_2(can_plus, decltype(std::declval<T>() + std::declval<U>()));
NSM_DETECT_2(can_less, decltype(std::declval<T>() < std::declval<U>()));
NSM_DETECT_2(can_subscript, decltype(std::declval<T>()[std::declval<U>()]));

template <class T>
using no_qual = std::remove_cv_t<std::remove_reference_t<T>>;

/******************************************************************************************/

// Range-type iterator that returns itself when dereferenced
template <class T>
struct ValueIter {
    using difference_type = if_t<std::is_unsigned_v<T>, std::make_signed_t<if_t<std::is_unsigned_v<T>, T, int>>, std::ptrdiff_t>;
    using value_type = T;
    using pointer = T;
    using reference = T const &;
    using iterator_category = if_t<traits::can_minus<T, T>, std::random_access_iterator_tag, std::bidirectional_iterator_tag>;

    T t;
    ValueIter(T t) : t(std::move(t)) {}
    bool operator==(ValueIter const &b) {return t == b.t;}
    bool operator!=(ValueIter const &b) {return t != b.t;}

    template <bool B=true, NSM_IF(B && traits::can_minus<T, T>)>
    friend auto operator-(ValueIter const &e, ValueIter const &b) {return e.t - b.t;}
    
    template <bool B=true, NSM_IF(B && traits::can_less<T, T>)>
    friend auto operator<(ValueIter const &b, ValueIter const &e) {return b.t < e.t;}
    
    template <class I, NSM_IF(traits::can_plus<T, I>)>
    friend auto operator+(ValueIter const &v, I const &i) {return ValueIter(v.t + i);}
    
    template <class I, NSM_IF(traits::can_plus<I, T>)>
    friend auto operator+(I const &i, ValueIter const &v) {return ValueIter(i + v.t);}
    
    template <class I, NSM_IF(traits::can_plus<T, I>)>
    T operator[](I const &i) {return t + i;}

    ValueIter & operator++() {++t; return *this;}
    ValueIter & operator--() {--t; return *this;}

    template <class D>
    ValueIter & operator+=(D d) {t += d; return *this;}
    T const & operator *() const {return t;}
};

/******************************************************************************************/

// View holding two iterators
template <class Iter>
struct View {
    Iter b, e;
    View(Iter b, Iter e) : b(std::move(b)), e(std::move(e)) {}

    auto begin() const {return b;}
    auto end() const {return b;}
    std::size_t size() const {return end() - begin();}

    template <class I>
    auto operator[](I const &i) -> decltype(b[i]) {return b[i];}

    auto operator~() const {return View<decltype(std::make_reverse_iterator(e))>(
        std::make_reverse_iterator(e), std::make_reverse_iterator(b));}
};

/******************************************************************************************/

/// Binary search between two iterators
template <class It1, class It2, class T, class F=Identity>
It1 binary_it_search(It1 b, It2 e, T const &t, F const &f=Identity()) {
   auto mask = e - b - 1; auto ret = b;
   while (mask > 0) if ((mask >>= 1) + ret < e && f(ret[mask]) < t) ret += mask + 1;
   return ret;
}

/// Binary search in a container
template <class V, class T, class F=Identity>
auto binary_search(V &&v, T const &t, F const &f=Identity()) -> decltype(std::begin(v)) {
   return binary_it_search(std::begin(v), std::end(v), t, f);
}

/******************************************************************************************/

template <class T> decltype(auto) first_of(T &&t) {return std::get<0>(std::forward<T>(t));}
template <class T> decltype(auto) second_of(T &&t) {return std::get<1>(std::forward<T>(t));}
template <class T> decltype(auto) third_of(T &&t) {return std::get<2>(std::forward<T>(t));}
template <class T> decltype(auto) fourth_of(T &&t) {return std::get<3>(std::forward<T>(t));}

/// Iterate through a tuple of iterators, the visitor function is the last argument; returns iterators when done
template <class T, std::size_t ...Is>
auto zip_tuple(std::index_sequence<Is...>, T t) {
    auto &&f = std::get<sizeof...(Is)>(t);
    auto iters = std::make_tuple(std::begin(std::get<Is>(t))...);
    for (; first_of(iters) != std::end(first_of(t)); (++std::get<Is>(iters), ...))
        f((*std::get<Is>(iters))...);
    return iters;
}

/// Iterate through a zip of iterators, the visitor function is the last argument; returns iterators when done
template <class ...Cs>
auto zip(Cs &&...cs) {return zip_tuple(std::make_index_sequence<sizeof...(Cs)-1>(), std::forward_as_tuple(cs...));}

template <class T, std::size_t ...Is>
std::size_t izip_tuple(std::index_sequence<Is...>, T t) {
    std::size_t n = 0;
    auto &&f = std::get<sizeof...(Is)>(t);
    for (auto is = std::make_tuple(std::begin(std::get<Is>(t))...); first_of(is) != std::end(first_of(t)); (++std::get<Is>(is), ...), ++n)
        f(n, (*std::get<Is>(is))...);
    return n;
}

/// Iterate through a zip of iterators plus a prepended counter, the visitor function is the last argument
template <class ...Cs, std::enable_if_t<sizeof...(Cs) != 2, int> = 0>
auto izip(Cs &&...cs) {
    return izip_tuple(std::make_index_sequence<sizeof...(Cs)-1>(), std::forward_as_tuple(cs...));
}

/// Iterate through a zip of iterators plus a prepended counter, the visitor function is the last argument
template <class C, class F>
auto izip(C &&c, F &&f) {
    std::size_t n = 0;
    for (auto it = std::begin(c); it != std::end(c); ++it) f(n++, *it);
    return n;
}

/******************************************************************************************/

// Range similar to Python range
template <class B, class E>
auto range(B b, E e) {
    using T = std::common_type_t<B, E>;
    return View(ValueIter(T(b)), ValueIter(T(e)));
}

template <class T>
auto range(T t) {return range(T(), t);}

// Range over indices in a container
template <class T>
auto indices(T const &t) {return range(len(t));}

// View from pointer and size
template <class P>
auto ptr_view(P p, std::size_t n) {return View(p, p+n);}

// Range over iterators through a container
template <class V>
auto iterators(V &&v) {return View(ValueIter(std::begin(v)), ValueIter(std::end(v)));}

/******************************************************************************************/

// Simple helper functions
template <class T, class U>
auto min(T &&t, U &&u) {return std::min<std::common_type_t<T, U>>(t, u);}

template <class T, class U>
auto max(T &&t, U &&u) {return std::max<std::common_type_t<T, U>>(t, u);}

template <class T>
auto sq(T &&t) {return t * t;}

/******************************************************************************************/

// Simple random distribution functions
template <class RNG>
std::size_t random_range(RNG &rng, std::size_t b, std::size_t e) {
    return std::uniform_int_distribution<std::size_t>(b, e-1)(rng); 
}

template <class V>
auto discrete_distribution(V const &v) {return std::discrete_distribution(std::begin(v), std::end(v));}

using DefaultRNG = std::mt19937;

/******************************************************************************************/

template <class ...Ts>
void print(Ts const &...ts) {
    ((std::cout << ts << ' '), ...);
    std::cout << std::endl;
}

/******************************************************************************************/

template <class V>
void sort(V &&v) {std::sort(std::begin(v), std::end(v));}

template <class V, class F>
void sort(V &&v, F &&f) {std::sort(std::begin(v), std::end(v), std::forward<F>(f));}

/******************************************************************************************/

template <class ...Ts>
auto move_as_tuple(Ts &&...ts) {return std::make_tuple(std::move(std::forward<Ts>(ts))...);}

/******************************************************************************************/

}

namespace std {

template <typename T>
struct iterator_traits<nsm::ValueIter<T>> {
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using pointer = T const *;
    using reference = T const &;
    using difference_type = typename nsm::ValueIter<T>::difference_type;
};

}