#pragma once
/**
Serialize.h: helper routines and ADL overloading for serialization to/from JSON
*/

#include <nlohmann/json.hpp>
#include "Matrix.h"

namespace nlohmann {

/******************************************************************************************/

template <class T>
struct arma_serializer {using defined = std::false_type;};

template <class T>
struct adl_serializer<T, std::enable_if_t< nsm::la::traits::has_eval<T> && !(arma_serializer<T>::defined::value)>> {
    static void to_json(json &j, T const &t) {
        adl_serializer<std::decay_t<decltype(t.eval())>>::to_json(j, t.eval());
    }
};

template <class T>
struct adl_serializer<T, std::enable_if_t<(arma_serializer<T>::defined::value)>> 
    : arma_serializer<T> {};

/******************************************************************************************/

template <class T>
struct arma_serializer<arma::Col<T>> {
    using defined = std::true_type;
    static void to_json(json &j, arma::Col<T> const &t) {
        j = nlohmann::json::array_t(t.begin(), t.end());
    }
    static void from_json(json const &j, arma::Col<T> &t) {
        t.set_size(j.size());
        std::copy(j.begin(), j.end(), t.begin());
    }
};

/******************************************************************************************/

template <class T>
struct arma_serializer<arma::Mat<T>> {
    using defined = std::true_type;
    static void to_json(json &j, arma::Mat<T> const &t) {
        j["shape"] = {t.n_rows, t.n_cols};
        j["data"] = nlohmann::json::array_t(t.begin(), t.end());
    }
    static void from_json(json const &j, arma::Mat<T> &t) {
        if (j.is_object()) {
            auto shape = j.at("shape").get<std::array<std::size_t, 2>>();
            t.set_size(shape[0], shape[1]);
            auto const &data = j.at("data");
            std::copy(data.begin(), data.end(), t.begin());
        } else if (j.is_array()) {
            if (!j.empty()) {
                t = arma::Mat<T>(j.size(), j[0].size(), arma::fill::none);
                arma::uword r = 0;
                for (auto const &x : j) {
                    NSM_REQUIRE(x.size(), ==, t.n_cols, "mismatch in matrix size");
                    std::copy(x.begin(), x.end(), t.begin_row(r++));
                }
            } else t = arma::Mat<T>();
        } else throw std::runtime_error("Matrix json could not be loaded");
    }
};

/******************************************************************************************/

template <class T>
struct arma_serializer<arma::Cube<T>> {
    using defined = std::true_type;
    static void to_json(json &j, arma::Cube<T> const &t) {
        j["shape"] = {t.n_rows, t.n_cols, t.n_slices};
        j["data"] = nlohmann::json::array_t(t.begin(), t.end());
    }
    static void from_json(json const &j, arma::Cube<T> &t) {
        auto shape = j.at("shape").get<std::array<std::size_t, 3>>();
        t.set_size(shape[0], shape[1], shape[2]);
        auto const &data = j.at("data");
        std::copy(data.begin(), data.end(), t.begin());
    }
};

/******************************************************************************************/

template <class T>
struct arma_serializer<arma::SpMat<T>> {
    using defined = std::true_type;
    static void to_json(json &j, arma::SpMat<T> const &t) {
        t.sync();
        j["shape"] = std::make_tuple(t.n_rows, t.n_cols);
        j["values"] = std::vector<T>(t.values, t.values + t.n_nonzero + 1);
        j["row_indices"] = std::vector<arma::uword>(t.row_indices, t.row_indices + t.n_nonzero + 1);
        j["col_ptrs"] = std::vector<arma::uword>(t.col_ptrs, t.col_ptrs + t.n_cols + 1);
    }
    static void from_json(json const &j, arma::SpMat<T> &t) {
        arma::Col<T> values;
        arma::uvec rows, cols;
        std::array<arma::uword, 2> shape;
        j.at("shape").get_to(shape);
        j.at("values").get_to(values);
        if (auto r = j.find("row_indices"); r != j.end()) {
            r->get_to(rows);
            j.at("col_ptrs").get_to(cols);
            t = arma::SpMat<T>(std::move(rows), std::move(cols), std::move(values), shape[0], shape[1]);
        } else {
            j.at("rows").get_to(rows);
            j.at("cols").get_to(cols);
            t = arma::SpMat<T>(arma::join_cols(rows.t(), cols.t()), std::move(values), shape[0], shape[1]);
        }
    }
};

/******************************************************************************************/

}
