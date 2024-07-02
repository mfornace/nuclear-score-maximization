#pragma once
/**
    Execution.h: Executor class for serial and multithreaded execution
*/

#include "Common.h"
#include <type_traits>
#include <iterator>
#include <tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb.h>

namespace nsm {

/******************************************************************************************/

// Error which can be set from multiple threads
class SharedError {
    struct Impl {
        std::atomic<bool> set = false; // set before ptr is stored
        std::atomic<bool> stored = false; // set after ptr is stored
        std::exception_ptr exception;
    };
    std::shared_ptr<Impl> ptr = std::make_shared<Impl>();

public:
    void set_to_current_exception() noexcept {
        if (!ptr->set.exchange(true, std::memory_order_relaxed)) {
            ptr->exception = std::current_exception();
            ptr->stored.store(true, std::memory_order_relaxed);
        }
    }

    bool clear() noexcept {
        bool out = ptr->stored.exchange(false, std::memory_order_relaxed);
        ptr->exception = nullptr;
        ptr->set.exchange(false, std::memory_order_relaxed);
        return out;
    }

    void rethrow_if_set() const {
        if (is_set()) {
            // this should practically never matter ... but theoretically it could
            while (!ptr->stored.load(std::memory_order_relaxed)) {}
            std::rethrow_exception(ptr->exception);
        }
    }

    bool is_set() const noexcept {
        return ptr->set.load(std::memory_order_relaxed);
    }

    template <class Exc>
    void set_exception(Exc exc) noexcept {
        try {throw std::move(exc);}
        catch (Exc const &) {set_to_current_exception();}
    }

    template <class F, class ...Args>
    void invoke_noexcept(F &&f, Args &&...args) noexcept {
        try {
            f(std::forward<Args>(args)...);
        } catch (...) {
            set_to_current_exception();
        }
    }
};


/******************************************************************************************/

// Serial or multithreaded executor using TBB
struct Executor {
    using Arena = oneapi::tbb::task_arena;

    std::shared_ptr<Arena> arena;
    unsigned n_workers() const {return arena ? arena->max_concurrency() : 1;}

    Executor(unsigned threads=1) {
        if (threads == 0) threads = std::thread::hardware_concurrency();
        if (threads != 1) arena = std::make_shared<Arena>(threads);
    }

    template <class V, class F>
    void implement_map(V &&v, F &&f) const {
        if (arena) arena->execute([&] {
            SharedError err;
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0u, len(v), 1), [&](auto const &b) {
                if (!err.is_set()) err.invoke_noexcept([&] {for (auto i : iterators(b)) f(v[i]);});
            }, tbb::auto_partitioner());
            err.rethrow_if_set();
        }); else for (auto &&x : v) f(std::forward<decltype(x)>(x));
    };

    template <class V, class F>
    auto map(V &&v, F &&f) const {
        if constexpr(!std::is_same_v<decltype(f(*std::begin(v))), void>) {
            vec<decltype(f(*std::begin(v)))> out(len(v));
            map(indices(out), [&](auto i) {out[i] = f(*std::next(std::begin(v)));});
            return out;
        } else {
            implement_map(v, f);
        }
    }
};

/******************************************************************************************/

}