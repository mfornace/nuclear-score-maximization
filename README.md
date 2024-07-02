# Efficient column selection algorithms for kernel approximation, CUR decomposition, and inverse Laplacian rank reduction in C++

This repository contains the production code used for *Column and row subset selection using nuclear scores: theory and applications for Nystrom approximation, CUR decomposition, and graph Laplacian reduction* by Mark Fornace and Michael Lindsey (arXiv, 2024). 

The presented code is in the form of header-only C++17 code: 

CMake may be used to build a test library or to provide an interface to your own codes. If CMake is used:
- You may add `-DNSM_RCHOL=OFF` to your `cmake` command to disable incorporation of the rchol factorization library
- You may add `-DNSM_BUILD_TESTS=ON` to your `cmake` command to build the unit test module. This usage will be documented more in the future.
- vcpkg is used to handle the dependencies (which are listed in vcpkg.json)

The main drivers of the column selection algorithms used in the above paper are the `nsm::matrix_free_selection()` and `nsm::deterministic_selection()` functions in `Selection.h`. A variety of operator types may be supplied for SPSD operators comprised of sparse/dense, factorized/unfactorized, and Laplacian/non-Laplacian forms. You may consult `Engines.h` for these types. For some usage examples, you may see `Test.cc`, but more explanations of usage will be provided in the future.

This repository will be updated soon with usage examples and additional documentation. Please stay tuned.


<!-- 
```bash
cmake
ninja
```

```bash

``` -->