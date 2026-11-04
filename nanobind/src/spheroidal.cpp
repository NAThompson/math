// Copyright 2026 Nick Thompson
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/math/special_functions/spheroidal.hpp>

#include "bindings.hpp"
#include "macros.hpp"

// Each angular and radial function, and its derivative (the C++ derivative pointer), as separate ufunc-like functions.
#define BMPY_SPHEROIDAL(fn, x, doc)                                                                                           \
   bmpy::def_vec<bmpy::Int<unsigned>, bmpy::Int<unsigned>, bmpy::Real, bmpy::Real>(                                        \
       m, #fn, []<class T>(unsigned n, unsigned mm, T c, T x0) { return bm::fn(n, mm, c, x0); }, {"n", "m", "c", x}, doc); \
   bmpy::def_vec<bmpy::Int<unsigned>, bmpy::Int<unsigned>, bmpy::Real, bmpy::Real>(                                        \
       m, #fn "_prime",                                                                                                      \
       []<class T>(unsigned n, unsigned mm, T c, T x0) { T d; bm::fn(n, mm, c, x0, &d); return d; }, {"n", "m", "c", x},     \
       "The derivative of " #fn " with respect to " x "." BMPY_URL("spheroidal"))

void bmpy::bind_spheroidal(nb::module_& m) {
   bmpy::def_vec<bmpy::Int<unsigned>, bmpy::Int<unsigned>, bmpy::Real>(
       m, "prolate_spheroidal_eigenvalue", []<class T>(unsigned n, unsigned mm, T c) { return bm::prolate_spheroidal_eigenvalue(n, mm, c); },
       {"n", "m", "c"}, "DLMF's eigenvalue λ_n^m(c²); scipy.special.pro_cv(m, n, c) is this plus c²." BMPY_URL("spheroidal"));
   bmpy::def_vec<bmpy::Int<unsigned>, bmpy::Int<unsigned>, bmpy::Real>(
       m, "oblate_spheroidal_eigenvalue", []<class T>(unsigned n, unsigned mm, T c) { return bm::oblate_spheroidal_eigenvalue(n, mm, c); },
       {"n", "m", "c"}, "DLMF's eigenvalue λ_n^m(-c²); scipy.special.obl_cv(m, n, c) is this minus c²." BMPY_URL("spheroidal"));
   BMPY_SPHEROIDAL(prolate_spheroidal_angular, "eta", "Ps_n^m(η, c²) (DLMF 30.4, Meixner-Schäfke normalization)." BMPY_URL("spheroidal"));
   BMPY_SPHEROIDAL(oblate_spheroidal_angular, "eta", "Ps_n^m(η, -c²) (DLMF 30.4, Meixner-Schäfke normalization)." BMPY_URL("spheroidal"));
   BMPY_SPHEROIDAL(prolate_spheroidal_radial_1, "xi", "R_mn^(1)(c, ξ), ξ >= 1; as scipy.special.pro_rad1(m, n, c, ξ)[0]." BMPY_URL("spheroidal"));
   BMPY_SPHEROIDAL(prolate_spheroidal_radial_2, "xi", "R_mn^(2)(c, ξ), ξ > 1; as scipy.special.pro_rad2(m, n, c, ξ)[0]." BMPY_URL("spheroidal"));
   BMPY_SPHEROIDAL(oblate_spheroidal_radial_1, "xi", "R_mn^(1)(-ic, iξ), ξ >= 0; as scipy.special.obl_rad1(m, n, c, ξ)[0]." BMPY_URL("spheroidal"));
   BMPY_SPHEROIDAL(oblate_spheroidal_radial_2, "xi", "R_mn^(2)(-ic, iξ), ξ >= 0; as scipy.special.obl_rad2(m, n, c, ξ)[0]." BMPY_URL("spheroidal"));
}
