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

namespace {

// prolate_spheroidal(n, m, c) and oblate_spheroidal(n, m, c): the eigenvalue and expansion coefficients are
// computed once, and each method evaluates elementwise.
template <class W>
void bind_wave(bmpy::nb::module_& m, const char* name, const char* doc) {
   bmpy::nb::class_<W> cls(m, name, doc);
   cls.def(bmpy::nb::init<unsigned, unsigned, double>(), bmpy::nb::arg("n"), bmpy::nb::arg("m"), bmpy::nb::arg("c"),
           bmpy::nb::sig("def __init__(self, n, m, c, /) -> None"))
       .def_prop_ro("eigenvalue", &W::eigenvalue, "The eigenvalue, as the free function *_spheroidal_eigenvalue.");
   bmpy::def_method_vec(cls, "angular", "eta", [](const W& w, double x) { return w.angular(x); }, "Ps_n^m(η).");
   bmpy::def_method_vec(cls, "angular_prime", "eta", [](const W& w, double x) { double d; w.angular(x, &d); return d; },
                        "dPs_n^m/dη.");
   bmpy::def_method_vec(cls, "radial_1", "xi", [](const W& w, double x) { return w.radial_1(x); }, "R_mn^(1)(ξ).");
   bmpy::def_method_vec(cls, "radial_1_prime", "xi", [](const W& w, double x) { double d; w.radial_1(x, &d); return d; },
                        "dR_mn^(1)/dξ.");
   bmpy::def_method_vec(cls, "radial_2", "xi", [](const W& w, double x) { return w.radial_2(x); }, "R_mn^(2)(ξ).");
   bmpy::def_method_vec(cls, "radial_2_prime", "xi", [](const W& w, double x) { double d; w.radial_2(x, &d); return d; },
                        "dR_mn^(2)/dξ.");
}

}  // namespace

void bmpy::bind_spheroidal(nb::module_& m) {
   bind_wave<bm::prolate_spheroidal<double>>(m, "prolate_spheroidal",
       "The prolate spheroidal wave functions of degree n and order m for c, with the eigenvalue and coefficients "
       "computed once; the same values as the prolate_spheroidal_* functions, and faster for many arguments." BMPY_URL("spheroidal"));
   bind_wave<bm::oblate_spheroidal<double>>(m, "oblate_spheroidal",
       "The oblate spheroidal wave functions of degree n and order m for c, with the eigenvalue and coefficients "
       "computed once; the same values as the oblate_spheroidal_* functions, and faster for many arguments." BMPY_URL("spheroidal"));
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
