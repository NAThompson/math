// Copyright 2026 Nick Thompson
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/math/special_functions/struve.hpp>

#include "bindings.hpp"
#include "macros.hpp"

void bmpy::bind_struve(nb::module_& m) {
   BMPY_R2(struve_h, "v", "x", "The Struve function H_v(x); x < 0 only for integer v." BMPY_URL("struve"));
   BMPY_R2(struve_l, "v", "x", "The modified Struve function L_v(x); x < 0 only for integer v." BMPY_URL("struve"));
   bmpy::def_vec<bmpy::Real, bmpy::Real>(
       m, "struve_h_prime", []<class T>(T v, T x) { T d; bm::struve_h(v, x, &d); return d; }, {"v", "x"},
       "H_v′(x), the derivative in x." BMPY_URL("struve"));
   bmpy::def_vec<bmpy::Real, bmpy::Real>(
       m, "struve_l_prime", []<class T>(T v, T x) { T d; bm::struve_l(v, x, &d); return d; }, {"v", "x"},
       "L_v′(x), the derivative in x." BMPY_URL("struve"));
}
