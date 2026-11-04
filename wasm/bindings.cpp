// Copyright 2026 Nick Thompson
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
//
// Embind bindings for the WebAssembly build. Without wasm exceptions a throw aborts the module, so errors
// return NaN or infinity, as JavaScript's Math does. Doubles stay doubles: long double is software quad here.

#define BOOST_MATH_DOMAIN_ERROR_POLICY errno_on_error
#define BOOST_MATH_POLE_ERROR_POLICY ignore_error
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_MATH_EVALUATION_ERROR_POLICY ignore_error
#define BOOST_MATH_ROUNDING_ERROR_POLICY ignore_error
#define BOOST_MATH_ASSERT_UNDEFINED_POLICY false
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false

#include <boost/math/distributions.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/math/filters/trend_filter.hpp>
#include <boost/math/tools/fft.hpp>
#include <boost/math/tools/multivariate_roots.hpp>
#include <boost/math/statistics/anderson_darling.hpp>
#include <boost/math/statistics/linear_regression.hpp>
#include <boost/math/statistics/ljung_box.hpp>
#include <boost/math/statistics/runs_test.hpp>
#include <boost/math/statistics/signal_statistics.hpp>
#include <boost/math/statistics/t_test.hpp>
#include <boost/math/statistics/z_test.hpp>

#include <emscripten/bind.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace bm = boost::math;
namespace st = boost::math::statistics;
using namespace emscripten;

// f(a..., x) for each x of an array, returned as a Float64Array: one call from JavaScript instead of one per x.
template <class F, class... A>
val map_last(F f, const val& xs, A... a)
{
   std::vector<double> x = convertJSArrayToNumberVector<double>(xs);
   for (double& v : x)
      v = f(a..., v);
   return val(typed_memory_view(x.size(), x.data())).call<val>("slice");
}

// An integer order or degree n of type I: NaN unless n is an integer that I can hold.
template <class I, class F>
double with_order(double n, F f)
{
   return n == std::trunc(n) && n >= std::numeric_limits<I>::min() && n <= std::numeric_limits<I>::max() ? double(f(I(n))) : NAN;
}

// Each function f also has f_array, whose last argument is an array: erf_array(xs), gamma_p_array(a, xs).
#define F1(f) function(#f, +[](double x) { return bm::f(x); }); \
   function(#f "_array", +[](const val& x) { return map_last([](double v) { return double(bm::f(v)); }, x); })
#define F2(f) function(#f, +[](double a, double x) { return bm::f(a, x); }); \
   function(#f "_array", +[](double a, const val& x) { return map_last([](double a, double v) { return double(bm::f(a, v)); }, x, a); })
#define F3(f) function(#f, +[](double a, double b, double x) { return bm::f(a, b, x); }); \
   function(#f "_array", +[](double a, double b, const val& x) { \
      return map_last([](double a, double b, double v) { return double(bm::f(a, b, v)); }, x, a, b); })
#define FN(f, I) function(#f, +[](double n, double x) { return with_order<I>(n, [x](I k) { return bm::f(k, x); }); }); \
   function(#f "_array", +[](double n, const val& x) { \
      return map_last([](double n, double v) { return with_order<I>(n, [v](I k) { return bm::f(k, v); }); }, x, n); })

// Only the constructor checks the parameters, so remember whether it set errno, and give NaN for everything
// about an invalid distribution: mean(geometric_distribution(2.2)) would otherwise be -0.55.
template <class D>
struct checked
{
   D d;
   bool valid;
};

template <class D, class... Params>
checked<D> make(Params... params)
{
   errno = 0;
   D d(params...);
   return {d, errno == 0};
}

// Not every distribution has an entropy.
template <class D>
auto entropy_of(const D& d, int) -> decltype(entropy(d)) { return entropy(d); }
template <class D>
double entropy_of(const D&, long) { return NAN; }

template <class D, class... Params>
void distribution(const char* name)
{
   using C = checked<D>;
   // pdf(x) and pdf_array(xs), and so on.
#define AT(f, e) function(f, +[](const C& c, double x) { return c.valid ? double(e) : NAN; }) \
      .function(f "_array", +[](const C& c, const val& xs) { return map_last([&c](double x) { return c.valid ? double(e) : NAN; }, xs); })
   class_<C>(name)
      .constructor(&make<D, Params...>)
      .AT("pdf", pdf(c.d, x))
      .AT("cdf", cdf(c.d, x))
      .AT("ccdf", cdf(complement(c.d, x)))
      .AT("quantile", quantile(c.d, x))
#undef AT
      .function("mean", +[](const C& c) { return c.valid ? mean(c.d) : NAN; })
      .function("variance", +[](const C& c) { return c.valid ? variance(c.d) : NAN; })
      .function("entropy", +[](const C& c) { return c.valid ? entropy_of(c.d, 0) : NAN; })
      .function("support", +[](const C& c) {
         auto s = support(c.d);
         const double values[] = {c.valid ? s.first : NAN, c.valid ? s.second : NAN};
         return val::array(values, values + 2);
      });
}

// The FFTs and statistics throw for bad sizes, which would abort the module, so check first and throw a RangeError.
[[noreturn]] void range_error(const std::string& what)
{
   val::global("RangeError").new_(what).throw_();
   __builtin_unreachable();
}

// f(x) for the multivariate solvers returns [F, J] or [F, J, d2]: F an array of n, J n arrays of n or n*n numbers
// row major, and d2(v) = F''(x)[v, v] an array of n. A wrong shape throws a RangeError, and so does anything f
// throws: both unwind through the solver, so its scratch memory leaks, a few times n^2 doubles.
struct js_system
{
   val f;
   std::size_t n;

   static std::vector<double> numbers(const val& a, std::size_t count, const char* what)
   {
      std::vector<double> v;
      if (a.isArray() && a["length"].as<std::size_t>() > 0 && a[0].isArray())
         v = convertJSArrayToNumberVector<double>(a.call<val>("flat"));
      else if (!a.isUndefined() && !a.isNull())
         v = convertJSArrayToNumberVector<double>(a);
      if (v.size() != count)
         range_error(std::string(what) + " has " + std::to_string(v.size()) + " numbers, expected " + std::to_string(count) + ".");
      return v;
   }

   auto operator()(const std::vector<double>& x) const
   {
      val r = f(to_js(x));
      if (!r.isArray() || r["length"].as<std::size_t>() < 2)
         range_error("f(x) must return [F, J] or [F, J, d2].");
      val d2 = r["length"].as<std::size_t>() > 2 ? r[2] : val::undefined();
      auto second = [d2, n = n](const std::vector<double>& v) {
         if (d2.typeOf().as<std::string>() != "function")
            range_error("multivariate_halley: f(x) must return [F, J, d2], with d2 a function.");
         return numbers(d2(to_js(v)), n, "d2(v)");
      };
      return std::make_tuple(numbers(r[0], n, "F(x)"), numbers(r[1], n * n, "J(x)"), second);
   }

   static val to_js(const std::vector<double>& x) { return val(typed_memory_view(x.size(), x.data())).call<val>("slice"); }
};

// {root, iterations}: the root is all NaN if the solver failed, and iterations == max_iterations if it did not converge.
template <bool Halley>
val multivariate_root(const val& f, const val& x0, double digits, double max_iterations)
{
   std::vector<double> x = convertJSArrayToNumberVector<double>(x0);
   std::uintmax_t it = max_iterations >= 1 && max_iterations < 1e15 ? std::uintmax_t(max_iterations) : 1000;
   const int d = digits >= 1 && digits <= 53 ? int(digits) : 53;
   const js_system sys{f, x.size()};
   x = Halley ? bm::tools::multivariate_halley(sys, std::move(x), d, it) : bm::tools::multivariate_newton(sys, std::move(x), d, it);
   val o = val::object();
   o.set("root", js_system::to_js(x));
   o.set("iterations", double(it));
   return o;
}

std::size_t length(double n)
{
   if (!(n >= 1 && n <= (1 << 24) && n == std::trunc(n)))
      range_error("The length must be an integer in [1, 2^24].");
   return static_cast<std::size_t>(n);
}

using cplx = std::complex<double>;

template <class T>
val to_js(std::vector<T>& z)
{
   return val(typed_memory_view(z.size() * sizeof(T) / sizeof(double), reinterpret_cast<double*>(z.data()))).call<val>("slice");
}

// Complex data are interleaved (re, im) Float64Arrays.
struct fft
{
   explicit fft(double n) : f(length(n)), z(f.size()) {}
   val run(const val& x, bool inverse)
   {
      std::vector<double> v = convertJSArrayToNumberVector<double>(x);
      if (v.size() != 2 * z.size())
         range_error("The input must hold 2n numbers.");
      for (std::size_t k = 0; k < z.size(); ++k)
         z[k] = cplx(v[2 * k], v[2 * k + 1]);
      inverse ? f.inverse_transform(z) : f.transform(z);
      return to_js(z);
   }
   bm::tools::fft<cplx> f;
   std::vector<cplx> z;
};

// n real values in, X[0], ..., X[n/2] out.
struct real_fft
{
   explicit real_fft(double n) : f(length(n)), z(f.size() / 2 + 1) {}
   val transform(const val& x)
   {
      std::vector<double> v = convertJSArrayToNumberVector<double>(x);
      if (v.size() != f.size())
         range_error("The input must hold n numbers.");
      f.transform(v, z);
      return to_js(z);
   }
   bm::tools::real_fft<cplx> f;
   std::vector<cplx> z;
};

// Transforms each lane along one axis of a C-order array with the given shape, whose elements are In (double or
// cplx). make(n) gives the transform of one lane of length n, a function from std::vector<In> to std::vector<Out>.
template <class In, class Make>
val along_axis(const val& x, const val& shape, double axis, Make make)
{
   std::vector<double> v = convertJSArrayToNumberVector<double>(x);
   std::vector<double> s = convertJSArrayToNumberVector<double>(shape);
   const double d = double(s.size());
   if (!(axis == std::trunc(axis) && axis >= -d && axis < d))
      range_error("The axis must be an integer in [-ndim, ndim).");
   const std::size_t a = std::size_t(axis < 0 ? axis + d : axis);
   double before = 1, after = 1;
   for (std::size_t i = 0; i < s.size(); ++i)
   {
      if (!(s[i] >= 1 && s[i] == std::trunc(s[i])))
         range_error("The shape must hold positive integers.");
      if (i != a)
         (i < a ? before : after) *= s[i];
   }
   if (before * s[a] * after * (sizeof(In) / sizeof(double)) != double(v.size()))
      range_error(sizeof(In) == sizeof(double) ? "The data must hold product(shape) numbers."
                                               : "The data must hold 2 product(shape) numbers.");
   const std::size_t outer = std::size_t(before), n = length(s[a]), inner = std::size_t(after);
   auto t = make(n);
   const In* in = reinterpret_cast<const In*>(v.data());
   std::vector<In> lane(n);
   std::vector<std::decay_t<decltype(t(lane)[0])>> out;
   for (std::size_t o = 0; o < outer; ++o)
      for (std::size_t i = 0; i < inner; ++i)
      {
         for (std::size_t j = 0; j < n; ++j)
            lane[j] = in[(o * n + j) * inner + i];
         const auto& y = t(lane);
         out.resize(outer * y.size() * inner);
         for (std::size_t k = 0; k < y.size(); ++k)
            out[(o * y.size() + k) * inner + i] = y[k];
      }
   return to_js(out);
}

val fft_axis(const val& x, const val& shape, double axis, bool inverse)
{
   return along_axis<cplx>(x, shape, axis, [inverse](std::size_t n) {
      return [f = bm::tools::fft<cplx>(n), inverse](std::vector<cplx>& z) mutable -> std::vector<cplx>& {
         inverse ? f.inverse_transform(z) : f.transform(z);
         return z;
      };
   });
}

val real_fft_axis(const val& x, const val& shape, double axis)
{
   return along_axis<double>(x, shape, axis, [](std::size_t n) {
      return [f = bm::tools::real_fft<cplx>(n), z = std::vector<cplx>(n / 2 + 1)](std::vector<double>& v) mutable -> std::vector<cplx>& {
         f.transform(v, z);
         return z;
      };
   });
}

// P[k] = c |X[k]|^2 / n^2 for k = 0, ..., n/2, with c = 1 at k = 0 and the Nyquist k = n/2 and c = 2 otherwise:
// sum P = mean(x^2), and A cos(2 pi k j / n) gives P[k] = A^2 / 2.
val power_spectrum_axis(const val& x, const val& shape, double axis)
{
   return along_axis<double>(x, shape, axis, [](std::size_t n) {
      return [f = bm::tools::real_fft<cplx>(n), z = std::vector<cplx>(n / 2 + 1), p = std::vector<double>(n / 2 + 1), n](
                std::vector<double>& v) mutable -> std::vector<double>& {
         f.transform(v, z);
         for (std::size_t k = 0; k < z.size(); ++k)
            p[k] = (k == 0 || 2 * k == n ? 1 : 2) * std::norm(z[k]) / (double(n) * double(n));
         return p;
      };
   });
}

// The statistics assert or throw on too few values, so check first.
std::vector<double> sample(const val& x, std::size_t at_least)
{
   std::vector<double> v = convertJSArrayToNumberVector<double>(x);
   if (v.size() < at_least)
      range_error("The array needs at least " + std::to_string(at_least) + (at_least == 1 ? " value." : " values."));
   return v;
}

std::vector<double> same_size(const val& y, const std::vector<double>& x)
{
   std::vector<double> v = sample(y, x.size());
   if (v.size() != x.size())
      range_error("The arrays must have the same length.");
   return v;
}

// Sorting and selection need an order, which NaN breaks.
bool has_nan(const std::vector<double>& v)
{
   return std::any_of(v.begin(), v.end(), [](double x) { return std::isnan(x); });
}

val test_result(std::pair<double, double> r)
{
   val o = val::object();
   o.set("statistic", r.first);
   o.set("p_value", r.second);
   return o;
}

// The statistics of one array, which needs at least n values.
#define S1(f, n) function(#f, +[](const val& x) { std::vector<double> v = sample(x, n); return double(st::f(v)); })
#define S1_ORDER(f, n) function(#f, +[](const val& x) { std::vector<double> v = sample(x, n); return has_nan(v) ? NAN : double(st::f(v)); })

EMSCRIPTEN_BINDINGS(boost_math)
{
   F1(tgamma); F2(tgamma); F1(lgamma); F1(digamma); F1(trigamma); FN(polygamma, int); F1(tgamma1pm1);
   F2(tgamma_lower); F2(tgamma_ratio); F2(tgamma_delta_ratio);
   F2(gamma_p); F2(gamma_q); F2(gamma_p_inv); F2(gamma_q_inv); F2(gamma_p_derivative);
   F2(beta); F3(beta); F3(betac); F3(ibeta); F3(ibetac); F3(ibeta_inv); F3(ibetac_inv); F3(ibeta_derivative);
   F1(erf); F1(erfc); F1(erf_inv); F1(erfc_inv);
   F1(expint); FN(expint, unsigned); F1(zeta);
   F2(cyl_bessel_j); F2(cyl_neumann); F2(cyl_bessel_i); F2(cyl_bessel_k); FN(sph_bessel, unsigned); FN(sph_neumann, unsigned);
   F1(airy_ai); F1(airy_bi); F1(airy_ai_prime); F1(airy_bi_prime);
   F1(ellint_1); F2(ellint_1); F1(ellint_2); F2(ellint_2); F2(ellint_3); F3(ellint_3);
   F1(lambert_w0); F1(lambert_wm1); F2(owens_t);
   FN(legendre_p, int); FN(laguerre, unsigned); FN(hermite, unsigned); FN(chebyshev_t, unsigned);
   F1(log1p); F1(expm1); F1(cbrt); F1(sqrt1pm1); F2(powm1); F1(sinc_pi); F1(sin_pi); F1(cos_pi);
   F2(hypergeometric_0F1); F3(hypergeometric_1F1); F3(hypergeometric_2F0);
   F2(jacobi_sn); F2(jacobi_cn); F2(jacobi_dn); F2(jacobi_zeta); F2(heuman_lambda); F1(ellint_d); F2(ellint_d);
   F2(jacobi_theta1); F2(jacobi_theta2); F2(jacobi_theta3); F2(jacobi_theta4);
   F3(ellint_rf); F3(ellint_rd); F2(ellint_rc); F3(ellint_rg);
   F2(cyl_bessel_j_prime); F2(cyl_neumann_prime); F2(cyl_bessel_i_prime); F2(cyl_bessel_k_prime);
   // Zeros are numbered m = 1, 2, ...; the order n of gegenbauer and spherical_harmonic_* comes first.
   function("cyl_bessel_j_zero", +[](double v, double m) { return with_order<int>(m, [v](int k) { return bm::cyl_bessel_j_zero(v, k); }); });
   function("cyl_neumann_zero", +[](double v, double m) { return with_order<int>(m, [v](int k) { return bm::cyl_neumann_zero(v, k); }); });
   function("airy_ai_zero", +[](double m) { return with_order<int>(m, [](int k) { return bm::airy_ai_zero<double>(k); }); });
   function("airy_bi_zero", +[](double m) { return with_order<int>(m, [](int k) { return bm::airy_bi_zero<double>(k); }); });
   function("gegenbauer", +[](double n, double l, double x) { return with_order<unsigned>(n, [l, x](unsigned k) { return bm::gegenbauer(k, l, x); }); });
   function("spherical_harmonic_r", +[](double n, double m, double theta, double phi) {
      return with_order<unsigned>(n, [&](unsigned k) { return with_order<int>(m, [&](int j) { return bm::spherical_harmonic_r(k, j, theta, phi); }); }); });
   function("spherical_harmonic_i", +[](double n, double m, double theta, double phi) {
      return with_order<unsigned>(n, [&](unsigned k) { return with_order<int>(m, [&](int j) { return bm::spherical_harmonic_i(k, j, theta, phi); }); }); });

   distribution<bm::normal_distribution<double>, double, double>("normal_distribution");
   distribution<bm::students_t_distribution<double>, double>("students_t_distribution");
   distribution<bm::chi_squared_distribution<double>, double>("chi_squared_distribution");
   distribution<bm::beta_distribution<double>, double, double>("beta_distribution");
   distribution<bm::gamma_distribution<double>, double, double>("gamma_distribution");
   distribution<bm::exponential_distribution<double>, double>("exponential_distribution");
   distribution<bm::lognormal_distribution<double>, double, double>("lognormal_distribution");
   distribution<bm::weibull_distribution<double>, double, double>("weibull_distribution");
   distribution<bm::cauchy_distribution<double>, double, double>("cauchy_distribution");
   distribution<bm::fisher_f_distribution<double>, double, double>("fisher_f_distribution");
   distribution<bm::uniform_distribution<double>, double, double>("uniform_distribution");
   distribution<bm::laplace_distribution<double>, double, double>("laplace_distribution");
   distribution<bm::logistic_distribution<double>, double, double>("logistic_distribution");
   distribution<bm::non_central_chi_squared_distribution<double>, double, double>("non_central_chi_squared_distribution");
   distribution<bm::non_central_t_distribution<double>, double, double>("non_central_t_distribution");
   distribution<bm::binomial_distribution<double>, double, double>("binomial_distribution");
   distribution<bm::poisson_distribution<double>, double>("poisson_distribution");
   distribution<bm::negative_binomial_distribution<double>, double, double>("negative_binomial_distribution");
   distribution<bm::geometric_distribution<double>, double>("geometric_distribution");
   distribution<bm::arcsine_distribution<double>, double, double>("arcsine_distribution");
   distribution<bm::bernoulli_distribution<double>, double>("bernoulli_distribution");
   distribution<bm::extreme_value_distribution<double>, double, double>("extreme_value_distribution");
   distribution<bm::generalized_extreme_value_distribution<double>, double, double, double>("generalized_extreme_value_distribution");
   distribution<bm::holtsmark_distribution<double>, double, double>("holtsmark_distribution");
   distribution<bm::hypergeometric_distribution<double>, unsigned, unsigned, unsigned>("hypergeometric_distribution");
   distribution<bm::inverse_chi_squared_distribution<double>, double, double>("inverse_chi_squared_distribution");
   distribution<bm::inverse_gamma_distribution<double>, double, double>("inverse_gamma_distribution");
   distribution<bm::inverse_gaussian_distribution<double>, double, double>("inverse_gaussian_distribution");
   distribution<bm::kolmogorov_smirnov_distribution<double>, double>("kolmogorov_smirnov_distribution");
   distribution<bm::landau_distribution<double>, double, double>("landau_distribution");
   distribution<bm::mapairy_distribution<double>, double, double>("mapairy_distribution");
   distribution<bm::non_central_beta_distribution<double>, double, double, double>("non_central_beta_distribution");
   distribution<bm::non_central_f_distribution<double>, double, double, double>("non_central_f_distribution");
   distribution<bm::pareto_distribution<double>, double, double>("pareto_distribution");
   distribution<bm::rayleigh_distribution<double>, double>("rayleigh_distribution");
   distribution<bm::saspoint5_distribution<double>, double, double>("saspoint5_distribution");
   distribution<bm::skew_normal_distribution<double>, double, double, double>("skew_normal_distribution");
   distribution<bm::triangular_distribution<double>, double, double, double>("triangular_distribution");
   distribution<bm::von_mises_distribution<double>, double, double>("von_mises_distribution");

   class_<fft>("fft")
      .constructor<double>()
      .function("transform", +[](fft& f, const val& x) { return f.run(x, false); })
      .function("inverse_transform", +[](fft& f, const val& x) { return f.run(x, true); });
   class_<real_fft>("real_fft").constructor<double>().function("transform", &real_fft::transform);
   function("fft_axis", +[](const val& x, const val& shape, double axis) { return fft_axis(x, shape, axis, false); });
   function("inverse_fft_axis", +[](const val& x, const val& shape, double axis) { return fft_axis(x, shape, axis, true); });
   function("real_fft_axis", &real_fft_axis);
   function("power_spectrum_axis", &power_spectrum_axis);
   function("power_spectrum", +[](const val& x) { return power_spectrum_axis(x, val::array(std::vector<double>{x["length"].as<double>()}), 0); });

   function("multivariate_newton", +[](const val& f, const val& x0) { return multivariate_root<false>(f, x0, 53, 1000); });
   function("multivariate_newton", +[](const val& f, const val& x0, double d) { return multivariate_root<false>(f, x0, d, 1000); });
   function("multivariate_newton", &multivariate_root<false>);
   function("multivariate_halley", +[](const val& f, const val& x0) { return multivariate_root<true>(f, x0, 53, 1000); });
   function("multivariate_halley", +[](const val& f, const val& x0, double d) { return multivariate_root<true>(f, x0, d, 1000); });
   function("multivariate_halley", &multivariate_root<true>);

   // A non-integer k becomes -1, which the filter rejects like any other bad k: the result is all NaN.
   function("trend_filter", +[](const val& x, double k, double lambda) {
      std::vector<double> v = convertJSArrayToNumberVector<double>(x);
      v = bm::filters::trend_filter(v, k == std::trunc(k) && std::abs(k) < 8 ? int(k) : -1, lambda);
      return val(typed_memory_view(v.size(), v.data())).call<val>("slice");
   });
   function("trend_filter", +[](const val& x) {
      std::vector<double> v = bm::filters::trend_filter(convertJSArrayToNumberVector<double>(x));
      return val(typed_memory_view(v.size(), v.data())).call<val>("slice");
   });

   S1(mean, 1); S1(variance, 1); S1(sample_variance, 2); S1(skewness, 1); S1(kurtosis, 1); S1(excess_kurtosis, 1);
   S1_ORDER(median, 1); S1_ORDER(median_absolute_deviation, 1); S1_ORDER(interquartile_range, 3);
   S1_ORDER(gini_coefficient, 1); S1_ORDER(sample_gini_coefficient, 2); S1(hoyer_sparsity, 2);
   function("mean_and_sample_variance", +[](const val& x) {
      std::vector<double> v = sample(x, 2);
      auto [m, s2] = st::mean_and_sample_variance(v);
      val o = val::object();
      o.set("mean", m);
      o.set("sample_variance", s2);
      return o;
   });
   function("covariance", +[](const val& x, const val& y) { std::vector<double> u = sample(x, 1); return st::covariance(u, same_size(y, u)); });
   function("correlation_coefficient", +[](const val& x, const val& y) {
      std::vector<double> u = sample(x, 1);
      return st::correlation_coefficient(u, same_size(y, u));
   });
   function("simple_ordinary_least_squares", +[](const val& x, const val& y) {
      std::vector<double> u = sample(x, 2), v = same_size(y, u);
      if (st::variance(u) <= 0)
         range_error("x must not be constant.");
      auto [c0, c1, r2] = st::simple_ordinary_least_squares_with_R_squared(u, v);
      val o = val::object();
      o.set("intercept", c0);
      o.set("slope", c1);
      o.set("r_squared", r2);
      return o;
   });
   function("one_sample_t_test", +[](const val& x, double mu) { return test_result(st::one_sample_t_test(sample(x, 2), mu)); });
   function("one_sample_z_test", +[](const val& x, double mu) { return test_result(st::one_sample_z_test(sample(x, 2), mu)); });
   function("two_sample_t_test", +[](const val& x, const val& y) { return test_result(st::two_sample_t_test(sample(x, 2), sample(y, 2))); });
   function("two_sample_z_test", +[](const val& x, const val& y) { return test_result(st::two_sample_z_test(sample(x, 2), sample(y, 2))); });
   function("paired_samples_t_test", +[](const val& x, const val& y) {
      std::vector<double> u = sample(x, 2);
      return test_result(st::paired_samples_t_test(u, same_size(y, u)));
   });
   function("runs_above_and_below_threshold", +[](const val& x, double t) {
      std::vector<double> v = sample(x, 2);
      return test_result(has_nan(v) ? std::pair<double, double>(NAN, NAN) : st::runs_above_and_below_threshold(v, t));
   });
   function("runs_above_and_below_median", +[](const val& x) {
      std::vector<double> v = sample(x, 2);
      return test_result(has_nan(v) ? std::pair<double, double>(NAN, NAN) : st::runs_above_and_below_median(v));
   });
   // Sorts a copy of the data, and estimates the mean and standard deviation from it.
   function("anderson_darling_normality_statistic", +[](const val& x) -> double {
      std::vector<double> v = sample(x, 2);
      if (has_nan(v))
         return NAN;
      std::sort(v.begin(), v.end());
      return st::anderson_darling_normality_statistic(v);
   });
   // lags = -1 takes ceil(log(n)) lags.
   function("ljung_box", +[](const val& x, double lags, double fit_dof) {
      std::vector<double> v = sample(x, 2);
      const double n = double(v.size()), l = lags == -1 ? std::ceil(std::log(n)) : lags;
      if (!(l == std::trunc(l) && l >= 1 && l < n))
         range_error("The lags must be -1 or an integer in [1, n).");
      if (!(fit_dof == std::trunc(fit_dof) && std::abs(fit_dof) < 1e15))
         range_error("fit_dof must be an integer.");
      return test_result(st::ljung_box(v, int64_t(l), int64_t(fit_dof)));
   });
}
