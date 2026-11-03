// Copyright 2026 Nick Thompson
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
//
// Smoke test of the bindings; the functions themselves are tested in test/.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const path = process.env.BOOST_MATH_WASM ?? 'build-wasm/boost_math.mjs';
const createModule = (await import(pathToFileURL(path))).default;
const bm = await createModule();

function close(actual, expected, tol = 1e-13) {
   assert.ok(Math.abs(actual - expected) <= tol * Math.abs(expected), `${actual} != ${expected}`);
}

// References from mpmath at 40 digits.
test('special functions', () => {
   close(bm.tgamma(4.5), 11.631728396567452);
   close(bm.lgamma(100.5), 361.43554046777762);
   close(bm.digamma(2.5), 0.70315664064524319);
   close(bm.gamma_p(3, 2), 0.32332358381693654);
   close(bm.ibeta(2, 3, 0.4), 0.5248);
   close(bm.erf(0.5), 0.52049987781304654);
   close(bm.erfc(5), 1.5374597944280349e-12);
   close(bm.expint(1), 1.8951178163559368);
   close(bm.zeta(3), 1.2020569031595943);
   close(bm.cyl_bessel_j(2.5, 4), 0.44088497455734117);
   close(bm.cyl_bessel_k(1, 2), 0.13986588181652243);
   close(bm.sph_bessel(3, 2), 0.060722097662874828);
   close(bm.airy_ai(1), 0.13529241631288142);
   close(bm.ellint_1(0.5), 1.685750354812596);
   close(bm.ellint_2(0.5, 1), 0.96487645426862749);
   close(bm.lambert_w0(1), 0.56714329040978387);
   close(bm.owens_t(0.5, 0.5), 0.064488602847503757);
   // The recurrence's error terms come from two_product.
   close(bm.laguerre(100, 3.5), 0.38531832348347825);
   close(bm.hypergeometric_1F1(1, 2, 1), Math.E - 1);
   close(bm.cyl_bessel_j_zero(0, 1), 2.404825557695773);
});

test('errors return NaN or infinity', () => {
   assert.ok(Number.isNaN(bm.tgamma(-2)));
   assert.ok(Number.isNaN(bm.gamma_p(-1, 2)));
   assert.ok(Number.isNaN(bm.laguerre(-1, 2)));
   assert.ok(Number.isNaN(bm.legendre_p(2.7, 0.5)));
   assert.equal(bm.tgamma(200), Infinity);
   assert.ok(Number.isNaN(new bm.normal_distribution(0, -1).cdf(0)));
   assert.ok(Number.isNaN(new bm.cauchy_distribution(0, 1).mean()));
   assert.ok(Number.isNaN(new bm.geometric_distribution(2.2).mean()));
});

test('distributions', () => {
   close(new bm.normal_distribution(0, 1).cdf(1.5), 0.93319279873114193);
   close(new bm.students_t_distribution(5).ccdf(2), 0.050969739414929178);
   close(new bm.chi_squared_distribution(4).quantile(0.95), 9.4877290367811568);
   close(new bm.poisson_distribution(3).pdf(2), 0.22404180765538774);
   close(new bm.binomial_distribution(10, 0.3).cdf(3), 0.6496107184);
   const g = new bm.gamma_distribution(2.5, 1.5);
   close(g.quantile(g.cdf(4)), 4, 1e-12);
   close(g.mean(), 3.75);
   close(g.variance(), 5.625);
   close(new bm.normal_distribution(0, 1).entropy(), 1.4189385332046727);  // log(2 pi e) / 2
   assert.ok(Number.isNaN(new bm.non_central_t_distribution(5, 1).entropy()));
   assert.deepEqual(new bm.beta_distribution(2, 3).support(), [0, 1]);
   assert.ok(Number.isNaN(new bm.landau_distribution(0, 1).mean()));  // undefined
});

test('fft', () => {
   for (const n of [8, 7, 1000]) {  // radix 2, Bluestein, mixed radix
      const x = Float64Array.from({ length: 2 * n }, (_, i) => Math.sin(i * i));
      const f = new bm.fft(n), y = f.inverse_transform(f.transform(x));
      x.forEach((v, i) => assert.ok(Math.abs(y[i] - v) < 1e-13));
      f.delete();
   }
   // cos(2 pi 3 j / 16) has X[3] = 8 and nothing else.
   const r = new bm.real_fft(16), X = r.transform(Float64Array.from({ length: 16 }, (_, j) => Math.cos(2 * Math.PI * 3 * j / 16)));
   assert.equal(X.length, 18);
   X.forEach((v, i) => assert.ok(Math.abs(v - (i === 6 ? 8 : 0)) < 1e-13, `X[${i >> 1}]`));
   assert.throws(() => r.transform(new Float64Array(15)), RangeError);
   assert.throws(() => new bm.fft(0), RangeError);
   r.delete();
});

test('arrays', () => {
   const xs = Float64Array.from({ length: 101 }, (_, i) => -2 + i / 25);
   const ys = bm.erf_array(xs);
   assert.ok(ys instanceof Float64Array);
   xs.forEach((x, i) => assert.equal(ys[i], bm.erf(x)));
   assert.deepEqual([...bm.gamma_p_array(3, [1, 2])], [bm.gamma_p(3, 1), bm.gamma_p(3, 2)]);
   assert.deepEqual([...bm.ibeta_array(2, 3, [0.4])], [bm.ibeta(2, 3, 0.4)]);
   assert.deepEqual([...bm.laguerre_array(100, [3.5])], [bm.laguerre(100, 3.5)]);
   assert.ok(bm.laguerre_array(-1, [1, 2]).every(Number.isNaN));
   assert.equal(bm.tgamma_array([]).length, 0);
});

// Native Boost references, from clang++ -O2 on arm64.
test('distribution arrays', () => {
   const g = new bm.gamma_distribution(2.5, 1.5);
   const xs = new Float32Array([3, 0.5, 9]);
   [['pdf', xs], ['cdf', xs], ['ccdf', xs], ['quantile', [0.9, 0.1]]].forEach(([f, x]) => {
      const y = g[f + '_array'](x);
      assert.ok(y instanceof Float64Array);
      assert.deepEqual([...y], [...x].map(v => g[f](v)));
   });
   close(g.pdf_array([3])[0], 0.19196788093577977);
   close(g.ccdf_array([3])[0], 0.54941595135278032);
   close(g.quantile_array([0.9])[0], 6.9272676748358384);
   assert.ok(new bm.normal_distribution(0, -1).pdf_array([0, 1]).every(Number.isNaN));
   g.delete();
});

test('power spectrum', () => {
   // 3 cos(2 pi 3 j / 16) has P[3] = 9/2; sum P = mean(x^2) for any n.
   const P = bm.power_spectrum(Float64Array.from({ length: 16 }, (_, j) => 3 * Math.cos(2 * Math.PI * 3 * j / 16)));
   assert.equal(P.length, 9);
   P.forEach((v, k) => assert.ok(Math.abs(v - (k === 3 ? 4.5 : 0)) < 1e-14, `P[${k}]`));
   for (const n of [1, 2, 7, 1000]) {
      const x = Array.from({ length: n }, (_, j) => Math.sin(j * j) + 0.25);
      close(bm.power_spectrum(x).reduce((a, b) => a + b), x.reduce((a, v) => a + v * v, 0) / n, 1e-13);
   }
});

// Lane (o, i) along the axis of a C-order array with the given shape, w numbers per element.
function lane(a, shape, axis, o, i, w) {
   const n = shape[axis], inner = shape.slice(axis + 1).reduce((p, q) => p * q, 1);
   return Array.from({ length: n }, (_, j) => [...a.slice(((o * n + j) * inner + i) * w, ((o * n + j) * inner + i + 1) * w)]).flat();
}

test('transforms along an axis', () => {
   const shape = [3, 4, 5], z = Float64Array.from({ length: 120 }, (_, i) => Math.sin(i * i)), x = z.subarray(0, 60);
   for (const axis of [0, 1, 2]) {
      const n = shape[axis], f = new bm.fft(n), r = new bm.real_fft(n), half = shape.with(axis, (n >> 1) + 1);
      const outer = shape.slice(0, axis).reduce((p, q) => p * q, 1), inner = 60 / outer / n;
      const Z = bm.fft_axis(z, shape, axis), X = bm.real_fft_axis(x, shape, axis), P = bm.power_spectrum_axis(x, shape, axis - 3);
      assert.deepEqual(bm.fft_axis(z, shape, axis - 3), Z);
      bm.inverse_fft_axis(Z, shape, axis).forEach((v, i) => assert.ok(Math.abs(v - z[i]) < 1e-14));
      for (let o = 0; o < outer; ++o)
         for (let i = 0; i < inner; ++i) {
            assert.deepEqual(lane(Z, shape, axis, o, i, 2), [...f.transform(lane(z, shape, axis, o, i, 2))]);
            assert.deepEqual(lane(X, half, axis, o, i, 2), [...r.transform(lane(x, shape, axis, o, i, 1))]);
            assert.deepEqual(lane(P, half, axis, o, i, 1), [...bm.power_spectrum(lane(x, shape, axis, o, i, 1))]);
         }
      f.delete();
      r.delete();
   }
   assert.throws(() => bm.fft_axis(z, shape, 3), RangeError);
   assert.throws(() => bm.fft_axis(z, shape, 0.5), RangeError);
   assert.throws(() => bm.fft_axis(x, shape, 0), RangeError);
   assert.throws(() => bm.real_fft_axis(x, [3, 0, 20], 0), RangeError);
   assert.throws(() => bm.power_spectrum_axis(x, [6, 2.5, 4], 0), RangeError);
   assert.throws(() => bm.power_spectrum([]), RangeError);
});

test('statistics', () => {
   const x = new Float32Array([2.1, 3.7, 1.4, 5.9, 4.2, 3.3, 6.8, 2.6, 4.9, 3.1]);
   const xd = Float64Array.from([2.1, 3.7, 1.4, 5.9, 4.2, 3.3, 6.8, 2.6, 4.9, 3.1]), y = [1.0, 2.9, 0.7, 5.2, 3.1, 2.2, 6.9, 1.8, 4.4, 2.0];
   assert.equal(bm.mean(x), bm.mean([...x]));
   const near = (a, e) => close(a, e, 1e-12);
   near(bm.mean(xd), 3.8);
   near(bm.variance(xd), 2.562);
   near(bm.sample_variance(xd), 2.8466666666666662);
   const ms = bm.mean_and_sample_variance(xd);
   near(ms.mean, 3.8);
   near(ms.sample_variance, 2.8466666666666662);
   near(bm.skewness(xd), 0.40777399964924838);
   near(bm.kurtosis(xd), 2.2219266637049864);
   near(bm.excess_kurtosis(xd), -0.77807333629501363);
   near(bm.median(xd), 3.5);
   assert.equal(xd[0], 2.1);  // not sorted in place
   near(bm.median_absolute_deviation(xd), 1.15);
   near(bm.interquartile_range(xd), 2.3);
   near(bm.gini_coefficient(xd), 0.23789473684210555);
   near(bm.sample_gini_coefficient(xd), 0.2643274853801173);
   near(bm.covariance(xd, y), 2.937);
   near(bm.correlation_coefficient(xd, y), 0.98937381873520702);
   const ols = bm.simple_ordinary_least_squares(xd, y);
   near(ols.intercept, -1.3362060889929757);
   near(ols.slope, 1.146370023419204);
   near(ols.r_squared, 0.97886055319868626);
   const tests = {
      one_sample_t_test: [[xd, 3], 1.4994144055993899, 0.16799963235871171],
      one_sample_z_test: [[xd, 3], 1.4994144055993899, 0.13376615872757328],
      two_sample_t_test: [[xd, y], 0.95517365168784873, 0.35214299405867849],
      two_sample_z_test: [[xd, y], 0.95517365168784873, 0.33948988789717377],
      paired_samples_t_test: [[xd, y], 6.5507423781834921, 0.00010508080594175347],
      runs_above_and_below_threshold: [[xd, 3.5], 2.0124611797498106, 0.044171344908442642],
      runs_above_and_below_median: [[xd], 2.0124611797498106, 0.044171344908442642],
      ljung_box: [[xd, -1, 0], 3.9444372042038913, 0.267526751594658],
   };
   for (const [f, [args, statistic, p]] of Object.entries(tests)) {
      const r = bm[f](...args);
      near(r.statistic, statistic);
      near(r.p_value, p);
   }
   near(bm.ljung_box(xd, 3, 1).p_value, 0.13914779990398185);
   near(bm.anderson_darling_normality_statistic(xd), 0.16543707021126064);
   assert.ok(Number.isNaN(bm.median([1, NaN, 2])));
   assert.ok(Number.isNaN(bm.runs_above_and_below_threshold([1, NaN, 2], 1.5).p_value));
   for (const bad of [() => bm.mean([]), () => bm.sample_variance([1]), () => bm.interquartile_range([1, 2]),
                      () => bm.covariance([1, 2], [1]), () => bm.simple_ordinary_least_squares([1, 1], [1, 2]),
                      () => bm.two_sample_t_test([1, 2], [1]), () => bm.ljung_box(xd, 10, 0), () => bm.ljung_box(xd, 1.5, 0)])
      assert.throws(bad, RangeError);
});

test('boost_math.d.ts declares every export', () => {
   const dts = readFileSync(path.replace(/\.mjs$/, '.d.ts'), 'utf8');
   for (const name of Object.keys(bm))
      assert.match(dts, new RegExp(`^  ${name}[(:]`, 'm'), name);
});

test('trend_filter', () => {
  const n = 256, y = new Float64Array(n);
  for (let i = 0; i < n; ++i) y[i] = (i < n / 2 ? 0 : 1) + 0.1 * Math.sin(37 * i);
  const x0 = bm.trend_filter(y, 0, 2);
  assert.equal(x0.length, n);
  assert.ok(Math.abs(x0[10]) < 0.1 && Math.abs(x0[n - 10] - 1) < 0.1);
  assert.ok(Number.isNaN(bm.trend_filter(y, 3, 1)[0]) && Number.isNaN(bm.trend_filter(y, 1.5, 1)[0]));
  assert.ok(Number.isNaN(bm.trend_filter(y, 1, -1)[0]));
  const x = bm.trend_filter(y);
  assert.equal(x.length, n);
  assert.ok(Math.abs(x[n - 10] - 1) < 0.2);
});
