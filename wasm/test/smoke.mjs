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
   close(bm.fresnel_c(1.5), 0.4452611760398215);
   close(bm.fresnel_s(1.5), 0.6975049600820931);
   const cs = bm.fresnel_cis_array([1.5, -1000.3]);
   assert.equal(cs.length, 4);
   close(cs[0], 0.4452611760398215);
   close(cs[1], 0.6975049600820931);
   close(cs[2], -0.5000448367037948);
   close(cs[3], -0.49968496017986497);
   assert.deepEqual([...bm.fresnel_cis(1.5)], [bm.fresnel_c(1.5), bm.fresnel_s(1.5)]);
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
   near(bm.hoyer_sparsity(xd), 0.11468560732221353);
   near(bm.hoyer_sparsity([0, 0, 3, 0]), 1);
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

test('wavelet_denoise, wavelet_transform, denoise', () => {
  // An odd, non-dyadic length: the transform on the interval needs no padding.
  const n = 301, f = new Float64Array(n), y = new Float64Array(n);
  let seed = 12345;
  const u = () => (seed = (1103515245 * seed + 12345) % 2147483648) / 2147483648;
  for (let i = 0; i < n; ++i) { f[i] = i < 150 ? 0 : 4; y[i] = f[i] + 0.3 * Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u()); }
  const c = bm.wavelet_transform(y, 'symlet', 4);
  assert.equal(c.length, n);
  let e2 = 0, c2 = 0;
  for (let i = 0; i < n; ++i) { e2 += y[i] ** 2; c2 += c[i] ** 2; }
  close(c2, e2, 1e-12);  // orthonormal
  const back = bm.inverse_wavelet_transform(c, 'symlet', 4);
  for (let i = 0; i < n; ++i) assert.ok(Math.abs(back[i] - y[i]) < 1e-12);
  assert.ok(Math.abs(bm.inverse_wavelet_transform(bm.wavelet_transform(y, 'daubechies', 3, 2), 'daubechies', 3, 2)[7] - y[7]) < 1e-12);
  const mse = (x) => x.reduce((s, v, i) => s + (v - f[i]) ** 2, 0) / n;
  const noisy = mse(y);
  for (const x of [bm.wavelet_denoise(y), bm.wavelet_denoise(y, 'daubechies', 2), bm.wavelet_denoise(y, 'symlet', 6, 'sure'),
                   bm.wavelet_denoise(y, 'symlet', 6, 'universal_soft', 0.3), bm.denoise(y)]) {
    assert.equal(x.length, n);
    assert.ok(mse(x) < 0.4 * noisy, `${mse(x)} vs ${noisy}`);  // universal_soft, the weakest at the jump, gets 0.3
  }
  for (const bad of [bm.wavelet_transform(y, 'haar', 4), bm.wavelet_transform(y, 'symlet', 11), bm.wavelet_transform(y, 'symlet', 4, 99),
                     bm.wavelet_denoise(y, 'symlet', 4, 'hard'), bm.wavelet_denoise(new Float64Array(5))])
    assert.ok(Number.isNaN(bad[0]));
});

test('multivariate_newton and multivariate_halley', () => {
  // x^2 + y^2 = 4 and x = y: the root (sqrt 2, sqrt 2) from (1, 2).
  const circle = (x) => [[x[0] ** 2 + x[1] ** 2 - 4, x[0] - x[1]], [[2 * x[0], 2 * x[1]], [1, -1]],
                         (v) => [2 * v[0] ** 2 + 2 * v[1] ** 2, 0]];
  for (const solve of [bm.multivariate_newton, bm.multivariate_halley]) {
    const { root, iterations } = solve(circle, [1, 2]);
    close(root[0], Math.SQRT2, 2e-16);
    close(root[1], Math.SQRT2, 2e-16);
    assert.ok(iterations > 0 && iterations < 10);
  }
  // J flat row major, Newton only (no d2): z^3 = 1 as a real 2x2 system, from near the root exp(2 pi i / 3).
  const cube = (x) => {
    const [a, b] = x;
    return [[a ** 3 - 3 * a * b * b - 1, 3 * a * a * b - b ** 3], [3 * a * a - 3 * b * b, -6 * a * b, 6 * a * b, 3 * a * a - 3 * b * b]];
  };
  const r = bm.multivariate_newton(cube, [-0.4, 0.8], 53, 100).root;
  close(r[0], -0.5, 2e-16);
  close(r[1], Math.sqrt(3) / 2, 2e-16);
  assert.throws(() => bm.multivariate_halley(cube, [-0.4, 0.8]), RangeError);
  assert.throws(() => bm.multivariate_newton((x) => [[x[0]], [[1, 0], [0, 1]]], [1, 2]), RangeError);
  assert.throws(() => bm.multivariate_newton(() => { throw new TypeError('mine'); }, [1]), TypeError);
  // x^2 + 1 = 0 has no real root: NaN.
  assert.ok(Number.isNaN(bm.multivariate_newton((x) => [[x[0] ** 2 + 1], [2 * x[0]]], [0.5]).root[0]));
  // Halley's steps more than Newton's are a cubic vs quadratic rate: fewer iterations from far away.
  const far = [3, 0.5];
  assert.ok(bm.multivariate_halley(circle, far).iterations < bm.multivariate_newton(circle, far).iterations);
});

// ZDT1 in 10 variables, whose front is f2 = 1 - sqrt(f1); the hypervolume of the front against (1.1, 1.1) is 1.21 - 1/3.
function zdt1(x) {
  const g = 1 + 9 * x.slice(1).reduce((a, b) => a + b, 0) / (x.length - 1);
  return [x[0], g * (1 - Math.sqrt(x[0] / g))];
}
const dominates = (a, b) => a.every((v, i) => v <= b[i]) && a.some((v, i) => v < b[i]);
// f is in increasing order of f1, so decreasing in f2.
const hypervolume = (f) => f.reduce((s, v, i) => s + (1.1 - v[0]) * ((i ? f[i - 1][1] : 1.1) - v[1]), 0);
function check_front({ x, f }, objective, lower, upper) {
  assert.ok(x.length > 0 && x.length === f.length);
  x.forEach((p, i) => {
    assert.ok(p instanceof Float64Array && f[i] instanceof Float64Array);
    assert.ok(p.every((v, k) => v >= lower[k] && v <= upper[k]));
    assert.deepEqual([...f[i]], [...objective(p)]);
    assert.ok(!f.some((g) => dominates(g, f[i])));
  });
}

test('nsga2', () => {
  const lower = new Array(10).fill(0), upper = new Float64Array(10).fill(1);
  const options = { population_size: 40, max_generations: 100, seed: 3 };
  const front = bm.nsga2(zdt1, lower, upper, options);
  check_front(front, zdt1, lower, upper);
  assert.ok(front.x.length >= 20 && front.x.length <= 40);
  assert.ok(hypervolume(front.f) > 0.8, `${hypervolume(front.f)}`);
  assert.deepEqual(bm.nsga2(zdt1, lower, upper, options), front);
  assert.notDeepEqual(bm.nsga2(zdt1, lower, upper, { ...options, seed: 4 }), front);
  // The defaults, on Schaffer's problem, whose Pareto set is 0 <= x <= 2; then a warm start from its front.
  const schaffer = (x) => [x[0] ** 2, (x[0] - 2) ** 2];
  const s = bm.nsga2(schaffer, [-10], [10]);
  check_front(s, schaffer, [-10], [10]);
  assert.ok(s.x.length > 50 && s.x.every((p) => p[0] > -0.01 && p[0] < 2.01));
  const warm = bm.nsga2(schaffer, [-10], [10], { max_generations: 1, initial_population: s.x, mutation_probability: NaN, seed: null });
  assert.ok(warm.x.every((p) => p[0] > -0.01 && p[0] < 2.01));
  for (const bad of [{ population_size: 3 }, { population_size: 4.5 }, { max_generations: 0 }, { crossover_probability: 2 },
                     { mutation_probability: -0.1 }, { crossover_distribution_index: -1 }, { mutation_distribution_index: Infinity },
                     { seed: -1 }, { seed: 0.5 }, { threads: 2 }, { populationSize: 50 }, { initial_population: [[11]] },
                     { initial_population: [[NaN]] }, { initial_population: [[0, 0]] }, { initial_population: [1, 2, 3, 4, 5], population_size: 4 }])
    assert.throws(() => bm.nsga2(schaffer, [-10], [10], bad), RangeError, JSON.stringify(bad));
  for (const [lo, hi] of [[[], []], [[0], [1, 2]], [[1], [0]], [[0], [Infinity]], [[NaN], [1]]])
    assert.throws(() => bm.nsga2(schaffer, lo, hi), RangeError);
  assert.throws(() => bm.nsga2(() => [], [-10], [10]), RangeError);
  assert.throws(() => bm.nsga2((x) => (x[0] > 0 ? [1, 2] : [1]), [-10], [10]), RangeError);
  assert.throws(() => bm.nsga2(() => { throw new TypeError('mine'); }, [-10], [10]), TypeError);
  assert.throws(() => bm.nsga2(() => undefined, [-10], [10]));
  // Errors don't use up the stack, and one caught inside f leaves the run as it was.
  for (let i = 0; i < 1000; ++i) {
    assert.throws(() => bm.nsga2(schaffer, [-10], [10], { population_size: 3 }), RangeError);
    assert.throws(() => bm.nsga2((x) => (x[0] > 0 ? [1, 2] : [1]), [-10], [10]), RangeError);
    assert.throws(() => bm.nsga2(() => { throw new TypeError('mine'); }, [-10], [10]), TypeError);
  }
  const careless = (x) => { assert.throws(() => bm.mean([]), RangeError); return zdt1(x); };
  assert.deepEqual(bm.nsga2(careless, lower, upper, options), front);
});

test('nsga2_optimizer', () => {
  const lower = new Array(10).fill(0), upper = new Array(10).fill(1);
  const run = (flat) => {
    const opt = new bm.nsga2_optimizer(lower, upper, { population_size: 40, seed: 3 });
    assert.equal(opt.front().x.length + opt.population().length + opt.objectives().length + opt.ranks().length, 0);
    assert.throws(() => opt.tell(new Float64Array(80)), RangeError);  // nothing was asked for
    for (let generation = 0; generation <= 100; ++generation) {
      const x = opt.ask();
      assert.equal(x.length, 40);
      assert.deepEqual(opt.ask(), x);
      const f = x.map(zdt1);
      opt.tell(flat ? Float64Array.from(f.flat()) : f);
    }
    return opt;
  };
  const opt = run(false), front = opt.front();
  check_front(front, zdt1, lower, upper);
  assert.ok(hypervolume(front.f) > 0.8, `${hypervolume(front.f)}`);
  const flat = run(true);
  assert.deepEqual(flat.front(), front);
  flat.delete();
  const x = opt.population(), f = opt.objectives(), ranks = opt.ranks();
  assert.ok(ranks instanceof Uint32Array && x.length === 40 && f.length === 40 && ranks.length === 40);
  x.forEach((p, i) => assert.deepEqual([...f[i]], zdt1(p)));
  assert.ok(ranks.every((r, i) => r >= (i ? ranks[i - 1] : 0)) && ranks.filter((r) => r === 0).length >= front.x.length);
  // A wrong number of points or of objectives leaves the optimizer as it was.
  const points = opt.ask();
  for (const bad of [points.slice(1).map(zdt1), points.map((p) => [p[0]]), points.map(() => []), new Float64Array(79), [], 7,
                     points.map((p, i) => (i === 5 ? [1, 2, 3] : zdt1(p)))])
    assert.throws(() => opt.tell(bad), RangeError);
  assert.deepEqual(opt.ask(), points);
  opt.tell(points.map(zdt1));
  check_front(opt.front(), zdt1, lower, upper);
  opt.delete();
  // The rows given come first in the first generation.
  const warm = new bm.nsga2_optimizer(lower, upper, { population_size: 40, initial_population: front.x.slice(0, 10) });
  assert.deepEqual(warm.ask().slice(0, 10), front.x.slice(0, 10));
  warm.delete();
  const defaults = new bm.nsga2_optimizer([-10], [10]);
  assert.equal(defaults.ask().length, 100);
  defaults.delete();
  for (const bad of [{ population_size: 3 }, { max_generations: 10 }, { initial_population: [new Array(10).fill(2)] },
                     { initial_population: new Array(41).fill(lower), population_size: 40 }])
    assert.throws(() => new bm.nsga2_optimizer(lower, upper, bad), RangeError, JSON.stringify(bad));
  assert.throws(() => new bm.nsga2_optimizer([1], [0]), RangeError);
});
