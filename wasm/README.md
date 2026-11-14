# Boost.Math for WebAssembly

Special functions, distributions and FFTs in double precision, built with [Emscripten](https://emscripten.org):

```sh
emcmake cmake -S wasm -B build-wasm && cmake --build build-wasm
BOOST_MATH_WASM=build-wasm/boost_math.mjs node --test wasm/test/smoke.mjs
```

```js
import createModule from './boost_math.mjs';
const bm = await createModule();
bm.tgamma(4.5);                      // 11.631728396567452
bm.tgamma_array([0.5, 1.5]);         // Float64Array [1.7724538509055159, 0.8862269254527579]
const t = new bm.students_t_distribution(5);
t.quantile(0.975);                   // 2.5705818356363146
t.delete();                          // embind objects live in the wasm heap
```

Arrays in (`Float64Array`, `Float32Array` or plain) give new `Float64Array`s out, which need no freeing. Most
functions have an `_array` twin over their last argument, and every distribution has `pdf_array`, `cdf_array`,
`ccdf_array` and `quantile_array`. Transforms of N-dimensional arrays, stored row major, run along one axis:

```js
bm.power_spectrum(x);                         // P[0..n/2], with sum P = mean(x^2)
bm.real_fft_axis(image, [rows, cols], -1);  // each row; complex (re, im) pairs out, shape [rows, floor(cols/2) + 1]
bm.fft_axis(z, [rows, cols], 0);              // each column of complex (re, im) pairs
bm.one_sample_t_test(x, 0);                   // {statistic, p_value}; also z, two sample, paired, runs, Ljung-Box
bm.simple_ordinary_least_squares(x, y);       // {intercept, slope, r_squared}
bm.trend_filter(y, 1, 10);                    // piecewise linear fit; bm.trend_filter(y) picks k and lambda
bm.wavelet_denoise(y);                        // wavelet and rule by cross-validation; or (y, "symlet", 8, "sure", sigma)
bm.wavelet_transform(y, "daubechies", 4);     // any length: [a_L | d_L | ... | d_1]; inverse_wavelet_transform undoes it
bm.multivariate_halley(x => [F(x), J(x), v => d2(x, v)], x0);  // {root, iterations}; also multivariate_newton
bm.nsga2(x => [f1(x), f2(x)], lower, upper, { seed: 1 });      // {x, f}: the Pareto front; nsga2_optimizer has ask and tell
```

`boost_math.d.ts` documents the whole API and its conventions. The build directory is an npm package (`npm pack`).
Errors return NaN or infinity rather than throwing, except that bad sizes, shapes, axes and `nsga2` parameters
throw a `RangeError`.

`-DBOOST_MATH_WASM_RELAXED_SIMD=ON` gives the double-double arithmetic a hardware fma, but relaxed SIMD needs
Chrome or Edge 114+ or Firefox 146+, and Safari can't load it.
