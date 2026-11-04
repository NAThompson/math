// Copyright Nick Thompson 2026.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/*
 * Google Benchmark suite for the spheroidal wave functions: one evaluation of each function for n = 4, m = 1
 * and c = 1, 4, 16, 64 (the argument is the benchmark's second parameter). Every call computes the eigenvalue
 * and the expansion coefficients, which dominate at small c.
 *
 * Example:
 *   c++ -O2 -DNDEBUG -std=c++14 -Iinclude \
 *     reporting/performance/spheroidal_performance.cpp \
 *     -lbenchmark -lpthread -o spheroidal_bench
 */

#include <boost/math/special_functions/spheroidal.hpp>
#include <benchmark/benchmark.h>

namespace {

using namespace boost::math;

template <class F>
void run(benchmark::State& state, F f, double x)
{
    const double c = static_cast<double>(state.range(0));
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(f(4u, 1u, c, x));
    }
}

void prolate_eigenvalue(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double) { return prolate_spheroidal_eigenvalue(n, m, c); }, 0.0); }
void prolate_angular(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double x) { return prolate_spheroidal_angular(n, m, c, x); }, 0.5); }
void prolate_angular_near_one(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double x) { return prolate_spheroidal_angular(n, m, c, x); }, 0.99); }
void prolate_radial_1(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double x) { return prolate_spheroidal_radial_1(n, m, c, x); }, 1.5); }
void prolate_radial_2(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double x) { return prolate_spheroidal_radial_2(n, m, c, x); }, 1.5); }
void prolate_radial_2_near_one(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double x) { return prolate_spheroidal_radial_2(n, m, c, x); }, 1.001); }
void oblate_angular(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double x) { return oblate_spheroidal_angular(n, m, c, x); }, 0.5); }
void oblate_angular_near_zero(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double x) { return oblate_spheroidal_angular(n, m, c, x); }, 0.05); }
void oblate_radial_1(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double x) { return oblate_spheroidal_radial_1(n, m, c, x); }, 0.5); }
void oblate_radial_2(benchmark::State& state) { run(state, [](unsigned n, unsigned m, double c, double x) { return oblate_spheroidal_radial_2(n, m, c, x); }, 0.5); }

} // namespace

BENCHMARK(prolate_eigenvalue)->RangeMultiplier(4)->Range(1, 64);
BENCHMARK(prolate_angular)->RangeMultiplier(4)->Range(1, 64);
BENCHMARK(prolate_angular_near_one)->RangeMultiplier(4)->Range(1, 64);
BENCHMARK(prolate_radial_1)->RangeMultiplier(4)->Range(1, 64);
BENCHMARK(prolate_radial_2)->RangeMultiplier(4)->Range(1, 64);
BENCHMARK(prolate_radial_2_near_one)->RangeMultiplier(4)->Range(1, 64);
BENCHMARK(oblate_angular)->RangeMultiplier(4)->Range(1, 64);
BENCHMARK(oblate_angular_near_zero)->RangeMultiplier(4)->Range(1, 64);
BENCHMARK(oblate_radial_1)->RangeMultiplier(4)->Range(1, 64);
BENCHMARK(oblate_radial_2)->RangeMultiplier(4)->Range(1, 64);

BENCHMARK_MAIN();
