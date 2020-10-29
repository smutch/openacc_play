#include <cmath>
#include <cstdio>
#include <vector>
#include <chrono>

void sumit(const std::vector<float>& __restrict x, std::vector<float>& __restrict y) {
    const int N = x.size();
#pragma acc parallel loop
    for (int ii=0; ii < N; ++ii)
        y[ii] += x[ii];
}

int main(int argc, char *argv[])
{

    const int N = 5000000;
    const int loops = 1000;

    std::vector<float> x(N, 1.0f);
    std::vector<float> y(N, 2.0f);

    {
        auto time_start = std::chrono::high_resolution_clock::now();

        for(int ii=0; ii < loops; ++ii)
            sumit(x, y);

        auto time_stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start) / 100;

        auto duration_ms = static_cast<double>(duration.count());
        printf("duration = %.2f µs (%.2f μs per loop)\n", duration_ms, duration_ms/loops);
    }

    auto total = 0.0;
    for (auto v: y)
        total += static_cast<double>(v);

    constexpr auto expected = (2.0+static_cast<double>(loops)) * static_cast<double>(N);
    printf("%-10s = %.0f\n%-10s = %.0f\n", "total", total, "expected", expected);

    return 0;
}
