#pragma once
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <immintrin.h>
#include <fstream>
#include <random>

struct Tensor {
    int rows, cols;
    std::vector<float> val, grad;

    Tensor(int r = 0, int c = 0)
        : rows(r), cols(c), val(r * c, 0.0f), grad(r * c, 0.0f) {}

    inline float& operator()(int i, int j) {
        assert(i >= 0 && i < rows && j >= 0 && j < cols);
        return val[i * cols + j];
    }

    inline const float& operator()(int i, int j) const {
        assert(i >= 0 && i < rows && j >= 0 && j < cols);
        return val[i * cols + j];
    }

    void zero_grad() {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }

    void randomize(float s = 0.02f) {
        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> d(-s, s);
        for (auto& x : val) x = d(gen);
    }

    static float dot_simd(const float* a, const float* b, int n) {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;

        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
        #ifdef __FMA__
            acc = _mm256_fmadd_ps(va, vb, acc);
        #else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        #endif
        }

        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);

        float sum =
            tmp[0] + tmp[1] + tmp[2] + tmp[3] +
            tmp[4] + tmp[5] + tmp[6] + tmp[7];

        for (; i < n; ++i)
            sum += a[i] * b[i];

        return sum;
    }

    static Tensor transpose(const Tensor& t) {
        Tensor out(t.cols, t.rows);
        for (int i = 0; i < t.rows; ++i)
            for (int j = 0; j < t.cols; ++j)
                out(j, i) = t(i, j);
        return out;
    }

    static Tensor matmul(const Tensor& A, const Tensor& B) {
        assert(A.cols == B.rows);

        Tensor Bt = transpose(B);
        Tensor C(A.rows, B.cols);

        for (int i = 0; i < A.rows; ++i) {
            const float* arow = &A.val[i * A.cols];
            for (int j = 0; j < B.cols; ++j) {
                const float* brow = &Bt.val[j * Bt.cols];
                C(i, j) = dot_simd(arow, brow, A.cols);
            }
        }

        return C;
    }

    void save(std::ofstream& f) const {
        f.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        f.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        f.write(reinterpret_cast<const char*>(val.data()),
                val.size() * sizeof(float));
    }

    void load(std::ifstream& f) {
        f.read(reinterpret_cast<char*>(&rows), sizeof(int));
        f.read(reinterpret_cast<char*>(&cols), sizeof(int));
        val.resize(rows * cols);
        grad.assign(rows * cols, 0.0f);
        f.read(reinterpret_cast<char*>(val.data()),
               val.size() * sizeof(float));
    }
};
