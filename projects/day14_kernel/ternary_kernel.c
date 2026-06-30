/* AetherCore Phase-1 kernel: packed 2-bit ternary mat-vec for CPU decode.
 *
 * Computes  y[n] = scale[n] * sum_k sign(W[k,n]) * x[k]
 * where W is ternary {-1,0,+1} packed at 2 bits/weight, column-major:
 *   packed: N columns, each ceil(K/4) bytes; code 0=zero, 1=+1, 2=-1.
 *
 * The win is BANDWIDTH: weights resident at ~2 bits (16x less than fp32) and the
 * fp32 weight matrix is never materialised. The sign is turned into +x/-x/skip
 * (no float multiply by the weight). gcc -O3 -mavx2 -ffast-math auto-vectorises
 * the branch-free inner reduction.
 *
 * Build (MinGW-w64):
 *   gcc -O3 -mavx2 -mfma -funroll-loops -ffast-math -shared -o libternary.dll ternary_kernel.c
 */

#include <stdint.h>

/* single-token decode: x[K] -> y[N] */
void ternary_matvec(const float* x, const uint8_t* packed, const float* scale,
                    float* y, int K, int N)
{
    const int KB = (K + 3) / 4;          /* bytes per column */
    for (int n = 0; n < N; ++n) {
        const uint8_t* col = packed + (size_t)n * KB;
        float acc = 0.0f;
        int k = 0;
        for (int b = 0; b < KB; ++b) {
            uint8_t byte = col[b];
            /* unpack 4 ternary codes; sign = (c==1) - (c==2) in {-1,0,+1} */
            int c0 =  byte        & 3;
            int c1 = (byte >> 2)  & 3;
            int c2 = (byte >> 4)  & 3;
            int c3 = (byte >> 6)  & 3;
            float s0 = (float)((c0 == 1) - (c0 == 2));
            float s1 = (float)((c1 == 1) - (c1 == 2));
            float s2 = (float)((c2 == 1) - (c2 == 2));
            float s3 = (float)((c3 == 1) - (c3 == 2));
            if (k + 3 < K) {
                acc += s0 * x[k] + s1 * x[k+1] + s2 * x[k+2] + s3 * x[k+3];
            } else {
                if (k   < K) acc += s0 * x[k];
                if (k+1 < K) acc += s1 * x[k+1];
                if (k+2 < K) acc += s2 * x[k+2];
                if (k+3 < K) acc += s3 * x[k+3];
            }
            k += 4;
        }
        y[n] = acc * scale[n];
    }
}

/* batched: X[M,K] row-major -> Y[M,N] row-major */
void ternary_matmul(const float* X, const uint8_t* packed, const float* scale,
                    float* Y, int M, int K, int N)
{
    for (int m = 0; m < M; ++m)
        ternary_matvec(X + (size_t)m * K, packed, scale, Y + (size_t)m * N, K, N);
}
