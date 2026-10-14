// Immutable ADAS tables, independent axes per table; no changed physical law.
// Only bilinear log coefficients are compiled. NumPy retains rowwise power.
#include <algorithm>
#include <cstdint>

extern "C" std::int64_t native_atomic_interpolation_abi() { return 1; }

extern "C" std::int64_t native_atomic_log_interpolate(
    std::int64_t count, std::int64_t tables, std::int64_t data_size,
    const std::int64_t* meta, const double* data,
    const double* logn, const double* logt, double* output) {
    if (count < 0 || (tables != 3 && tables != 4)) return 1;
    for (std::int64_t k = 0; k < tables; ++k) {
        const auto* m = meta + 5*k;
        if (m[0] < 0 || m[1] < 2 || m[2] < 0 || m[3] < 2 || m[4] < 0 ||
            m[0] + m[1] > data_size || m[2] + m[3] > data_size ||
            m[4] + 10*m[1]*m[3] > data_size) return 1;
    }
    for (std::int64_t p = 0; p < count; ++p) {
        for (std::int64_t k = 0; k < tables; ++k) {
            const auto* m = meta + 5*k;
            const auto nn = m[1], nt = m[3];
            const double* na = data + m[0];
            const double* ta = data + m[2];
            const double nq = std::min(std::max(logn[p], na[0]), na[nn-1]);
            const double tq = std::min(std::max(logt[p], ta[0]), ta[nt-1]);
            // NumPy searchsorted(side='left') - 1, including exact knots.
            const auto i = std::min(std::max<std::int64_t>(
                std::lower_bound(na, na+nn, nq)-na-1, 0), nn-2);
            const auto j = std::min(std::max<std::int64_t>(
                std::lower_bound(ta, ta+nt, tq)-ta-1, 0), nt-2);
            const double dn = (nq-na[i])/(na[i+1]-na[i]);
            const double dt = (tq-ta[j])/(ta[j+1]-ta[j]);
            const double* coefficients = data + m[4];
            for (std::int64_t z = 0; z < 10; ++z) {
                const double* c = coefficients + (z*nt+j)*nn+i;
                // Same four terms and left-associative sums as the row oracle.
                // Compile with -fno-fast-math -ffp-contract=off.
                output[(k*10+z)*count+p] =
                    c[0]*(1-dn)*(1-dt) + c[1]*dn*(1-dt)
                    + c[nn]*(1-dn)*dt + c[nn+1]*dn*dt;
            }
        }
    }
    return 0;
}
