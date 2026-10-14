// Diagnostic FP64 bicubic function evaluation. No fast-math or contraction.
#include <cmath>
#include <cstdint>

static inline double cubic(const double* a, double x) {
    return ((a[3]*x+a[2])*x+a[1])*x+a[0];
}

static inline void jet(const double* a, double s, double t,
                       double& value, double& ds, double& dt) {
    double row[4], derivative[4];
    for (int i=0; i<4; ++i) {
        const double* r=a+4*i;
        row[i]=cubic(r,t);
        derivative[i]=(3*r[3]*t+2*r[2])*t+r[1];
    }
    value=cubic(row,s);
    ds=(3*row[3]*s+2*row[2])*s+row[1];
    dt=cubic(derivative,s);
}

extern "C" std::int64_t native_element_function_abi() { return 1; }

extern "C" std::int64_t native_element_function_sample(
    std::int64_t n, const std::int64_t* elements, const double* st,
    const double* geometry, const double* fields, const double* feedback,
    double* values, double* gradients, double* projected, double* coordinates) {
    for (std::int64_t p=0; p<n; ++p) {
        const std::int64_t e=elements[p];
        const double s=st[2*p], t=st[2*p+1];
        double r,rs,rt,z,zs,zt;
        jet(geometry+e*32,s,t,r,rs,rt);
        jet(geometry+e*32+16,s,t,z,zs,zt);
        const double jac=rs*zt-rt*zs;
        if (!std::isfinite(jac) || jac<=0 || !std::isfinite(r) ||
            !std::isfinite(z) || r<=0) return p+1;
        coordinates[2*p]=r; coordinates[2*p+1]=z;
        // Only channels consumed by the native marker formula are returned.
        for (int f=0; f<9; ++f) {
            const bool need_value=(f==4 || f==6 || f==7 || f==8);
            const bool need_gradient=(f==0 || f==1 || f==7);
            if (!need_value && !need_gradient) continue;
            double v,vs,vt;
            jet(fields+(e*9+f)*16,s,t,v,vs,vt);
            if (need_value) values[p*9+f]=v;
            if (need_gradient) {
                gradients[(p*9+f)*2]=(zt*vs-zs*vt)/jac;
                gradients[(p*9+f)*2+1]=(-rt*vs+rs*vt)/jac;
            }
        }
        for (int f=3; f<5; ++f) {
            const double* a=feedback+(e*5+f)*16;
            double row[4];
            for (int i=0; i<4; ++i) row[i]=cubic(a+4*i,t);
            projected[p*5+f]=cubic(row,s);
        }
    }
    return 0;
}
