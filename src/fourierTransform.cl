#include <math.h>
#define pi2 3.1415

typedef struct {
    double x,y;
} complex;

complex cmul(complex a, complex b) {
   return (complex){ a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x };
}

complex cadd(complex a, complex b) {
    return (complex){ a.x+b.x, a.y+b.y };
}

complex cdiv(complex a, double d) {
    return (complex){ a.x*d, a.y*d };
}

complex csub(complex a, complex b) {
    return (complex){ a.x-b.x, a.y-b.y };
}

complex ekN(long k, long N){
    double x = -pi2*((double)k)/N;
    return (complex){ cos(x), sin(x) };
}

void FFT(long N, const complex *v, complex *res){
    //TODO add security to check if v.size() is a power of 2
    long N2 = N/2;
    complex tmp[N];
    for(long i = 0;i<N;i++){
        res[i] = v[i];
    }

    //reorder the values
    for(long revi = N2;revi>0;revi >>= 1){
        for(long j = 0;j<N2;j++){
            long revk = j + (j/revi)*revi;
            tmp[revk] = res[2*j];
            tmp[revk+revi] = res[2*j+1];
        }
        for(long i = 0;i<N;i++) {
            res[i] = tmp[i];
        }
    }

    //compute the fft
    for(long i = 1;i<N;i <<= 1){
        long i2 = 2*i;
        for(long j = 0;j<N2;j++){
            long k = j + (j/i)*i;
            complex e = res[k], oexp = cmul(res[k+i],ekN(j%i, i2));
            res[k] = cadd(e, oexp);
            res[k+i] = csub(e, oexp);
        }
    }

    //normalize the fft
    for(long i = 0;i<N;i++)
        res[i] = cdiv(res[i],N);
}
