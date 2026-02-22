package kireiko.dev.millennium.ml.logic.rnn.layers;

import kireiko.dev.millennium.ml.logic.rnn.util.MathOps;

import java.util.Random;

public final class LSTMLayer {
    public final int inputSize;
    public final int hiddenSize;

    public final double[] Wf, Wi, Wc, Wo;
    public final double[] Uf, Ui, Uc, Uo;
    public final double[] bf, bi, bc, bo;

    public final double[] lnGamma, lnBeta;
    private final int lnOffset;

    public static final class Cache {
        public int T;
        public int[] stepToTime;
        public double[][] xByStep;
        public double[][] h;
        public double[][] c;
        public double[][] f, i, g, o;
        public double[][] tanhC;
        public double[][] preLN;
        public LayerNorm.Cache[] lnCache;
        public double[] recDropMask;
    }

    public LSTMLayer(int inputSize, int hiddenSize, Random rng, int lnOffset) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.lnOffset = lnOffset;

        Wf = MathOps.xavierUniform(rng, hiddenSize * inputSize, inputSize, hiddenSize);
        Wi = MathOps.xavierUniform(rng, hiddenSize * inputSize, inputSize, hiddenSize);
        Wc = MathOps.xavierUniform(rng, hiddenSize * inputSize, inputSize, hiddenSize);
        Wo = MathOps.xavierUniform(rng, hiddenSize * inputSize, inputSize, hiddenSize);

        Uf = MathOps.orthogonal(rng, hiddenSize);
        Ui = MathOps.orthogonal(rng, hiddenSize);
        Uc = MathOps.orthogonal(rng, hiddenSize);
        Uo = MathOps.orthogonal(rng, hiddenSize);

        bf = new double[hiddenSize];
        bi = new double[hiddenSize];
        bc = new double[hiddenSize];
        bo = new double[hiddenSize];

        for (int k = 0; k < hiddenSize; k++) bf[k] = 1.0;

        lnGamma = new double[hiddenSize];
        lnBeta = new double[hiddenSize];
        for (int k = 0; k < hiddenSize; k++) lnGamma[k] = 1.0;
    }

    public double[][] forward(double[][] xTime, boolean reverse, boolean training, double recurrentDropoutRate, Random rng, Cache cache) {
        int T = xTime.length;
        double[][] out = new double[T][hiddenSize];

        int[] stepToTime = new int[T];
        if (!reverse) {
            for (int s = 0; s < T; s++) stepToTime[s] = s;
        } else {
            for (int s = 0; s < T; s++) stepToTime[s] = T - 1 - s;
        }

        double[] hPrev = new double[hiddenSize];
        double[] cPrev = new double[hiddenSize];

        double[] recMask = null;
        if (training && recurrentDropoutRate > 0) {
            recMask = new double[hiddenSize];
            double keep = 1.0 - recurrentDropoutRate;
            for (int i = 0; i < hiddenSize; i++) recMask[i] = rng.nextDouble() < keep ? (1.0 / keep) : 0.0;
        }

        double[][] h = cache != null ? new double[T + 1][hiddenSize] : null;
        double[][] c = cache != null ? new double[T + 1][hiddenSize] : null;

        double[][] f = cache != null ? new double[T][hiddenSize] : null;
        double[][] it = cache != null ? new double[T][hiddenSize] : null;
        double[][] g = cache != null ? new double[T][hiddenSize] : null;
        double[][] o = cache != null ? new double[T][hiddenSize] : null;

        double[][] tanhC = cache != null ? new double[T][hiddenSize] : null;
        double[][] preLN = cache != null ? new double[T][hiddenSize] : null;
        LayerNorm.Cache[] lnCache = cache != null ? new LayerNorm.Cache[T] : null;
        double[][] xByStep = cache != null ? new double[T][inputSize] : null;

        if (cache != null) {
            System.arraycopy(hPrev, 0, h[0], 0, hiddenSize);
            System.arraycopy(cPrev, 0, c[0], 0, hiddenSize);
        }

        for (int s = 0; s < T; s++) {
            int t = stepToTime[s];
            double[] x = xTime[t];

            double[] hIn = hPrev;
            if (training && recMask != null) {
                double[] tmp = new double[hiddenSize];
                for (int i = 0; i < hiddenSize; i++) tmp[i] = hPrev[i] * recMask[i];
                hIn = tmp;
            }

            double[] ft = gateSigmoid(x, hIn, Wf, Uf, bf);
            double[] ii = gateSigmoid(x, hIn, Wi, Ui, bi);
            double[] gg = gateTanh(x, hIn, Wc, Uc, bc);
            double[] oo = gateSigmoid(x, hIn, Wo, Uo, bo);

            double[] cNow = new double[hiddenSize];
            double[] tanhNow = new double[hiddenSize];
            double[] pre = new double[hiddenSize];

            for (int k = 0; k < hiddenSize; k++) {
                cNow[k] = ft[k] * cPrev[k] + ii[k] * gg[k];
                tanhNow[k] = Math.tanh(cNow[k]);
                pre[k] = oo[k] * tanhNow[k];
            }

            LayerNorm.Cache lnc = cache != null ? (lnCache[s] = new LayerNorm.Cache()) : null;
            double[] hNow = LayerNorm.forward(pre, lnGamma, lnBeta, 0, lnc);

            out[t] = hNow;

            hPrev = hNow;
            cPrev = cNow;

            if (cache != null) {
                System.arraycopy(hNow, 0, h[s + 1], 0, hiddenSize);
                System.arraycopy(cNow, 0, c[s + 1], 0, hiddenSize);
                f[s] = ft;
                it[s] = ii;
                g[s] = gg;
                o[s] = oo;
                tanhC[s] = tanhNow;
                preLN[s] = pre;
                System.arraycopy(x, 0, xByStep[s], 0, inputSize);
            }
        }

        if (cache != null) {
            cache.T = T;
            cache.stepToTime = stepToTime;
            cache.xByStep = xByStep;
            cache.h = h;
            cache.c = c;
            cache.f = f;
            cache.i = it;
            cache.g = g;
            cache.o = o;
            cache.tanhC = tanhC;
            cache.preLN = preLN;
            cache.lnCache = lnCache;
            cache.recDropMask = recMask;
        }

        return out;
    }

    public static final class Grad {
        public final double[] dWf, dWi, dWc, dWo;
        public final double[] dUf, dUi, dUc, dUo;
        public final double[] dbf, dbi, dbc, dbo;
        public final double[] dLnGamma, dLnBeta;

        public Grad(int inputSize, int hiddenSize) {
            int wSize = hiddenSize * inputSize;
            int uSize = hiddenSize * hiddenSize;
            dWf = new double[wSize]; dWi = new double[wSize]; dWc = new double[wSize]; dWo = new double[wSize];
            dUf = new double[uSize]; dUi = new double[uSize]; dUc = new double[uSize]; dUo = new double[uSize];
            dbf = new double[hiddenSize]; dbi = new double[hiddenSize]; dbc = new double[hiddenSize]; dbo = new double[hiddenSize];
            dLnGamma = new double[hiddenSize]; dLnBeta = new double[hiddenSize];
        }
    }

    public double[][] backward(Cache cch, double[][] dH_time, Grad gAcc) {
        int T = cch.T;
        double[][] dX_time = new double[T][inputSize];

        double[] dhNext = new double[hiddenSize];
        double[] dcNext = new double[hiddenSize];

        for (int s = T - 1; s >= 0; s--) {
            int t = cch.stepToTime[s];

            double[] dh = new double[hiddenSize];
            for (int k = 0; k < hiddenSize; k++) dh[k] = dH_time[t][k] + dhNext[k];

            double[] dPre = LayerNorm.backward(dh, lnGamma, 0, cch.lnCache[s], gAcc.dLnGamma, gAcc.dLnBeta);

            double[] ft = cch.f[s];
            double[] it = cch.i[s];
            double[] gg = cch.g[s];
            double[] oo = cch.o[s];
            double[] tanhC = cch.tanhC[s];

            double[] cPrev = cch.c[s];
            double[] hPrev = cch.h[s];

            double[] dOo = new double[hiddenSize];
            double[] dFt = new double[hiddenSize];
            double[] dIt = new double[hiddenSize];
            double[] dGg = new double[hiddenSize];
            double[] dc = new double[hiddenSize];

            for (int k = 0; k < hiddenSize; k++) {
                dOo[k] = dPre[k] * tanhC[k];
                dc[k] = dcNext[k] + dPre[k] * oo[k] * (1.0 - tanhC[k] * tanhC[k]);
                dFt[k] = dc[k] * cPrev[k];
                dIt[k] = dc[k] * gg[k];
                dGg[k] = dc[k] * it[k];
            }

            for (int k = 0; k < hiddenSize; k++) {
                dOo[k] = dOo[k] * oo[k] * (1.0 - oo[k]);
                dFt[k] = dFt[k] * ft[k] * (1.0 - ft[k]);
                dIt[k] = dIt[k] * it[k] * (1.0 - it[k]);
                dGg[k] = dGg[k] * (1.0 - gg[k] * gg[k]);
            }

            double[] x = cch.xByStep[s];

            for (int n = 0; n < hiddenSize; n++) {
                gAcc.dbf[n] += dFt[n];
                gAcc.dbi[n] += dIt[n];
                gAcc.dbc[n] += dGg[n];
                gAcc.dbo[n] += dOo[n];

                int wOff = n * inputSize;
                for (int k = 0; k < inputSize; k++) {
                    gAcc.dWf[wOff + k] += dFt[n] * x[k];
                    gAcc.dWi[wOff + k] += dIt[n] * x[k];
                    gAcc.dWc[wOff + k] += dGg[n] * x[k];
                    gAcc.dWo[wOff + k] += dOo[n] * x[k];
                }

                int uOff = n * hiddenSize;
                for (int k = 0; k < hiddenSize; k++) {
                    double hpk = hPrev[k];
                    if (cch.recDropMask != null) hpk *= cch.recDropMask[k];
                    gAcc.dUf[uOff + k] += dFt[n] * hpk;
                    gAcc.dUi[uOff + k] += dIt[n] * hpk;
                    gAcc.dUc[uOff + k] += dGg[n] * hpk;
                    gAcc.dUo[uOff + k] += dOo[n] * hpk;
                }
            }

            for (int k = 0; k < inputSize; k++) {
                double sdx = 0.0;
                for (int n = 0; n < hiddenSize; n++) {
                    int wOff = n * inputSize + k;
                    sdx += Wf[wOff] * dFt[n];
                    sdx += Wi[wOff] * dIt[n];
                    sdx += Wc[wOff] * dGg[n];
                    sdx += Wo[wOff] * dOo[n];
                }
                dX_time[t][k] += sdx;
            }

            double[] dhPrev = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++) {
                double sdh = 0.0;
                for (int n = 0; n < hiddenSize; n++) {
                    int idx = n * hiddenSize + j;
                    sdh += Uf[idx] * dFt[n];
                    sdh += Ui[idx] * dIt[n];
                    sdh += Uc[idx] * dGg[n];
                    sdh += Uo[idx] * dOo[n];
                }
                if (cch.recDropMask != null) sdh *= cch.recDropMask[j];
                dhPrev[j] = sdh;
            }

            for (int k = 0; k < hiddenSize; k++) {
                dhNext[k] = dhPrev[k];
                dcNext[k] = dc[k] * ft[k];
            }
        }

        return dX_time;
    }

    private double[] gateSigmoid(double[] x, double[] h, double[] W, double[] U, double[] b) {
        double[] r = new double[hiddenSize];
        for (int n = 0; n < hiddenSize; n++) {
            double sum = b[n];
            int wOff = n * inputSize;
            for (int k = 0; k < inputSize; k++) sum += W[wOff + k] * x[k];
            int uOff = n * hiddenSize;
            for (int k = 0; k < hiddenSize; k++) sum += U[uOff + k] * h[k];
            sum = MathOps.clamp(sum, -50.0, 50.0);
            r[n] = MathOps.sigmoid(sum);
        }
        return r;
    }

    private double[] gateTanh(double[] x, double[] h, double[] W, double[] U, double[] b) {
        double[] r = new double[hiddenSize];
        for (int n = 0; n < hiddenSize; n++) {
            double sum = b[n];
            int wOff = n * inputSize;
            for (int k = 0; k < inputSize; k++) sum += W[wOff + k] * x[k];
            int uOff = n * hiddenSize;
            for (int k = 0; k < hiddenSize; k++) sum += U[uOff + k] * h[k];
            sum = MathOps.clamp(sum, -50.0, 50.0);
            r[n] = Math.tanh(sum);
        }
        return r;
    }
}
