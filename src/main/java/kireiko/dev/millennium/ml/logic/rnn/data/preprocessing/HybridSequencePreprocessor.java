package kireiko.dev.millennium.ml.logic.rnn.data.preprocessing;

import kireiko.dev.millennium.ml.logic.rnn.data.SequenceData;

import java.util.List;

public final class HybridSequencePreprocessor implements SequencePreprocessor {
    private final int inputSize;
    private final StatisticalSequencePreprocessor stat;

    public HybridSequencePreprocessor(int inputSize) {
        this.inputSize = inputSize;
        this.stat = new StatisticalSequencePreprocessor(inputSize);
    }

    @Override
    public SequenceData prepare(List<Double> raw) {
        int totalLen = raw.size();
        int numWindows = Math.max(2, Math.min(15, totalLen / 5));
        int windowSize = Math.max(1, totalLen / numWindows);

        double[][] x = new double[numWindows][inputSize];
        int used = 0;

        for (int w = 0; w < numWindows; w++) {
            int start = w * windowSize;
            if (start >= totalLen) break;
            int end = (w == numWindows - 1) ? totalLen : Math.min(totalLen, start + windowSize);
            List<Double> window = raw.subList(start, end);
            if (window.size() < 3) continue;

            double[] s = stat.prepare(window).x[0];
            double[] r = rawFeatures(window);
            x[used++] = combine(s, r);
        }

        if (used < 2) {
            double[][] xx = new double[Math.max(2, used)][inputSize];
            if (used >= 0) System.arraycopy(x, 0, xx, 0, used);
            x = xx;
            used = x.length;
        }

        double[] mask = new double[used];
        for (int i = 0; i < used; i++) mask[i] = 1.0;

        double[][] out = new double[used][inputSize];
        System.arraycopy(x, 0, out, 0, used);
        return new SequenceData(out, mask);
    }

    private double[] rawFeatures(List<Double> w) {
        double[] f = new double[inputSize];
        int step = Math.max(1, w.size() / inputSize);
        for (int i = 0; i < inputSize && i * step < w.size(); i++) f[i] = Math.tanh(w.get(i * step) / 100.0);
        return f;
    }

    private double[] combine(double[] stat, double[] raw) {
        double[] c = new double[inputSize];
        int half = inputSize / 2;
        for (int i = 0; i < half && i < stat.length; i++) c[i] = stat[i];
        for (int i = 0; i < half && i < raw.length; i++) c[half + i] = raw[i] * 0.5;
        return c;
    }
}
