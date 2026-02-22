package kireiko.dev.millennium.ml.logic.rnn.pooling;

import kireiko.dev.millennium.ml.logic.rnn.pooling.PoolingCache;
import kireiko.dev.millennium.ml.logic.rnn.pooling.PoolingGrad;
import kireiko.dev.millennium.ml.logic.rnn.pooling.PoolingStrategy;

public final class LastHiddenPooling implements PoolingStrategy {
    private final int outSize;

    public LastHiddenPooling(int outSize) { this.outSize = outSize; }

    @Override public int outputSize() { return outSize; }

    @Override
    public double[] forward(double[][] hTime, double[] mask, PoolingCache cache) {
        int last = hTime.length - 1;
        return hTime[last].clone();
    }

    @Override
    public double[][] backward(double[][] hTime, double[] mask, double[] dPooled, PoolingCache cache, PoolingGrad gAcc) {
        double[][] dH = new double[hTime.length][outSize];
        int last = hTime.length - 1;
        System.arraycopy(dPooled, 0, dH[last], 0, outSize);
        return dH;
    }
}
