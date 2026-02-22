package kireiko.dev.millennium.ml.logic;

import kireiko.dev.millennium.ml.data.ObjectML;
import kireiko.dev.millennium.ml.data.ResultML;
import kireiko.dev.millennium.ml.logic.rnn.*;
import kireiko.dev.millennium.ml.logic.rnn.data.SequenceData;
import kireiko.dev.millennium.ml.logic.rnn.data.preprocessing.RawSequencePreprocessor;
import kireiko.dev.millennium.ml.logic.rnn.data.preprocessing.SequencePreprocessor;
import kireiko.dev.millennium.ml.logic.rnn.data.preprocessing.StatisticalSequencePreprocessor;
import kireiko.dev.millennium.ml.logic.rnn.layers.BinaryHead;
import kireiko.dev.millennium.ml.logic.rnn.layers.LSTMLayer;
import kireiko.dev.millennium.ml.logic.rnn.layers.StackedBiLSTM;
import kireiko.dev.millennium.ml.logic.rnn.optim.AdamW;
import kireiko.dev.millennium.ml.logic.rnn.pooling.*;
import kireiko.dev.millennium.ml.logic.rnn.util.ModelIO;
import kireiko.dev.millennium.vectors.Pair;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class RNNModelML implements Millennium {

    public enum InputMode { RAW_SEQUENCE, STATISTICAL_FEATURES, HYBRID }
    public enum PoolingMode { LAST_HIDDEN, MEAN_POOLING, MAX_POOLING, ATTENTION }

    private static final int MAGIC = 0x524E4E35;
    private static final int VERSION = 5;

    private final RNNConfig cfg;
    private final Random rng;

    private final RawSequencePreprocessor rawPre;
    private final StatisticalSequencePreprocessor statPre;
    private final SequencePreprocessor hybridPre;
    private SequencePreprocessor activePre;

    private final StackedBiLSTM encoder;

    private final AttentionPooling attnPooling;
    private final PoolingStrategy lastPooling;
    private final PoolingStrategy meanPooling;
    private final PoolingStrategy maxPooling;
    private PoolingStrategy activePooling;

    private final BinaryHead head;
    private final AdamW opt;

    private boolean training;
    private long step;
    private int batchSize = 16;

    private final double[] dHeadV;
    private final double[] mHeadV;
    private final double[] vHeadV;
    private final AdamW.DoubleRef headBias;
    private final AdamW.DoubleRef mHeadBias;
    private final AdamW.DoubleRef vHeadBias;

    private final double[] dAttnW;
    private final double[] mAttnW;
    private final double[] vAttnW;
    private final AdamW.DoubleRef attnB;
    private final AdamW.DoubleRef mAttnB;
    private final AdamW.DoubleRef vAttnB;

    private final double[] dEmbW;
    private final double[] mEmbW;
    private final double[] vEmbW;

    private final double[] dEmbB;
    private final double[] mEmbB;
    private final double[] vEmbB;

    private final StackedBiLSTM.Grad encGradAcc;

    private final LayerState[] fwdState;
    private final LayerState[] bwdState;

    public RNNModelML(int inputSize, int hiddenSize) {
        this(RNNConfig.builder().inputSize(inputSize).hiddenSize(hiddenSize).build());
    }

    public RNNModelML(RNNConfig cfg) {
        this.cfg = cfg;
        this.rng = new Random(cfg.seed);

        this.rawPre = new RawSequencePreprocessor(cfg.inputSize, rng);
        this.statPre = new StatisticalSequencePreprocessor(cfg.inputSize);
        this.hybridPre = new HybridPre(rawPre, statPre);
        this.activePre = pickPre(cfg.inputMode);

        this.encoder = new StackedBiLSTM(cfg.inputSize, cfg.hiddenSize, cfg.numLayers, cfg.bidirectional, rng);

        int outSize = encoder.outputSize();

        this.attnPooling = new AttentionPooling(outSize, rng);
        this.lastPooling = new LastHiddenPooling(outSize);
        this.meanPooling = new MeanPooling(outSize);
        this.maxPooling = new MaxPooling(outSize);
        this.activePooling = pickPool(cfg.poolingMode);

        this.head = new BinaryHead(outSize, rng);
        this.opt = new AdamW();

        this.training = false;
        this.step = 0L;

        this.dHeadV = new double[outSize];
        this.mHeadV = new double[outSize];
        this.vHeadV = new double[outSize];
        this.headBias = new AdamW.DoubleRef(head.bias);
        this.mHeadBias = new AdamW.DoubleRef(0.0);
        this.vHeadBias = new AdamW.DoubleRef(0.0);

        this.dAttnW = new double[outSize];
        this.mAttnW = new double[outSize];
        this.vAttnW = new double[outSize];
        this.attnB = new AdamW.DoubleRef(attnPooling.b);
        this.mAttnB = new AdamW.DoubleRef(0.0);
        this.vAttnB = new AdamW.DoubleRef(0.0);

        this.dEmbW = new double[cfg.inputSize];
        this.mEmbW = new double[cfg.inputSize];
        this.vEmbW = new double[cfg.inputSize];

        this.dEmbB = new double[cfg.inputSize];
        this.mEmbB = new double[cfg.inputSize];
        this.vEmbB = new double[cfg.inputSize];

        this.encGradAcc = new StackedBiLSTM.Grad(encoder);

        int L = cfg.numLayers;
        this.fwdState = new LayerState[L];
        this.bwdState = cfg.bidirectional ? new LayerState[L] : null;

        for (int l = 0; l < L; l++) {
            fwdState[l] = new LayerState(encoder.fwd[l]);
            if (cfg.bidirectional) bwdState[l] = new LayerState(encoder.bwd[l]);
        }
    }

    private SequencePreprocessor pickPre(InputMode m) {
        if (m == InputMode.STATISTICAL_FEATURES) return statPre;
        if (m == InputMode.HYBRID) return hybridPre;
        return rawPre;
    }

    private PoolingStrategy pickPool(PoolingMode m) {
        if (m == PoolingMode.MEAN_POOLING) return meanPooling;
        if (m == PoolingMode.MAX_POOLING) return maxPooling;
        if (m == PoolingMode.ATTENTION) return attnPooling;
        return lastPooling;
    }

    public void setLearningRate(double v) { cfg.learningRate = v; }
    public void setDropoutRate(double v) { cfg.dropoutRate = v; }
    public void setRecurrentDropoutRate(double v) { cfg.recurrentDropoutRate = v; }
    public void setWeightDecay(double v) { cfg.weightDecay = v; }
    public void setGradientClip(double v) { cfg.gradientClip = v; }
    public void setLabelSmoothing(double v) { cfg.labelSmoothing = v; }
    public void setBatchSize(int v) { this.batchSize = Math.max(1, v); }

    public void setInputMode(InputMode mode) {
        cfg.inputMode = mode;
        activePre = pickPre(mode);
    }

    public void setPoolingMode(PoolingMode mode) {
        cfg.poolingMode = mode;
        activePooling = pickPool(mode);
    }

    public double getLearningRate() { return cfg.learningRate; }
    public double getDropoutRate() { return cfg.dropoutRate; }
    public double getRecurrentDropoutRate() { return cfg.recurrentDropoutRate; }
    public double getWeightDecay() { return cfg.weightDecay; }
    public double getGradientClip() { return cfg.gradientClip; }
    public double getLabelSmoothing() { return cfg.labelSmoothing; }

    public InputMode getInputMode() { return cfg.inputMode; }
    public PoolingMode getPoolingMode() { return cfg.poolingMode; }

    @Override
    public ResultML checkData(List<ObjectML> o) {
        ResultML r = new ResultML();
        if (o == null || o.isEmpty()) return r;

        boolean prev = training;
        training = false;

        double sum = 0.0;
        int n = 0;

        for (ObjectML obj : o) {
            if (obj == null || obj.getValues() == null || obj.getValues().size() < 2) continue;
            sum += forwardProbability(obj.getValues());
            n++;
        }

        training = prev;

        double avg = n == 0 ? 0.0 : sum / n;

        r.statisticsResult.UNUSUAL = (float) avg;
        r.statisticsResult.STRANGE = 0;
        r.statisticsResult.SUSPECTED = 0;
        r.statisticsResult.SUSPICIOUSLY = 0;

        return r;
    }

    @Override
    public void learnByData(List<ObjectML> o, boolean isMustBeBlocked) {
        if (o == null || o.isEmpty()) return;

        training = true;

        double y = isMustBeBlocked ? 1.0 : 0.0;
        if (cfg.labelSmoothing > 0.0) y = y * (1.0 - cfg.labelSmoothing) + 0.5 * cfg.labelSmoothing;

        List<Sample> samples = new ArrayList<>();
        for (ObjectML obj : o) {
            if (obj == null || obj.getValues() == null || obj.getValues().size() < 2) continue;
            samples.add(new Sample(obj.getValues(), y));
        }
        if (samples.isEmpty()) {
            training = false;
            return;
        }

        int bs = Math.max(1, batchSize);
        for (int i = 0; i < samples.size(); i += bs) {
            int end = Math.min(samples.size(), i + bs);
            trainBatch(samples.subList(i, end));
        }

        training = false;
    }

    private void trainBatch(List<Sample> batch) {
        zeroBatchGrads();

        int used = 0;

        for (Sample s : batch) {
            ForwardCache fc = forwardCache(s.values);
            if (fc == null) continue;

            double p = fc.prob;
            double dLogit = (p - s.label);

            BinaryHead.Grad hg = new BinaryHead.Grad(head.in);
            double[] dPooled = head.backward(fc.headCache, dLogit, hg);
            add(dHeadV, hg.dV);
            fc.dHeadBias += hg.dBias;

            PoolingGrad pg = new PoolingGrad();
            pg.dAttentionW = dAttnW;
            pg.dAttentionB = 0.0;

            double[][] dH = activePooling.backward(fc.hTime, fc.seq.mask, dPooled, fc.poolCache, pg);
            fc.dAttnB += pg.dAttentionB;

            double[][] dX = encoder.backward(fc.encCache, dH, encGradAcc);

            if (activePre == rawPre) {
                accumulateEmbeddingGrad(s.values, dX);
            }

            used++;
        }

        if (used <= 0) return;

        applyUpdate(used);
        step++;
    }

    private ForwardCache forwardCache(List<Double> raw) {
        SequenceData seq = activePre.prepare(raw);
        if (seq == null || seq.length() < 2) return null;

        StackedBiLSTM.Cache encCache = new StackedBiLSTM.Cache();
        double[][] hTime = encoder.forward(seq.x, training, cfg.dropoutRate, cfg.recurrentDropoutRate, rng, encCache);

        PoolingCache pc = new PoolingCache();
        double[] pooled = activePooling.forward(hTime, seq.mask, pc);

        BinaryHead.Cache hc = new BinaryHead.Cache();
        double prob = head.forward(pooled, hc);

        ForwardCache fc = new ForwardCache();
        fc.seq = seq;
        fc.hTime = hTime;
        fc.encCache = encCache;
        fc.poolCache = pc;
        fc.headCache = hc;
        fc.prob = prob;
        return fc;
    }

    private double forwardProbability(List<Double> raw) {
        SequenceData seq = activePre.prepare(raw);
        if (seq == null || seq.length() < 2) return 0.5;

        double[][] hTime = encoder.forward(seq.x, false, 0.0, 0.0, rng, null);
        double[] pooled = activePooling.forward(hTime, seq.mask, null);
        return head.forward(pooled, null);
    }

    private void accumulateEmbeddingGrad(List<Double> raw, double[][] dX) {
        double[] embW = rawPre.getEmbeddingW();
        double[] embB = rawPre.getEmbeddingB();

        int T = Math.min(raw.size(), dX.length);
        for (int t = 0; t < T; t++) {
            double v = Math.tanh(raw.get(t) / 100.0);
            for (int i = 0; i < cfg.inputSize; i++) {
                dEmbW[i] += dX[t][i] * v;
                dEmbB[i] += dX[t][i];
            }
        }

        for (int i = 0; i < cfg.inputSize; i++) {
            if (Double.isNaN(dEmbW[i]) || Double.isInfinite(dEmbW[i])) dEmbW[i] = 0.0;
            if (Double.isNaN(dEmbB[i]) || Double.isInfinite(dEmbB[i])) dEmbB[i] = 0.0;
            if (Double.isNaN(embW[i]) || Double.isInfinite(embW[i])) embW[i] = 0.0;
            if (Double.isNaN(embB[i]) || Double.isInfinite(embB[i])) embB[i] = 0.0;
        }
    }

    private void applyUpdate(int used) {
        double inv = 1.0 / used;

        scaleInPlace(dHeadV, inv);
        headBias.value = head.bias;
        opt.step(head.V, dHeadV, mHeadV, vHeadV, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.stepScalar(headBias, headBiasGrad * inv, mHeadBias, vHeadBias, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        head.bias = headBias.value;

        scaleInPlace(dAttnW, inv);
        attnB.value = attnPooling.b;
        opt.step(attnPooling.W, dAttnW, mAttnW, vAttnW, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.stepScalar(attnB, attnBGrad * inv, mAttnB, vAttnB, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        attnPooling.b = attnB.value;

        if (activePre == rawPre) {
            scaleInPlace(dEmbW, inv);
            scaleInPlace(dEmbB, inv);
            opt.step(rawPre.getEmbeddingW(), dEmbW, mEmbW, vEmbW, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
            opt.step(rawPre.getEmbeddingB(), dEmbB, mEmbB, vEmbB, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        }

        for (int l = 0; l < cfg.numLayers; l++) {
            updateLayer(encoder.fwd[l], encGradAcc.fwd[l], fwdState[l], inv);
            if (cfg.bidirectional) updateLayer(encoder.bwd[l], encGradAcc.bwd[l], bwdState[l], inv);
        }
    }

    private void updateLayer(LSTMLayer layer, LSTMLayer.Grad g, LayerState s, double inv) {
        scaleInPlace(g.dWf, inv); scaleInPlace(g.dWi, inv); scaleInPlace(g.dWc, inv); scaleInPlace(g.dWo, inv);
        scaleInPlace(g.dUf, inv); scaleInPlace(g.dUi, inv); scaleInPlace(g.dUc, inv); scaleInPlace(g.dUo, inv);
        scaleInPlace(g.dbf, inv); scaleInPlace(g.dbi, inv); scaleInPlace(g.dbc, inv); scaleInPlace(g.dbo, inv);
        scaleInPlace(g.dLnGamma, inv); scaleInPlace(g.dLnBeta, inv);

        opt.step(layer.Wf, g.dWf, s.mWf, s.vWf, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.Wi, g.dWi, s.mWi, s.vWi, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.Wc, g.dWc, s.mWc, s.vWc, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.Wo, g.dWo, s.mWo, s.vWo, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);

        opt.step(layer.Uf, g.dUf, s.mUf, s.vUf, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.Ui, g.dUi, s.mUi, s.vUi, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.Uc, g.dUc, s.mUc, s.vUc, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.Uo, g.dUo, s.mUo, s.vUo, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);

        opt.step(layer.bf, g.dbf, s.mbf, s.vbf, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.bi, g.dbi, s.mbi, s.vbi, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.bc, g.dbc, s.mbc, s.vbc, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.bo, g.dbo, s.mbo, s.vbo, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);

        opt.step(layer.lnGamma, g.dLnGamma, s.mLnG, s.vLnG, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
        opt.step(layer.lnBeta, g.dLnBeta, s.mLnB, s.vLnB, cfg.learningRate, cfg.weightDecay, cfg.gradientClip);
    }

    private void zeroBatchGrads() {
        zero(dHeadV);
        headBiasGrad = 0.0;

        zero(dAttnW);
        attnBGrad = 0.0;

        zero(dEmbW);
        zero(dEmbB);

        zeroGrad(encGradAcc);
    }

    private void zeroGrad(StackedBiLSTM.Grad g) {
        for (int l = 0; l < g.fwd.length; l++) {
            zeroLstmGrad(g.fwd[l]);
            if (cfg.bidirectional) zeroLstmGrad(g.bwd[l]);
        }
    }

    private void zeroLstmGrad(LSTMLayer.Grad g) {
        zero(g.dWf); zero(g.dWi); zero(g.dWc); zero(g.dWo);
        zero(g.dUf); zero(g.dUi); zero(g.dUc); zero(g.dUo);
        zero(g.dbf); zero(g.dbi); zero(g.dbc); zero(g.dbo);
        zero(g.dLnGamma); zero(g.dLnBeta);
    }

    private static void zero(double[] a) { for (int i = 0; i < a.length; i++) a[i] = 0.0; }

    private static void add(double[] dst, double[] src) {
        for (int i = 0; i < dst.length; i++) dst[i] += src[i];
    }

    private static void scaleInPlace(double[] a, double s) {
        for (int i = 0; i < a.length; i++) a[i] *= s;
    }

    private double headBiasGrad;
    private double attnBGrad;

    @Override
    public void saveToFile(String fileName) {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(Paths.get(fileName))))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);

            out.writeInt(cfg.inputSize);
            out.writeInt(cfg.hiddenSize);
            out.writeInt(cfg.numLayers);
            out.writeBoolean(cfg.bidirectional);

            out.writeInt(cfg.inputMode.ordinal());
            out.writeInt(cfg.poolingMode.ordinal());

            out.writeDouble(cfg.learningRate);
            out.writeDouble(cfg.dropoutRate);
            out.writeDouble(cfg.recurrentDropoutRate);
            out.writeDouble(cfg.weightDecay);
            out.writeDouble(cfg.gradientClip);
            out.writeDouble(cfg.labelSmoothing);

            out.writeInt(batchSize);
            out.writeLong(step);
            out.writeLong(opt.t);

            ModelIO.writeArr(out, rawPre.getEmbeddingW());
            ModelIO.writeArr(out, rawPre.getEmbeddingB());

            writeEncoder(out);
            ModelIO.writeArr(out, attnPooling.W);
            out.writeDouble(attnPooling.b);

            ModelIO.writeArr(out, head.V);
            out.writeDouble(head.bias);
        } catch (Exception ignored) {}
    }

    public void load(InputStream in) {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(in))) {
            int magic = dis.readInt();
            int ver = dis.readInt();
            if (magic != MAGIC || ver != VERSION) throw new IllegalStateException("bad model");

            int inSize = dis.readInt();
            int hid = dis.readInt();
            int layers = dis.readInt();
            boolean bi = dis.readBoolean();

            if (inSize != cfg.inputSize || hid != cfg.hiddenSize || layers != cfg.numLayers || bi != cfg.bidirectional) {
                throw new IllegalStateException("arch mismatch");
            }

            int im = dis.readInt();
            int pm = dis.readInt();
            cfg.inputMode = InputMode.values()[clamp(im, InputMode.values().length)];
            cfg.poolingMode = PoolingMode.values()[clamp(pm, PoolingMode.values().length)];
            activePre = pickPre(cfg.inputMode);
            activePooling = pickPool(cfg.poolingMode);

            cfg.learningRate = dis.readDouble();
            cfg.dropoutRate = dis.readDouble();
            cfg.recurrentDropoutRate = dis.readDouble();
            cfg.weightDecay = dis.readDouble();
            cfg.gradientClip = dis.readDouble();
            cfg.labelSmoothing = dis.readDouble();

            batchSize = Math.max(1, dis.readInt());
            step = dis.readLong();
            opt.t = dis.readLong();

            readInto(dis, rawPre.getEmbeddingW());
            readInto(dis, rawPre.getEmbeddingB());

            readEncoder(dis);

            readInto(dis, attnPooling.W);
            attnPooling.b = dis.readDouble();

            readInto(dis, head.V);
            head.bias = dis.readDouble();
        } catch (Exception ignored) {}
    }

    private static int clamp(int v, int n) {
        if (n <= 0) return 0;
        if (v < 0) return 0;
        if (v >= n) return n - 1;
        return v;
    }

    private void writeEncoder(DataOutputStream out) throws Exception {
        out.writeInt(cfg.numLayers);
        out.writeBoolean(cfg.bidirectional);

        for (int l = 0; l < cfg.numLayers; l++) {
            writeLayer(out, encoder.fwd[l]);
            if (cfg.bidirectional) writeLayer(out, encoder.bwd[l]);
        }
    }

    private void readEncoder(DataInputStream in) throws Exception {
        int layers = in.readInt();
        boolean bi = in.readBoolean();
        if (layers != cfg.numLayers || bi != cfg.bidirectional) throw new IllegalStateException("encoder mismatch");

        for (int l = 0; l < cfg.numLayers; l++) {
            readLayer(in, encoder.fwd[l]);
            if (cfg.bidirectional) readLayer(in, encoder.bwd[l]);
        }
    }

    private void writeLayer(DataOutputStream out, LSTMLayer l) throws Exception {
        ModelIO.writeArr(out, l.Wf); ModelIO.writeArr(out, l.Wi); ModelIO.writeArr(out, l.Wc); ModelIO.writeArr(out, l.Wo);
        ModelIO.writeArr(out, l.Uf); ModelIO.writeArr(out, l.Ui); ModelIO.writeArr(out, l.Uc); ModelIO.writeArr(out, l.Uo);
        ModelIO.writeArr(out, l.bf); ModelIO.writeArr(out, l.bi); ModelIO.writeArr(out, l.bc); ModelIO.writeArr(out, l.bo);
        ModelIO.writeArr(out, l.lnGamma); ModelIO.writeArr(out, l.lnBeta);
    }

    private void readLayer(DataInputStream in, LSTMLayer l) throws Exception {
        readInto(in, l.Wf); readInto(in, l.Wi); readInto(in, l.Wc); readInto(in, l.Wo);
        readInto(in, l.Uf); readInto(in, l.Ui); readInto(in, l.Uc); readInto(in, l.Uo);
        readInto(in, l.bf); readInto(in, l.bi); readInto(in, l.bc); readInto(in, l.bo);
        readInto(in, l.lnGamma); readInto(in, l.lnBeta);
    }

    private static void readInto(DataInputStream in, double[] dst) throws Exception {
        double[] a = ModelIO.readArr(in);
        if (a == null) return;
        System.arraycopy(a, 0, dst, 0, Math.min(a.length, dst.length));
    }

    @Override
    public int parameters() {
        long p = 0;

        p += rawPre.getEmbeddingW().length;
        p += rawPre.getEmbeddingB().length;

        for (int l = 0; l < cfg.numLayers; l++) {
            p += countLayer(encoder.fwd[l]);
            if (cfg.bidirectional) p += countLayer(encoder.bwd[l]);
        }

        p += attnPooling.W.length + 1;
        p += head.V.length + 1;

        if (p > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        return (int) p;
    }

    private static long countLayer(LSTMLayer l) {
        long p = 0;
        p += l.Wf.length + l.Wi.length + l.Wc.length + l.Wo.length;
        p += l.Uf.length + l.Ui.length + l.Uc.length + l.Uo.length;
        p += l.bf.length + l.bi.length + l.bc.length + l.bo.length;
        p += l.lnGamma.length + l.lnBeta.length;
        return p;
    }

    private static final class Sample {
        final List<Double> values;
        final double label;
        Sample(List<Double> values, double label) {
            this.values = values;
            this.label = label;
        }
    }

    private static final class ForwardCache {
        SequenceData seq;
        double[][] hTime;
        StackedBiLSTM.Cache encCache;
        PoolingCache poolCache;
        BinaryHead.Cache headCache;
        double prob;
        double dHeadBias;
        double dAttnB;
    }

    private static final class LayerState {
        final double[] mWf, vWf, mWi, vWi, mWc, vWc, mWo, vWo;
        final double[] mUf, vUf, mUi, vUi, mUc, vUc, mUo, vUo;
        final double[] mbf, vbf, mbi, vbi, mbc, vbc, mbo, vbo;
        final double[] mLnG, vLnG, mLnB, vLnB;

        LayerState(LSTMLayer l) {
            mWf = new double[l.Wf.length]; vWf = new double[l.Wf.length];
            mWi = new double[l.Wi.length]; vWi = new double[l.Wi.length];
            mWc = new double[l.Wc.length]; vWc = new double[l.Wc.length];
            mWo = new double[l.Wo.length]; vWo = new double[l.Wo.length];

            mUf = new double[l.Uf.length]; vUf = new double[l.Uf.length];
            mUi = new double[l.Ui.length]; vUi = new double[l.Ui.length];
            mUc = new double[l.Uc.length]; vUc = new double[l.Uc.length];
            mUo = new double[l.Uo.length]; vUo = new double[l.Uo.length];

            mbf = new double[l.bf.length]; vbf = new double[l.bf.length];
            mbi = new double[l.bi.length]; vbi = new double[l.bi.length];
            mbc = new double[l.bc.length]; vbc = new double[l.bc.length];
            mbo = new double[l.bo.length]; vbo = new double[l.bo.length];

            mLnG = new double[l.lnGamma.length]; vLnG = new double[l.lnGamma.length];
            mLnB = new double[l.lnBeta.length]; vLnB = new double[l.lnBeta.length];
        }
    }

    private static final class HybridPre implements SequencePreprocessor {
        private final SequencePreprocessor raw;
        private final SequencePreprocessor stat;
        HybridPre(SequencePreprocessor raw, SequencePreprocessor stat) {
            this.raw = raw;
            this.stat = stat;
        }
        @Override
        public SequenceData prepare(List<Double> r) {
            if (r == null) return null;
            if (r.size() >= 40) return stat.prepare(r);
            return raw.prepare(r);
        }
    }

    private static final class LastHiddenPooling implements PoolingStrategy {
        private final int outSize;
        LastHiddenPooling(int outSize) { this.outSize = outSize; }
        @Override public int outputSize() { return outSize; }
        @Override
        public double[] forward(double[][] hTime, double[] mask, PoolingCache cache) {
            int last = -1;
            for (int t = hTime.length - 1; t >= 0; t--) {
                if (mask[t] > 0.5) { last = t; break; }
            }
            if (last < 0) last = hTime.length - 1;
            double[] pooled = new double[outSize];
            System.arraycopy(hTime[last], 0, pooled, 0, outSize);
            return pooled;
        }
        @Override
        public double[][] backward(double[][] hTime, double[] mask, double[] dPooled, PoolingCache cache, PoolingGrad gAcc) {
            double[][] dH = new double[hTime.length][outSize];
            int last = -1;
            for (int t = hTime.length - 1; t >= 0; t--) {
                if (mask[t] > 0.5) { last = t; break; }
            }
            if (last < 0) last = hTime.length - 1;
            System.arraycopy(dPooled, 0, dH[last], 0, outSize);
            return dH;
        }
    }

    @Override
    public void trainEpochs(List<Pair<List<ObjectML>, Boolean>> dataset, int epochs) {
        if (dataset == null || dataset.isEmpty()) return;
        int bs = Math.max(1, batchSize);
        for (int e = 0; e < epochs; e++) {
            java.util.Collections.shuffle(dataset, rng);
            training = true;
            List<Sample> currentBatch = new ArrayList<>();
            for (Pair<List<ObjectML>, Boolean> dataPair : dataset) {
                double y = dataPair.getY() ? 1.0 : 0.0;
                if (cfg.labelSmoothing > 0.0) {
                    y = y * (1.0 - cfg.labelSmoothing) + 0.5 * cfg.labelSmoothing;
                }
                for (ObjectML obj : dataPair.getX()) {
                    if (obj == null || obj.getValues() == null || obj.getValues().size() < 2) continue;
                    currentBatch.add(new Sample(obj.getValues(), y));
                }
                if (currentBatch.size() >= bs) {
                    trainBatch(currentBatch);
                    currentBatch.clear();
                }
            }
            if (!currentBatch.isEmpty()) {
                trainBatch(currentBatch);
            }
        }
        training = false;
    }
}
