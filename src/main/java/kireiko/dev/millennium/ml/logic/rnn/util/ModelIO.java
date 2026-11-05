package kireiko.dev.millennium.ml.logic.rnn.util;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public final class ModelIO {
    public static final int MAGIC = 0xDEAD0003;
    public static final int VERSION = 4;
    public static final int MAX_ARRAY_LENGTH = 1 << 24;

    public static void writeArr(DataOutputStream out, double[] a) throws IOException {
        if (a == null) { out.writeInt(-1); return; }
        out.writeInt(a.length);
        for (double v : a) out.writeDouble(v);
    }

    public static double[] readArr(DataInputStream in) throws IOException {
        return readArr(in, MAX_ARRAY_LENGTH);
    }

    public static double[] readArr(DataInputStream in, int maxLength) throws IOException {
        int n = in.readInt();
        if (n < 0) return null;
        if (n > maxLength) throw new IOException("bad array size: " + n + " (max " + maxLength + ")");
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = in.readDouble();
        return a;
    }
}
