package kireiko.dev.millennium.ml.logic.rnn.data.preprocessing;

import kireiko.dev.millennium.ml.logic.rnn.data.SequenceData;

import java.util.List;

public interface SequencePreprocessor {
    SequenceData prepare(List<Double> raw);
}
