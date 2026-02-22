package kireiko.dev.anticheat.checks.aim.ml.modules.v4_5;

import kireiko.dev.millennium.math.Simplification;
import kireiko.dev.millennium.ml.data.ResultML;
import kireiko.dev.millennium.ml.data.module.FlagType;
import kireiko.dev.millennium.ml.data.module.ModuleML;
import kireiko.dev.millennium.ml.data.module.ModuleResultML;
import kireiko.dev.millennium.ml.logic.ModelVer;
import lombok.var;

public class M4Module implements ModuleML {

    private static final double M = 2.0;

    @Override
    public String getName() {
        return "m4";
    }

    @Override
    public ModuleResultML getResult(ResultML resultML) {
        var stats = resultML.statisticsResult;
        final double UNUSUAL = stats.UNUSUAL / M;
        final double STRANGE = stats.STRANGE / M;
        final double SUSPECTED = stats.SUSPECTED / M;
        final double SUSPICIOUSLY = stats.SUSPICIOUSLY / M;
        String scaledUnusual = String.valueOf(Simplification.scaleVal(UNUSUAL, 3));

        if (UNUSUAL > 0.25 && STRANGE > 0.07 && SUSPECTED > 0.018 && SUSPICIOUSLY > 0)
            return new ModuleResultML(10, FlagType.SUSPECTED, scaledUnusual);

        if (UNUSUAL > 0.25 && STRANGE > 0.04 && SUSPECTED > 0)
            return new ModuleResultML(10, FlagType.STRANGE, scaledUnusual);

        return new ModuleResultML(0, FlagType.NORMAL, scaledUnusual);
    }

    @Override
    public int getParameterBuffer() {
        return 15;
    }

    @Override
    public ModelVer getVersion() {
        return ModelVer.VERSION_4_5;
    }
}
