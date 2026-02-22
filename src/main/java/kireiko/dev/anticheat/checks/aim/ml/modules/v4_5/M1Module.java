package kireiko.dev.anticheat.checks.aim.ml.modules.v4_5;

import kireiko.dev.millennium.math.Simplification;
import kireiko.dev.millennium.ml.data.ResultML;
import kireiko.dev.millennium.ml.data.module.FlagType;
import kireiko.dev.millennium.ml.data.module.ModuleML;
import kireiko.dev.millennium.ml.data.module.ModuleResultML;
import kireiko.dev.millennium.ml.logic.ModelVer;

public class M1Module implements ModuleML {

    private static final double M = 2.0;

    @Override
    public String getName() {
        return "m1";
    }

    @Override
    public ModuleResultML getResult(ResultML resultML) {
        ResultML.CheckResultML checkResult = resultML.statisticsResult;
        final double UNUSUAL = checkResult.UNUSUAL / M;
        final double STRANGE = checkResult.STRANGE / M;
        final double SUSPECTED = checkResult.SUSPECTED / M;
        final double SUSPICIOUSLY = checkResult.SUSPICIOUSLY / M;
        String scaledUnusual = String.valueOf(Simplification.scaleVal(UNUSUAL, 3));

        if (UNUSUAL > 0.8 && STRANGE > 0.15)
            return new ModuleResultML(20, FlagType.SUSPECTED, "Blatant prohibition (" + scaledUnusual + ")");
        if (UNUSUAL > 0.5 && STRANGE > 0.11 && SUSPICIOUSLY > 0.1)
            return new ModuleResultML(10, FlagType.SUSPECTED, "Suspected prohibition (" + scaledUnusual + ")");
        if (UNUSUAL > 0.3 && STRANGE > 0.11 && SUSPICIOUSLY > 0.11)
            return new ModuleResultML(15, FlagType.SUSPECTED, "Suspected prohibition (" + scaledUnusual + ")");
        if (UNUSUAL > 0.45 && STRANGE > 0.21 && SUSPECTED > 0)
            return new ModuleResultML(10, FlagType.STRANGE, "Strange prohibition (" + scaledUnusual + ")");
        if (UNUSUAL > 0.3 && STRANGE > 0.085 && SUSPECTED > 0.06 && SUSPICIOUSLY > 0)
            return new ModuleResultML(10, FlagType.STRANGE, "Strange prohibition (" + scaledUnusual + ")");
        if (UNUSUAL > 0.27 && STRANGE > 0.06 && SUSPECTED > 0.04 && SUSPICIOUSLY > 0)
            return new ModuleResultML(8, FlagType.UNUSUAL, "Unusual prohibition (" + scaledUnusual + ")");

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
