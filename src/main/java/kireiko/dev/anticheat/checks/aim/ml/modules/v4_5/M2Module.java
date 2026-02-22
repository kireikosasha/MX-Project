package kireiko.dev.anticheat.checks.aim.ml.modules.v4_5;

import kireiko.dev.millennium.math.Simplification;
import kireiko.dev.millennium.ml.data.ResultML;
import kireiko.dev.millennium.ml.data.module.FlagType;
import kireiko.dev.millennium.ml.data.module.ModuleML;
import kireiko.dev.millennium.ml.data.module.ModuleResultML;
import kireiko.dev.millennium.ml.logic.ModelVer;
import lombok.var;

public class M2Module implements ModuleML {

    private static final double M = 2.0;

    @Override
    public String getName() {
        return "m2";
    }

    @Override
    public ModuleResultML getResult(ResultML resultML) {
        var stats = resultML.statisticsResult;
        final double UNUSUAL = stats.UNUSUAL / M;
        final double SUSPECTED = stats.SUSPECTED / M;

        FlagType type = (UNUSUAL > 0.5 || (UNUSUAL > 0.4 && SUSPECTED > 0.15)) ? FlagType.SUSPECTED :
                (UNUSUAL > 0.4) ? FlagType.STRANGE :
                        (UNUSUAL > 0.3) ? FlagType.UNUSUAL :
                                FlagType.NORMAL;

        int score = (type == FlagType.NORMAL) ? 0 : 10;
        String message = String.valueOf(Simplification.scaleVal(UNUSUAL, 3));

        return new ModuleResultML(score, type, message);
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
