package kireiko.dev.anticheat.checks.aim.ml.modules.v4_5;

import kireiko.dev.millennium.ml.data.ResultML;
import kireiko.dev.millennium.ml.data.module.FlagType;
import kireiko.dev.millennium.ml.data.module.ModuleML;
import kireiko.dev.millennium.ml.data.module.ModuleResultML;
import kireiko.dev.millennium.ml.logic.ModelVer;

public class MHuge2Module implements ModuleML {

    private static final double M = 2.0;

    @Override
    public String getName() {
        return "m_huge2";
    }

    @Override
    public ModuleResultML getResult(ResultML resultML) {
        ResultML.CheckResultML checkResult = resultML.statisticsResult;
        double ab1 = checkResult.UNUSUAL / M;
        double ab2 = checkResult.STRANGE / M;
        double ab3 = checkResult.SUSPECTED / M;
        double ab4 = checkResult.SUSPICIOUSLY / M;
        FlagType type = FlagType.NORMAL;
        if (ab1 > 0.6) {
            type = FlagType.SUSPECTED;
        } else if (ab1 > 0.5 && ab2 < 0.042) {
            type = FlagType.STRANGE;
        } else if (ab4 > 0 && ab2 > 0.17 && ab1 > 0.35) {
            type = FlagType.UNUSUAL;
        }
        return new ModuleResultML(30, type, checkResult.toString());
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
