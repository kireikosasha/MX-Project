package kireiko.dev.anticheat.checks.aim.ml.modules.v5;

import kireiko.dev.millennium.ml.data.ResultML;
import kireiko.dev.millennium.ml.data.module.FlagType;
import kireiko.dev.millennium.ml.data.module.ModuleML;
import kireiko.dev.millennium.ml.data.module.ModuleResultML;
import kireiko.dev.millennium.ml.logic.ModelVer;
import org.bukkit.Bukkit;

public class RNN1Module implements ModuleML {

    @Override
    public String getName() {
        return "m1-rnn";
    }

    @Override
    public ModuleResultML getResult(ResultML resultML) {
        double p = resultML.statisticsResult.UNUSUAL;
        if (p > 0.90) return new ModuleResultML(20, FlagType.SUSPECTED, "Insane Probability " + f(p));
        if (p > 0.80) return new ModuleResultML(12, FlagType.SUSPECTED, "Suspicious Probability " + f(p));
        if (p > 0.70) return new ModuleResultML(8, FlagType.STRANGE, "Strange Behavior " + f(p));
        if (p > 0.60) return new ModuleResultML(4, FlagType.UNUSUAL, "Unusual patterns " + f(p));

        return new ModuleResultML(0, FlagType.NORMAL, f(p));
    }

    private String f(double v) {
        return String.format("%.1f%%", v * 100);
    }

    @Override
    public int getParameterBuffer() {
        return 32;
    }

    @Override
    public ModelVer getVersion() {
        return ModelVer.VERSION_5;
    }
}