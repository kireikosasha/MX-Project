package kireiko.dev.anticheat.checks.aim.heuristic;

import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.RotationEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.checks.aim.AimHeuristicCheck;
import kireiko.dev.millennium.math.Euler;
import kireiko.dev.millennium.math.Statistics;
import kireiko.dev.millennium.ml.data.reasoning.MathML;
import kireiko.dev.millennium.vectors.Vec2f;

import java.util.*;
import java.util.function.Function;

public final class AimSmoothCheck implements HeuristicComponent {
    private final AimHeuristicCheck check;
    private int buffer = 0;
    private Map<String, Object> localCfg = new TreeMap<>();
    private final List<Double> stack = new ArrayList<>();
    public AimSmoothCheck(final AimHeuristicCheck check) {
        this.check = check;
    }

    @Override
    public ConfigLabel config() {
        localCfg.put("addGlobalVl", 35);
        return new ConfigLabel("smooth_check", localCfg);
    }
    @Override
    public void applyConfig(Map<String, Object> params) {
        localCfg = params;
    }

    @Override
    public void process(final RotationEvent event) {
        if (check.getProfile().ignoreCinematic()) return;
        if (event.getAbsDelta().getY() == 0 && event.getAbsDelta().getY() == 0) return;
        final PlayerProfile profile = check.getProfile();
        final Vec2f delta = event.getDelta();
        double angle = Euler.getAngleInDegrees(delta) % 90;
        { // check logic
            if ((event.getAbsDelta().getY() > 1.5 && event.getAbsDelta().getX() > 0.32) || event.getAbsDelta().getX() > 1.5)
                stack.add(angle);
            if (stack.size() >= 20) {
                {
                    List<Float> jiff = Statistics.getJiffDelta(stack, 1);
                    float prev = 999;
                    float prePrev = 999;
                    for (float f : jiff) {
                        if (f == 0.0 && prev == 0.0 && prePrev == 0) {
                            profile.punish("Aim", "Smooth", "Invalid smoothing " + Arrays.toString(jiff.toArray()), getNumCfg("addGlobalVl") / 10);
                            break;
                        }
                        prePrev = prev;
                        prev = f;
                    }
                }
                stack.clear();
            }
        }
    }

    private float getNumCfg(String key) {
        return ((Number) localCfg.get(key)).floatValue();
    }
}