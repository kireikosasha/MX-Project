package kireiko.dev.anticheat.checks.aim.heuristic;

import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.RotationEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.checks.aim.AimHeuristicCheck;
import kireiko.dev.millennium.math.Simplification;
import kireiko.dev.millennium.math.Statistics;
import kireiko.dev.millennium.types.EvictingList;
import kireiko.dev.millennium.vectors.Pair;
import kireiko.dev.millennium.vectors.Vec2f;

import java.util.*;

public final class AimFactorCheck implements HeuristicComponent {
    private final AimHeuristicCheck check;
    private boolean lastIsNoRotation = false;
    private double lastHash = 0;
    private float buffer = 0;
    private int ticksToReset = 0;
    private final List<Double> stack = new EvictingList<>(3);
    private Map<String, Object> localCfg = new TreeMap<>();


    public AimFactorCheck(final AimHeuristicCheck check) {
        this.check = check;
    }

    @Override
    public ConfigLabel config() {
        localCfg.put("addGlobalVl", 25);
        localCfg.put("buffer", 3.0);
        localCfg.put("ticksToReset", 2500);
        return new ConfigLabel("factor_check", localCfg);
    }
    @Override
    public void applyConfig(Map<String, Object> params) {
        localCfg = params;
    }

    @Override
    public void process(final RotationEvent rotationUpdate) {
        if (rotationUpdate.getAbsDelta().getY() == 0 && rotationUpdate.getAbsDelta().getY() == 0) {
            if (!lastIsNoRotation) stack.add(0.0);
            check();
            lastIsNoRotation = true;
        } else {
            Vec2f delta = rotationUpdate.getAbsDelta();
            stack.add(Simplification.scaleVal(delta.getX(), 2));
            check();
            lastIsNoRotation = false;
        }
    }

    private void check() {
        if (stack.size() != 3) return;
        double hash = stack.get(0) + stack.get(1) + stack.get(2);
        if (hash == lastHash) return;
        double centre = stack.get(1);
        boolean hugeRotation = centre > 35;
        //check.getProfile().getPlayer().sendMessage("p: " + Arrays.toString(stack.toArray()) + " " + buffer);
        if (hugeRotation && centre != 360.0f) {
            double compare = 1.2;
            boolean invalid = (stack.get(0) < compare && stack.get(2) < compare)
                            || (stack.get(0) > 55 && stack.get(1) < 2 && stack.get(2) > 55)
                            || Statistics.getMax(stack) > 70 && Statistics.getMin(stack) < compare && Statistics.getDistinct(stack) != 3
                            ;
            if (invalid) {
                float vl = ((Number) localCfg.get("addGlobalVl")).floatValue() / 10f;
                float bufferLimit = ((Number) localCfg.get("buffer")).floatValue();
                float localVl = (centre > 160) ? 3 : (centre < 60) ? 1 : 2;
                buffer += localVl;
                check.getProfile().debug("&7Factor analysis: " + Arrays.toString(stack.toArray()) + " (" + buffer + ")");
                if (buffer >= bufferLimit) {
                    check.getProfile().punish("Aim", "Factor", "Factor analysis ("
                                    + centre + "/" + Simplification.scaleVal(stack.get(0) + stack.get(2), 2) + ")", vl);
                    buffer = bufferLimit - 1;
                }
            }
        } else {
            ticksToReset++;
            if (ticksToReset >= ((Number) localCfg.get("ticksToReset")).intValue()) {
                ticksToReset = 0;
                buffer = 0;
            }
        }
        lastHash = hash;
    }
}
