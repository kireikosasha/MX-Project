package kireiko.dev.anticheat.api.player;

import com.google.common.collect.Lists;
import kireiko.dev.anticheat.api.events.RotationEvent;
import kireiko.dev.anticheat.utils.GraphUtil;
import kireiko.dev.millennium.math.Statistics;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class CinematicComponent {
    private final List<Double
            > yawSamples = Lists.newArrayList(),
            pitchSamples = Lists.newArrayList();
    private final PlayerProfile profile;
    private long lastSmooth = 0L, lastHighRate = 0L;
    private double lastDeltaYaw = 0.0D, lastDeltaPitch = 0.0D;
    private int isTotallyNotCinematic = 0;

    public CinematicComponent(PlayerProfile profile) {
        this.profile = profile;
    }

    public void process(RotationEvent rotationUpdate) {
        long now = System.currentTimeMillis();
        double deltaYaw = rotationUpdate.getDelta().getX();
        double deltaPitch = rotationUpdate.getDelta().getY();
        double differenceYaw = Math.abs(deltaYaw - this.lastDeltaYaw);
        double differencePitch = Math.abs(deltaPitch - this.lastDeltaPitch);
        double joltYaw = Math.abs(differenceYaw - deltaYaw);
        double joltPitch = Math.abs(differencePitch - deltaPitch);
        boolean cinematic = (now - this.lastHighRate > 250L || now - this.lastSmooth < 9000L);
        if (joltYaw > 1.0D && joltPitch > 1.0D)
            this.lastHighRate = now;
        this.yawSamples.add(deltaYaw);
        this.pitchSamples.add(deltaPitch);
        if (this.yawSamples.size() >= 20 && this.pitchSamples.size() >= 20) {
            Set<Double> shannonYaw = new HashSet<>(), shannonPitch = new HashSet<>();
            List<Double> stackYaw = new ArrayList<>(), stackPitch = new ArrayList<>();
            for (int i = 0; i < this.yawSamples.size(); i++) {
                stackYaw.add(this.yawSamples.get(i));
                stackPitch.add(this.pitchSamples.get(i));
                if (stackYaw.size() >= 10 && stackPitch.size() >= 10) {
                    shannonYaw.add(Statistics.getShannonEntropy(stackYaw));
                    shannonPitch.add(Statistics.getShannonEntropy(stackPitch));
                    stackYaw.clear();
                    stackPitch.clear();
                }
            }
            if (shannonYaw.size() != 1 || shannonPitch.size() != 1 || (
                    (Double) shannonYaw.toArray()[0]).doubleValue() != ((Double) shannonPitch.toArray()[0]).doubleValue()) {
                this.isTotallyNotCinematic = 20;
            }
            GraphUtil.GraphResult resultsYaw = GraphUtil.getGraph(this.yawSamples);
            GraphUtil.GraphResult resultsPitch = GraphUtil.getGraph(this.pitchSamples);
            int negativesYaw = resultsYaw.getNegatives();
            int negativesPitch = resultsPitch.getNegatives();
            int positivesYaw = resultsYaw.getPositives();
            int positivesPitch = resultsPitch.getPositives();
            if (positivesYaw > negativesYaw || positivesPitch > negativesPitch)
                this.lastSmooth = now;
            this.yawSamples.clear();
            this.pitchSamples.clear();
        }
        if (this.isTotallyNotCinematic > 0) {
            this.isTotallyNotCinematic--;
            this.profile.setCinematic(false);
        } else {
            this.profile.setCinematic(cinematic);
        }
        this.lastDeltaYaw = deltaYaw;
        this.lastDeltaPitch = deltaPitch;
    }
}