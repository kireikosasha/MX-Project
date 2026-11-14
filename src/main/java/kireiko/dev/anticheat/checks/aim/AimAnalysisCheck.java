package kireiko.dev.anticheat.checks.aim;

import kireiko.dev.anticheat.api.PacketCheckHandler;
import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.RotationEvent;
import kireiko.dev.anticheat.api.events.UseEntityEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.managers.CheckManager;
import kireiko.dev.millennium.math.Statistics;
import kireiko.dev.millennium.types.ConcurrentEvictingList;
import kireiko.dev.millennium.vectors.Vec2f;

import java.util.*;

public final class AimAnalysisCheck implements PacketCheckHandler {
    private final List<Float> buffer;
    private final PlayerProfile profile;
    private final List<Vec2f> rawRotations, limitedRotations;
    private final List<Float> longTermAnalysis;
    private long lastAttack;
    private boolean query = false;
    private Map<String, Object> localCfg = new TreeMap<>();

    public AimAnalysisCheck(PlayerProfile profile) {
        this.profile = profile;
        this.rawRotations = new ConcurrentEvictingList<>(128);
        this.limitedRotations = new ConcurrentEvictingList<>(128);
        this.longTermAnalysis = new ConcurrentEvictingList<>(16);
        this.lastAttack = 0L;
        this.buffer = new ConcurrentEvictingList<>(16);
        for (int i = 0; i < 16; i++) this.buffer.add(0.0f);
        if (CheckManager.classCheck(this.getClass()))
            this.localCfg = CheckManager.getConfig(this.getClass());
    }

    @Override
    public ConfigLabel config() {
        localCfg.put("addGlobalVl(linear)", 35);
        localCfg.put("addGlobalVl(rank)", 20);
        localCfg.put("addGlobalVl(longterm)", 55);
        localCfg.put("localVlLimit(rank)", 6.0f);
        return new ConfigLabel("aim_analysis", localCfg);
    }

    @Override
    public void applyConfig(Map<String, Object> params) {
        localCfg = params;
    }

    @Override
    public Map<String, Object> getConfig() {
        return localCfg;
    }

    @Override
    public void event(Object o) {
        if (o instanceof RotationEvent) {
            RotationEvent event = (RotationEvent) o;
            if (System.currentTimeMillis() > this.lastAttack + 3500) return;
            Vec2f delta = event.getDelta();
            this.rawRotations.add(delta);
            if (this.rawRotations.size() >= 100) this.checkRaw();
            if (Math.abs(delta.getX()) > 1.35 || Math.abs(delta.getY()) > 1.35 && Math.abs(delta.getX()) > 0.32) {
                this.limitedRotations.add(delta);
                if (this.limitedRotations.size() >= 100) this.checkLimited();
            }
        } else if (o instanceof UseEntityEvent) {
            UseEntityEvent event = (UseEntityEvent) o;
            if (event.isAttack()) {
                this.lastAttack = System.currentTimeMillis();
            }
        }
    }

    private void checkLimited() {
        {
            final List<Float> x = new ArrayList<>(), xAbs = new ArrayList<>(), y = new ArrayList<>();
            final int sens = profile.calculateSensitivity();
            for (Vec2f vec2 : this.limitedRotations) {
                x.add(vec2.getX());
                xAbs.add(vec2.getX());
                y.add(vec2.getY());
            }
            { // limited analysis
                final List<Float> yawStack = new ArrayList<>();
                int resultDistinct = 0;
                for (final float yaw : x) {
                    yawStack.add(yaw);
                    if (yawStack.size() >= 10) {
                        resultDistinct += Statistics.getDistinct(Statistics.getJiffDelta(yawStack, 4));
                        yawStack.clear();
                    }
                }
                final float distinctRank = (float) resultDistinct / 60;
                longTermAnalysis.add(distinctRank);
                profile.debug("&7Long-Term Aim Analysis " + longTermAnalysis.size() + "/10");
                if (longTermAnalysis.size() >= 10) {
                    final double avg = Statistics.getAverage(longTermAnalysis);
                    double normal = 0;
                    for (double d : longTermAnalysis) if (d > 0.97) normal++;
                    profile.debug("&7Long-Term Aim Analysis | avg: " + avg + " | normal: " + normal + "/10");
                    if (avg < 0.95 && normal < 4) {
                        final float vl = ((Number) localCfg.get("addGlobalVl(longterm)")).floatValue() / 10f;
                        if (vl > 0) {
                            this.profile.punish("Aim", "Analysis", "Long term analysis (average rank: " + avg + ", normal: " + normal + " /10)", vl);
                        }
                    }
                    //profile.getPlayer().sendMessage("avg: " + avg + " | distinct: " + normal);

                    longTermAnalysis.clear();
                }
            }
        }
        this.limitedRotations.clear();
    }

    private void checkRaw() {
        { // uh
            final List<Float> x = new ArrayList<>(), xAbs = new ArrayList<>(), y = new ArrayList<>();
            final int sens = profile.calculateSensitivity();
            for (Vec2f vec2 : this.rawRotations) {
                x.add(vec2.getX());
                xAbs.add(vec2.getX());
                y.add(vec2.getY());
            }
            { // score
                final List<Float> yawStack = new ArrayList<>();
                final List<Double> resultDeviation = new ArrayList<>();
                int resultDistinct = 0;
                for (final float yaw : x) {
                    yawStack.add(yaw);
                    if (yawStack.size() >= 10) {
                        resultDeviation.add(Statistics.getStandardDeviation(Statistics.getJiffDelta(yawStack, 5)));
                        resultDistinct += Statistics.getDistinct(Statistics.getJiffDelta(yawStack, 4));
                        yawStack.clear();
                    }
                }
                final List<Double> outliers5 = Statistics.getZScoreOutliers(resultDeviation, 0.5f);
                final float distinctRank = (float) resultDistinct / 60;
                { // linear
                    if (outliers5.isEmpty() || outliers5.size() == 1 && Math.abs(outliers5.get(0)) > 10 && Math.abs(outliers5.get(0)) < 100) {
                        final float vl = ((Number) localCfg.get("addGlobalVl(linear)")).floatValue() / 10f;
                        if (vl > 0) {
                            if (!query) {
                                query = true;
                            } else {
                                this.profile.punish("Aim", "Linear", "[Analysis] Invalid outliers "
                                                + Arrays.toString(outliers5.toArray()), vl);
                            }
                        }
                    } else {
                        query = false;
                    }
                }
                { // rank
                    final boolean valid = profile.calculateSensitivity() > 20 && sens < 140;
                    if (distinctRank < 1.0 && distinctRank > 0.7 && Statistics.getAverage(xAbs) > 1.8 && valid) {
                        if (this.buffer.get(1) < 0.01) {
                            if (distinctRank < 0.8) this.increaseBuffer(1, 0.2f);
                        } else {
                            final float limit = ((Number) localCfg.get("localVlLimit(rank)")).floatValue();
                            this.increaseBuffer(1, (distinctRank > 0.9) ? 0.08f : (distinctRank > 0.8) ? 2f : 3f);
                            profile.debug("&7Aim Incorrect rank: " + this.buffer.get(1) + " (" + distinctRank + ")");
                            if (this.buffer.get(1) >= limit) {
                                final float vl = ((Number) localCfg.get("addGlobalVl(rank)")).floatValue() / 10f;
                                if (vl > 0) {
                                    this.profile.punish("Aim", "Rank", "[Analysis] Incorrect rank " + distinctRank, vl);
                                }
                                this.buffer.set(1, limit - 1);
                            }
                        }

                    } else this.increaseBuffer(1, -2.25f);
                }
            }
        }
        this.rawRotations.clear();
    }

    private void increaseBuffer(int index, float v) {
        float r = this.buffer.get(index) + v;
        this.buffer.set(index, (r < 0) ? 0 : r);
    }

}
