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
import java.util.function.Function;

public final class AimStatisticsCheck implements PacketCheckHandler {
    private final List<Float> buffer;
    private final PlayerProfile profile;
    private final List<Vec2f> rawRotations;
    private final List<Double> shannonAnalysis;
    private long lastAttack;
    private Map<String, Object> localCfg = new TreeMap<>();

    public AimStatisticsCheck(PlayerProfile profile) {
        this.profile = profile;
        this.rawRotations = new ConcurrentEvictingList<>(32);
        this.lastAttack = 0L;
        this.buffer = new ConcurrentEvictingList<>(16);
        this.shannonAnalysis = new ConcurrentEvictingList<>(16);
        for (int i = 0; i < 16; i++) this.buffer.add(0.0f);
        if (CheckManager.classCheck(this.getClass()))
            this.localCfg = CheckManager.getConfig(this.getClass());
    }

    @Override
    public ConfigLabel config() {
        localCfg.put("addGlobalVl(iqr)", 25);
        localCfg.put("addGlobalVl(bot_pattern)", 35);
        localCfg.put("addGlobalVl(zfactor)", 0);
        localCfg.put("addGlobalVl(improbable)", 30);
        localCfg.put("hitCancelTimeMS(iqr)", 0);
        localCfg.put("hitCancelTimeMS(bot_pattern)", 0);
        localCfg.put("hitCancelTimeMS(zfactor)", 5500);
        localCfg.put("hitCancelTimeMS(improbable)", 4000);
        localCfg.put("localVlLimit(iqr)", 11.0f);
        localCfg.put("localVlLimit(zfactor)", 7.0f);
        localCfg.put("localVlLimit(improbable)", 15.0f);
        return new ConfigLabel("aim_statistics", localCfg);
    }
    @Override
    public void applyConfig(Map<String, Object> params) {
        localCfg = params;
    }

    @Override
    public Map<String, Object> getConfig() {
        return localCfg;
    }

    private static double getDifference(double a, double b) {
        return Math.abs(Math.abs(a) - Math.abs(b));
    }

    @Override
    public void event(Object o) {
        if (o instanceof RotationEvent) {
            RotationEvent event = (RotationEvent) o;
            if (System.currentTimeMillis() > this.lastAttack + 3500) return;
            Vec2f delta = event.getDelta();
            this.rawRotations.add(delta);
            if (this.rawRotations.size() >= 25) this.checkRaw();
        } else if (o instanceof UseEntityEvent) {
            UseEntityEvent event = (UseEntityEvent) o;
            if (event.isAttack()) {
                this.lastAttack = System.currentTimeMillis();
            }
        }
    }

    private void checkRaw() {
        float total = 0;
        { // uh
            //profile.getPlayer().sendMessage("i: " + profile.getSensitivityProcessor().getSensitivity());
            List<Float> x = new ArrayList<>(), y = new ArrayList<>();
            for (Vec2f vec2 : this.rawRotations) {
                x.add(vec2.getX());
                y.add(vec2.getY());
            }
            final List<Double> zFactorYaw = Statistics.getZScoreOutliers(x, 2.0);
            final List<Float> jiffYaw = Statistics.getJiffDelta(x, 5);
            final List<Float> jiffPitch = Statistics.getJiffDelta(y, 5);
            final List<Float> jiffOmni = new ArrayList<>();
            for (int i = 0; i < Math.min(jiffYaw.size(), jiffPitch.size()); i++) {
                jiffOmni.add(jiffYaw.get(i) / (jiffPitch.get(i)));
            }
            { // omni check
                int infs = 0;
                for (float j : jiffOmni) if (Float.isInfinite(j)) infs++;
                final double iqr = Statistics.getIQR(jiffOmni);
                if (iqr > 12.5 && iqr < 96 && infs > 0) {
                    this.increaseBuffer(8, (iqr > 20) ? 1.4f : 0.8f);
                    this.profile.debug("&7Aim IQR: " + this.buffer.get(8) + " infs: " + infs);
                    final float vl = ((Number) localCfg.get("addGlobalVl(iqr)")).floatValue() / 10f;
                    final long cancel = ((Number) localCfg.get("hitCancelTimeMS(iqr)")).longValue();
                    final float vlLimit = ((Number) localCfg.get("localVlLimit(iqr)")).floatValue();
                    if (this.buffer.get(8) > vlLimit) {
                        if (cancel > 0 || vl > 0) {
                            this.profile.punish("Aim", "IQR", "[Statistics] IQR " + iqr, vl);
                            this.profile.setAttackBlockToTime(System.currentTimeMillis() + cancel);
                        }
                        this.buffer.set(8, vlLimit - 2f);
                    }
                } else if (iqr < 13 || infs == 0) {
                    if (iqr < 7) {
                        this.increaseBuffer(8, -5.0f);
                    } else this.increaseBuffer(8, -3.5f);
                }
            }
            //profile.getPlayer().sendMessage("j: " + Arrays.toString(jiffOmni.toArray()));
            final double kTest = Statistics.kolmogorovSmirnovTest(Statistics.getJiffDelta(x, 6), Function.identity());
            { // Kolmogorov Smirnov Test
                if (kTest > 10 && Math.abs(Statistics.getAverage(x)) < 13) {
                    total++;
                }
            }
            shannonAnalysis.add(Statistics.getShannonEntropy(jiffYaw));
            if (shannonAnalysis.size() > 9) {
                final Set<Double> uniq = new HashSet<>(shannonAnalysis);
                final double diff =
                        getDifference(Statistics.getMin(uniq), Statistics.getMax(uniq));
                if (uniq.size() <= 5 && uniq.size() > 3 && diff < 0.38) {
                    // this.profile.punish("Aim", "[Statistics] AimBot Entropy Heuristic " + uniq.size() + " " + diff, 3.0f);
                }
                shannonAnalysis.clear();
            }
            int jiffPatterns = 0;
            for (int i = 0; i < jiffYaw.size(); i++) {
                float f = jiffYaw.get(i);
                if (!String.valueOf(f).contains("E") || f == 0) continue;
                for (int r = 0; r < jiffYaw.size(); r++) {
                    if (r == i) continue;
                    if (f == jiffYaw.get(r))
                        jiffPatterns++;
                }
            }
            if (jiffPatterns > 2 && Statistics.getAverage(x) > 3.0
                    && jiffPatterns != 6 && jiffPatterns != 12 && jiffPatterns != 4) {
                final float vl = ((Number) localCfg.get("addGlobalVl(bot_pattern)")).floatValue() / 10f;
                final long cancel = ((Number) localCfg.get("hitCancelTimeMS(bot_pattern)")).longValue();
                if (vl > 0 || cancel > 0) {
                    this.profile.punish("Aim", "Pattern", "[Statistics] AimBot pattern " + jiffPatterns, vl);
                    this.profile.setAttackBlockToTime(System.currentTimeMillis() + cancel);
                }
            }
            //profile.getPlayer().sendMessage("j: " + Arrays.toString(jiffYaw.toArray()));

            boolean positive = false, negative = false;
            for (double d : zFactorYaw) {
                if (d > 10) positive = true;
                if (d < -10) negative = true;
            }
            if (zFactorYaw.size() == 2 && positive && negative
                    && Statistics.getMax(zFactorYaw) < 55) {
                this.increaseBuffer(0, 1.5f);
                if (this.buffer.get(0) > 4)
                    total++;
                profile.debug("&7Aim zFactor: " + this.buffer.get(0));
                final float vlLimit = ((Number) localCfg.get("localVlLimit(zfactor)")).floatValue();
                if (this.buffer.get(0) > vlLimit) {
                    final float vl = ((Number) localCfg.get("addGlobalVl(zfactor)")).floatValue() / 10f;
                    final long cancel = ((Number) localCfg.get("hitCancelTimeMS(zfactor)")).longValue();
                    if (vl > 0 || cancel > 0) {
                        this.profile.punish("Aim", "Factor", "[Statistics] Suspicious zFactor " + zFactorYaw, vl);
                        this.profile.setAttackBlockToTime(System.currentTimeMillis() + cancel);
                    }
                    this.buffer.set(0, vlLimit - 1f);
                }
            } else this.increaseBuffer(0, -1.2f);
        }
        { // total
            if (total > 0) profile.debug("&7Aim Statistics Total: " + total);
            if (total < 2.0) {
                this.increaseBuffer(10, -2f);
            } else if (total > 2.0) {
                this.increaseBuffer(10, 5f);
                final float vlLimit = ((Number) localCfg.get("localVlLimit(improbable)")).floatValue();
                if (this.buffer.get(10) >= vlLimit) {
                    final float vl = ((Number) localCfg.get("addGlobalVl(improbable)")).floatValue() / 10f;
                    final long cancel = ((Number) localCfg.get("hitCancelTimeMS(improbable)")).longValue();
                    if (vl > 0 || cancel > 0) {
                        this.profile.punish("Aim", "Statistics", "[Statistics] Improbable " + this.buffer.get(10), 3.0f);
                        this.profile.setAttackBlockToTime(System.currentTimeMillis() + cancel);
                    }
                    this.increaseBuffer(10, vlLimit - 2.0f);
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
