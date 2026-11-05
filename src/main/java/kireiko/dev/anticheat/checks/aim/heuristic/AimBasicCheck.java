package kireiko.dev.anticheat.checks.aim.heuristic;

import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.RotationEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.checks.aim.AimHeuristicCheck;
import kireiko.dev.millennium.math.Simplification;
import kireiko.dev.millennium.math.Statistics;
import kireiko.dev.millennium.types.ConcurrentEvictingList;
import kireiko.dev.millennium.vectors.Vec2;

import java.util.*;

public final class AimBasicCheck implements HeuristicComponent {
    private final AimHeuristicCheck check;
    private final List<Vec2> rawRotations;
    private int streak = 0;
    private float vl = 0, vlL2 = 0;
    private String reason = "";
    private Map<String, Object> localCfg = new TreeMap<>();

    public AimBasicCheck(final AimHeuristicCheck check) {
        this.check = check;
        this.rawRotations = new ConcurrentEvictingList<>(16);
    }

    @Override
    public void process(final RotationEvent event) {
        if (check.getProfile().ignoreCinematic()) return;
        if (event.getAbsDelta().getY() == 0 && event.getAbsDelta().getY() == 0) return;
        this.rawRotations.add(new Vec2(event.getTo().getX(), event.getTo().getY()));
        final PlayerProfile profile = check.getProfile();
        //profile.getPlayer().sendMessage("x: " + event.getDelta().getX() + " y: " + event.getDelta().getY());
        final int randomizerFlawVl = (int) localCfg.get("randomizer_flaw_addVl");
        if (randomizerFlawVl > 0 && (((event.getDelta().getY() > 1.5f || event.getDelta().getX() > 3.0f)
                && (profile.getTo().getPitch() == 0
                || profile.getTo().getPitch() % 0.01f == 0)))) {
            profile.punish("Aim", "Randomizer", "[Heuristic] Randomizer flaw", (randomizerFlawVl) / 10f);
        }
        if (this.rawRotations.size() >= 10) checkDefaultAim();
    }

    @Override
    public ConfigLabel config() {
        localCfg.put("heuristic(sync)_addLocalVl", 125);
        localCfg.put("heuristic(aggressive)_addLocalVl", 50);
        localCfg.put("heuristic(aim)_addLocalVl", 100);
        localCfg.put("heuristic(constant)_addLocalVl", 65);
        localCfg.put("heuristic(interpolation)_addLocalVl", 55);
        localCfg.put("pattern(snap)_addLocalVl", 55);
        localCfg.put("pattern(random)_addLocalVl", 25);
        localCfg.put("addGlobalVl(component)", 0);
        localCfg.put("addGlobalVl(interpolation)", 0);
        localCfg.put("hitCancelTimeMS(component)", 3500);
        localCfg.put("hitCancelTimeMS(interpolation)", 2500);
        localCfg.put("localVl_limit(component)", 400);
        localCfg.put("localVl_limit(interpolation)", 400);
        localCfg.put("localVl_fade(component)", 5);
        localCfg.put("localVl_fade(interpolation)", 5);
        localCfg.put("randomizer_flaw_addVl", 25);
        return new ConfigLabel("basic_component", localCfg);
    }

    @Override
    public void applyConfig(Map<String, Object> params) {
        localCfg = params;
    }

    private void checkDefaultAim() {
        final List<Double> robotizedStack = new ArrayList<>();
        { // standard checks for bad aim assist
            final List<Vec2> rotations = this.rawRotations;
            Set<Double> yaws = new HashSet<>();
            {
                double oldYaw = rotations.get(0).getX();
                for (Vec2 r : rotations) {
                    yaws.add(Math.abs(r.getX() - oldYaw));
                    oldYaw = r.getX();
                }
            }
            double oldYawResult = rotations.get(0).getX();
            double oldPitchResult = rotations.get(0).getY();
            double oldYawChange = Math.abs(rotations.get(0).getX() - oldYawResult);
            double yawChangeFirst = Math.abs(rotations.get(0).getX() - rotations.get(1).getX());
            int machineKnownMovement = 0,
                    constantRotations = 0, gcd = 0, aggressivePatternI = 0,
                    aggressivePatternD = 0, aggressivePatternI2 = 0, aggressivePatternD2 = 0,
                    robotizedAmount = 0, aggressiveAim = 0, infinitives = 0;
            for (Vec2 rotation : rotations) {
                double yawChange = Math.abs(rotation.getX() - oldYawResult);
                double pitchChange = Math.abs(rotation.getY() - oldPitchResult);
                double robotized = Math.abs(yawChange - yawChangeFirst);
                double diffBetweenYawChanges = yawChange - oldYawChange;
                double interpolation;
                float yaw = (float) rotation.getX();

                if (robotized < 2 && yawChange > 2.5) robotizedAmount += 1;
                if (robotized < 0.99 && yawChange > 4) machineKnownMovement++;
                if (robotized < 0.02 && yawChange > 3) constantRotations++;
                if (robotized < 2 && yawChange > 3) aggressiveAim++;
                interpolation = Simplification.scaleVal(yawChange / robotized, 2);
                if (Double.isInfinite(interpolation) && yawChange > 0) {
                    infinitives++;
                    if (infinitives > 1 && yawChange < 0.4) {
                        infinitives--;
                    }
                }
                if (robotized != 0) robotizedStack.add(robotized);
                if (yawChange == 0.1 || pitchChange == 0.1) gcd++;
                if (yawChange == 0.01 || pitchChange == 0.01) gcd++;
                if ((diffBetweenYawChanges > 0.01 && diffBetweenYawChanges < 2)) aggressivePatternI++;
                if ((diffBetweenYawChanges < -0.01 && diffBetweenYawChanges > -2)) aggressivePatternD++;
                if (diffBetweenYawChanges > 2) aggressivePatternI2++;
                if (diffBetweenYawChanges < -2) aggressivePatternD2++;
                oldYawResult = yaw;
                oldPitchResult = rotation.getY();
                oldYawChange = yawChange;
            }
            final int sens = check.getProfile().calculateSensitivity();
            if (sens > 65) {
                if (robotizedAmount > 8) addNewPunish("heuristic(sync)", getNumCfg("heuristic(sync)_addLocalVl"));
                if (aggressiveAim > 8) addNewPunish("heuristic(aggressive)", getNumCfg("heuristic(aggressive)_addLocalVl"));
                if (machineKnownMovement > 7) addNewPunish("heuristic(aim)", getNumCfg("heuristic(aim)_addLocalVl"));
                if (constantRotations > 3) addNewPunish("heuristic(constant)", getNumCfg("heuristic(constant)_addLocalVl"));
            } else {
                if (machineKnownMovement > 8) addNewPunish("heuristic(aim)", getNumCfg("heuristic(aim)_addLocalVl"));
                if (constantRotations > 6) addNewPunish("heuristic(constant)", getNumCfg("heuristic(constant)_addLocalVl"));
            }
            if (infinitives > 1 && Math.abs(Statistics.getAverage(yaws)) > 3.2) {
                addNewPunishL2("heuristic(interpolation)", getNumCfg("heuristic(interpolation)_addLocalVl"));
                //profile.debug("avg: " + Math.abs(Statistics.getAverage(yaws)));
            }
            if (gcd > 0) addNewPunish("pattern(gcd)", 1000);
            if (aggressivePatternI > 3 && aggressivePatternD > 3)
                addNewPunishL2("pattern(random)", getNumCfg("pattern(random)_addLocalVl"));
            if (aggressivePatternI2 > 3 && aggressivePatternD2 > 3
                    && (aggressivePatternI2 + aggressivePatternD2) > 8) {
                streak++;
                if (streak > 2) addNewPunish("pattern(snap)", getNumCfg("pattern(snap)_addLocalVl"));
            } else streak = 0;
            if (this.vl > ((Number) localCfg.get("localVl_limit(component)")).intValue()) {
                this.check.getProfile().punish("Aim", "Heuristic", "[Component] " + this.reason,
                                getNumCfg("addGlobalVl(component)") / 10);
                this.check.getProfile().setAttackBlockToTime(System.currentTimeMillis()
                                + ((Number) localCfg.get("hitCancelTimeMS(component)")).longValue());
                this.vl = 360;
            }
            if (this.vlL2 > ((Number) localCfg.get("localVl_limit(interpolation)")).intValue()) {
                this.check.getProfile().punish("Aim", "Flaw", "[Flaw] Interpolation",
                                getNumCfg("addGlobalVl(interpolation)") / 10);
                this.check.getProfile().setAttackBlockToTime(System.currentTimeMillis()
                                + ((Number) localCfg.get("hitCancelTimeMS(interpolation)")).longValue());
                this.vlL2 -= 65;
            }
            if (this.vl > 0) this.vl -= getNumCfg("localVl_fade(component)");
            if (this.vl > 400) this.vl -= 10;
            if (this.vlL2 > 0) this.vlL2 -= getNumCfg("localVl_fade(interpolation)");
            if (this.vlL2 > 380) this.vlL2 -= 10;
        }
        this.rawRotations.clear();
        /*
        this.check.getProfile().getPlayer().sendMessage("r: " +
                        Statistics.getStandardDeviation(Statistics.getJiffDelta(robotizedStack, 1)));
         */
    }

    private void addNewPunish(String reason, float vl) {
        this.reason = reason;
        this.vl += vl;
        check.getProfile().debug("&7Aim Component: " + reason
                + " " + this.vl + " (+" + vl + ")");
    }

    private void addNewPunishL2(String reason, float vl) {
        this.vlL2 += vl;
        check.getProfile().debug("&7Interpolation Component: " + reason
                + " " + this.vlL2 + " (+" + vl + ")");
    }
    private float getNumCfg(String key) {
        return ((Number) localCfg.get(key)).floatValue();
    }
}