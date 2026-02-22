package kireiko.dev.anticheat.checks.aim.ml;

import kireiko.dev.anticheat.api.PacketCheckHandler;
import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.RotationEvent;
import kireiko.dev.anticheat.api.events.UseEntityEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.core.AsyncScheduler;
import kireiko.dev.anticheat.managers.CheckManager;
import kireiko.dev.anticheat.managers.DatasetManager;
import kireiko.dev.millennium.ml.ClientML;
import kireiko.dev.millennium.ml.FactoryML;
import kireiko.dev.millennium.ml.data.ObjectML;
import kireiko.dev.millennium.ml.data.ResultML;
import kireiko.dev.millennium.ml.data.module.FlagType;
import kireiko.dev.millennium.ml.data.module.ModuleML;
import kireiko.dev.millennium.ml.data.module.ModuleResultML;
import kireiko.dev.millennium.vectors.Vec2f;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public final class AimMLCheck implements PacketCheckHandler {

    private static final boolean TEST_MODE = false;
    public static final Map<UUID, Boolean> RECORDING = new ConcurrentHashMap<>();

    private final PlayerProfile profile;
    private final List<Vec2f> rawRotations;
    private long lastAttack;
    private Map<String, Object> localCfg = new TreeMap<>();

    public AimMLCheck(PlayerProfile profile) {
        this.profile = profile;
        this.rawRotations = new CopyOnWriteArrayList<>();
        this.lastAttack = 0L;
        if (CheckManager.classCheck(this.getClass()))
            this.localCfg = CheckManager.getConfig(this.getClass());
    }

    @Override
    public ConfigLabel config() {
        localCfg.put("enabled", true);
        localCfg.put("unusual_vl", 10);
        localCfg.put("strange_vl", 20);
        localCfg.put("suspected_vl", 40);
        return new ConfigLabel("aim_ml", localCfg);
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
            if (profile.isCinematic()) return;
            if (!((boolean) getConfig().get("enabled"))) return;
            RotationEvent event = (RotationEvent) o;
            if (System.currentTimeMillis() > this.lastAttack + 2500) return;

            Vec2f delta = event.getDelta();
            this.rawRotations.add(delta);

            if (TEST_MODE && !RECORDING.containsKey(profile.getPlayer().getUniqueId())) {
                profile.getPlayer().sendActionBar("§aChecking: " + this.rawRotations.size() + "/600");
            } else if (RECORDING.containsKey(profile.getPlayer().getUniqueId())) {
                profile.getPlayer().sendActionBar("§cRECORDING: " + this.rawRotations.size() + "/600");
            }

            if (this.rawRotations.size() >= 600) this.check();
        } else if (o instanceof UseEntityEvent) {
            UseEntityEvent event = (UseEntityEvent) o;
            if (event.isAttack()) {
                this.lastAttack = System.currentTimeMillis();
            }
        }
    }

    private void checkResult(List<ObjectML> objectML) {
        ModuleResultML finalModuleResult = new ModuleResultML(0, FlagType.NORMAL, null);
        final Set<String> modelsThatFlagged = new HashSet<>();

        for (int i = 0; i < ClientML.MODEL_LIST.size(); i++) {
            final ResultML resultML = FactoryML.getModel(i).checkData(objectML);
            ModuleML moduleML = ClientML.MODEL_LIST.get(i);
            final ModuleResultML moduleResultML = moduleML.getResult(resultML);

            if (i == 7) {
                profile.debug("&dRNN Prob: " + moduleResultML.getInfo() + " (" + moduleResultML.getType() + ")");
            }

            if (moduleResultML.getType() != FlagType.NORMAL) {
                modelsThatFlagged.add(moduleML.getName());
            }

            if (finalModuleResult.getInfo() == null) {
                finalModuleResult = moduleResultML;
            } else {
                final int finalLevel = finalModuleResult.getType().getLevel();
                final int tempLevel = moduleResultML.getType().getLevel();
                if (finalLevel < tempLevel || (finalLevel == tempLevel && finalModuleResult.getPriority() < moduleResultML.getPriority())) {
                    finalModuleResult = moduleResultML;
                }
            }
        }

        profile.debug("&8ML Result: " + finalModuleResult.getType() + " " + finalModuleResult.getInfo());

        if (finalModuleResult.getType() != FlagType.NORMAL) {
            final FlagType type = finalModuleResult.getType();
            float vl = 0f;
            String color = "&a";

            switch (type) {
                case UNUSUAL:
                    color = "&e";
                    vl = ((Number) localCfg.get("unusual_vl")).floatValue() / 10f;
                    break;
                case STRANGE:
                    color = "&6";
                    vl = ((Number) localCfg.get("strange_vl")).floatValue() / 10f;
                    break;
                case SUSPECTED:
                    color = "&c";
                    vl = ((Number) localCfg.get("suspected_vl")).floatValue() / 10f;
                    break;
                case NORMAL:
                    break;
            }
            profile.punish("Aim", "ML", "&fResult: " + color + type + " &8" + Arrays.toString(modelsThatFlagged.toArray()), vl);
        }
    }

    private void check() {
        final List<ObjectML> objectMLStack = new ArrayList<>();
        ObjectML yaw = new ObjectML(new ArrayList<>());
        ObjectML pitch = new ObjectML(new ArrayList<>());

        for(Vec2f rot : this.rawRotations) {
            yaw.getValues().add((double) rot.getX());
            pitch.getValues().add((double) rot.getY());
        }

        objectMLStack.add(yaw);
        objectMLStack.add(pitch);

        Boolean recordType = RECORDING.get(profile.getPlayer().getUniqueId());

        if (recordType != null) {
            AsyncScheduler.run(() -> {
                DatasetManager.saveSample(objectMLStack, recordType);
                profile.getPlayer().sendMessage("§a[ML] Saved sample! Total in dataset: " + DatasetManager.getCount());
            });
        } else {
            AsyncScheduler.run(() -> checkResult(objectMLStack));
        }

        this.rawRotations.clear();
    }
}