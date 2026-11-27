package kireiko.dev.anticheat.checks.clicks;

import com.comphenix.protocol.PacketType;
import com.comphenix.protocol.events.PacketEvent;
import kireiko.dev.anticheat.api.PacketCheckHandler;
import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.*;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.managers.CheckManager;
import kireiko.dev.millennium.math.Statistics;

import java.util.*;

public final class AutoClickerCheck implements PacketCheckHandler {
    private final PlayerProfile profile;
    private long oldTime = System.currentTimeMillis(),
                    lastMove = System.currentTimeMillis(), lastAttack = System.currentTimeMillis();
    private boolean enabled = false;
    private final List<Long> stack = new ArrayList<>();
    private boolean entropyQuery = false;
    private Map<String, Object> localCfg = new TreeMap<>();

    @Override
    public ConfigLabel config() {
        localCfg.put("enabled", false);
        localCfg.put("addGlobalVl", 20);
        return new ConfigLabel("auto_clicker", localCfg);
    }
    @Override
    public void applyConfig(Map<String, Object> params) {
        localCfg = params;
    }

    @Override
    public Map<String, Object> getConfig() {
        return localCfg;
    }


    public AutoClickerCheck(PlayerProfile profile) {
        this.profile = profile;
        if (CheckManager.classCheck(this.getClass()))
            this.localCfg = CheckManager.getConfig(this.getClass());
    }

    @Override
    public void event(Object o) {
        if (!(boolean) localCfg.get("enabled")) return;
        if (o instanceof CPacketEvent) {
            PacketEvent event = ((CPacketEvent) o).getPacketEvent();
            PacketType type = event.getPacket().getType();
            if (type.equals(PacketType.Play.Client.BLOCK_DIG)) {
                enabled = false;
            } else if (type.equals(PacketType.Play.Client.USE_ENTITY)) {
                lastAttack = System.currentTimeMillis();
                enabled = true;
            } else if (type.equals(PacketType.Play.Client.ARM_ANIMATION)) {
                long delay = (System.currentTimeMillis() - oldTime) / 50;
                if (delay < 25
                     && enabled
                     && lastMove + 500 > System.currentTimeMillis()
                     && lastAttack + 7000 > System.currentTimeMillis()
                ) {
                    stack.add(delay);
                    if (stack.size() > 100) {
                        check();
                    }
                }
                oldTime = System.currentTimeMillis();
            }
        } else if (o instanceof NoRotationEvent || o instanceof RotationEvent || o instanceof MoveEvent) {
            lastMove = System.currentTimeMillis();
        }
    }

    private void check() {
        { // analysis
            final List<Double> kurtosisStack = new ArrayList<>();
            final List<Double> shannonStack = new ArrayList<>();
            final List<Double> localDeltaStack = new ArrayList<>();
            for (double delay : stack) {
                localDeltaStack.add(delay);
                if (localDeltaStack.size() >= 20) {
                    kurtosisStack.add(Statistics.getKurtosis(localDeltaStack));
                    shannonStack.add(Statistics.getShannonEntropy(localDeltaStack));
                    localDeltaStack.clear();
                }
            }
            final float vl = ((Number) localCfg.get("addGlobalVl")).floatValue() / 10;
            if (Statistics.getMax(kurtosisStack) < 0) {
                profile.punish("AutoClicker", "Kurtosis", "Analysis <negative> [" + kurtosisStack + "]", vl);
            } else {
                final List<Float> jiff = Statistics.getJiffDelta(shannonStack, 2);
                double min = Statistics.getMin(jiff);
                if (min < 0.04 && Statistics.getMax(jiff) < 0.06) {
                    if (!entropyQuery) {
                        entropyQuery = true;
                    } else {
                        profile.punish("AutoClicker", "Entropy", "Analysis <min> (" + jiff + ") => " + min, vl);
                    }
                } else {
                    entropyQuery = false;
                }
            }
        }
        stack.clear();
    }
}
