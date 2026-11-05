package kireiko.dev.anticheat.api.player;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.CheckPacketRegister;
import kireiko.dev.anticheat.api.PacketCheckHandler;
import kireiko.dev.anticheat.api.events.MXFlagEvent;
import kireiko.dev.anticheat.managers.CheckManager;
import kireiko.dev.anticheat.services.AnimatedPunishService;
import kireiko.dev.anticheat.utils.ConfigCache;
import kireiko.dev.anticheat.utils.MessageUtils;
import kireiko.dev.anticheat.utils.protocol.ProtocolLib;
import kireiko.dev.millennium.math.Statistics;
import kireiko.dev.millennium.types.ConcurrentEvictingList;
import kireiko.dev.millennium.vectors.Pair;
import lombok.Data;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.entity.Player;

import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ThreadLocalRandom;

@Data
public final class PlayerProfile {

    private final Object stateLock = new Object();

    private final Player player;
    private final Set<PacketCheckHandler> checks = new HashSet<>();
    private final List<Location> pastLoc = new ConcurrentEvictingList<>(20);
    private final List<Long> ping = new ConcurrentEvictingList<>(10);
    private final List<Integer> sensitivity = new ConcurrentEvictingList<>(14);
    private final SensitivityProcessor sensitivityProcessor = new SensitivityProcessor(this);
    private final CinematicComponent cinematicComponent = new CinematicComponent(this);
    private final List<String> logs = new CopyOnWriteArrayList<>();
    public volatile boolean transactionSentKeep;
    public volatile boolean transactionBoot = true;
    public volatile long transactionTime, transactionLastTime, transactionPing;
    public volatile short transactionId;
    public volatile int airTicks, flagCount;
    public volatile int punishAnimation, teleportTicks;
    public volatile boolean sneaking = false, sprinting = false, ground = false;
    private volatile boolean cinematic = false;
    private volatile Location to;
    private volatile Location from;
    private volatile float vl;
    private volatile long attackBlockToTime, lastTeleport = 0;
    private volatile boolean alerts, debug, ignoreExitBan, ignoreFirstTick = true;
    private volatile Pair<String, String> banAnimInfo;
    private volatile Pair<Location, Location> banAnimPositions;
    private volatile boolean valid = true;
    private final Object instance;

    public PlayerProfile(Player player) {
        this.player = player;
        this.to = this.from = player.getLocation();
        this.instance = this;
    }

    public void punish(final String check, final String component, final String info, final float m) {
        if (!ConfigCache.BYPASS.equalsIgnoreCase("none")
                && this.player.hasPermission(ConfigCache.BYPASS)) {
            return;
        }
        synchronized (stateLock) {
            if (!this.valid || !this.player.isOnline()) return;
            final float tempVl = this.vl + 10.0f * m;
            final double vlLimit = ConfigCache.VL_LIMIT;
            MXFlagEvent event = new MXFlagEvent(this.player, check, component, info, tempVl, vlLimit);
            Bukkit.getPluginManager().callEvent(event);
            if (event.isCancelled()) {
                return;
            }
            this.vl = tempVl;
            this.flagCount += (m == 0.0) ? 0 : 1;
            String builder = this.wrapString(ConfigCache.ALERT_MSG
                    .replace("%check%", check)
                    .replace("%component%", component)
                    .replace("%info%", info));
            MessageUtils.sendMessagesToPlayers(MX.permission, builder);
            if (ConfigCache.LOG_IN_FILES) {
                logs.add("[" + MessageUtils.getDate() + "] "
                        + this.getPlayer().getName()
                        + " >> " + check + " (" + component + ") " + info + " ["
                        + ((int) this.vl) + "/"
                        + ConfigCache.VL_LIMIT
                        + "]");
            }
            if (this.vl >= vlLimit) {
                if (ConfigCache.PUNISH_EFFECT) {
                    AnimatedPunishService.punish(this, new Pair<>(check, info));
                } else {
                    forcePunish(check, info);
                }
            } else if (this.vl >= vlLimit / 1.8) {
                if (flagCount > 2 && !ConfigCache.SUSPECTED.isEmpty()) {
                    MessageUtils.sendMessagesToPlayersNative(
                            MX.permissionHead + "personal",
                            MX.permission,
                            this.wrapString(ConfigCache.SUSPECTED
                                    .replace("%check%", check)
                                    .replace("%info%", info))
                    );
                    this.flagCount = 0;
                }
            } else if (flagCount == 2 && !ConfigCache.UNUSUAL.isEmpty()) {
                MessageUtils.sendMessagesToPlayersNative(
                        MX.permissionHead + "personal",
                        MX.permission,
                        this.wrapString(ConfigCache.UNUSUAL
                                .replace("%check%", check)
                                .replace("%info%", info))
                );
            }
        }
    }

    public void fade(float vl) {
        synchronized (stateLock) {
            this.vl -= vl;
            if (this.vl < 0) this.vl = 0;
        }
    }

    public void invalidate() {
        synchronized (stateLock) {
            this.valid = false;
        }
    }

    public void initChecks(Object dependency) {
        for (Class<? extends PacketCheckHandler> checkHandler : CheckManager.getChecks()) {
            try {
                this.checks.add(checkHandler.getConstructor(dependency.getClass()).newInstance(this));
            } catch (ReflectiveOperationException e) {
                MX.getInstance().getLogger().severe(
                        "Failed to initialise check " + checkHandler.getName() + ": " + e);
            }
        }
    }

    public void run(Object handler) {
        if (!this.valid) return;
        CheckPacketRegister.runCustom(handler, checks);
    }

    private String wrapString(String v) {
        return MessageUtils.wrapColors(v.replace("%player%", this.getPlayer().getName())
                .replace("%vl%", String.valueOf(this.vl))
                .replace("%vlLimit%", String.valueOf(ConfigCache.VL_LIMIT))
        );
    }

    public boolean ignoreCinematic() {
        return cinematic && ConfigCache.IGNORE_CINEMATIC;
    }

    public boolean toggleAlerts() {
        this.alerts = !this.alerts;
        return this.alerts;
    }

    public boolean toggleDebug() {
        this.debug = !this.debug;
        return this.debug;
    }

    public void forcePunish(String check, String info) {
        MX.bannedPerMinuteCount.incrementAndGet();
        this.ignoreExitBan = true;
        synchronized (stateLock) {
            this.vl = 0;
        }
        Bukkit.getScheduler().runTask(MX.getInstance(), () -> {
            String banMsg = this.wrapString(ConfigCache.BAN_COMMAND
                    .replace("%check%", check)
                    .replace("%info%", info));
            Bukkit.dispatchCommand(Bukkit.getConsoleSender(), banMsg);
            this.setBanAnimInfo(null);
        });
    }

    public void debug(String msg) {
        if (debug && valid && this.player.isOnline())
            this.player.sendMessage(wrapString("&9&l[Debug] &f" + msg));
    }

    public void setAttackBlockToTime(long time) {
        if (time < System.currentTimeMillis() + 10) return;
        if (!ConfigCache.BYPASS.equalsIgnoreCase("none")
                && this.player.hasPermission(ConfigCache.BYPASS)) {
            return;
        }
        this.attackBlockToTime = time;
    }

    public int getEntityId() {
        return ProtocolLib.isTemporary(this.getPlayer())
                ? ThreadLocalRandom.current().nextInt()
                : this.getPlayer().getEntityId();
    }

    public int calculateSensitivity() {
        if (Statistics.getDistinct(getSensitivity()) != getSensitivity().size()) {
            final Set<Integer> prev = new HashSet<>();
            for (int i : getSensitivity()) {
                if (prev.contains(i / 5)) {
                    return i;
                } else {
                    prev.add(i / 5);
                }
            }
        }
        return -1;
    }
    @Override public int hashCode() {
        return System.identityHashCode(this);
    }
    @Override public boolean equals(Object o) {
        return this == o;
    }
}
