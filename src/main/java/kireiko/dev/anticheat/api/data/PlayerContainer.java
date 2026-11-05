package kireiko.dev.anticheat.api.data;

import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.core.AsyncScheduler;
import kireiko.dev.anticheat.services.AnimatedPunishService;
import kireiko.dev.anticheat.services.SimulationFlagService;
import kireiko.dev.anticheat.utils.LogUtils;
import lombok.Getter;
import org.bukkit.entity.Player;
import org.jetbrains.annotations.Nullable;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public final class PlayerContainer {

    private static final Object INIT_LOCK = new Object();

    @Getter
    private static final Map<UUID, PlayerProfile> uuidPlayerProfileMap = new ConcurrentHashMap<>();

    private static final Map<UUID, Object> pendingInits = new ConcurrentHashMap<>();

    public static void init(Player player) {
        final UUID uuid = player.getUniqueId();
        final Object token = new Object();
        pendingInits.put(uuid, token);
        AsyncScheduler.run(() -> {
            try {
                PlayerProfile profile = new PlayerProfile(player);
                profile.initChecks(profile.getInstance());
                synchronized (INIT_LOCK) {
                    if (pendingInits.remove(uuid, token) && player.isOnline()) {
                        uuidPlayerProfileMap.compute(uuid, (id, previous) -> {
                            if (previous != null && previous != profile) previous.invalidate();
                            return profile;
                        });
                    } else {
                        profile.invalidate();
                    }
                }
            } finally {
                pendingInits.remove(uuid, token);
            }
        });
    }

    public static void unload(Player player) {
        final UUID uuid = player.getUniqueId();
        final PlayerProfile profile;
        synchronized (INIT_LOCK) {
            pendingInits.remove(uuid);
            final PlayerProfile[] removed = new PlayerProfile[1];
            uuidPlayerProfileMap.compute(uuid, (id, previous) -> {
                removed[0] = previous;
                if (previous != null) previous.invalidate();
                return null;
            });
            profile = removed[0];
            RotationsContainer.unload(uuid);
        }
        if (profile == null) return;
        saveLogs(profile);
        if (profile.getBanAnimInfo() != null && !profile.isIgnoreExitBan()) {
            profile.forcePunish(profile.getBanAnimInfo().getX(), profile.getBanAnimInfo().getY());
        }
        AnimatedPunishService.unload(profile);
        SimulationFlagService.unload(profile);
    }

    private static void saveLogs(PlayerProfile profile) {
        if (profile.getLogs().isEmpty()) return;
        final String playerName = profile.getPlayer().getName();
        final StringBuilder logBuilder = new StringBuilder();
        LogUtils.createLog(playerName);
        for (final String l : profile.getLogs()) logBuilder.append("\n").append(l);
        LogUtils.addLog(playerName, logBuilder.toString());
        profile.getLogs().clear();
    }

    @Nullable
    public static PlayerProfile getProfile(Player player) {
        return uuidPlayerProfileMap.get(player.getUniqueId());
    }

    @Nullable
    public static PlayerProfile getProfileByName(String name) {
        for (PlayerProfile profile : uuidPlayerProfileMap.values()) {
            if (profile.getPlayer().getName().equalsIgnoreCase(name)) {
                return profile;
            }
        }
        return null;
    }
}
