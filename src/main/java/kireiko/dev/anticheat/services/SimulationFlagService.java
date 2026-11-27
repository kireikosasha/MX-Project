package kireiko.dev.anticheat.services;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.utils.BlockSnapshot;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.util.Vector;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;

public final class SimulationFlagService {

    @Getter
    private static final List<Flag> flags = new CopyOnWriteArrayList<>();

    public static void init() {
        // World access is only safe on the main thread, so the whole simulation
        // step runs there instead of being offloaded to AsyncScheduler.
        Bukkit.getScheduler().runTaskTimer(MX.getInstance(), () -> {
            final Set<Flag> toRemove = new HashSet<>();
            for (Flag flag : flags) {
                flag.getLocation().add(flag.getVector());
                if (!isPointWall(flag.getLocation(), 0.3)) {
                    final Location finalLoc = flag.getLocation().clone();
                    Bukkit.getScheduler().runTask(MX.getInstance(), () -> {
                        flag.getProfile().getPlayer().teleport(finalLoc);
                    });
                    flag.setVector(new Vector(
                            flag.vector.getX() * 0.91,
                            flag.vector.getY() - (0.08 * 0.98),
                            flag.vector.getZ() * 0.91));
                } else toRemove.add(flag);
            }
            flags.removeAll(toRemove);
        }, 0L, 1L);
    }

    private static boolean isPointWall(Location location, final double scale) {
        return BlockSnapshot.capture(
                location.getWorld(),
                location.getX(),
                location.getY() + 0.1,
                location.getZ(),
                scale
        ).anyMatch(material -> !material.toString().equals("AIR"));
    }

    public static void unload(PlayerProfile profile) {
        flags.removeIf(flag -> flag.getProfile() == profile);
    }

    @Data
    @AllArgsConstructor
    public static class Flag {
        private final PlayerProfile profile;
        private Location location;
        private Vector vector;
    }
}
