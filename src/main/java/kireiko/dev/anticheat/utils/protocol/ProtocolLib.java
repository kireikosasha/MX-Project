package kireiko.dev.anticheat.utils.protocol;

import com.comphenix.protocol.injector.temporary.TemporaryPlayer;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.OfflinePlayer;
import org.bukkit.World;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;

import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;

public final class ProtocolLib {

    private static boolean
            temporaryClass = false;

    public static boolean isTemporary(OfflinePlayer player) {
        return temporaryClass && player instanceof TemporaryPlayer;
    }

    public static UUID getUUID(Entity entity) {
        if (entity instanceof Player) {
            if (ProtocolLib.isTemporary((Player) entity)) {
                return UUID.randomUUID();
            } else {
                return entity.getUniqueId();
            }
        } else {
            return entity.getUniqueId();
        }
    }

    public static int getEntityID(Entity entity) {
        if (entity instanceof Player) {
            if (ProtocolLib.isTemporary((Player) entity)) {
                return ThreadLocalRandom.current().nextInt();
            } else {
                return entity.getEntityId();
            }
        } else {
            return entity.getEntityId();
        }
    }

    public static Location getLocationOrNull(Entity entity) {
        if (entity instanceof Player) {
            return getLocationOrNull((Player) entity);
        } else {
            return entity.getLocation();
        }
    }

    public static Location getLocationOrNull(Player player) {
        if (ProtocolLib.isTemporary(player)) {
            return null;
        } else {
            return player.getLocation();
        }
    }

    public static Entity getVehicle(Entity entity) {
        if (entity instanceof Player) {
            if (ProtocolLib.isTemporary((Player) entity)) {
                return null;
            } else {
                return entity.getVehicle();
            }
        } else {
            return entity.getVehicle();
        }
    }

    public static World getWorld(Entity entity) {
        if (entity instanceof Player) {
            if (ProtocolLib.isTemporary((Player) entity)) {
                return Bukkit.getWorlds().get(0);
            } else {
                return entity.getWorld();
            }
        } else {
            return entity.getWorld();
        }
    }

}
