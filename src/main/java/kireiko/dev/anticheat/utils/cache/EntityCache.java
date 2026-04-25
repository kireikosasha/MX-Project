package kireiko.dev.anticheat.utils.cache;

import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Per-player entity tracking cache populated from outgoing spawn/destroy packets.
 * Lookup is fully async-safe and O(1) — no NMS / reflection on the hot path.
 */
public final class EntityCache {

    private EntityCache() {}

    private static final Map<UUID, Map<Integer, WeakReference<Entity>>> BY_PLAYER = new ConcurrentHashMap<>();

    public static Entity get(Player player, int entityId) {
        Map<Integer, WeakReference<Entity>> m = BY_PLAYER.get(player.getUniqueId());
        if (m == null) return null;
        WeakReference<Entity> ref = m.get(entityId);
        if (ref == null) return null;
        Entity e = ref.get();
        if (e == null) {
            m.remove(entityId);
            return null;
        }
        return e;
    }

    public static void track(Player player, int entityId, Entity entity) {
        if (entity == null) return;
        BY_PLAYER.computeIfAbsent(player.getUniqueId(), k -> new ConcurrentHashMap<>())
                 .put(entityId, new WeakReference<>(entity));
    }

    public static void untrack(Player player, int entityId) {
        Map<Integer, WeakReference<Entity>> m = BY_PLAYER.get(player.getUniqueId());
        if (m != null) m.remove(entityId);
    }

    public static void clearTracked(Player player) {
        Map<Integer, WeakReference<Entity>> m = BY_PLAYER.get(player.getUniqueId());
        if (m != null) m.clear();
    }

    public static void forget(Player player) {
        BY_PLAYER.remove(player.getUniqueId());
    }
}
