package kireiko.dev.anticheat.utils.cache;

import org.bukkit.entity.Entity;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Global entity-id → Entity cache. Bukkit assigns entity IDs from a single static
 * counter, so they are unique across all worlds — a flat map is enough.
 *
 * <p>Lookups are O(1) and lock-free. Populated proactively from Bukkit spawn
 * events (no packet interception), so first-attack latency is zero.
 */
public final class EntityCache {

    private EntityCache() {}

    private static final Map<Integer, WeakReference<Entity>> CACHE = new ConcurrentHashMap<>();

    public static Entity get(int entityId) {
        WeakReference<Entity> ref = CACHE.get(entityId);
        if (ref == null) return null;
        Entity e = ref.get();
        if (e == null) {
            CACHE.remove(entityId);
            return null;
        }
        return e;
    }

    public static void track(Entity entity) {
        if (entity == null) return;
        CACHE.put(entity.getEntityId(), new WeakReference<>(entity));
    }

    public static void untrack(int entityId) {
        CACHE.remove(entityId);
    }

    public static void clear() {
        CACHE.clear();
    }
}
