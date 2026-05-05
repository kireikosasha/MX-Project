package kireiko.dev.anticheat.listeners;

import com.destroystokyo.paper.event.entity.EntityAddToWorldEvent;
import com.destroystokyo.paper.event.entity.EntityRemoveFromWorldEvent;
import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.utils.cache.EntityCache;
import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.entity.Entity;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;

/**
 * Populates {@link EntityCache} from Paper's add/remove-to-world events. We do
 * NOT touch the netty pipeline — packet-only entities from hologram plugins
 * (DecentHolograms etc.) are simply not real Bukkit entities and never enter
 * the cache, which is exactly what combat checks want.
 */
public final class EntityTrackerListener implements Listener {

    public static void register() {
        Bukkit.getPluginManager().registerEvents(new EntityTrackerListener(), MX.getInstance());
        // Pre-populate with entities already loaded at plugin enable.
        for (World world : Bukkit.getWorlds()) {
            for (Entity e : world.getEntities()) EntityCache.track(e);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onAdd(EntityAddToWorldEvent e) {
        EntityCache.track(e.getEntity());
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onRemove(EntityRemoveFromWorldEvent e) {
        EntityCache.untrack(e.getEntity().getEntityId());
    }
}
