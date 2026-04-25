package kireiko.dev.anticheat.listeners;

import com.comphenix.protocol.PacketType;
import com.comphenix.protocol.ProtocolLibrary;
import com.comphenix.protocol.events.ListenerPriority;
import com.comphenix.protocol.events.PacketAdapter;
import com.comphenix.protocol.events.PacketContainer;
import com.comphenix.protocol.events.PacketEvent;
import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.utils.cache.EntityCache;
import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerQuitEvent;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Populates {@link EntityCache} via outgoing server packets.
 * Compatible with 1.8 .. 1.21+ (uses isSupported() guards on legacy packet types).
 */
public final class EntityTrackerListener implements Listener {

    public static void register() {
        registerSpawnListener();
        registerDestroyListener();
        registerRespawnListener();
        Bukkit.getPluginManager().registerEvents(new EntityTrackerListener(), MX.getInstance());
    }

    private static void registerSpawnListener() {
        List<PacketType> spawn = new ArrayList<>();
        addIfSupported(spawn, () -> PacketType.Play.Server.SPAWN_ENTITY);
        addIfSupported(spawn, () -> PacketType.Play.Server.SPAWN_ENTITY_LIVING);
        addIfSupported(spawn, () -> PacketType.Play.Server.NAMED_ENTITY_SPAWN);
        addIfSupported(spawn, () -> PacketType.Play.Server.SPAWN_ENTITY_PAINTING);
        addIfSupported(spawn, () -> PacketType.Play.Server.SPAWN_ENTITY_EXPERIENCE_ORB);
        if (spawn.isEmpty()) return;

        ProtocolLibrary.getProtocolManager().addPacketListener(new PacketAdapter(
                MX.getInstance(),
                ListenerPriority.MONITOR,
                spawn.toArray(new PacketType[0])
        ) {
            @Override
            public void onPacketSending(PacketEvent event) {
                // Hot path on netty thread — keep it microsecond-cheap.
                final int id;
                final Player player;
                final World world;
                try {
                    PacketContainer p = event.getPacket();
                    if (p.getIntegers().size() == 0) return;
                    id = p.getIntegers().read(0);
                    player = event.getPlayer();
                    if (player == null) return;
                    world = player.getWorld();
                } catch (Throwable t) { return; }

                // Heavy entity resolution off the netty pipeline.
                Bukkit.getScheduler().runTask(MX.getInstance(), () -> {
                    try {
                        Entity entity = ProtocolLibrary.getProtocolManager()
                                                       .getEntityFromID(world, id);
                        if (entity != null) EntityCache.track(player, id, entity);
                    } catch (Throwable ignored) {}
                });
            }
        });
    }

    private static void registerDestroyListener() {
        ProtocolLibrary.getProtocolManager().addPacketListener(new PacketAdapter(
                MX.getInstance(),
                ListenerPriority.MONITOR,
                PacketType.Play.Server.ENTITY_DESTROY
        ) {
            @Override
            public void onPacketSending(PacketEvent event) {
                try {
                    PacketContainer p = event.getPacket();
                    Player player = event.getPlayer();
                    if (player == null) return;

                    // 1.17.1+ : List<Integer>
                    try {
                        if (p.getIntLists().size() > 0) {
                            List<Integer> ids = p.getIntLists().read(0);
                            if (ids != null) for (int id : ids) EntityCache.untrack(player, id);
                            return;
                        }
                    } catch (Throwable ignored) {}

                    // <= 1.16.5 : int[]
                    try {
                        if (p.getIntegerArrays().size() > 0) {
                            int[] ids = p.getIntegerArrays().read(0);
                            if (ids != null) for (int id : ids) EntityCache.untrack(player, id);
                            return;
                        }
                    } catch (Throwable ignored) {}

                    // 1.17.0 : single int
                    if (p.getIntegers().size() > 0) {
                        EntityCache.untrack(player, p.getIntegers().read(0));
                    }
                } catch (Throwable ignored) {}
            }
        });
    }

    private static void registerRespawnListener() {
        ProtocolLibrary.getProtocolManager().addPacketListener(new PacketAdapter(
                MX.getInstance(),
                ListenerPriority.MONITOR,
                PacketType.Play.Server.RESPAWN
        ) {
            @Override
            public void onPacketSending(PacketEvent event) {
                try {
                    Player player = event.getPlayer();
                    if (player != null) EntityCache.clearTracked(player);
                } catch (Throwable ignored) {}
            }
        });
    }

    private static void addIfSupported(List<PacketType> list, Supplier<PacketType> s) {
        PacketType t;
        try { t = s.get(); } catch (Throwable ignored) { return; }
        if (t == null) return;
        try { if (t.isSupported()) list.add(t); } catch (Throwable ignored) {}
    }

    @EventHandler
    public void onQuit(PlayerQuitEvent e) {
        EntityCache.forget(e.getPlayer());
    }
}
