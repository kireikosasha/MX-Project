package kireiko.dev.anticheat.listeners;

import com.comphenix.protocol.PacketType;
import com.comphenix.protocol.ProtocolLibrary;
import com.comphenix.protocol.events.*;
import com.comphenix.protocol.wrappers.EnumWrappers;
import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.data.PlayerContainer;
import kireiko.dev.anticheat.api.events.UseEntityEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.utils.ConfigCache;
import kireiko.dev.anticheat.utils.version.VersionUtil;
import lombok.SneakyThrows;
import org.bukkit.Bukkit;
import org.bukkit.entity.Entity;
import org.bukkit.entity.LivingEntity;
import org.bukkit.entity.Player;

import java.util.Collections;

public final class UseEntityListener extends PacketAdapter {

    private static final boolean modern = VersionUtil.is1_13orAbove();

    public UseEntityListener() {
        super(MX.getInstance(), ListenerPriority.HIGHEST, Collections.singletonList(PacketType.Play.Client.USE_ENTITY));
    }

    @SneakyThrows
    @Override
    public void onPacketReceiving(PacketEvent event) {
        Player player = event.getPlayer();
        PlayerProfile profile = PlayerContainer.getProfile(player);
        if (profile == null) {
            return;
        }
        PacketContainer packet = event.getPacket();
        boolean attack = !packet.getEntityUseActions().getValues().isEmpty() ?
                packet.getEntityUseActions().read(0).toString().equals("ATTACK")
                : packet.getEnumEntityUseActions().read(0).getAction().equals(
                EnumWrappers.EntityUseAction.ATTACK);
        if (packet.getIntegers().getValues().isEmpty()) return;
        int entityId = packet.getIntegers().read(0);
        Entity fetchEntity = null;
        try {
            if (Bukkit.isPrimaryThread()) {
                fetchEntity = ProtocolLibrary.getProtocolManager().getEntityFromID(player.getWorld(), entityId);
            } else {
                fetchEntity = Bukkit.getScheduler().callSyncMethod(MX.getInstance(), () ->
                        ProtocolLibrary.getProtocolManager().getEntityFromID(player.getWorld(), entityId)
                ).get();
            }
        } catch (Exception ex) {
            fetchEntity = null;
        }
        final Entity entity = fetchEntity;
        if (profile.getAttackBlockToTime() > System.currentTimeMillis()) {
            if (ConfigCache.PREVENTION > 0) {
                event.setCancelled(true);
                if (ConfigCache.PREVENTION >= 3) {
                    Bukkit.getScheduler().runTask(MX.getInstance(), () -> {
                        player.teleport(player.getLocation());
                    });
                } else if (ConfigCache.PREVENTION == 1
                        && attack
                        && entity instanceof LivingEntity
                        && player.getLocation().toVector().distance(entity.getLocation().toVector()) < 3.3) {
                    Bukkit.getScheduler().runTask(MX.getInstance(), () -> {
                        ((LivingEntity) entity).damage(0.5, player);
                    });
                }
                profile.debug("UseEntity packet blocked");
                MX.blockedPerMinuteCount++;
            }
        }
        UseEntityEvent e = new UseEntityEvent(entity, attack, entityId, false);
        profile.run(e);
        if (e.isCancelled()) {
            event.setCancelled(true);
            profile.debug("UseEntity packet blocked after checking");
            MX.blockedPerMinuteCount++;
        }
    }
}
