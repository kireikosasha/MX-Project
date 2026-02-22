package kireiko.dev.anticheat.listeners;

import com.comphenix.protocol.PacketType;
import com.comphenix.protocol.events.*;
import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.data.PlayerContainer;
import kireiko.dev.anticheat.api.data.RotationsContainer;
import kireiko.dev.anticheat.api.events.MoveEvent;
import kireiko.dev.anticheat.api.events.NoRotationEvent;
import kireiko.dev.anticheat.api.events.RotationEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.api.player.SensitivityProcessor;
import kireiko.dev.anticheat.utils.ConfigCache;
import kireiko.dev.anticheat.utils.protocol.ProtocolLib;
import kireiko.dev.anticheat.utils.protocol.ProtocolTools;
import kireiko.dev.anticheat.utils.version.VersionUtil;
import kireiko.dev.millennium.vectors.Vec2f;
import org.bukkit.Location;
import org.bukkit.entity.Player;

import java.util.Arrays;

public final class RawMovementListener extends PacketAdapter {
    public RawMovementListener() {
        super(
                MX.getInstance(),
                ListenerPriority.LOWEST,
                Arrays.asList(
                        PacketType.Play.Server.POSITION,
                        PacketType.Play.Client.POSITION,
                        PacketType.Play.Client.POSITION_LOOK,
                        PacketType.Play.Client.LOOK,
                        VersionUtil.is1_17orAbove()
                                ? PacketType.Play.Client.GROUND
                                : PacketType.Play.Client.FLYING
                ),
                ListenerOptions.ASYNC
        );
    }

    @Override
    public void onPacketSending(PacketEvent event) {
        final Player player = event.getPlayer();
        final PlayerProfile profile = PlayerContainer.getProfile(player);
        if (profile == null) {
            return;
        }
        profile.setLastTeleport(System.currentTimeMillis());
        profile.setIgnoreFirstTick(true);
    }

    @Override
    public void onPacketReceiving(PacketEvent event) {
        final Player player = event.getPlayer();
        final PlayerProfile profile = PlayerContainer.getProfile(player);
        if (profile == null) {
            return;
        }
        profile.setGround(event.getPacket().getBooleans().read(0));
        profile.setAirTicks((profile.isGround()) ? 0 : profile.getAirTicks() + 1);
        final PacketContainer packet = event.getPacket();
        profile.setFrom(profile.getTo().clone());
        Location l = profile.getTo().clone();
        boolean hasPosition = ProtocolTools.hasPosition(packet.getType());
        boolean hasRotation = ProtocolTools.hasRotation(packet.getType());
        if (hasPosition) {
            Location r = ProtocolTools.readLocation(event);
            if (r == null) return;
            double[] v = new double[]{r.getX(), r.getY(), r.getZ()};
            for (Double check : v)
                if (check.isNaN() || check.isInfinite() || Math.abs(check) > 3E8) {
                    return;
                }
            l.setX(r.getX());
            l.setY(r.getY());
            l.setZ(r.getZ());
        }
        l.setWorld(ProtocolLib.getWorld(player));
        if (hasRotation) {
            for (Float check : Arrays.asList(packet.getFloat().read(0), packet.getFloat().read(1)))
                if (check.isNaN() || check.isInfinite() || Math.abs(check) > 3E8) {
                    return;
                }
            l.setYaw(packet.getFloat().read(0));
            l.setPitch(packet.getFloat().read(1));
        }
        profile.setTo(l.clone());
        if (hasRotation) {
            SensitivityProcessor controller = profile.getSensitivityProcessor();
            controller.setLastDeltaPitch(controller.getLastDeltaPitch());
            Vec2f from = new Vec2f(profile.getFrom().getYaw(), profile.getFrom().getPitch());
            Vec2f to = new Vec2f(profile.getTo().getYaw(), profile.getTo().getPitch());
            RotationEvent rotationEvent = new RotationEvent(profile, to, from);
            controller.setDeltaPitch(rotationEvent.getDelta().getY());
            controller.processSensitivity();
            boolean isTeleporting = (System.currentTimeMillis() - profile.getLastTeleport() < 500) || profile.isIgnoreFirstTick();

            if (ConfigCache.ROTATIONS_CONTAINER
                            && !profile.isIgnoreFirstTick()
                            && !isTeleporting) {
                RotationsContainer.register(ProtocolLib.getUUID(profile.getPlayer()), rotationEvent.getDelta());
            }

            profile.getCinematicComponent().process(rotationEvent);
            if (!isTeleporting) {
                profile.run(rotationEvent);
            }
            profile.setIgnoreFirstTick(false);
        } else {
            if (!profile.isIgnoreFirstTick() && profile.getLastTeleport() + 1000 < System.currentTimeMillis()) {
                if (profile.getTo().toVector().distance(profile.getFrom().toVector()) > 1e-4) {
                    profile.run(new NoRotationEvent(profile));
                }
            }
        }

        profile.getPastLoc().add(profile.getTo());
        profile.run(new MoveEvent(profile, profile.getTo(), profile.getFrom()));

        if (profile.transactionBoot) LatencyHandler.startChecking(profile);
    }
}
