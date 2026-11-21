package kireiko.dev.anticheat.listeners;

import com.comphenix.protocol.PacketType;
import com.comphenix.protocol.events.ListenerOptions;
import com.comphenix.protocol.events.ListenerPriority;
import com.comphenix.protocol.events.PacketAdapter;
import com.comphenix.protocol.events.PacketEvent;
import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.data.PlayerContainer;
import kireiko.dev.anticheat.api.player.PlayerProfile;

import java.util.Collections;

public final class VehicleTeleportListener extends PacketAdapter {

    public VehicleTeleportListener() {
        super(
                MX.getInstance(),
                ListenerPriority.HIGHEST,
                Collections.singletonList(PacketType.Play.Client.STEER_VEHICLE),
                ListenerOptions.ASYNC
        );
    }

    @Override
    public void onPacketReceiving(PacketEvent event) {
        PlayerProfile protocol = PlayerContainer.getProfile(event.getPlayer());
        if (protocol == null) {
            return;
        }
        protocol.setLastTeleport(System.currentTimeMillis());
        protocol.setIgnoreFirstTick(true);
    }

}