package kireiko.dev.anticheat.listeners;

import com.comphenix.protocol.PacketType;
import com.comphenix.protocol.events.*;
import com.comphenix.protocol.reflect.StructureModifier;
import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.data.PlayerContainer;
import kireiko.dev.anticheat.api.events.SVelocityEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import org.bukkit.entity.Player;
import org.bukkit.util.Vector;

import java.util.Collections;

public final class VelocityListener extends PacketAdapter {


    public VelocityListener() {
        super(
                MX.getInstance(),
                ListenerPriority.MONITOR,
                Collections.singletonList(PacketType.Play.Server.ENTITY_VELOCITY),
                ListenerOptions.ASYNC
        );
    }

    @Override
    public void onPacketSending(PacketEvent event) {
        final Player player = event.getPlayer();
        final PlayerProfile protocol = PlayerContainer.getProfile(player);
        if (protocol == null) {
            return;
        }
        PacketContainer packet = event.getPacket();
        if (!packet.getIntegers().getValues().isEmpty()) {
            int id = packet.getIntegers().getValues().get(0);
            if (protocol.getEntityId() == id) {
                if (packet.getIntegers().getValues().size() > 1) {
                    double x = packet.getIntegers().read(1).doubleValue() / 8000.0D,
                                    y = packet.getIntegers().read(2).doubleValue() / 8000.0D,
                                    z = packet.getIntegers().read(3).doubleValue() / 8000.0D;
                    SVelocityEvent velocityEvent = new SVelocityEvent(new Vector(x, y, z));
                    protocol.run(velocityEvent);
                } else if (packet.getModifier().size() > 1) {
                    Object vec3 = packet.getModifier().read(1);

                    if (vec3 != null) {
                        StructureModifier<Double> vecStruct = new StructureModifier<>(vec3.getClass())
                                        .withTarget(vec3)
                                        .withType(double.class);

                        if (vecStruct.size() >= 3) {
                            double x = vecStruct.read(0),
                                            y = vecStruct.read(1),
                                            z = vecStruct.read(2);
                            protocol.run(new SVelocityEvent(new Vector(x, y, z)));
                        }
                    }
                }
            }
        }
    }
}
