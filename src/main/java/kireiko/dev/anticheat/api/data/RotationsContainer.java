package kireiko.dev.anticheat.api.data;

import kireiko.dev.millennium.types.ConcurrentEvictingList;
import kireiko.dev.millennium.vectors.Vec2f;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public final class RotationsContainer {
    private static final Map<UUID, List<Vec2f>> uuidPlayerRotationsMap = new ConcurrentHashMap<>();
    public static void register(final UUID uuid, final Vec2f vec2f) {
        uuidPlayerRotationsMap.computeIfAbsent(uuid, k -> new ConcurrentEvictingList<>(600)).add(vec2f);
    }
    public static String getJson(final UUID uuid) {
        List<Vec2f> rotations = uuidPlayerRotationsMap.get(uuid);
        if (rotations == null) return "[]";
        final StringBuilder sb = new StringBuilder();
        sb.append("[");
        int i = 0;
        for (Vec2f vec : rotations) {
            if (i++ > 0) sb.append(",");
            sb.append("[")
                            .append(vec.getX()).append(",")
                            .append(vec.getY())
                            .append("]");
        }
        sb.append("]");
        return sb.toString();
    }

    public static void unload(final UUID uuid) {
        uuidPlayerRotationsMap.remove(uuid);
    }
}
