package kireiko.dev.anticheat.utils;

import org.bukkit.Bukkit;
import org.bukkit.Material;
import org.bukkit.World;

import java.util.function.Predicate;

public final class BlockSnapshot {

    private static final int SIZE = 3;
    private static final int VOLUME = SIZE * SIZE * SIZE;

    private final Material[] materials;

    private BlockSnapshot(Material[] materials) {
        this.materials = materials;
    }

    public static BlockSnapshot capture(World world, double x, double y, double z, double scale) {
        return capture(world, x, y, z, scale, scale, scale);
    }

    public static BlockSnapshot capture(World world, double x, double y, double z,
                                        double scaleX, double scaleY, double scaleZ) {
        if (!Bukkit.isPrimaryThread())
            throw new IllegalStateException("BlockSnapshot must be captured on the main thread");
        if (world == null) return new BlockSnapshot(new Material[VOLUME]);

        final Material[] materials = new Material[VOLUME];
        int i = 0;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    materials[i++] = world.getBlockAt(
                            (int) Math.floor(x + dx * scaleX),
                            (int) Math.floor(y + dy * scaleY),
                            (int) Math.floor(z + dz * scaleZ)
                    ).getType();
                }
            }
        }
        return new BlockSnapshot(materials);
    }

    public boolean anyMatch(Predicate<Material> predicate) {
        for (Material material : materials) {
            if (material != null && predicate.test(material)) return true;
        }
        return false;
    }
}
