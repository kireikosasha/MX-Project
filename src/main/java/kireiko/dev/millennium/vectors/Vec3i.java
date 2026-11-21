package kireiko.dev.millennium.vectors;

import lombok.Getter;

@Getter
public class Vec3i implements Comparable {
    /**
     * X coordinate
     * -- GETTER --
     * Get the X coordinate
     */
    private final int x;

    /**
     * Y coordinate
     * -- GETTER --
     * Get the Y coordinate
     */
    private final int y;

    /**
     * Z coordinate
     * -- GETTER --
     * Get the Z coordinate
     */
    private final int z;

    public Vec3i(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public Vec3i(double x, double y, double z) {
        this((int) Math.floor(x), (int) Math.floor(y), (int) Math.floor(z));
    }

    public boolean equals(Object p_equals_1_) {
        if (this == p_equals_1_) {
            return true;
        } else if (!(p_equals_1_ instanceof Vec3i)) {
            return false;
        } else {
            Vec3i var2 = (Vec3i) p_equals_1_;
            return this.getX() != var2.getX() ? false : (this.getY() != var2.getY() ? false : this.getZ() == var2.getZ());
        }
    }

    public int hashCode() {
        return (this.getY() + this.getZ() * 31) * 31 + this.getX();
    }

    public int compareTo(Vec3i vec) {
        return this.getY() == vec.getY() ? (this.getZ() == vec.getZ() ? this.getX() - vec.getX() : this.getZ() - vec.getZ()) : this.getY() - vec.getY();
    }

    /**
     * Calculate the cross product of this and the given Vector
     */
    public Vec3i crossProduct(Vec3i vec) {
        return new Vec3i(this.getY() * vec.getZ() - this.getZ() * vec.getY(), this.getZ() * vec.getX() - this.getX() * vec.getZ(), this.getX() * vec.getY() - this.getY() * vec.getX());
    }

    /**
     * Calculate squared distance to the given coordinates
     *
     * @param toX X Coordinate
     * @param toY Y Coordinate
     * @param toZ Z Coordinate
     */
    public double distanceSq(double toX, double toY, double toZ) {
        double var7 = (double) this.getX() - toX;
        double var9 = (double) this.getY() - toY;
        double var11 = (double) this.getZ() - toZ;
        return var7 * var7 + var9 * var9 + var11 * var11;
    }

    /**
     * Compute square of distance from point x, y, z to center of this Block
     */
    public double distanceSqToCenter(double x, double y, double z) {
        double var7 = (double) this.getX() + 0.5D - x;
        double var9 = (double) this.getY() + 0.5D - y;
        double var11 = (double) this.getZ() + 0.5D - z;
        return var7 * var7 + var9 * var9 + var11 * var11;
    }

    /**
     * Calculate squared distance to the given Vector
     */
    public double distanceSq(Vec3i to) {
        return this.distanceSq((double) to.getX(), (double) to.getY(), (double) to.getZ());
    }

    public int compareTo(Object p_compareTo_1_) {
        return this.compareTo((Vec3i) p_compareTo_1_);
    }
}
