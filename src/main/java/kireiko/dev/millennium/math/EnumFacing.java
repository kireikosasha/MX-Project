package kireiko.dev.millennium.math;

import com.google.common.base.Predicate;
import com.google.common.collect.Iterators;
import com.google.common.collect.Maps;
import kireiko.dev.millennium.vectors.Vec3i;
import lombok.Getter;

import java.util.Iterator;
import java.util.Map;
import java.util.Random;

public enum EnumFacing {
    DOWN(0, 1, -1, "down", AxisDirection.NEGATIVE, Axis.Y, new Vec3i(0, -1, 0)),
    UP(1, 0, -1, "up", AxisDirection.POSITIVE, Axis.Y, new Vec3i(0, 1, 0)),
    NORTH(2, 3, 2, "north", AxisDirection.NEGATIVE, Axis.Z, new Vec3i(0, 0, -1)),
    SOUTH(3, 2, 0, "south", AxisDirection.POSITIVE, Axis.Z, new Vec3i(0, 0, 1)),
    WEST(4, 5, 1, "west", AxisDirection.NEGATIVE, Axis.X, new Vec3i(-1, 0, 0)),
    EAST(5, 4, 3, "east", AxisDirection.POSITIVE, Axis.X, new Vec3i(1, 0, 0));

    /**
     * All facings in D-U-N-S-W-E order
     */
    public static final EnumFacing[] VALUES = new EnumFacing[6];
    /**
     * All Facings with horizontal axis in order S-W-N-E
     */
    private static final EnumFacing[] HORIZONTALS = new EnumFacing[4];
    private static final Map NAME_LOOKUP = Maps.newHashMap();
    private static final EnumFacing[] $VALUES = new EnumFacing[]{DOWN, UP, NORTH, SOUTH, WEST, EAST};

    static {
        EnumFacing[] var0 = values();
        int var1 = var0.length;

        for (EnumFacing var3 : var0) {
            VALUES[var3.index] = var3;

            if (var3.getAxis().isHorizontal()) {
                HORIZONTALS[var3.horizontalIndex] = var3;
            }

            NAME_LOOKUP.put(var3.getName2().toLowerCase(), var3);
        }
    }

    /**
     * Ordering index for D-U-N-S-W-E
     * -- GETTER --
     * Get the Index of this Facing (0-5). The order is D-U-N-S-W-E
     */
    @Getter
    private final int index;
    /**
     * Index of the opposite Facing in the VALUES array
     */
    private final int opposite;
    /**
     * Oredering index for the HORIZONTALS field (S-W-N-E)
     * -- GETTER --
     * Get the index of this horizontal facing (0-3). The order is S-W-N-E
     */
    @Getter
    private final int horizontalIndex;
    @Getter
    private final String name;
    @Getter
    private final Axis axis;
    // private static final String __OBFID = "CL_00001201";
    /**
     * -- GETTER --
     * Get the AxisDirection of this Facing.
     */
    @Getter
    private final AxisDirection axisDirection;
    /**
     * Normalized Vector that points in the direction of this Facing
     * -- GETTER --
     * Get a normalized Vector that points in the direction of this Facing.
     */
    @Getter
    private final Vec3i directionVec;

    EnumFacing(int indexIn, int oppositeIn, int horizontalIndexIn, String nameIn, AxisDirection axisDirectionIn, Axis axisIn, Vec3i directionVecIn) {
        this.index = indexIn;
        this.horizontalIndex = horizontalIndexIn;
        this.opposite = oppositeIn;
        this.name = nameIn;
        this.axis = axisIn;
        this.axisDirection = axisDirectionIn;
        this.directionVec = directionVecIn;
    }

    /**
     * Get the facing specified by the given name
     */
    public static EnumFacing byName(String name) {
        return name == null ? null : (EnumFacing) NAME_LOOKUP.get(name.toLowerCase());
    }

    /**
     * Get a Facing by it's index (0-5). The order is D-U-N-S-W-E. Named getFront for legacy reasons.
     */
    public static EnumFacing getFront(int index) {
        return VALUES[Math.abs(index % VALUES.length)];
    }

    /**
     * Get a Facing by it's horizontal index (0-3). The order is S-W-N-E.
     */
    public static EnumFacing getHorizontal(int p_176731_0_) {
        return HORIZONTALS[Math.abs(p_176731_0_ % HORIZONTALS.length)];
    }

    /**
     * Get the Facing corresponding to the given angle (0-360). An angle of 0 is SOUTH, an angle of 90 would be WEST.
     */
    public static EnumFacing fromAngle(double angle) {
        return getHorizontal(FastMath.floor(angle / 90.0D + 0.5D) & 3);
    }

    /**
     * Choose a random Facing using the given Random
     */
    public static EnumFacing random(Random rand) {
        return values()[rand.nextInt(values().length)];
    }

    public static EnumFacing func_176737_a(float p_176737_0_, float p_176737_1_, float p_176737_2_) {
        EnumFacing var3 = NORTH;
        float var4 = -Float.MAX_VALUE;
        EnumFacing[] var5 = values();
        int var6 = var5.length;

        for (EnumFacing var8 : var5) {
            float var9 = p_176737_0_ * (float) var8.directionVec.getX() + p_176737_1_ * (float) var8.directionVec.getY() + p_176737_2_ * (float) var8.directionVec.getZ();

            if (var9 > var4) {
                var4 = var9;
                var3 = var8;
            }
        }

        return var3;
    }

    /**
     * Get the opposite Facing (e.g. DOWN => UP)
     */
    public EnumFacing getOpposite() {
        return VALUES[this.opposite];
    }

    /**
     * Rotate this Facing around the given axis clockwise. If this facing cannot be rotated around the given axis,
     * returns this facing without rotating.
     */
    public EnumFacing rotateAround(Axis axis) {
        switch (SwitchPlane.AXIS_LOOKUP[axis.ordinal()]) {
            case 1:
                if (this != WEST && this != EAST) {
                    return this.rotateX();
                }

                return this;

            case 2:
                if (this != UP && this != DOWN) {
                    return this.rotateY();
                }

                return this;

            case 3:
                if (this != NORTH && this != SOUTH) {
                    return this.rotateZ();
                }

                return this;

            default:
                throw new IllegalStateException("Unable to get CW facing for axis " + axis);
        }
    }

    /**
     * Rotate this Facing around the Y axis clockwise (NORTH => EAST => SOUTH => WEST => NORTH)
     */
    public EnumFacing rotateY() {
        switch (SwitchPlane.FACING_LOOKUP[this.ordinal()]) {
            case 1:
                return EAST;

            case 2:
                return SOUTH;

            case 3:
                return WEST;

            case 4:
                return NORTH;

            default:
                throw new IllegalStateException("Unable to get Y-rotated facing of " + this);
        }
    }

    /**
     * Rotate this Facing around the X axis (NORTH => DOWN => SOUTH => UP => NORTH)
     */
    private EnumFacing rotateX() {
        switch (SwitchPlane.FACING_LOOKUP[this.ordinal()]) {
            case 1:
                return DOWN;

            case 2:
            case 4:
            default:
                throw new IllegalStateException("Unable to get X-rotated facing of " + this);

            case 3:
                return UP;

            case 5:
                return NORTH;

            case 6:
                return SOUTH;
        }
    }

    /**
     * Rotate this Facing around the Z axis (EAST => DOWN => WEST => UP => EAST)
     */
    private EnumFacing rotateZ() {
        switch (SwitchPlane.FACING_LOOKUP[this.ordinal()]) {
            case 2:
                return DOWN;

            case 3:
            default:
                throw new IllegalStateException("Unable to get Z-rotated facing of " + this);

            case 4:
                return UP;

            case 5:
                return EAST;

            case 6:
                return WEST;
        }
    }

    /**
     * Rotate this Facing around the Y axis counter-clockwise (NORTH => WEST => SOUTH => EAST => NORTH)
     */
    public EnumFacing rotateYCCW() {
        switch (SwitchPlane.FACING_LOOKUP[this.ordinal()]) {
            case 1:
                return WEST;

            case 2:
                return NORTH;

            case 3:
                return EAST;

            case 4:
                return SOUTH;

            default:
                throw new IllegalStateException("Unable to get CCW facing of " + this);
        }
    }

    /**
     * Returns a offset that addresses the block in front of this facing.
     */
    public int getFrontOffsetX() {
        return this.axis == Axis.X ? this.axisDirection.getOffset() : 0;
    }

    public int getFrontOffsetY() {
        return this.axis == Axis.Y ? this.axisDirection.getOffset() : 0;
    }

    /**
     * Returns a offset that addresses the block in front of this facing.
     */
    public int getFrontOffsetZ() {
        return this.axis == Axis.Z ? this.axisDirection.getOffset() : 0;
    }

    /**
     * Same as getName, but does not override the method from Enum.
     */
    public String getName2() {
        return this.name;
    }

    public String toString() {
        return this.name;
    }

    @Getter
    public static enum Axis implements Predicate {
        X("x", Plane.HORIZONTAL),
        Y("y", Plane.VERTICAL),
        Z("z", Plane.HORIZONTAL);
        private static final Map NAME_LOOKUP = Maps.newHashMap();
        private static final Axis[] $VALUES = new Axis[]{X, Y, Z};
        private static final Axis[] $VALUES$ = new Axis[]{X, Y, Z};

        static {
            Axis[] var0 = values();
            int var1 = var0.length;

            for (Axis var3 : var0) {
                NAME_LOOKUP.put(var3.getName2().toLowerCase(), var3);
            }
        }
        // private static final String __OBFID = "CL_00002321";

        private final String name;
        private final Plane plane;

        Axis(String name, Plane plane) {
            this.name = name;
            this.plane = plane;
        }

        public static Axis byName(String name) {
            return name == null ? null : (Axis) NAME_LOOKUP.get(name.toLowerCase());
        }

        public String getName2() {
            return this.name;
        }

        public boolean isVertical() {
            return this.plane == Plane.VERTICAL;
        }

        public boolean isHorizontal() {
            return this.plane == Plane.HORIZONTAL;
        }

        public String toString() {
            return this.name;
        }

        public boolean apply(EnumFacing facing) {
            return facing != null && facing.getAxis() == this;
        }

        public boolean apply(Object p_apply_1_) {
            return this.apply((EnumFacing) p_apply_1_);
        }
    }

    public static enum AxisDirection {
        POSITIVE(1, "Towards positive"),
        NEGATIVE(-1, "Towards negative");
        private static final AxisDirection[] $VALUES = new AxisDirection[]{POSITIVE, NEGATIVE};
        @Getter
        private final int offset;
        private final String description;


        AxisDirection(int offset, String description) {
            this.offset = offset;
            this.description = description;
        }

        public String toString() {
            return this.description;
        }
    }

    public static enum Plane implements Predicate, Iterable {
        HORIZONTAL(),
        VERTICAL();
        private static final Plane[] $VALUES = new Plane[]{HORIZONTAL, VERTICAL};


        Plane() {
        }

        public EnumFacing[] facings() {
            switch (SwitchPlane.PLANE_LOOKUP[this.ordinal()]) {
                case 1:
                    return new EnumFacing[]{EnumFacing.NORTH, EnumFacing.EAST, EnumFacing.SOUTH, EnumFacing.WEST};
                case 2:
                    return new EnumFacing[]{EnumFacing.UP, EnumFacing.DOWN};
                default:
                    throw new Error("Someone\'s been tampering with the universe!");
            }
        }

        public EnumFacing random(Random rand) {
            EnumFacing[] var2 = this.facings();
            return var2[rand.nextInt(var2.length)];
        }

        public boolean apply(EnumFacing facing) {
            return facing != null && facing.getAxis().getPlane() == this;
        }

        public Iterator iterator() {
            return Iterators.forArray(this.facings());
        }

        public boolean apply(Object p_apply_1_) {
            return this.apply((EnumFacing) p_apply_1_);
        }
    }

    static final class SwitchPlane {
        static final int[] AXIS_LOOKUP;
        static final int[] FACING_LOOKUP;
        static final int[] PLANE_LOOKUP = new int[Plane.values().length];
        // private static final String __OBFID = "CL_00002322";

        static {
            try {
                PLANE_LOOKUP[Plane.HORIZONTAL.ordinal()] = 1;
            } catch (NoSuchFieldError var11) {
                ;
            }

            try {
                PLANE_LOOKUP[Plane.VERTICAL.ordinal()] = 2;
            } catch (NoSuchFieldError var10) {
            }

            FACING_LOOKUP = new int[EnumFacing.values().length];

            try {
                FACING_LOOKUP[EnumFacing.NORTH.ordinal()] = 1;
            } catch (NoSuchFieldError var9) {
            }

            try {
                FACING_LOOKUP[EnumFacing.EAST.ordinal()] = 2;
            } catch (NoSuchFieldError var8) {
            }

            try {
                FACING_LOOKUP[EnumFacing.SOUTH.ordinal()] = 3;
            } catch (NoSuchFieldError ignored) {
            }

            try {
                FACING_LOOKUP[EnumFacing.WEST.ordinal()] = 4;
            } catch (NoSuchFieldError var6) {
            }

            try {
                FACING_LOOKUP[EnumFacing.UP.ordinal()] = 5;
            } catch (NoSuchFieldError var5) {
            }

            try {
                FACING_LOOKUP[EnumFacing.DOWN.ordinal()] = 6;
            } catch (NoSuchFieldError var4) {
            }

            AXIS_LOOKUP = new int[Axis.values().length];

            try {
                AXIS_LOOKUP[Axis.X.ordinal()] = 1;
            } catch (NoSuchFieldError var3) {
            }

            try {
                AXIS_LOOKUP[Axis.Y.ordinal()] = 2;
            } catch (NoSuchFieldError var2) {
            }

            try {
                AXIS_LOOKUP[Axis.Z.ordinal()] = 3;
            } catch (NoSuchFieldError var1) {
            }
        }
    }
}
