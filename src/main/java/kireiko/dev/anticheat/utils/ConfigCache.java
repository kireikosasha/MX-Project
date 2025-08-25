package kireiko.dev.anticheat.utils;

import kireiko.dev.anticheat.MX;

public final class ConfigCache {

    public static double VL_LIMIT;
    public static float VL_RESET;
    public static String ALERT_MSG;
    public static String UNUSUAL;
    public static String SUSPECTED;
    public static String BAN_COMMAND;
    public static String BYPASS;
    public static String BC_MSG;
    public static boolean PUNISH_EFFECT;
    public static boolean INTERACT_SPELL;
    public static boolean IGNORE_CINEMATIC;
    public static boolean LOG_IN_FILES;
    public static boolean ROTATIONS_CONTAINER;
    public static boolean PREVENT_GHOST_BLOCK_ABUSE;
    public static int PREVENTION;
    public static boolean ENABLE_ALERTS_ON_JOIN;


    public static void loadConfig() {
        VL_LIMIT = MX.getInstance().getConfig().getDouble("vlLimit", 100);
        VL_RESET = (float) MX.getInstance().getConfig().getDouble("vlReset", 15);
        PREVENTION = MX.getInstance().getConfig().getInt("prevention", 2);
        ALERT_MSG = MX.getInstance().getConfig().getString("alertMsg", "&9&l[MX] &e%player% &8>>&c %check% &7(&c%component%&7) &8%info% &f[%vl%/%vlLimit%]");
        UNUSUAL = MX.getInstance().getConfig().getString("unusual", "&9&l[MX] &e%player% &8>>&6 Playing suspiciously");
        SUSPECTED = MX.getInstance().getConfig().getString("suspected", "&9&l[MX] &e%player% &8>>&4 Looks like a cheater!");
        BAN_COMMAND = MX.getInstance().getConfig().getString("banCommand", "ban %player% 1d Unfair advantage");
        BYPASS = MX.getInstance().getConfig().getString("bypass", "mx.bypass");
        BC_MSG = MX.getInstance().getConfig().getString("bcMsg", "&c&l[MX]&f %message%");
        PUNISH_EFFECT = MX.getInstance().getConfig().getBoolean("punishEffect", false);
        INTERACT_SPELL = MX.getInstance().getConfig().getBoolean("interactSpell", false);
        IGNORE_CINEMATIC = MX.getInstance().getConfig().getBoolean("ignoreCinematic", false);
        LOG_IN_FILES = MX.getInstance().getConfig().getBoolean("logInFiles", true);
        ROTATIONS_CONTAINER = MX.getInstance().getConfig().getBoolean("rotationsContainer", true);
        PREVENT_GHOST_BLOCK_ABUSE = MX.getInstance().getConfig().getBoolean("preventGhostBlockAbuse", false);
        ENABLE_ALERTS_ON_JOIN = MX.getInstance().getConfig().getBoolean("enableAlertsOnJoin", true);
    }
}
