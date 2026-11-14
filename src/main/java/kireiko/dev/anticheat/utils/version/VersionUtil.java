package kireiko.dev.anticheat.utils.version;

import lombok.Getter;
import org.bukkit.Bukkit;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class VersionUtil {

    private static final Pattern VERSION_PATTERN = Pattern.compile("MC: ([0-9]+(?:\\.[0-9]+)+)");
    @Getter
    private static final String version = extractVersion(Bukkit.getVersion());

    private static String extractVersion(String fullVersion) {
        Matcher matcher = VERSION_PATTERN.matcher(fullVersion);
        if (matcher.find()) {
            return matcher.group(1);
        }
        return "unknown";
    }

    public static String getBukkitVersion() {
        String[] parts = Bukkit.getServer().getClass().getPackage().getName().split("\\.");
        return parts.length > 3 ? parts[3] : "";
    }

    public static boolean is1_8() {
        return version.startsWith("1.8");
    }

    public static boolean is1_8orAbove() {
        return compareVersion("1.8") >= 0;
    }

    public static boolean is1_9orAbove() {
        return compareVersion("1.9") >= 0;
    }

    public static boolean is1_12orAbove() {
        return compareVersion("1.12") >= 0;
    }

    public static boolean is1_13orAbove() {
        return compareVersion("1.13") >= 0;
    }

    public static boolean is1_14orAbove() {
        return compareVersion("1.14") >= 0;
    }

    public static boolean is1_15orAbove() {
        return compareVersion("1.15") >= 0;
    }

    public static boolean is1_16orAbove() {
        return compareVersion("1.16") >= 0;
    }

    public static boolean is1_17orAbove() {
        return compareVersion("1.17") >= 0;
    }

    public static boolean is1_18orAbove() {
        return compareVersion("1.18") >= 0;
    }

    public static boolean is1_19orAbove() {
        return compareVersion("1.19") >= 0;
    }

    public static boolean is1_20orAbove() {
        return compareVersion("1.20") >= 0;
    }

    public static boolean is1_21orAbove() {
        return compareVersion("1.21") >= 0;
    }

    public static boolean is26orAbove() {
        return compareVersion("26") >= 0;
    }

    public static boolean is26_3orAbove() {
        return compareVersion("26.3") >= 0;
    }

    private static int compareVersion(String v2) {
        String[] v1Parts = VersionUtil.version.split("\\.");
        String[] v2Parts = v2.split("\\.");

        int len = Math.max(v1Parts.length, v2Parts.length);
        for (int i = 0; i < len; i++) {
            int v1Part = i < v1Parts.length ? Integer.parseInt(v1Parts[i]) : 0;
            int v2Part = i < v2Parts.length ? Integer.parseInt(v2Parts[i]) : 0;

            if (v1Part != v2Part) {
                return v1Part - v2Part;
            }
        }

        return 0;
    }
}
