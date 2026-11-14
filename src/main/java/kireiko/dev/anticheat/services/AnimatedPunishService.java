package kireiko.dev.anticheat.services;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.core.AsyncScheduler;
import kireiko.dev.millennium.vectors.Pair;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.Particle;
import org.bukkit.World;

import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public final class AnimatedPunishService {
    private static final Set<PlayerProfile> punished
            = ConcurrentHashMap.newKeySet();
    private static final List<Object[]> endAnim = new CopyOnWriteArrayList<>();

    public static void punish(PlayerProfile profile, Pair<String, String> bane) {
        profile.setBanAnimPositions(new Pair<>(profile.getTo().clone(), profile.getTo().clone()));
        profile.setBanAnimInfo(bane);
        punished.add(profile);
    }

    public static void unload(PlayerProfile profile) {
        punished.remove(profile);
    }

    public static void init() {
        Bukkit.getScheduler().runTaskTimer(MX.getInstance(), () -> {
            // async performance
            AsyncScheduler.run(() -> {
                punishAnim();
                outAnim();
            });
        }, 0L, 1L);
    }

    private static synchronized void punishAnim() {
        Set<PlayerProfile> rm = ConcurrentHashMap.newKeySet();
        for (PlayerProfile profile : punished) {
            if (profile.punishAnimation > 100) {
                // end, go to out animation
                Bukkit.getScheduler().runTask(MX.getInstance(), () -> {
                    for (int i = 0; i < 10; i++) {
                        p(profile.getPlayer().getWorld(), profile.getTo(), Particle.FLAME, 10);
                    }
                    endAnim.add(new Object[]{profile.getTo(), 0, profile.getPlayer().getWorld()});
                    profile.getPlayer().teleport(profile.getBanAnimPositions().getY());
                    Bukkit.getScheduler().runTaskLater(MX.getInstance(), () -> {
                        if (profile.getBanAnimInfo() != null) {
                            profile.getPlayer().damage(900);
                            profile.forcePunish(profile.getBanAnimInfo().getX(), profile.getBanAnimInfo().getY());
                        }
                    }, 1L);
                });
                rm.add(profile);
            } else {
                playAnimation(profile, profile.punishAnimation);
            }
            profile.punishAnimation += 2;
        }
        for (PlayerProfile profile : rm)
            punished.remove(profile);
        rm.clear();
    }

    private static synchronized void outAnim() {
        for (Object[] object : endAnim) {
            Location l = (Location) object[0];
            World w = (World) object[2];
            int progress = (int) object[1];
            double d = circIn(0, 20, progress);
            if (progress > 100) {
                endAnim.remove(object);
            } else {
                for (int i = 0; i < 360; i += 30) {
                    double angle = Math.toRadians(i);
                    for (double y = -7; y < 4; y++) {
                        p(w, new Location(w, l.getX() + -Math.sin(angle) * d,
                                        l.getY() + y, l.getZ() + Math.cos(angle) * d),
                                Particle.CRIT_MAGIC, 1);
                    }
                }
                object[1] = ((int) object[1]) + 4;
            }
        }
    }

    private static void playAnimation(PlayerProfile iPlayer, int percent) {
        World w = iPlayer.getPlayer().getWorld();
        Location l = iPlayer.getBanAnimPositions().getY().clone().add(0, circOut(0, 5, percent), 0);
        l.setWorld(iPlayer.getPlayer().getWorld());
        { // chest anim
            Location l1 = l.clone().add(0, -0.4, 0);
            Particle p = Particle.DRIP_LAVA;
            double d = 2.5 - circOut(0, 2, percent); // alpha value
            double circular = 2; // count of circular rotations (2)
            double angle1 = Math.toRadians(percent * (3.6 * circular));
            double angle2 = Math.toRadians((percent * (3.6 * circular)) + 180);
            p(w, new Location(w, l1.getX() + -Math.sin(angle1) * d, l1.getY(), l1.getZ() + Math.cos(angle1) * d), p, 1);
            p(w, new Location(w, l1.getX() + -Math.sin(angle2) * d, l1.getY(), l1.getZ() + Math.cos(angle2) * d), p, 1);
        }

        { // coming explosion anim
            Location l1 = l.clone().add(0, 1, 0);
            Particle p = Particle.VILLAGER_ANGRY;
            double d = 12 - circOut(0, 11.9, percent); // alpha value
            double circular = 1; // count of circular rotations (1)
            double angle1 = Math.toRadians(percent * (3.6 * circular));
            double angle2 = Math.toRadians((percent * (3.6 * circular)) + 180);
            p(w, new Location(w, l1.getX() + -Math.sin(angle1) * d, l1.getY(), l1.getZ() + Math.cos(angle1) * d), p, 2);
            p(w, new Location(w, l1.getX() + -Math.sin(angle2) * d, l1.getY(), l1.getZ() + Math.cos(angle2) * d), p, 2);
        }

        { // mystic particles
            for (double d = 0; d < 5; d += 0.4) {
                Location l1 = l.clone().add(0, d, 0);
                p(w, l1, Particle.ENCHANTMENT_TABLE, 1);
            }
        }
        Bukkit.getScheduler().runTask(MX.getInstance(),
                () -> iPlayer.getPlayer().teleport(l));
    }

    private static void p(World w, Location l, Particle p, int c) {
        // thread-safe
        Bukkit.getScheduler().runTask(MX.getInstance(), () -> w.spawnParticle(p, l, c));
    }

    // interpolation
    private static double circOut(double from, double to, int percent) {
        percent = Math.max(0, Math.min(100, percent));
        double change = to - from;
        double progress = percent / 100.0;
        return from + change * Math.sqrt(1 - Math.pow(progress - 1, 2));
    }

    public static double circIn(double from, double to, int percent) {
        percent = Math.max(0, Math.min(100, percent));
        double change = to - from;
        double progress = percent / 100.0;
        return from - change * (Math.sqrt(1 - Math.pow(progress, 2)) - 1);
    }
}
