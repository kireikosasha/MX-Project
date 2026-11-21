package kireiko.dev.anticheat.checks.movement;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.PacketCheckHandler;
import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.MoveEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.core.MainThread;
import kireiko.dev.anticheat.managers.CheckManager;
import kireiko.dev.anticheat.utils.BlockSnapshot;
import kireiko.dev.anticheat.utils.ConfigCache;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;

import java.util.Map;
import java.util.TreeMap;

public class GhostBlockAbuseCheck implements PacketCheckHandler {
    private final PlayerProfile profile;
    private int flags;
    private Location buffLoc;
    private Map<String, Object> localCfg = new TreeMap<>();

    public GhostBlockAbuseCheck(PlayerProfile profile) {
        this.profile = profile;
        this.flags = 0;
        this.buffLoc = null;
        if (CheckManager.classCheck(this.getClass()))
            this.localCfg = CheckManager.getConfig(this.getClass());
    }

    @Override
    public ConfigLabel config() {
        localCfg.put("buffer", 2);
        return new ConfigLabel("ghost_block_abuse", localCfg);
    }
    @Override
    public void applyConfig(Map<String, Object> params) {
        localCfg = params;
    }

    @Override
    public Map<String, Object> getConfig() {
        return localCfg;
    }

    @Override
    public void event(Object o) {
        if (!(o instanceof MoveEvent)) return;
        if (!ConfigCache.PREVENT_GHOST_BLOCK_ABUSE || profile.isIgnoreFirstTick()) return;
        final MoveEvent event = (MoveEvent) o;
        final Location to = event.getTo().clone();
        final Location from = event.getFrom().clone();
        final boolean onGround = event.getProfile().isGround();
        final int buffer = Math.max(2, ((Number) localCfg.get("buffer")).intValue());
        MainThread.run(() -> {
            if (!onGround || !profile.getPlayer().isOnline()) return;
            final boolean flag = !(isOnSolidGround(to) || isOnSolidGround(from));
            if (flag) {
                this.flags++;
                profile.debug("&7GhostBlock flags: " + this.flags + "/" + buffer);
                if (this.buffLoc == null) {
                    this.buffLoc = from.clone();
                    this.buffLoc.setWorld(profile.getPlayer().getWorld());
                }
                if (this.flags < buffer) {
                    final Location l = to.clone();
                    l.setWorld(profile.getPlayer().getWorld());
                    this.buffLoc = l;
                } else {
                    if (this.buffLoc.getWorld().equals(profile.getPlayer().getWorld())
                    ) {
                        if (this.buffLoc.getY() <= to.getY() + 1.0) {
                            final Location setback = this.buffLoc.clone();
                            Bukkit.getScheduler().runTask(MX.getInstance(), () -> {
                                this.profile.getPlayer().teleport(setback);
                            });
                        }
                        this.flags = 1;
                    } else {
                        this.profile.debug("&7GhostBlock setback invalid world: "
                                        + this.buffLoc.getWorld().getName()
                                        + " " + profile.getPlayer().getWorld().getName());
                        this.flags = 0;
                        this.buffLoc = null;
                    }
                }
            } else this.flags = 0;
        });
    }

    private boolean isOnSolidGround(final Location location) {
        final World world = location.getWorld();
        if (world == null) return true;
        final Material center = world.getBlockAt(
                location.getBlockX(),
                (int) Math.floor(location.getY() - 0.1),
                location.getBlockZ()
        ).getType();
        if (isSolid(center)) return true;
        return BlockSnapshot.capture(
                world,
                location.getX(),
                location.getY() - 0.1,
                location.getZ(),
                0.3,
                0.5,
                0.3
        ).anyMatch(this::isSolid);
    }

    private boolean isSolid(final Material material) {
        if (material == null) return false;
        final String name = material.toString().toLowerCase();
        return !name.contains("air") || material.isSolid() || ignore(name);
    }

    private static boolean ignore(String block) {
        return block.matches(".*(snow|step|frame|table|slab|stair|ladder|vine|waterlily|wall|carpet|fence|rod|bed|skull|pot|hopper|door|piston|lily).*");
    }
}
