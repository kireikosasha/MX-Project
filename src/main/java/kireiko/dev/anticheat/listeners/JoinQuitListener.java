package kireiko.dev.anticheat.listeners;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.data.PlayerContainer;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.utils.ConfigCache;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.scheduler.BukkitRunnable;

import static kireiko.dev.anticheat.utils.MessageUtils.wrapColors;

public final class JoinQuitListener implements Listener {
    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        PlayerContainer.init(event.getPlayer());

        new BukkitRunnable() {
            int attempts = 0;

            @Override
            public void run() {
                PlayerProfile profile = PlayerContainer.getProfile(event.getPlayer());
                if (profile != null) {
                    if (event.getPlayer().hasPermission(MX.permission) && ConfigCache.ENABLE_ALERTS_ON_JOIN) {
                        profile.setAlerts(true);
                        event.getPlayer().sendMessage(wrapColors("&cAlerts: &etrue"));
                    }
                    cancel();
                    return;
                }
                if (++attempts >= 40) { // ~2s timeout
                    cancel();
                }
            }
        }.runTaskTimer(MX.getInstance(), 1L, 1L);
    }

    @EventHandler
    public void onPlayerQuit(PlayerQuitEvent event) {
        PlayerContainer.unload(event.getPlayer());
    }
}
