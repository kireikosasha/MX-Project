package kireiko.dev.anticheat.listeners;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.data.PlayerContainer;
import kireiko.dev.anticheat.api.events.UseEntityEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.utils.ConfigCache;
import org.bukkit.entity.Entity;
import org.bukkit.entity.LivingEntity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;

public final class CombatListener implements Listener {

    @EventHandler(priority = EventPriority.LOWEST)
    public void onDamage(EntityDamageByEntityEvent event) {
        if (!(event.getDamager() instanceof Player)) return;

        Player player = (Player) event.getDamager();
        PlayerProfile profile = PlayerContainer.getProfile(player);
        if (profile == null) return;

        Entity entity = event.getEntity();

        if (profile.getAttackBlockToTime() > System.currentTimeMillis()) {
            if (ConfigCache.PREVENTION > 0) {
                event.setCancelled(true);
                if (ConfigCache.PREVENTION >= 3) {
                    player.teleport(player.getLocation());
                } else if (ConfigCache.PREVENTION == 1
                        && entity instanceof LivingEntity
                        && player.getLocation().toVector().distance(entity.getLocation().toVector()) < 3.3) {
                    ((LivingEntity) entity).damage(0.5, player);
                }
                profile.debug("Attack blocked by prevention");
                MX.blockedPerMinuteCount.incrementAndGet();
                return;
            }
        }

        UseEntityEvent e = new UseEntityEvent(entity, true, entity.getEntityId(), event.isCancelled());
        profile.run(e);

        if (e.isCancelled()) {
            event.setCancelled(true);
            profile.debug("Attack blocked after checking");
            MX.blockedPerMinuteCount.incrementAndGet();
        }
    }
}
