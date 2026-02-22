package kireiko.dev.anticheat.checks.protocol;

import kireiko.dev.anticheat.api.PacketCheckHandler;
import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.EntityActionEvent;
import kireiko.dev.anticheat.api.events.UseEntityEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.listeners.EntityActionListener;
import kireiko.dev.anticheat.managers.CheckManager;
import org.bukkit.entity.Player;

import java.util.Map;
import java.util.TreeMap;

public class SprintCheck implements PacketCheckHandler {
    private final PlayerProfile profile;
    private long lastAction, activeTo;
    private int zeros;
    private boolean stab;
    private Map<String, Object> localCfg = new TreeMap<>();
    public SprintCheck(PlayerProfile profile) {
        this.profile = profile;
        this.lastAction = System.currentTimeMillis();
        this.activeTo = System.currentTimeMillis();
        this.zeros = 0;
        this.stab = false;
        if (CheckManager.classCheck(this.getClass()))
            this.localCfg = CheckManager.getConfig(this.getClass());
    }

    @Override
    public ConfigLabel config() {
        localCfg.put("buffer", 40);
        localCfg.put("addGlobalVl", 25);
        return new ConfigLabel("sprint", localCfg);
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
        if (o instanceof EntityActionEvent) {
            EntityActionEvent event = (EntityActionEvent) o;
            if (event.getAbilitiesEnum() == null) return;
            if (event.getAbilitiesEnum()
                            .equals(EntityActionListener.AbilitiesEnum.START_SPRINTING)
                            || event.getAbilitiesEnum().equals(EntityActionListener
                            .AbilitiesEnum.STOP_SPRINTING)) {
                final long dev = System.currentTimeMillis() - this.lastAction;
                this.lastAction = System.currentTimeMillis();
                if (activeTo < System.currentTimeMillis()) return;
                if (dev < 10 && !stab) {
                    this.zeros += 11;
                    this.stab = true;
                    if (this.zeros > ((Number) localCfg.get("buffer")).intValue()) {
                        /*
                        Goofy KillAura Zero's flaw
                        Detect invalid sprint deviation
                         */
                        profile.punish("Sprint", "Invalid", "[Flaw] Zero's sprint",
                                        ((Number) localCfg.get("addGlobalVl")).floatValue() / 10f);
                        this.zeros -= 6;
                    }
                } else {
                    this.stab = false;
                    if (this.zeros > 0) this.zeros--;
                }
            }
            //profile.getPlayer().sendMessage("action " + dev);
        } else if (o instanceof UseEntityEvent) {
            UseEntityEvent e = (UseEntityEvent) o;
            if (e.isAttack() && e.getTarget() instanceof Player) {
                this.activeTo = System.currentTimeMillis() + 3500;
                if (this.zeros > 0) this.zeros -= 2;
            }
        }
    }
}
