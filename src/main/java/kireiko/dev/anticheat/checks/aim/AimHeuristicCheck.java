package kireiko.dev.anticheat.checks.aim;

import kireiko.dev.anticheat.api.PacketCheckHandler;
import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.NoRotationEvent;
import kireiko.dev.anticheat.api.events.RotationEvent;
import kireiko.dev.anticheat.api.events.UseEntityEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.checks.aim.heuristic.*;
import kireiko.dev.anticheat.managers.CheckManager;
import kireiko.dev.millennium.vectors.Vec2f;
import lombok.Getter;

import java.util.*;

public final class AimHeuristicCheck implements PacketCheckHandler {

    @Getter
    private final PlayerProfile profile;
    private final Set<HeuristicComponent> components;
    private long lastAttack;
    private Map<String, Object> localCfg = new HashMap<>();
    private final Map<String, Map<String, Object>> defaultConfigs = new TreeMap<>();

    @Override
    public ConfigLabel config() {
        for (final HeuristicComponent component : components) {
            ConfigLabel label = component.config();
            localCfg.put(label.getName(), label.getParameters());
        }
        return new ConfigLabel("aim_heuristic", localCfg);
    }

    @Override
    public void applyConfig(Map<String, Object> params) {
        this.localCfg = params;
        for (HeuristicComponent comp : components) {
            ConfigLabel label    = comp.config();
            String section       = label.getName();
            Map<String, Object> defaults = defaultConfigs.getOrDefault(section,
                            Collections.emptyMap());
            Object rawSection = params.get(section);
            Map<String, Object> fileParams = rawSection instanceof Map
                            ? new TreeMap<>((Map<String, Object>) rawSection)
                            : Collections.emptyMap();
            Map<String, Object> merged = new TreeMap<>(defaults);
            merged.putAll(fileParams);
            comp.applyConfig(merged);
        }
    }

    @Override
    public Map<String, Object> getConfig() {
        return localCfg;
    }

    public AimHeuristicCheck(PlayerProfile profile) {
        this.profile = profile;
        this.lastAttack = 0L;
        this.components = new HashSet<>();
        { // components
            this.components.add(new AimBasicCheck(this));
            this.components.add(new AimConstantCheck(this));
            this.components.add(new AimInvalidCheck(this));
            this.components.add(new AimInconsistentCheck(this));
            this.components.add(new AimPatternCheck(this));
            this.components.add(new AimFactorCheck(this));
            this.components.add(new AimSmoothCheck(this));
        }
        for (HeuristicComponent comp : components) {
            ConfigLabel lbl = comp.config();
            defaultConfigs.put(
                            lbl.getName(),
                            new HashMap<>(lbl.getParameters())
            );
        }
        if (CheckManager.classCheck(this.getClass())) {
            applyConfig(CheckManager.getConfig(this.getClass()));
        }
    }

    @Override
    public void event(Object o) {
        if (o instanceof RotationEvent) {
            RotationEvent event = (RotationEvent) o;
            if (System.currentTimeMillis() > this.lastAttack + 3500 || profile.isIgnoreFirstTick()) return;
            for (HeuristicComponent component : components) component.process(event);
        } else if (o instanceof NoRotationEvent) {
            if (System.currentTimeMillis() > this.lastAttack + 3500 || profile.isIgnoreFirstTick()) return;
            for (HeuristicComponent component : components) component.process(new RotationEvent(profile, new Vec2f(0, 0), new Vec2f(0, 0)));
        } else if (o instanceof UseEntityEvent) {
            UseEntityEvent event = (UseEntityEvent) o;
            if (event.isAttack()) {
                this.lastAttack = System.currentTimeMillis();
            }
        }
    }
}
