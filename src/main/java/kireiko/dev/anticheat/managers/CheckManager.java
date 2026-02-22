package kireiko.dev.anticheat.managers;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.api.PacketCheckHandler;
import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.checks.aim.AimAnalysisCheck;
import kireiko.dev.anticheat.checks.aim.AimComplexCheck;
import kireiko.dev.anticheat.checks.aim.AimHeuristicCheck;
import kireiko.dev.anticheat.checks.aim.AimStatisticsCheck;
import kireiko.dev.anticheat.checks.aim.ml.AimMLCheck;
import kireiko.dev.anticheat.checks.clicks.AutoClickerCheck;
import kireiko.dev.anticheat.checks.movement.BaritoneCheck;
import kireiko.dev.anticheat.checks.movement.GhostBlockAbuseCheck;
import kireiko.dev.anticheat.checks.protocol.SprintCheck;
import kireiko.dev.anticheat.checks.velocity.VelocityCheck;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.experimental.UtilityClass;
import org.bukkit.configuration.ConfigurationSection;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.File;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@UtilityClass
public class CheckManager {
    @Getter
    private Set<Class<? extends PacketCheckHandler>> checks = new HashSet<>();
    @Getter
    private final Map<String, PacketCheckHandler> instances = new ConcurrentHashMap<>();

    static {
        checks.addAll(Arrays.asList(
                        AimHeuristicCheck.class,
                        AimComplexCheck.class,
                        AimAnalysisCheck.class,
                        AimStatisticsCheck.class,
                        AimMLCheck.class,
                        VelocityCheck.class,
                        AutoClickerCheck.class,
                        BaritoneCheck.class,
                        GhostBlockAbuseCheck.class,
                        SprintCheck.class
        ));
    }

    @SneakyThrows
    public void init() {
        instances.clear();
        JavaPlugin plugin = MX.getInstance();
        File file = new File(plugin.getDataFolder(), "checks.yml");
        YamlConfiguration cfg = file.exists()
                        ? YamlConfiguration.loadConfiguration(file)
                        : new YamlConfiguration();

        for (Class<? extends PacketCheckHandler> handlerClass : checks) {
            PacketCheckHandler check = handlerClass
                            .getConstructor(PlayerProfile.class)
                            .newInstance((Object) null);
            ConfigLabel defaultLabel = check.config();

            String sectionName = defaultLabel.getName();
            Map<String, Object> defaultParams = defaultLabel.getParameters();

            ConfigurationSection section = cfg.getConfigurationSection(sectionName);
            if (section == null) {
                section = cfg.createSection(sectionName);
            }
            for (Map.Entry<String, Object> e : defaultParams.entrySet()) {
                String key = e.getKey();
                Object val = e.getValue();
                if (val instanceof Map) {
                    if (!section.isConfigurationSection(key)) {
                        section.createSection(key, (Map<?, ?>) val);
                    }
                } else {
                    if (!section.contains(key)) {
                        section.set(key, val);
                    }
                }
            }
            Map<String, Object> mergedParams = new HashMap<>();
            for (Map.Entry<String, Object> e : defaultParams.entrySet()) {
                String key = e.getKey();
                Object val = e.getValue();

                if (val instanceof Map) {
                    ConfigurationSection sub = section.getConfigurationSection(key);
                    mergedParams.put(key, sub != null
                                    ? sub.getValues(false)
                                    : new TreeMap<>());
                } else {
                    mergedParams.put(key, section.get(key));
                }
            }
            check.applyConfig(mergedParams);
            instances.put(check.getClass().getName(), check);
        }
        cfg.save(file);
    }
    public boolean classCheck(Class<?> clazz) {
        return (CheckManager.getInstances().containsKey(clazz.getName()));
    }
    public Map<String, Object> getConfig(Class<?> clazz) {
        return (CheckManager.getInstances().get(clazz.getName())).getConfig();
    }
}
