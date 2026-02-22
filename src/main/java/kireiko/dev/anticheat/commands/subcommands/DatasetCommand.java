package kireiko.dev.anticheat.commands.subcommands;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.checks.aim.ml.AimMLCheck;
import kireiko.dev.anticheat.commands.MXSubCommand;
import org.bukkit.Bukkit;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class DatasetCommand extends MXSubCommand {

    public DatasetCommand() {
        super("dataset");
    }

    @Override
    public String getDescription() {
        return "Manage dataset recording for ML";
    }

    @Override
    public String getUsage() {
        return "/" + MX.command + " dataset <legit|cheat|off> <player|all>";
    }

    @Override
    public int getMinArgs() {
        return 2;
    }

    @Override
    public int getMaxArgs() {
        return 2;
    }

    @Override
    public boolean onlyPlayerCanUse() {
        return false;
    }

    @Override
    public boolean onCommand(@NotNull CommandSender sender, String[] args) {
        String mode = args[0].toLowerCase();
        String targetArg = args[1].toLowerCase();

        if (!mode.equals("legit") && !mode.equals("cheat") && !mode.equals("off")) {
            sender.sendMessage("§cUse: legit, cheat, or off.");
            return true;
        }

        Boolean isCheat = null;
        if (mode.equals("cheat")) {
            isCheat = Boolean.TRUE;
        } else if (mode.equals("legit")) {
            isCheat = Boolean.FALSE;
        }

        if (targetArg.equals("all") || targetArg.equals("*")) {
            int count = 0;
            for (Player p : Bukkit.getOnlinePlayers()) {
                if (isCheat == null) {
                    AimMLCheck.RECORDING.remove(p.getUniqueId());
                } else {
                    AimMLCheck.RECORDING.put(p.getUniqueId(), isCheat);
                }
                count++;
            }
            sender.sendMessage("§aApplied dataset mode §e" + mode.toUpperCase() + " §ato §e" + count + " §aonline players.");
        } else {
            Player target = Bukkit.getPlayer(args[1]);
            if (target == null) {
                sender.sendMessage("§cPlayer not found.");
                return true;
            }

            if (isCheat == null) {
                AimMLCheck.RECORDING.remove(target.getUniqueId());
                sender.sendMessage("§eStopped recording for " + target.getName());
            } else {
                AimMLCheck.RECORDING.put(target.getUniqueId(), isCheat);
                sender.sendMessage(isCheat ? "§cNow recording CHEAT data for " + target.getName() : "§aNow recording LEGIT data for " + target.getName());
            }
        }
        return true;
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, String[] args) {
        if (args.length == 1) return Arrays.asList("legit", "cheat", "off");
        if (args.length == 2) {
            List<String> suggestions = new ArrayList<>();
            suggestions.add("all");
            for (Player p : Bukkit.getOnlinePlayers()) {
                suggestions.add(p.getName());
            }
            return suggestions;
        }
        return null;
    }
}