package kireiko.dev.anticheat.commands;

import com.google.common.collect.ImmutableList;
import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.commands.subcommands.*;
import net.md_5.bungee.api.ChatColor;
import net.md_5.bungee.api.chat.ClickEvent;
import net.md_5.bungee.api.chat.ComponentBuilder;
import net.md_5.bungee.api.chat.HoverEvent;
import net.md_5.bungee.api.chat.TextComponent;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabExecutor;
import org.bukkit.entity.Player;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;

import static kireiko.dev.anticheat.utils.MessageUtils.wrapColors;

public final class MXCommandHandler implements TabExecutor {

    private final Set<MXSubCommand> subCommands = new LinkedHashSet<>();

    public MXCommandHandler() {
        this.subCommands.add(new AlertCommand());
        this.subCommands.add(new LogCommand());
        this.subCommands.add(new BroadcastCommand());
        this.subCommands.add(new DebugCommand());
        this.subCommands.add(new FunCommand());
        this.subCommands.add(new InfoCommand());
        this.subCommands.add(new PunishCommand());
        this.subCommands.add(new ReloadCommand());
        this.subCommands.add(new StatsCommand());
        this.subCommands.add(new ActivityCommand());
        this.subCommands.add(new MLCommand());
        this.subCommands.add(new DatasetCommand());
        this.subCommands.add(new TrainCommand());
    }

    @Override
    public boolean onCommand(@NotNull CommandSender commandSender, @NotNull Command command, @NotNull String label, @NotNull String[] args) {
        // check permission first
        if (!commandSender.hasPermission(MX.permission)) {
            commandSender.sendMessage("You don't have permission!");
            return true;
        }
        if (!label.equalsIgnoreCase(MX.command)) {
            commandSender.sendMessage("Usage: /" + MX.command);
            return true;
        }

        if (args.length == 0) {
            this.showHelps(commandSender);
            return true;
        }
        String sCommand = args[0];
        // get the sub command
        for (MXSubCommand subCommand : subCommands) {
            if (!subCommand.getName().equalsIgnoreCase(sCommand)) {
                continue;
            }
            // check if the sub command can only be used by player
            if (subCommand.onlyPlayerCanUse() && !(commandSender instanceof Player)) {
                commandSender.sendMessage("This command can only be used by player!");
                return true;
            }
            // check permission
            if (!subCommand.hasPermission(commandSender)) {
                commandSender.sendMessage("You don't have permission!");
                return true;
            }
            String[] processedArgs = this.processArgs(args);
            if (processedArgs.length > subCommand.getMaxArgs() || processedArgs.length < subCommand.getMinArgs()) {
                commandSender.sendMessage("Usage: " + subCommand.getUsage());
                return true;
            }
            return subCommand.onCommand(commandSender, processedArgs);
        }
        // if not found, just show help
        this.showHelps(commandSender);
        return true;
    }

    @Override
    public @Nullable List<String> onTabComplete(@NotNull CommandSender commandSender, @NotNull Command command, @NotNull String label, @NotNull String[] args) {
        // check permission first
        if (!commandSender.hasPermission(MX.permission)) {
            return ImmutableList.of(); // return empty list
        }
        if (!label.equalsIgnoreCase(MX.command)) {
            return ImmutableList.of();
        }
        if (args.length == 1) {
            List<String> possibleCommands = new ArrayList<>();
            String currentInput = args[0];

            for (MXSubCommand subCommand : this.subCommands) {
                if (subCommand.onlyPlayerCanUse() && !(commandSender instanceof Player)) {
                    continue;
                }
                if (!subCommand.hasPermission(commandSender)) {
                    continue;
                }
                if (currentInput.isEmpty()) {
                    possibleCommands.add(subCommand.getName());
                } else if (subCommand.getName().startsWith(currentInput.toLowerCase(Locale.ROOT))) {
                    possibleCommands.add(subCommand.getName());
                }
            }
            return possibleCommands;
        }
        // or if already have a sub command, then do sub command's tab complete
        for (MXSubCommand subCommand : this.subCommands) {
            if (!subCommand.getName().equalsIgnoreCase(args[0])) {
                continue;
            }
            if (subCommand.onlyPlayerCanUse() && !(commandSender instanceof Player)) {
                return ImmutableList.of(); // return empty list
            }
            if (!subCommand.hasPermission(commandSender)) {
                return ImmutableList.of(); // return empty list
            }
            String[] processedArgs = this.processArgs(args);
            return subCommand.onTabComplete(commandSender, processedArgs);
        }
        // if not found, just return empty list
        return ImmutableList.of();
    }

    private String[] processArgs(String[] args) {
        String[] tempArgs = new String[args.length - 1];
        System.arraycopy(args, 1, tempArgs, 0, args.length - 1);
        return tempArgs;
    }

    /*private static final String[] s = new String[]{
            wrapColors("&9&l" + MX.name + " &fCommands"),
            "",
            wrapColors("&e/" + MX.command + " alerts &f- &cturn on/off alerts"),
            wrapColors("&e/" + MX.command + " info <player> &f- &cplayer info"),
            wrapColors("&e/" + MX.command + " ban <player> &f- &cforce ban"),
            wrapColors("&e/" + MX.command + " reload &f- &cconfig reload"),
            wrapColors("&e/" + MX.command + " stat &f- &cglobal statistics"),
            wrapColors("&e/" + MX.command + " bc &f- &cmessage for all players"),
            wrapColors("&e/" + MX.command + " debug &f- &cverbose checks"),
            wrapColors("&e/" + MX.command + " fun &f- &cfun things"),
            ""
    };*/
    private void showHelps(CommandSender sender) {
        sender.sendMessage(wrapColors("&9&l" + MX.name + " &fCommands"));
        sender.sendMessage("");
        for (MXSubCommand subCommand : subCommands) {
            // check only player can use
            if (subCommand.onlyPlayerCanUse() && !(sender instanceof Player)) {
                continue;
            }
            // check permission
            if (!subCommand.hasPermission(sender)) {
                continue;
            }
            String message = wrapColors(ChatColor.YELLOW + "/" + MX.command + " " + subCommand.getName() + " " + ChatColor.WHITE + "- " + ChatColor.RED + subCommand.getDescription());
            if (sender instanceof Player) {
                TextComponent textComponent = new TextComponent(message);
                textComponent.setClickEvent(new ClickEvent(ClickEvent.Action.SUGGEST_COMMAND, "/" + MX.command + " " + subCommand.getName()));
                textComponent.setHoverEvent(new HoverEvent(HoverEvent.Action.SHOW_TEXT,
                        new ComponentBuilder("Command: ").color(ChatColor.YELLOW).append(subCommand.getName()).color(ChatColor.RED)
                                .append("\nDescription: ").color(ChatColor.YELLOW).append(subCommand.getDescription()).color(ChatColor.RED)
                                .append("\nUsage: ").color(ChatColor.YELLOW).append(subCommand.getUsage()).color(ChatColor.RED)
                                .create()));
                ((Player) sender).spigot().sendMessage(textComponent);
            } else {
                sender.sendMessage(message);
            }
        }
    }
}
