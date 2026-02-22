package kireiko.dev.anticheat.commands.subcommands;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.commands.MXSubCommand;
import kireiko.dev.millennium.ml.ClientML;
import kireiko.dev.millennium.ml.FactoryML;
import kireiko.dev.millennium.ml.logic.Millennium;
import kireiko.dev.millennium.ml.logic.RNNModelML;
import kireiko.dev.millennium.ml.logic.RNNModelML.InputMode;
import kireiko.dev.millennium.ml.logic.RNNModelML.PoolingMode;
import org.bukkit.command.CommandSender;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public final class MLCommand extends MXSubCommand {

    public MLCommand() {
        super("ml");
    }

    @Override
    public String getDescription() {
        return "Manage RNN Model parameters by index";
    }

    @Override
    public String getUsage() {
        return "/" + MX.command + " ml <index> <param> <value> (Params: lr, dr, rdr, wd, gc, ls, im, pm)";
    }

    @Override
    public int getMinArgs() {
        return 3; // <index> <param> <value>
    }

    @Override
    public int getMaxArgs() {
        return 3;
    }

    @Override
    public boolean onlyPlayerCanUse() {
        return false;
    }

    @Override
    public boolean onCommand(@NotNull CommandSender sender, String[] args) {
        int index;
        try {
            index = Integer.parseInt(args[0]);
        } catch (NumberFormatException e) {
            sender.sendMessage("§cInvalid index: " + args[0]);
            return true;
        }

        Millennium modelRaw = FactoryML.getModel(index);
        if (modelRaw == null) {
            sender.sendMessage("§cModel at index " + index + " not found in CACHE!");
            return true;
        }

        if (!(modelRaw instanceof RNNModelML)) {
            sender.sendMessage("§cModel at index " + index + " is not an RNN (v5). This command only supports RNNs.");
            return true;
        }

        RNNModelML rnn = (RNNModelML) modelRaw;
        String param = args[1].toLowerCase();
        String value = args[2];

        try {
            switch (param) {
                case "lr":
                    rnn.setLearningRate(Double.parseDouble(value));
                    sender.sendMessage("§a[ML #" + index + "] learningRate -> " + value);
                    break;
                case "dr":
                    rnn.setDropoutRate(Double.parseDouble(value));
                    sender.sendMessage("§a[ML #" + index + "] dropoutRate -> " + value);
                    break;
                case "rdr":
                    rnn.setRecurrentDropoutRate(Double.parseDouble(value));
                    sender.sendMessage("§a[ML #" + index + "] recurrentDropoutRate -> " + value);
                    break;
                case "wd":
                    rnn.setWeightDecay(Double.parseDouble(value));
                    sender.sendMessage("§a[ML #" + index + "] weightDecay -> " + value);
                    break;
                case "gc":
                    rnn.setGradientClip(Double.parseDouble(value));
                    sender.sendMessage("§a[ML #" + index + "] gradientClip -> " + value);
                    break;
                case "ls":
                    rnn.setLabelSmoothing(Double.parseDouble(value));
                    sender.sendMessage("§a[ML #" + index + "] labelSmoothing -> " + value);
                    break;
                case "im":
                    rnn.setInputMode(InputMode.valueOf(value.toUpperCase()));
                    sender.sendMessage("§a[ML #" + index + "] inputMode -> " + value.toUpperCase());
                    break;
                case "pm":
                    rnn.setPoolingMode(PoolingMode.valueOf(value.toUpperCase()));
                    sender.sendMessage("§a[ML #" + index + "] poolingMode -> " + value.toUpperCase());
                    break;
                default:
                    sender.sendMessage("§cUnknown param. Use: lr, dr, rdr, wd, gc, ls, im, pm");
                    break;
            }
        } catch (Exception e) {
            sender.sendMessage("§cError setting " + param + " to " + value + ": " + e.getMessage());
        }

        return true;
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, String[] args) {
        if (args.length == 1) {
            List<String> indices = new ArrayList<>();
            for (int i = 0; i < ClientML.MODEL_LIST.size(); i++) indices.add(String.valueOf(i));
            return indices;
        }
        if (args.length == 2) {
            return Arrays.asList("lr", "dr", "rdr", "wd", "gc", "ls", "im", "pm");
        }
        if (args.length == 3) {
            if (args[1].equalsIgnoreCase("im")) {
                return Arrays.stream(InputMode.values()).map(Enum::name).collect(Collectors.toList());
            }
            if (args[1].equalsIgnoreCase("pm")) {
                return Arrays.stream(PoolingMode.values()).map(Enum::name).collect(Collectors.toList());
            }
        }
        return null;
    }
}