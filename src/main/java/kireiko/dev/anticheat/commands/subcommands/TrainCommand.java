package kireiko.dev.anticheat.commands.subcommands;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.commands.MXSubCommand;
import kireiko.dev.anticheat.core.AsyncScheduler;
import kireiko.dev.anticheat.managers.DatasetManager;
import kireiko.dev.millennium.ml.FactoryML;
import kireiko.dev.millennium.ml.data.ObjectML;
import kireiko.dev.millennium.ml.logic.Millennium;
import kireiko.dev.millennium.vectors.Pair;
import org.bukkit.command.CommandSender;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public final class TrainCommand extends MXSubCommand {

    public TrainCommand() {
        super("train");
    }

    @Override
    public String getDescription() {
        return "Train ML model from saved dataset";
    }

    @Override
    public String getUsage() {
        return "/" + MX.command + " train <modelIndex> <epochs>";
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
        int index;
        int epochs;
        try {
            index = Integer.parseInt(args[0]);
            epochs = Integer.parseInt(args[1]);
        } catch (Exception e) {
            sender.sendMessage("§cInvalid numbers.");
            return true;
        }

        Millennium model = FactoryML.getModel(index);
        if (model == null) {
            sender.sendMessage("§cModel not found.");
            return true;
        }

        sender.sendMessage("§eStarting training process... This will take a while and run async.");

        AsyncScheduler.run(() -> {
            try {
                List<Pair<List<ObjectML>, Boolean>> dataset = DatasetManager.loadDataset();
                if (dataset.isEmpty()) {
                    sender.sendMessage("§cDataset is empty.");
                    return;
                }
                
                int cheats = 0;
                for(Pair<List<ObjectML>, Boolean> p : dataset) if(p.getY()) cheats++;
                
                sender.sendMessage("§aLoaded " + dataset.size() + " samples (" + cheats + " cheats, " + (dataset.size() - cheats) + " legit).");
                sender.sendMessage("§eTraining " + epochs + " epochs...");

                model.trainEpochs(dataset, epochs);
                model.saveToFile("plugins/MX/models/m1-rnn.dat");

                sender.sendMessage("§aTraining complete and model saved!");
            } catch (Exception e) {
                sender.sendMessage("§cError during training: " + e.getMessage());
                e.printStackTrace();
            }
        });

        return true;
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, String[] args) {
        return null;
    }
}