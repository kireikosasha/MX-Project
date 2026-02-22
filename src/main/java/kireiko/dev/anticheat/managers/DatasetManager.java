package kireiko.dev.anticheat.managers;

import kireiko.dev.anticheat.MX;
import kireiko.dev.millennium.ml.data.ObjectML;
import kireiko.dev.millennium.vectors.Pair;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class DatasetManager {

    private static final File FOLDER = new File(MX.getInstance().getDataFolder(), "dataset");

    public static void init() {
        if (!FOLDER.exists()) {
            FOLDER.mkdirs();
        }
    }

    public static void saveSample(List<ObjectML> data, boolean isCheater) {
        String prefix = isCheater ? "cheat_" : "legit_";
        File file = new File(FOLDER, prefix + UUID.randomUUID().toString() + ".dat");
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(data);
        } catch (Exception ignored) {
        }
    }

    @SuppressWarnings("unchecked")
    public static List<Pair<List<ObjectML>, Boolean>> loadDataset() {
        List<Pair<List<ObjectML>, Boolean>> dataset = new ArrayList<>();
        File[] files = FOLDER.listFiles((dir, name) -> name.endsWith(".dat"));
        if (files == null) return dataset;

        for (File file : files) {
            boolean isCheater = file.getName().startsWith("cheat_");
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                List<ObjectML> data = (List<ObjectML>) ois.readObject();
                dataset.add(new Pair<>(data, isCheater));
            } catch (Exception ignored) {
            }
        }
        return dataset;
    }

    public static int getCount() {
        File[] files = FOLDER.listFiles((dir, name) -> name.endsWith(".dat"));
        return files == null ? 0 : files.length;
    }
}