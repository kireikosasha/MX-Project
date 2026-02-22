package kireiko.dev.millennium.ml;

import kireiko.dev.anticheat.MX;
import kireiko.dev.millennium.ml.logic.*;
import lombok.SneakyThrows;
import lombok.experimental.UtilityClass;

import java.io.File;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@UtilityClass
public class FactoryML {
    private static final Map<Integer, Millennium> CACHE = new ConcurrentHashMap<>();

    public static void createModel(int id, int tableSize, ModelVer ver) {
        createModel(id, tableSize, 10, ver);
    }

    public static Millennium createModel(int id, int tableSize, int stackSize, ModelVer ver) {
        Millennium m;
        switch (ver) {
            case VERSION_5:
                m = new RNNModelML(16, 32);
                break;
            default:
                m = new ModelML(tableSize, stackSize);
                break;
        }
        CACHE.put(id, m);
        return m;
    }

    public static Millennium getModel(int id) {
        return CACHE.get(id);
    }

    public static void removeModel(int id) {
        CACHE.remove(id);
    }

    @SneakyThrows
    public static Millennium loadFromFile(int id, String name, int tSize, int sSize, ModelVer ver) {
        File folder = new File(MX.getInstance().getDataFolder(), "models");
        if (!folder.exists()) folder.mkdirs();
        File file = new File(folder, name);

        if (ver == ModelVer.VERSION_5) {
            RNNModelML m = new RNNModelML(16, 32);
            if (file.exists()) {
                try (InputStream in = Files.newInputStream(file.toPath())) {
                    m.load(in);
                    Logger.info("Model loaded: " + name);
                } catch (Exception e) {
                    Logger.error("Bad v5 model file, overwriting with fresh weights.");
                    m.saveToFile(file.getAbsolutePath());
                }
            } else {
                Logger.warn("Model " + name + " not found. Creating new model file...");
                m.saveToFile(file.getAbsolutePath());
            }
            CACHE.put(id, m);
            return m;
        }

        if (file.exists()) {
            try (ObjectInputStream ois = new ObjectInputStream(Files.newInputStream(file.toPath()))) {
                Millennium m = (Millennium) ois.readObject();
                CACHE.put(id, m);
                Logger.info("Model loaded: " + name);
                return m;
            } catch (Exception e) {
                Logger.error("Failed to deserialize model " + name + ", creating new.");
            }
        }

        Millennium m = createModel(id, tSize, sSize, ver);
        Logger.info("Creating and saving new model: " + file.getAbsolutePath());
        m.saveToFile(file.getAbsolutePath());
        return m;
    }

    @SneakyThrows
    public static Millennium loadFromResources(int id, String name, int tSize, int sSize, ModelVer ver) {
        String path = "/ml/" + name;

        try (InputStream is = MX.class.getResourceAsStream(path)) {
            if (is != null) {
                if (ver == ModelVer.VERSION_5) {
                    RNNModelML m = new RNNModelML(16, 32);
                    m.load(is);
                    CACHE.put(id, m);
                    Logger.info("Model loaded from JAR: " + name);
                    return m;
                } else {
                    try (ObjectInputStream ois = new ObjectInputStream(is)) {
                        Millennium m = (Millennium) ois.readObject();
                        CACHE.put(id, m);
                        Logger.info("Model loaded from JAR: " + name);
                        return m;
                    }
                }
            }
        } catch (Exception ignored) {}

        return loadFromFile(id, name, tSize, sSize, ver);
    }
}