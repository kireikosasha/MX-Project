package kireiko.dev.anticheat.utils;

import kireiko.dev.anticheat.MX;
import kireiko.dev.anticheat.core.AsyncScheduler;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public final class LogUtils {

    private static final File LOGS_FOLDER;

    static {
        LOGS_FOLDER = new File(MX.getInstance().getDataFolder(), "logs");
        if (!LOGS_FOLDER.exists()) {
            LOGS_FOLDER.mkdirs();
        }
    }

    public static boolean createLog(String playerName) {
        File logFile = new File(LOGS_FOLDER, playerName + ".log");
        if (!logFile.exists()) {
            try {
                return logFile.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return false;
    }

    public static void addLog(String playerName, String logMessage) {
        File logFile = new File(LOGS_FOLDER, playerName + ".log");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(logFile, true))) {
            writer.write(logMessage);
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void overwriteLog(String playerName, String logMessage) {
        AsyncScheduler.runIO(() -> {
            File logFile = new File(LOGS_FOLDER, playerName + ".log");
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(logFile, false))) {
                writer.write(logMessage);
                writer.newLine();
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    public static List<String> getLog(String playerName) {
        File logFile = new File(LOGS_FOLDER, playerName + ".log");
        List<String> logs = new ArrayList<>();
        if (logFile.exists()) {
            try (BufferedReader reader = new BufferedReader(new FileReader(logFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    logs.add(line);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return logs;
    }

    public static String getSimpleLog(String playerName) {
        List<String> logs = getLog(playerName);
        StringBuilder r = new StringBuilder();
        for (String s : logs) {
            r.append("\n ").append(s);
        }
        return r.toString();
    }

    public static boolean deleteLog(String playerName) {
        File logFile = new File(LOGS_FOLDER, playerName + ".log");
        return logFile.delete();
    }
}
