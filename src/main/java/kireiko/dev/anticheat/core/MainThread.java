package kireiko.dev.anticheat.core;

import kireiko.dev.anticheat.MX;
import org.bukkit.Bukkit;

/**
 * Runs work that touches the Bukkit world on the main server thread.
 * Packet listeners are executed on netty threads, so any block/entity access
 * originating from a check must be dispatched through this helper.
 */
public final class MainThread {

    private MainThread() {
    }

    public static void run(Runnable task) {
        if (Bukkit.isPrimaryThread()) {
            task.run();
        } else {
            Bukkit.getScheduler().runTask(MX.getInstance(), task);
        }
    }
}
