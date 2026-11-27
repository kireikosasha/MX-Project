package kireiko.dev.anticheat.core;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import lombok.extern.java.Log;

import java.util.concurrent.*;
import java.util.function.Function;
import java.util.logging.Level;

@Log
public final class AsyncScheduler {

    private static volatile int stopWatchTimeMillis = 1500;
    private static final int WORKER_THREADS =
            Math.max(2, Math.min(4, Runtime.getRuntime().availableProcessors()));
    @Getter
    private static ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(WORKER_THREADS,
            new ThreadFactoryBuilder().setNameFormat("MX Thread %d").build());
    @Getter
    private static final ExecutorService ioExecutor = Executors.newSingleThreadExecutor(
            new ThreadFactoryBuilder().setNameFormat("MX IO Thread").build());

    private AsyncScheduler() {

    }

    public static void shutdown() {
        TryIgnore.ignore(() -> scheduler.shutdownNow());
        TryIgnore.ignore(() -> ioExecutor.shutdownNow());
    }

    public static Future<?> run(Runnable runnable) {
        return scheduler.submit(new DecoratedRunnable(runnable));
    }

    public static <T> Future<T> run(Callable<T> callable) {
        return scheduler.submit(new DecoratedCallable<>(callable));
    }

    public static Future<?> runIO(Runnable runnable) {
        return ioExecutor.submit(new DecoratedRunnable(runnable));
    }

    public static <T> Future<T> runIO(Callable<T> callable) {
        return ioExecutor.submit(new DecoratedCallable<>(callable));
    }

    public static ScheduledFuture<?> later(Runnable runnable, long delay, TimeUnit time) {
        return scheduler.schedule(new DecoratedRunnable(runnable), delay, time);
    }

    public static ScheduledFuture<?> timer(Runnable runnable, long delay, long period, TimeUnit time) {
        return scheduler.scheduleAtFixedRate(new DecoratedRunnable(runnable), delay, period, time);
    }

    public static void cancel(ScheduledFuture<?> timer) {
        try {
            if (timer != null) {
                timer.cancel(true);
            }
        } catch (Exception ignored) {
        }
    }

    private static String describe(Object task) {
        return task.getClass().getName();
    }

    @ToString
    public static class DecoratedRunnable implements Runnable {
        @Setter
        private static Function<Runnable, Runnable> hotfixDecorator = runnable -> runnable;

        private final Runnable originalRunnable;
        private final Runnable decoratedRunnable;

        public DecoratedRunnable(Runnable originalRunnable) {
            this.originalRunnable = originalRunnable;
            this.decoratedRunnable = hotfixDecorator.apply(originalRunnable);
        }

        @Override
        public void run() {
            long start = System.currentTimeMillis();
            try {
                decoratedRunnable.run();
            } catch (Throwable e) {
                if (e instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                    log.log(Level.SEVERE, "Async task interrupted: " + describe(originalRunnable), e);
                } else {
                    log.log(Level.SEVERE, "Error in async task: " + describe(originalRunnable), e);
                }
            } finally {
                long after = System.currentTimeMillis() - start;
                if (after > stopWatchTimeMillis) {
                    log.warning("Busy task " + describe(originalRunnable)
                            + " took " + after + "ms");
                }
            }
        }
    }

    @ToString
    public static class DecoratedCallable<T> implements Callable<T> {
        @Setter
        private static Function<Callable<?>, Callable<?>> hotfixDecorator = callable -> callable;

        private final Callable<T> originalCallable;
        private final Callable<T> decoratedCallable;

        @SuppressWarnings("unchecked")
        public DecoratedCallable(Callable<T> originalCallable) {
            this.originalCallable = originalCallable;
            this.decoratedCallable = (Callable<T>) hotfixDecorator.apply(originalCallable);
        }

        @Override
        public T call() throws Exception {
            long start = System.currentTimeMillis();
            try {
                return decoratedCallable.call();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.log(Level.SEVERE, "Async task interrupted: " + describe(originalCallable), e);
                throw e;
            } catch (Throwable e) {
                log.log(Level.SEVERE, "Error in async task: " + describe(originalCallable), e);
                throw e;
            } finally {
                long after = System.currentTimeMillis() - start;
                if (after > stopWatchTimeMillis) {
                    log.warning("Busy task " + describe(originalCallable)
                            + " took " + after + "ms");
                }
            }
        }
    }
}
