package kireiko.dev.millennium.types;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Spliterator;

public final class ConcurrentEvictingList<T> extends AbstractList<T> {

    private final int maxSize;
    private final LinkedList<T> delegate = new LinkedList<>();

    public ConcurrentEvictingList(final int maxSize) {
        if (maxSize <= 0) throw new IllegalArgumentException("maxSize must be > 0");
        this.maxSize = maxSize;
    }

    public ConcurrentEvictingList(final Collection<? extends T> c, final int maxSize) {
        this(maxSize);
        for (T t : c) add(t);
    }

    @Override
    public synchronized boolean add(final T t) {
        if (delegate.size() >= maxSize) delegate.removeFirst();
        final boolean changed = delegate.add(t);
        if (changed) modCount++;
        return changed;
    }

    @Override
    public synchronized void add(final int index, final T element) {
        int target = index;
        if (delegate.size() >= maxSize) {
            delegate.removeFirst();
            if (target > 0) target--;
        }
        if (target > delegate.size()) target = delegate.size();
        delegate.add(target, element);
        modCount++;
    }

    @Override
    public synchronized T get(final int index) {
        return delegate.get(index);
    }

    @Override
    public synchronized T set(final int index, final T element) {
        final T previous = delegate.set(index, element);
        modCount++;
        return previous;
    }

    @Override
    public synchronized int size() {
        return delegate.size();
    }

    @Override
    public synchronized T remove(final int index) {
        final T removed = delegate.remove(index);
        modCount++;
        return removed;
    }

    @Override
    public synchronized boolean remove(final Object o) {
        final boolean removed = delegate.remove(o);
        if (removed) modCount++;
        return removed;
    }

    @Override
    public synchronized void clear() {
        delegate.clear();
        modCount++;
    }

    @Override
    public synchronized boolean isEmpty() {
        return delegate.isEmpty();
    }

    public synchronized boolean isFull() {
        return delegate.size() >= maxSize;
    }

    public int getMaxSize() {
        return maxSize;
    }

    public synchronized List<T> snapshot() {
        return new ArrayList<>(delegate);
    }

    @Override
    public synchronized Iterator<T> iterator() {
        return snapshot().iterator();
    }

    @Override
    public synchronized Spliterator<T> spliterator() {
        return snapshot().spliterator();
    }

    @Override
    public synchronized Object[] toArray() {
        return delegate.toArray();
    }

    @Override
    public synchronized <E> E[] toArray(final E[] a) {
        return delegate.toArray(a);
    }
}
