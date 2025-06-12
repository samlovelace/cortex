#ifndef CONCURRENTQUEUE_HPP
#define CONCURRENTQUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>

#include "plog/Log.h"

template <typename T>
class ConcurrentQueue {
public:
    void push(const T& item) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mQueue.push(item);
        }
        mCond.notify_one();
    }

    void push(T&& item) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mQueue.push(std::move(item));
        }
        mCond.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mMutex);
        mCond.wait(lock, [this]() { return !mQueue.empty(); });

        item = std::move(mQueue.front());
        mQueue.pop();
        return true;
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mQueue.empty()) return false;

        item = std::move(mQueue.front());
        mQueue.pop();
        return true;
    }

    bool peek(T& item) const {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mQueue.empty()) return false;
        item = mQueue.front();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mMutex);
        return mQueue.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mMutex);
        return mQueue.size();
    }

private:
    mutable std::mutex mMutex;
    std::queue<T> mQueue;
    std::condition_variable mCond;
};

#endif
