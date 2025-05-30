\
#ifndef SORTED_MAP_HPP
#define SORTED_MAP_HPP

#include <map>
#include <vector>
#include <algorithm> // For std::sort
#include <functional> // For std::function, if needed for more complex comparators

// Forward declaration for a potential custom iterator if we go that route in the future
// template<typename Key, typename ValueContainer> class SortedMapIterator;

// --- Bidirectional map helper ---
template <typename K, typename V>
class bidict {
public:
    std::map<K, V> forward;
    std::map<V, K> inverse;

    bidict(const std::initializer_list<std::pair<const K, V>>& init_list) {
        for (const auto& pair : init_list) {
            forward[pair.first] = pair.second;
            inverse[pair.second] = pair.first;
        }
    }

    const V& at_key(const K& key) const {
        return forward.at(key);
    }

    const K& at_val(const V& val) const {
        return inverse.at(val);
    }

    V& operator[](const K& key) {
        // This will insert if key doesn't exist, then we need to update inverse.
        // For simplicity, assume keys are pre-populated or use .at_key for lookup.
        // Proper handling of [] for insertion would require more logic.
        return forward[key];
    }
};

template<typename Key, typename ValueContainer>
class SortedByValueSizeMap {
public:
    using key_type = Key;
    using mapped_type = ValueContainer;
    using value_type = std::pair<const Key, ValueContainer>; // For compatibility with map-like interfaces
    // Assuming ValueContainer has a size() method and size_type.
    // If ValueContainer could be a raw array, this would need adjustment.
    // For std::vector, this is fine.
    using container_size_type = decltype(std::declval<ValueContainer>().size());

private:
    std::map<Key, ValueContainer> data_;
    bool ascending_; // True for ascending size, false for descending

    mutable std::vector<Key> sorted_keys_cache_;
    mutable bool cache_valid_ = false;

    void _invalidate_cache() const {
        cache_valid_ = false;
        // Clearing the cache vector might be too aggressive if it causes frequent reallocations.
        // For now, just marking as invalid is enough, _build_cache will clear and refill.
        // sorted_keys_cache_.clear(); 
    }

    void _build_cache() const {
        if (cache_valid_) return;
        
        sorted_keys_cache_.clear(); // Clear before refilling
        for (const auto& pair : data_) {
            sorted_keys_cache_.push_back(pair.first);
        }

        std::sort(sorted_keys_cache_.begin(), sorted_keys_cache_.end(),
            [this](const Key& a_key, const Key& b_key) {
                // Ensure keys exist before accessing, though they should if they are in sorted_keys_cache_
                // and came from data_.
                const auto& a_val_container = data_.at(a_key);
                const auto& b_val_container = data_.at(b_key);
                
                container_size_type a_size = a_val_container.size();
                container_size_type b_size = b_val_container.size();

                if (a_size == b_size) {
                    return a_key < b_key; // Tie-breaking: sort by key (ascending)
                }
                return ascending_ ? (a_size < b_size) : (a_size > b_size);
            });
        cache_valid_ = true;
    }

public:
    // Constructor: specifies sort order
    explicit SortedByValueSizeMap(bool ascending = true) : ascending_(ascending) {}

    // Modifiers
    ValueContainer& operator[](const Key& key) {
        _invalidate_cache();
        // This will default-construct ValueContainer if key doesn't exist.
        return data_[key]; 
    }

    ValueContainer& operator[](Key&& key) {
        _invalidate_cache();
        return data_[std::move(key)];
    }
    
    void insert(const Key& key, const ValueContainer& value) {
        _invalidate_cache();
        data_[key] = value;
    }

    void insert(Key&& key, ValueContainer&& value) {
        _invalidate_cache();
        data_[std::move(key)] = std::move(value);
    }

    size_t erase(const Key& key) {
        auto num_erased = data_.erase(key);
        if (num_erased > 0) {
            _invalidate_cache();
        }
        return num_erased;
    }

    void clear() {
        if (!data_.empty()) {
            _invalidate_cache();
            data_.clear();
        }
    }

    // Accessors
    ValueContainer& at(const Key& key) {
        // .at() is non-const on std::map if map is non-const, allows modification.
        // If modification through at() should invalidate cache, this needs thought.
        // However, typical use of at() is for access. If it modifies the ValueContainer
        // in a way that changes its size, the cache would be stale.
        // For simplicity, assume .at() is for read or for modifications that don't change size.
        // If ValueContainer size can change via its reference, cache could become invalid.
        // This is a general problem with caching based on mutable sub-object properties.
        // A safer approach might be to always rebuild cache on access if non-const, or provide const_at.
        return data_.at(key); 
    }

    const ValueContainer& at(const Key& key) const {
        return data_.at(key);
    }

    bool empty() const noexcept { // Added noexcept
        return data_.empty();
    }

    size_t size() const noexcept { // Added noexcept
        return data_.size();
    }
    
    size_t count(const Key& key) const {
        return data_.count(key);
    }

    // Sorted access methods
    std::vector<Key> get_sorted_keys() const {
        _build_cache();
        return sorted_keys_cache_;
    }

    std::vector<value_type> get_sorted_items() const {
        _build_cache();
        std::vector<value_type> items;
        items.reserve(sorted_keys_cache_.size());
        for (const auto& key : sorted_keys_cache_) {
            // Use find to get a const iterator, then construct the pair for items.
            // This ensures we don't accidentally create new entries if a key was in cache but somehow gone from data.
            auto it = data_.find(key);
            if (it != data_.end()) { // Should always be true if cache is consistent
                 items.emplace_back(it->first, it->second);
            }
        }
        return items;
    }
    
    // Provide map-like find, begin, end for convenience. These operate on the underlying std::map
    // and thus iterate in key-sorted order, not value-size-sorted order.
    typename std::map<Key, ValueContainer>::iterator find(const Key& key) {
        return data_.find(key);
    }

    typename std::map<Key, ValueContainer>::const_iterator find(const Key& key) const {
        return data_.find(key);
    }
    
    typename std::map<Key, ValueContainer>::iterator begin() { return data_.begin(); }
    typename std::map<Key, ValueContainer>::const_iterator begin() const { return data_.begin(); }
    typename std::map<Key, ValueContainer>::const_iterator cbegin() const { return data_.cbegin(); } // Added cbegin

    typename std::map<Key, ValueContainer>::iterator end() { return data_.end(); }
    typename std::map<Key, ValueContainer>::const_iterator end() const { return data_.end(); }
    typename std::map<Key, ValueContainer>::const_iterator cend() const { return data_.cend(); } // Added cend
};

#endif // SORTED_MAP_HPP
