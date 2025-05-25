import collections

class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key) 

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key) 
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)        

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]: 
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)

class SortedByValueLengthDict(collections.UserDict):
    """
    A dictionary-like class that sorts its items based on the length
    of the values associated with each key.

    Iteration methods (keys(), items(), values(), __iter__()) will
    yield items in the specified sorted order.
    The __repr__ and __str__ methods also display the dictionary
    in this sorted order.
    """

    def __init__(self, *args, ascending=True, **kwargs):
        """
        Initializes the SortedByValueLengthDict.

        Args:
            *args: Arguments to pass to the underlying UserDict constructor
                   (e.g., an existing dictionary or an iterable of key-value pairs).
            ascending (bool): If True (default), sorts from shortest value length
                              to longest. If False, sorts from longest to shortest.
            **kwargs: Keyword arguments to pass to the underlying UserDict constructor.
        """
        super().__init__(*args, **kwargs)
        self.ascending = ascending
        self._sorted_keys_cache = None # Cache for the sorted keys

    def _invalidate_cache(self):
        """Marks the sorted keys cache as invalid."""
        self._sorted_keys_cache = None

    def _get_sorted_keys(self):
        """
        Returns a list of keys sorted by the length of their corresponding values.
        Uses a cache to avoid re-sorting if the dictionary hasn't changed.
        """
        if self._sorted_keys_cache is None:
            # Ensure values have a length; otherwise, len() will raise an error.
            # This assumes values are lists, strings, or other objects with __len__.
            try:
                self._sorted_keys_cache = sorted(
                    self.data.keys(),
                    key=lambda k: len(self.data[k]),
                    reverse=not self.ascending
                )
            except TypeError as e:
                # Handle cases where a value might not have a len()
                # For simplicity, we'll re-raise, but you could add specific error handling
                # or default behavior (e.g., treat non-sizable items as having length 0 or infinity).
                raise TypeError(
                    "All values in SortedByValueLengthDict must support len(). "
                    f"Error encountered: {e}"
                ) from e
        return self._sorted_keys_cache

    # --- Methods that modify the dictionary ---
    # These must invalidate the cache.

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._invalidate_cache()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._invalidate_cache()

    def pop(self, key, *args):
        # The *args is for the optional default value in pop.
        value = super().pop(key, *args)
        self._invalidate_cache() # Invalidate even if key wasn't found and default was returned
                                 # or if KeyError was raised by super().pop if no default.
        return value

    def popitem(self):
        """
        Removes and returns a (key, value) pair.
        The pair returned is the "last" item according to the defined sort order.
        """
        if not self.data:
            raise KeyError(f'{self.__class__.__name__} is empty')
        
        # Get sorted keys to determine which item is "last"
        sorted_keys = self._get_sorted_keys()
        if not sorted_keys: # Should not happen if self.data is not empty
             raise RuntimeError("Internal error: sorted keys empty despite data presence.")

        key_to_pop = sorted_keys[-1]
        
        # Pop from the underlying data store
        value = self.data.pop(key_to_pop) # This modifies self.data
        
        # Invalidate cache as data has changed and sorted order might be affected
        self._invalidate_cache() # Important: re-cache will use the modified self.data
        
        return (key_to_pop, value)

    def clear(self):
        super().clear()
        self._invalidate_cache()

    def update(self, *args, **kwargs):
        # UserDict.update calls self.__setitem__ for each item,
        # so the cache will be invalidated by those calls.
        super().update(*args, **kwargs)
        # If super().update modified self.data directly without calling self.__setitem__,
        # then an explicit _invalidate_cache() would be needed here.
        # However, collections.UserDict's update does use self[k] = v.

    def setdefault(self, key, default=None):
        # UserDict.setdefault calls self.__setitem__ if the key is new.
        if key not in self.data:
            # __setitem__ will be called, invalidating the cache.
            return super().setdefault(key, default)
        return self.data[key] # Key exists, no change, cache remains valid.


    # --- Methods that provide views/iterations in sorted order ---

    def __iter__(self):
        """Iterates over keys in sorted order."""
        return iter(self._get_sorted_keys())

    def keys(self):
        """Returns a list of keys in sorted order."""
        return self._get_sorted_keys() # Returns the cached list

    def items(self):
        """Yields (key, value) pairs in sorted order."""
        for k in self._get_sorted_keys():
            yield (k, self.data[k])

    def values(self):
        """Yields values in sorted order."""
        for k in self._get_sorted_keys():
            yield self.data[k]

    # --- Representation ---

    def __repr__(self):
        """Returns a string representation of the dictionary, sorted."""
        if not self.data:
            return f"{self.__class__.__name__}({{}})"
        
        # Use self.items() to get sorted items for representation
        item_strs = [f"{repr(k)}: {repr(v)}" for k, v in self.items()]
        return f"{self.__class__.__name__}({{{', '.join(item_strs)}}})"

    def __str__(self):
        """Returns a user-friendly string representation, sorted."""
        if not self.data:
            return "{}"
        item_strs = [f"{repr(k)}: {repr(v)}" for k, v in self.items()]
        return f"{{{', '.join(item_strs)}}}"

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Example 1: Ascending Order (default) ---")
    data1 = {
        'positionC': ['matchA', 'matchB'],  # 2 items
        'positionA': ['matchD', 'matchE', 'matchF', 'matchG'],  # 4 items
        'positionB': ['matchH'],  # 1 item
        'positionD': ['matchI', 'matchJ', 'matchK']  # 3 items
    }
    sorted_dict_asc = SortedByValueLengthDict(data1)

    print(f"Original data: {data1}")
    print(f"SortedDict (ascending) repr: {repr(sorted_dict_asc)}")
    print(f"SortedDict (ascending) str: {str(sorted_dict_asc)}")

    print("Keys (ascending):", list(sorted_dict_asc.keys()))
    print("Items (ascending):", list(sorted_dict_asc.items()))
    print("Values (ascending):", list(sorted_dict_asc.values()))

    print("\nModifying the dictionary (ascending):")
    sorted_dict_asc['positionE'] = ['x'] # 1 item, should come first or second
    print(f"After adding 'positionE': {sorted_dict_asc}")
    
    popped_item = sorted_dict_asc.popitem() # Should pop the one with the most items ('positionA')
    print(f"Popped item (should be 'positionA' related): {popped_item}")
    print(f"After popitem: {sorted_dict_asc}")

    sorted_dict_asc.pop('positionC')
    print(f"After popping 'positionC': {sorted_dict_asc}")


    print("\n--- Example 2: Descending Order ---")
    data2 = {
        'short': [1],
        'medium': [1, 2, 3],
        'long': [1, 2, 3, 4, 5],
        'zero': []
    }
    sorted_dict_desc = SortedByValueLengthDict(data2, ascending=False)
    print(f"Original data: {data2}")
    print(f"SortedDict (descending): {sorted_dict_desc}")
    print("Keys (descending):", list(sorted_dict_desc.keys()))

    print("\n--- Example 3: Empty and single item dict ---")
    empty_dict = SortedByValueLengthDict()
    print(f"Empty dict: {empty_dict}")
    empty_dict['a'] = [1,2]
    print(f"After adding one item: {empty_dict}")
    empty_dict.popitem()
    print(f"After popitem on single item dict: {empty_dict}")

    print("\n--- Example 4: Values that are not lists (e.g. strings) ---")
    data3 = {
        'greeting': "hello", # len 5
        'name': "world",   # len 5
        'char': "a",       # len 1
        'empty_str': ""    # len 0
    }
    sorted_dict_strings = SortedByValueLengthDict(data3)
    print(f"Original string data: {data3}")
    print(f"SortedDict (strings, ascending): {sorted_dict_strings}")
    # Expected: {'empty_str': '', 'char': 'a', 'greeting': 'hello', 'name': 'world'} (or name then greeting for tie)

    print("\n--- Example 5: Cache check (debug prints in _get_sorted_keys would show this) ---")
    # To truly test cache, you'd add print statements in _get_sorted_keys
    # and see it's not called repeatedly for multiple iterations without modification.
    cache_test_dict = SortedByValueLengthDict({'a':[1,2], 'b':[1]})
    print("First iteration (builds cache):", list(cache_test_dict.items()))
    print("Second iteration (uses cache):", list(cache_test_dict.items()))
    cache_test_dict['c'] = [1,2,3] # Modifies, invalidates cache
    print("Third iteration (rebuilds cache):", list(cache_test_dict.items()))

