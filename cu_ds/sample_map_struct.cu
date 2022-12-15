#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>

// User-defined value type
struct custom_value_type {
  int32_t f;
  int32_t s;

  __host__ __device__ custom_value_type() {}
  __host__ __device__ custom_value_type(int32_t x) : f{x}, s{x} {}
};

// User-defined device value equal callable
struct custom_value_equals {
  __device__ bool operator()(custom_value_type lhs, custom_value_type rhs)
  {
    return lhs.f == rhs.f;
  }
};

int main(void)
{
    using Key   = int;

    // Empty slots are represented by reserved "sentinel" values. These values should be selected such
    // that they never occur in your input data.
    Key constexpr empty_key_sentinel     = -1;
    auto const empty_value_sentinel = custom_value_type{-1};

    // Number of key/value pairs to be inserted
    std::size_t constexpr num_keys = 50'000;
        
    // Compute capacity based on a 50% load factor
    auto constexpr load_factor = 0.5;
    std::size_t const capacity = std::ceil(num_keys / load_factor);

    // Constructs a map with "capacity" slots using -1 and -1 as the empty key/value sentinels.
    cuco::static_map<Key, custom_value_type> map{
            capacity, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};

    // Create an iterator of input key/value pairs {1, (1, 1)}, {2, (2, 2)}
    auto pairs_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int32_t>(0),
        [] __device__(auto i) { return cuco::make_pair(i, custom_value_type{i}); }
    );

    // Inserts all pairs into the map
    map.insert(pairs_begin, pairs_begin + num_keys);

    // Storage for found values
    thrust::device_vector<custom_value_type> found_values(num_keys);

    // Reproduce inserted keys and values
    auto insert_keys =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int32_t>(0),
                                    [] __device__(auto i) { return i; });
    auto insert_values =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int32_t>(0),
                                    [] __device__(auto i) { return custom_value_type{i}; });

    // Finds all keys {0, 1, 2, ...} and stores associated values into `found_values`
    // If a key `keys_to_find[i]` doesn't exist, `found_values[i] == empty_value_sentinel`
    map.find(insert_keys, insert_keys + num_keys, found_values.begin());

    // Verify that all the found values match the inserted values
    // Use custom equals function for the value type
    bool const all_values_match =
            thrust::equal(found_values.begin(), found_values.end(), insert_values, custom_value_equals());

    if (all_values_match) { std::cout << "Success! Found all values.\n"; }

    return 0;
}
