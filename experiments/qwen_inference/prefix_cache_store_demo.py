from common.prefix_cache import PrefixCache



def main():
    cache = PrefixCache()

    cached_prefix = (11, 22, 33)
    cache.put(
        token_ids=cached_prefix,
        past_key_values="KV cache for 11, 22, 33",
    )

    request_token_ids = (11, 22, 33, 44, 55)

    entry = cache.get_longest_prefix(request_token_ids)

    if entry is None:
        raise RuntimeError("Expected a prefix-cache hit")

    matched_tokens = len(entry.token_ids)
    suffix_token_ids = request_token_ids[matched_tokens:]

    print(f"Matched prefix: {entry.token_ids}")
    print(f"Remaining suffix: {suffix_token_ids}")
    print(f"Cache hits: {cache.hits}")
    print(f"Cache misses: {cache.misses}")
    print(f"Cache entries: {cache.entry_count}")
    unrelated_request = (99, 100)

    unrelated_entry = cache.get_longest_prefix(
        unrelated_request
    )

    print(f"Unrelated request entry: {unrelated_entry}")
    print(f"Cache hits after miss: {cache.hits}")
    print(f"Cache misses after miss: {cache.misses}")

if __name__ == "__main__":
    main()