# Summary

This PR improves docs search in three ways:

- Better search scoring for query relevance, so direct API symbols are ranked ahead of broad concept pages and lower-value internal matches.
- Deduplication of search results for algorithm overloads, so repeated entries collapse to a cleaner set of results.
- Breadcrumbs in search results, so users can see where a result lives in the CCCL docs hierarchy and use search as a way to discover related sections of the documentation.

# Impact

The goal is to make search results feel more navigable and more intentional:

- common API queries are more likely to surface the symbol users expect
- overload-heavy algorithms produce less noisy result lists
- result context makes it easier to explore CUB, Thrust, libcudacxx, and related doc structure from search alone
