#pragma once

#include "pdx/common.hpp"
#include <cstring>
#include <limits>
#include <queue>

namespace PDX {

// BOND pruner.
//
// Restored from PDX commit 4a2e65e (removed in #12 "SuperKMeans and Full Refactor").
// Adapted to the post-refactor searcher API used by ADSamplingPruner.
//
// BOND ("Bound by Order N Decoding") performs vertical, dimension-by-dimension
// distance accumulation and uses the running k-th best distance in the heap as
// the pruning threshold. Unlike ADSampling, it does not require a random
// orthogonal rotation of the data, and it does not scale the threshold by a
// per-dimension ratio. The savings come purely from short-circuiting on the
// vertical PDX layout once the partial accumulated distance exceeds the
// current heap top.
class BondPruner {
  public:
    const uint32_t num_dimensions;

    explicit BondPruner(const uint32_t num_dimensions) : num_dimensions(num_dimensions) {}

    // BOND does not rotate or otherwise transform the query; it just hands the
    // raw query through to the distance kernels.
    void PreprocessQuery(
        const float* PDX_RESTRICT const raw_query_embedding,
        float* PDX_RESTRICT const output_query_embedding
    ) const {
        std::memcpy(
            output_query_embedding,
            raw_query_embedding,
            static_cast<size_t>(num_dimensions) * sizeof(float)
        );
    }

    // Same identity transform applied in batch (used at index build time so the
    // searcher path is consistent with ADSampling's PreprocessEmbeddings).
    void PreprocessEmbeddings(
        const float* PDX_RESTRICT const input_embeddings,
        float* PDX_RESTRICT const output_embeddings,
        const size_t num_embeddings
    ) const {
        std::memcpy(
            output_embeddings,
            input_embeddings,
            static_cast<size_t>(num_embeddings) * num_dimensions * sizeof(float)
        );
    }

    // Pruning threshold = top of the heap once we have k candidates,
    // otherwise +inf so nothing is pruned during the warmup phase.
    // The current_dimension_idx parameter is unused (kept for API compatibility
    // with ADSamplingPruner).
    float GetPruningThreshold(
        uint32_t k,
        std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator>& heap,
        const uint32_t /*current_dimension_idx*/
    ) const {
        return heap.size() >= k ? heap.top().distance : std::numeric_limits<float>::max();
    }

    BondPruner(const BondPruner&) = delete;
    BondPruner& operator=(const BondPruner&) = delete;
};

} // namespace PDX
