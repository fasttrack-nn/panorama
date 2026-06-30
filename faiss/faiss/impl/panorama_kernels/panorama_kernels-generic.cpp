/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This TU provides:
// 1. _impl specializations for NONE, using scalar code.
// 2. Non-templated Panorama kernel dispatch wrappers
//    (process_level, process_filtering, process_code_compression) declared
//    in panorama_kernels.h. These use DISPATCH_SIMDLevel to route to the
//    best available SIMD implementation via the _impl function template
//    specializations defined in the per-SIMD .cpp files.

#include <faiss/impl/panorama_kernels/panorama_kernels-inl.h>

#include <cstring>

#ifdef __BMI2__
#include <immintrin.h>
#endif

namespace faiss {
namespace panorama_kernels {

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void process_level_impl<SIMDLevel::NONE>(
        size_t level_width_bytes,
        size_t max_batch_size,
        size_t num_active,
        float* sim_table,
        uint8_t* compressed_codes,
        float* exact_distances) {
    for (size_t byte_idx = 0; byte_idx < level_width_bytes; byte_idx++) {
        size_t byte_offset = byte_idx * max_batch_size;
        float* chunk_sim = sim_table + byte_idx * 256;
        for (size_t i = 0; i < num_active; i++) {
            exact_distances[i] += chunk_sim[compressed_codes[byte_offset + i]];
        }
    }
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
std::pair<uint8_t*, size_t> process_code_compression_impl<SIMDLevel::NONE>(
        size_t next_num_active,
        size_t max_batch_size,
        size_t level_width_bytes,
        uint8_t* compressed_codes_begin,
        uint8_t* byteset,
        const uint8_t* codes) {
    uint8_t* compressed_codes = compressed_codes_begin;
    size_t num_active = 0;

    // An important optimization is to skip the compression if all points
    // are active, as we can just use the compressed_codes_begin pointer.
    if (next_num_active < max_batch_size) {
        compressed_codes = compressed_codes_begin;
        for (size_t point_idx = 0; point_idx < max_batch_size;
             point_idx += 64) {
            // The byteset stores one byte per element, value 0 or 1.
            // Compute, in one pass, both:
            //  - byte_masks[g]: 8 bytes of 0xFF / 0x00 lane mask for
            //    PEXT-based byte-level stream compaction
            //  - mask: the 64-bit popcount-friendly bitmask
            // Trick: for each 8-byte chunk where every byte is 0x00 or
            // 0x01, broadcasting bit 0 across each byte lane is just a
            // sequence of OR-shifts within the same byte (no inter-lane
            // carry, since bits 1..7 are originally 0). This sidesteps
            // the old PEXT(byteset)->PDEP(byte_mask) roundtrip.
            uint64_t byte_masks[8];
            uint64_t mask = 0;
            for (int g = 0; g < 8; g++) {
                uint64_t bytes;
                memcpy(&bytes, byteset + point_idx + g * 8, 8);
                uint64_t bm = bytes | (bytes << 1);
                bm |= bm << 2;
                bm |= bm << 4;
                byte_masks[g] = bm;
#ifdef __BMI2__
                uint8_t bits =
                        (uint8_t)_pext_u64(bytes, 0x0101010101010101ULL);
#else
                uint8_t bits = 0;
                for (int b = 0; b < 8; b++) {
                    bits |= ((bytes >> (b * 8)) & 1u) << b;
                }
#endif
                mask |= ((uint64_t)bits << (g * 8));
            }

            // Byte-level stream compaction.
#ifdef __BMI2__
            // PEXT path: byte_mask is precomputed directly from the
            // byteset, no PDEP/multiply needed.
            for (size_t li = 0; li < level_width_bytes; li++) {
                size_t byte_offset = li * max_batch_size;
                const uint8_t* src = codes + byte_offset + point_idx;
                uint8_t* dst = compressed_codes + byte_offset + num_active;
                int write_pos = 0;
                for (int g = 0; g < 8; g++) {
                    uint64_t src_val;
                    memcpy(&src_val, src + g * 8, 8);
                    uint64_t compressed_val =
                            _pext_u64(src_val, byte_masks[g]);
                    int count = __builtin_popcount(
                            (uint8_t)((mask >> (g * 8)) & 0xFF));
                    memcpy(dst + write_pos, &compressed_val, 8);
                    write_pos += count;
                }
            }
#else
            // Scalar fallback: scan set bits one by one and copy
            // the corresponding code byte.
            for (size_t li = 0; li < level_width_bytes; li++) {
                size_t byte_offset = li * max_batch_size;
                const uint8_t* src = codes + byte_offset + point_idx;
                uint8_t* dst = compressed_codes + byte_offset + num_active;
                int write_pos = 0;
                uint64_t m = mask;
                while (m) {
                    int bit = __builtin_ctzll(m);
                    dst[write_pos++] = src[bit];
                    m &= m - 1;
                }
            }
#endif

            num_active += __builtin_popcountll(mask);
        }
    } else {
        num_active = next_num_active;
        compressed_codes = const_cast<uint8_t*>(codes);
    }

    return std::make_pair(compressed_codes, num_active);
}

void process_level(
        size_t level_width_bytes,
        size_t max_batch_size,
        size_t num_active,
        float* sim_table,
        uint8_t* compressed_codes,
        float* exact_distances) {
    DISPATCH_SIMDLevel(
            process_level_impl,
            level_width_bytes,
            max_batch_size,
            num_active,
            sim_table,
            compressed_codes,
            exact_distances);
}

size_t process_filtering(
        size_t num_active,
        float* exact_distances,
        uint32_t* active_indices,
        float* cum_sums,
        uint8_t* byteset,
        size_t batch_offset,
        float dis0,
        float query_cum_norm,
        float heap_max) {
    size_t next_num_active = 0;
    for (size_t i = 0; i < num_active; i++) {
        float exact_distance = exact_distances[i];
        float cum_sum = cum_sums[active_indices[i] - batch_offset];
        float lower_bound = exact_distance + dis0 - cum_sum * query_cum_norm;

        bool keep = heap_max > lower_bound;
        active_indices[next_num_active] = active_indices[i];
        exact_distances[next_num_active] = exact_distance;
        byteset[active_indices[i] - batch_offset] = keep;
        next_num_active += keep;
    }
    return next_num_active;
}

std::pair<uint8_t*, size_t> process_code_compression(
        size_t next_num_active,
        size_t max_batch_size,
        size_t level_width_bytes,
        uint8_t* compressed_codes_begin,
        uint8_t* byteset,
        const uint8_t* codes) {
    DISPATCH_SIMDLevel(
            process_code_compression_impl,
            next_num_active,
            max_batch_size,
            level_width_bytes,
            compressed_codes_begin,
            byteset,
            codes);
}

} // namespace panorama_kernels
} // namespace faiss
