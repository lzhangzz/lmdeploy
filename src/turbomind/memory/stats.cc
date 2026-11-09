#include "src/turbomind/memory/stats.h"

#include <format>
#include <iterator>

namespace turbomind {

namespace {

std::string human_bytes(size_t n)
{
    constexpr const char* kU[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double                v    = static_cast<double>(n);
    int                   i    = 0;
    while (v >= 1024.0 && i < 4) {
        v /= 1024.0;
        ++i;
    }
    return std::format("{:.2f}{}", v, kU[i]);
}

float pct(size_t used, size_t total)
{
    return total ? 100.f * static_cast<float>(used) / static_cast<float>(total) : 0.f;
}

}  // namespace

std::string FormatMemoryStats(const MemoryStats& s)
{
    std::string buf;

    std::format_to(std::back_inserter(buf),
                   "[cache] region={} live={} ({:.2f}%) live_alloc={} | pages {}/{} used ({:.2f}%) page_size={}",
                   human_bytes(s.region_bytes),
                   human_bytes(s.live_bytes),
                   pct(s.live_bytes, s.region_bytes),
                   s.live_allocations,
                   s.page.used_pages,
                   s.page.pages,
                   pct(s.page.used_pages, s.page.pages),
                   human_bytes(s.page.page_size));

    std::format_to(std::back_inserter(buf), "\n[cache]   free pages by order:");
    for (int k = 0; k <= s.page.max_order; ++k) {
        if (s.page.free_blocks_by_order[k]) {
            std::format_to(std::back_inserter(buf), " o{}={}", k, s.page.free_blocks_by_order[k]);
        }
    }

    for (const SlabStats& sl : s.slabs) {
        std::format_to(std::back_inserter(buf),
                       "\n[cache]   slab obj={} slab={} cap={} slabs={}(F{}/P{}/E{}) obj {}/{} used ({:.2f}%)",
                       human_bytes(sl.object_size),
                       human_bytes(sl.slab_size),
                       sl.object_count,
                       sl.n_slabs,
                       sl.n_full,
                       sl.n_partial,
                       sl.n_empty,
                       sl.used_objects,
                       sl.total_objects,
                       pct(sl.used_objects, sl.total_objects));
    }

    return buf;
}

}  // namespace turbomind
