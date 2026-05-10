#ifndef HOST_TAGS_HPP
#define HOST_TAGS_HPP

namespace pccl {
namespace host_tag {
struct Avx512Tag {};
struct Avx256Tag {};
struct NeonTag {};
struct StandardTag {};
}  // namespace host_tag

#if defined(__AVX512F__)
using HostTag = host_tag::Avx512Tag;
#elif defined(__AVX2__)
using HostTag = host_tag::Avx256Tag;
#elif defined(__ARM_NEON)
using HostTag = host_tag::NeonTag;
#else
using HostTag = host_tag::StandardTag;
#endif

}  // namespace pccl

#endif /* HOST_TAGS_HPP */
