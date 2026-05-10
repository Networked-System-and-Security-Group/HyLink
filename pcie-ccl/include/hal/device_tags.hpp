#ifndef DEVICE_TAGS_HPP
#define DEVICE_TAGS_HPP

namespace pccl {
namespace device_tag {
struct CannTag {};
struct CudaTag {};
}  // namespace device_tag

#if defined(PCCL_DEVICE_CUDA)
using DeviceTag = device_tag::CudaTag;
#elif defined(PCCL_DEVICE_ASCEND)
using DeviceTag = device_tag::CannTag;
#endif

}  // namespace pccl

#endif /* DEVICE_TAGS_HPP */
