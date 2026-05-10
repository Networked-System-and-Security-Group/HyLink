#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUF_LEN_32 = 8;

extern "C" __global__ __aicore__ void WaitCompletionKernel(GM_ADDR flag_ptr) {
  GlobalTensor<uint32_t> flagGm;
  flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(flag_ptr), BUF_LEN_32);

  TPipe pipe;
  TQue<QuePosition::VECIN, 1> que;
  pipe.InitBuffer(que, 1, BUF_LEN_32 * sizeof(uint32_t));

  uint32_t val = 0;
  while (val == 0) {
    LocalTensor<uint32_t> buf = que.AllocTensor<uint32_t>();
    DataCopy(buf, flagGm, BUF_LEN_32);
    que.EnQue(buf);
    buf = que.DeQue<uint32_t>();
    val = buf.GetValue(0);
    que.FreeTensor(buf);
  }
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void WaitCompletionKernel_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* flag_ptr) {
  WaitCompletionKernel<<<blockDim, l2ctrl, stream>>>(flag_ptr);
}
#endif
