#include <cstdint>

#include "memory/shared_state.hpp"

int main() {
  pccl::SharedStateManager manager(1);
  constexpr uint64_t hash = 0x123456789abcdef0ULL;

  const int first = manager.userReadyCommSlot(hash);
  const int second = manager.userReadyCommSlot(hash);
  if (first < 0 || second != first)
    return 1;

  // A release with a mismatched hash must not alter the live slot.
  const bool mismatched_release = manager.releaseUserReadyCommSlot(first, hash + 1);
  if (mismatched_release)
    return 1;
  const int third = manager.userReadyCommSlot(hash);
  if (third != first)
    return 2;

  // Releasing two of three references must keep the slot occupied.
  if (!manager.releaseUserReadyCommSlot(first, hash) || !manager.releaseUserReadyCommSlot(second, hash))
    return 3;
  const int fourth = manager.userReadyCommSlot(hash);
  if (fourth != first)
    return 4;

  // The final release clears the slot, allowing a new communicator hash to use it.
  if (!manager.releaseUserReadyCommSlot(third, hash) || !manager.releaseUserReadyCommSlot(fourth, hash))
    return 5;
  const int replacement = manager.userReadyCommSlot(hash);
  if (replacement != first || !manager.releaseUserReadyCommSlot(replacement, hash))
    return 6;
  return 0;
}
