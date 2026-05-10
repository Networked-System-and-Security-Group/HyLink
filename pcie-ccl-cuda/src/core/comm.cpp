#include "comm.hpp"

#include "../executor/graph_executor.hpp"
#include "../fastpath/small_message_executor.hpp"
#include "../memory/completion_flag.hpp"
#include "../util/log.hpp"
#include "process_state.hpp"

pcclComm::~pcclComm() {
  small_message_executor.reset();
  graph_executor.reset();
  small_completion_flag.reset();
  completion_flag.reset();
  LOG_INFO("pcclComm: Destroyed (rank=%d)", rank);
}
