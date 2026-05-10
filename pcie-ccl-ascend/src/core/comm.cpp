#include "comm.hpp"

#include "../executor/graph_executor.hpp"
#include "../memory/completion_flag.hpp"
#include "../util/log.hpp"
#include "process_state.hpp"

pcclComm::~pcclComm() {
  graph_executor.reset();
  completion_flag.reset();
  LOG_INFO("pcclComm: Destroyed (rank=%d)", rank);
}
