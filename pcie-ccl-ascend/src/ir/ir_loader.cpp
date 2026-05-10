#include "ir_loader.hpp"

#include <algorithm>
#include <cstdio>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "../third_party/tinyxml2/tinyxml2.h"
#include "../topology/topology.hpp"
#include "../util/log.hpp"

namespace pccl {

static OpCode parseOpCode(const char* str) {
  if (!str)
    throw std::runtime_error("Missing op attribute");
  std::string s(str);
  if (s == "NOP")
    return OpCode::NOP;
  if (s == "D2D")
    return OpCode::D2D;
  if (s == "D2H")
    return OpCode::D2H;
  if (s == "H2D")
    return OpCode::H2D;
  if (s == "H2H")
    return OpCode::H2H;
  if (s == "REDUCE" || s == "H2HReduce")
    throw std::runtime_error("Ascend PCCL IR does not support " + s);
  if (s == "D2DReduce")
    return OpCode::D2DReduce;
  throw std::runtime_error("Unknown OpCode: " + s);
}

static DeviceBufferKind parseDeviceBufferKind(const char* str, DeviceBufferKind default_kind) {
  if (!str)
    return default_kind;
  std::string s(str);
  if (s == "input")
    return DeviceBufferKind::INPUT;
  if (s == "output")
    return DeviceBufferKind::OUTPUT;
  if (s == "scratch")
    return DeviceBufferKind::SCRATCH;
  if (s == "internal")
    return DeviceBufferKind::INTERNAL;
  throw std::runtime_error("Unknown device buffer kind: " + s);
}

static void parseInstructionCommon(tinyxml2::XMLElement* elem, Instruction& inst, int rank_for_default_numa,
                                   int default_numa) {
  inst.op = parseOpCode(elem->Attribute("op"));
  inst.rank = rank_for_default_numa;
  inst.src_rank = elem->IntAttribute("src_rank", rank_for_default_numa);
  inst.src_buf = parseDeviceBufferKind(elem->Attribute("src_buf"), DeviceBufferKind::INPUT);
  inst.dst_buf = parseDeviceBufferKind(elem->Attribute("dst_buf"), DeviceBufferKind::OUTPUT);

  inst.src.chunk_idx = elem->IntAttribute("src_chunk", 0);
  int src_numa = elem->IntAttribute("src_numa", -1);
  inst.src.numa_id = (src_numa >= 0) ? src_numa : default_numa;

  inst.dst.chunk_idx = elem->IntAttribute("dst_chunk", 0);
  int dst_numa = elem->IntAttribute("dst_numa", -1);
  inst.dst.numa_id = (dst_numa >= 0) ? dst_numa : default_numa;

}

static IRProgram parseDocument(tinyxml2::XMLDocument& doc, int rank, const LocalTopology* topo) {
  auto* root = doc.FirstChildElement("ir_program");
  if (!root)
    throw std::runtime_error("Missing <ir_program> root element");

  IRProgram program;
  program.name = root->Attribute("name") ? root->Attribute("name") : "";
  program.nranks = root->IntAttribute("nranks", 0);
  program.input_chunk_count = 0;
  program.output_chunk_count = 0;

  // Find the <rank> element matching our rank
  tinyxml2::XMLElement* rank_elem = nullptr;
  for (auto* r = root->FirstChildElement("rank"); r; r = r->NextSiblingElement("rank")) {
    if (r->IntAttribute("id", -1) == rank) {
      rank_elem = r;
      break;
    }
  }

  if (!rank_elem) {
    throw std::runtime_error("No <rank id=\"" + std::to_string(rank) + "\"> found in IR");
  }

  // IR uses logical NUMA IDs (0, 1, ...) which now match LocalTopology's
  // logical NUMA directly — no mapping needed.
  int default_numa = topo ? topo->getNUMAForDevice(rank) : 0;

  int max_src_chunk = -1;
  int max_dst_chunk = -1;

  for (auto* inst_elem = rank_elem->FirstChildElement("instruction"); inst_elem;
       inst_elem = inst_elem->NextSiblingElement("instruction")) {
    Instruction inst;
    parseInstructionCommon(inst_elem, inst, rank, default_numa);

    // Track logical input chunks. A D2D/D2DReduce may read OUTPUT/INTERNAL; those source
    // chunk indices must not shrink the compiled input chunk size.
    if ((inst.op == OpCode::D2H && inst.src_buf == DeviceBufferKind::INPUT) ||
        ((inst.op == OpCode::D2D || inst.op == OpCode::D2DReduce) &&
         inst.src_buf == DeviceBufferKind::INPUT)) {
      max_src_chunk = std::max(max_src_chunk, inst.src.chunk_idx);
    }
    if ((inst.op == OpCode::H2D || inst.op == OpCode::D2D || inst.op == OpCode::D2DReduce) &&
        inst.dst_buf == DeviceBufferKind::OUTPUT) {
      max_dst_chunk = std::max(max_dst_chunk, inst.dst.chunk_idx);
    }

    program.instructions.push_back(inst);
  }

  program.input_chunk_count = max_src_chunk + 1;
  program.output_chunk_count = max_dst_chunk + 1;

  // Collect all ranks' instructions for cross-rank dependency analysis (ring buffer).
  int global_max_src_chunk = max_src_chunk;
  int global_max_dst_chunk = max_dst_chunk;
  for (auto* r = root->FirstChildElement("rank"); r; r = r->NextSiblingElement("rank")) {
    int r_id = r->IntAttribute("id", -1);
    int r_numa = topo ? topo->getNUMAForDevice(r_id) : 0;
    for (auto* ie = r->FirstChildElement("instruction"); ie; ie = ie->NextSiblingElement("instruction")) {
      Instruction inst;
      parseInstructionCommon(ie, inst, r_id, r_numa);
      if ((inst.op == OpCode::D2H && inst.src_buf == DeviceBufferKind::INPUT) ||
          ((inst.op == OpCode::D2D || inst.op == OpCode::D2DReduce) &&
           inst.src_buf == DeviceBufferKind::INPUT)) {
        global_max_src_chunk = std::max(global_max_src_chunk, inst.src.chunk_idx);
      }
      if ((inst.op == OpCode::H2D || inst.op == OpCode::D2D || inst.op == OpCode::D2DReduce) &&
          inst.dst_buf == DeviceBufferKind::OUTPUT) {
        global_max_dst_chunk = std::max(global_max_dst_chunk, inst.dst.chunk_idx);
      }
      program.all_instructions.push_back(inst);
    }
  }
  program.input_chunk_count = global_max_src_chunk + 1;
  program.output_chunk_count = global_max_dst_chunk + 1;

  LOG_INFO("IRLoader: Loaded %zu instructions for rank %d (%zu total across all ranks) from '%s'",
           program.instructions.size(), rank, program.all_instructions.size(), program.name.c_str());

  return program;
}

static bool fileExists(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f)
    return false;
  fclose(f);
  return true;
}

static std::string dirnameOf(const std::string& path) {
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos)
    return ".";
  if (pos == 0)
    return "/";
  return path.substr(0, pos);
}

static std::vector<std::string> defaultIRSearchPaths(const std::string& ir_path) {
  std::vector<std::string> search_paths;
  if (!ir_path.empty())
    search_paths.push_back(ir_path);

  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&IRLoader::findIRFile), &info) && info.dli_fname) {
    std::string lib_dir = dirnameOf(info.dli_fname);
    search_paths.push_back(lib_dir + "/ir");
    search_paths.push_back(lib_dir + "/../ir");
    search_paths.push_back(lib_dir + "/share/pccl/ir");
    search_paths.push_back(lib_dir + "/../share/pccl/ir");
  }

  search_paths.push_back("./ir");
  search_paths.push_back("/usr/local/share/pccl/ir");

  std::vector<std::string> unique_paths;
  for (const auto& path : search_paths) {
    if (std::find(unique_paths.begin(), unique_paths.end(), path) == unique_paths.end())
      unique_paths.push_back(path);
  }
  return unique_paths;
}

IRProgram IRLoader::loadFromFile(const std::string& path, int rank, const LocalTopology* topo) {
  tinyxml2::XMLDocument doc;
  auto err = doc.LoadFile(path.c_str());
  if (err != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("Failed to load IR file: " + path + " (error=" + std::to_string(err) + ")");
  }
  return parseDocument(doc, rank, topo);
}

IRProgram IRLoader::loadFromString(const std::string& xml, int rank, const LocalTopology* topo) {
  tinyxml2::XMLDocument doc;
  auto err = doc.Parse(xml.c_str(), xml.size());
  if (err != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("Failed to parse IR XML string (error=" + std::to_string(err) + ")");
  }
  return parseDocument(doc, rank, topo);
}

std::string IRLoader::findIRFile(const std::string& collective, const LocalTopology& topo,
                                 const std::string& ir_path) {
  // Build the logical NUMA pattern string from device_to_numa, e.g. [0,0,1,1] -> "0011".
  std::string pattern;
  for (int device = 0; device < topo.num_devices; ++device)
    pattern += static_cast<char>('0' + topo.device_to_numa[device]);

  std::string filename = collective + "_" + pattern + ".xml";

  // Search paths
  std::vector<std::string> search_paths = defaultIRSearchPaths(ir_path);

  for (const auto& dir : search_paths) {
    std::string full_path = dir + "/" + filename;
    if (fileExists(full_path)) {
      LOG_INFO("IRLoader: Found IR file: %s", full_path.c_str());
      return full_path;
    }
  }

  throw std::runtime_error("IR file not found: " + filename);
}

std::string IRLoader::findRelayIRFile(const std::string& collective, const LocalTopology& topo,
                                      const std::string& ir_path) {
  std::string pattern;
  for (int device = 0; device < topo.num_devices; ++device)
    pattern += static_cast<char>('0' + topo.device_to_numa[device]);

  std::string filename = collective + "_" + pattern + ".xml";
  std::vector<std::string> search_paths = defaultIRSearchPaths(ir_path);

  for (const auto& dir : search_paths) {
    std::string full_path = dir + "/" + filename;
    if (fileExists(full_path)) {
      LOG_INFO("IRLoader: Found relay IR file: %s", full_path.c_str());
      return full_path;
    }
  }
  throw std::runtime_error("IR file not found: " + filename);
}

}  // namespace pccl
