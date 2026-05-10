# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)
project(hccl_test)

message(${CMAKE_CURRENT_SOURCE_DIR})
set(HCCL_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

install(DIRECTORY
    ${HCCL_TEST_DIR}
    DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
)