/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <iostream>
#include <numeric>
#include <math.h>
#include <string>
#include "mkldnn.hpp"

using namespace mkldnn;

void simple_net()
{
    auto cpu_engine = engine(engine::cpu, 0);

    const int batch = 32;

    std::vector<float> net_src(batch * 8 * 54 * 54);
    std::vector<float> net_out_grad(batch * 8 * 54 * 54);
    std::vector<float> net_in_grad(batch * 8 * 54 * 54);
    std::vector<float> net_in_grad_custom(batch * 8 * 54 * 54);

    /* initializing non-zero values for src */
    for (size_t i = 0; i < net_src.size(); ++i)
        net_src[i] = sinf((float)i);

    /* initializing non-zero values for out_grad */
    for (size_t i = 0; i < net_src.size(); ++i)
      net_out_grad[i] = sinf((float)i);

    memory::dims lrn_src_tz = { batch, 8, 54, 54};



  /* out_grad is default and in_grad is default */
    auto lrn_src_mem = memory(
            { { { lrn_src_tz }, memory::data_type::f32, memory::format::nchw },
              cpu_engine },
            net_src.data());

    const uint32_t local_size = 5;
    const float alpha = 0.0001f;
    const float beta = 0.75f;
    const float k = 1.0f;

    auto lrn_desc = lrn_forward::desc(prop_kind::forward, lrn_across_channels,
                                      lrn_src_mem.get_primitive_desc().desc(),
                                      local_size, alpha, beta, k);
    auto lrn_pd = lrn_forward::primitive_desc(lrn_desc, cpu_engine);

    auto lrn_dst_memory = memory(lrn_pd.dst_primitive_desc());
    auto lrn_workspace_memory = memory(lrn_pd.workspace_primitive_desc());
    auto lrn = lrn_forward(lrn_pd, lrn_src_mem, lrn_workspace_memory,
                           lrn_dst_memory);

    memory::dims lrn_diff_dst_tz = { 32, 8, 54, 54};
    auto lrn_diff_dst_md = memory(
      { { { lrn_diff_dst_tz}, memory::data_type::f32, memory::format::nchw },
        cpu_engine },
      net_out_grad.data());

    auto lrn_bwd_desc = lrn_backward::desc(
            lrn_across_channels, lrn_pd.src_primitive_desc().desc(),
            lrn_diff_dst_md.get_primitive_desc().desc(), local_size, alpha, beta, k);

    auto lrn_bwd_pd
            = lrn_backward::primitive_desc(lrn_bwd_desc, cpu_engine, lrn_pd);

    auto lrn_diff_src_memory = memory(
      { { { lrn_src_tz }, memory::data_type::f32, memory::format::nchw },
        cpu_engine },
      net_in_grad.data());

    auto lrn_diff_out_mem = memory(
      { { { lrn_src_tz }, memory::data_type::f32, memory::format::nchw },
        cpu_engine },
      net_out_grad.data());

    auto lrn_bwd
            = lrn_backward(lrn_bwd_pd, lrn_src_mem, lrn_diff_out_mem,
                           lrn_workspace_memory, lrn_diff_src_memory);


  /* out_grad is default and in_grad is custom */
  auto lrn_src_mem_custom = memory(
      { { { lrn_src_tz }, memory::data_type::f32, memory::format::nChw8c },
        cpu_engine },
      net_src.data());

  auto lrn_desc_custom = lrn_forward::desc(prop_kind::forward, lrn_across_channels,
                                           lrn_src_mem_custom.get_primitive_desc().desc(),
                                    local_size, alpha, beta, k);
  auto lrn_pd_custom = lrn_forward::primitive_desc(lrn_desc_custom, cpu_engine);

  auto lrn_dst_memory_custom = memory(lrn_pd_custom.dst_primitive_desc());
  auto lrn_workspace_memory_custom = memory(lrn_pd_custom.workspace_primitive_desc());
  auto lrn_custom = lrn_forward(lrn_pd_custom, lrn_src_mem_custom, lrn_workspace_memory_custom,
                                lrn_dst_memory_custom);

  auto lrn_diff_dst_md_custom = memory(
      { { { lrn_diff_dst_tz}, memory::data_type::f32, memory::format::nchw },
        cpu_engine },
      net_out_grad.data());

  auto lrn_bwd_desc_custom = lrn_backward::desc(
      lrn_across_channels, lrn_pd_custom.src_primitive_desc().desc(),
      lrn_diff_dst_md_custom.get_primitive_desc().desc(), local_size, alpha, beta, k);


  auto lrn_bwd_pd_custom
      = lrn_backward::primitive_desc(lrn_bwd_desc_custom, cpu_engine, lrn_pd_custom);

  auto lrn_diff_src_memory_custom = memory(
      { { { lrn_src_tz }, memory::data_type::f32, memory::format::nChw8c },
        cpu_engine },
      net_in_grad_custom.data());

  auto lrn_diff_out_mem_custom = memory(
      { { { lrn_src_tz }, memory::data_type::f32, memory::format::nchw },
        cpu_engine },
      net_out_grad.data());

  auto lrn_bwd_custom
      = lrn_backward(lrn_bwd_pd_custom, lrn_src_mem_custom, lrn_diff_out_mem_custom,
                     lrn_workspace_memory_custom, lrn_diff_src_memory_custom);

    std::vector<primitive> net_bwd;
    net_bwd.push_back(lrn_bwd);
    net_bwd.push_back(lrn_bwd_custom);
    int n_iter = 1; //number of iterations for training
    /* execute */
    while (n_iter) {
        /* forward */
//        stream(stream::kind::eager).submit(net_fwd).wait();

        /* update net_diff_dst */
        // auto net_output = pool_user_dst_memory.get_data_handle();
        /*..user updates net_diff_dst using net_output...*/
        // some user defined func update_diff_dst(net_diff_dst.data(),
        // net_output)

        stream(stream::kind::eager).submit(net_bwd).wait();
        /* update weights and bias using diff weights and bias*/
        // auto net_diff_weights
        //     = conv_user_diff_weights_memory.get_data_handle();
        // auto net_diff_bias = conv_diff_bias_memory.get_data_handle();
        /* ...user updates weights and bias using diff weights and bias...*/
        // some user defined func update_weights(conv_weights.data(),
        // conv_bias.data(), net_diff_weights, net_diff_bias);

        --n_iter;
    }

  lrn_diff_src_memory.get_data_handle();
}

int main(int argc, char **argv)
{
    try
    {
        simple_net();
        std::cout << "passed" << std::endl;
    }
    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
