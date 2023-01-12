/******************************************************************************
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>

namespace ns_FakeQuantizeKernel
{
    struct Params
    {
        int64_t levels;
        int64_t level_low;
        int64_t level_high;
    };
}

bool register_fakequantize_fwd()
{
    // Registering custom_op::fakequantize_fwd
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::TENSOR, 2};
    habana::custom_op::InputDesc input_d_desc{
        habana::custom_op::input_type::USER_PARAMS, 3}; // shouldnt this be
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc, input_d_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack &inputs) -> std::vector<int64_t>
    {
        auto self = inputs[0].toTensor(); // input
        std::vector<int64_t> result_sizes = self.sizes().vec();
        return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // What does this means
    //  user param callback
    auto user_params_lambda = [](const at::Stack &inputs, size_t &size)
    {
        HPU_PARAMS_STUB(ns_FakeQuantizeKernel::Params);
        params->levels = inputs[3].toInt(); // levels
        return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::fakequantize_fwd", // schema name
        "custom_quantize_fwd_f32",     // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::fakequantize_fwd\n";
    return true;
}

bool register_fakequantize_bwd()
{
    // Registering custom_op::fakequantize_bwd
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::TENSOR, 2};
    habana::custom_op::InputDesc input_d_desc{
        habana::custom_op::input_type::TENSOR, 3};
    habana::custom_op::InputDesc input_e_desc{
        habana::custom_op::input_type::USER_PARAMS, 4};
    habana::custom_op::InputDesc input_f_desc{
        habana::custom_op::input_type::USER_PARAMS, 5};
    habana::custom_op::InputDesc input_g_desc{
        habana::custom_op::input_type::USER_PARAMS, 6};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc, input_d_desc,
        input_e_desc, input_f_desc, input_g_desc};

    // output desc
    // output shape callback
    auto grad_input_desc_size_lambda =
        [](const at::Stack &inputs) -> std::vector<int64_t>
    {
        auto self = inputs[0].toTensor(); // grad_output
        std::vector<int64_t> result_sizes = self.sizes().vec();
        return result_sizes;
    };

    auto grad_input_low_desc_size_lambda =
        [](const at::Stack &inputs) -> std::vector<int64_t>
    {
        auto self = inputs[2].toTensor(); // input_low
        std::vector<int64_t> result_sizes = self.sizes().vec();
        return result_sizes;
    };

    auto grad_input_range_desc_size_lambda =
        [](const at::Stack &inputs) -> std::vector<int64_t>
    {
        auto self = inputs[3].toTensor(); // input_range
        std::vector<int64_t> result_sizes = self.sizes().vec();
        return result_sizes;
    };

    habana::custom_op::OutputDesc grad_input_desc{
        0, c10::ScalarType::Float, grad_input_desc_size_lambda};
    habana::custom_op::OutputDesc grad_input_low_desc{
        1, c10::ScalarType::Float, grad_input_low_desc_size_lambda};
    habana::custom_op::OutputDesc grad_input_range_desc{
        2, c10::ScalarType::Float, grad_input_range_desc_size_lambda};
    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        grad_input_desc, grad_input_low_desc, grad_input_range_desc};

    // What does this means
    //  user param callback
    auto user_params_lambda = [](const at::Stack &inputs, size_t &size)
    {
        HPU_PARAMS_STUB(ns_FakeQuantizeKernel::Params);
        params->levels = inputs[4].toInt();     // levels
        params->level_low = inputs[5].toInt();  // level_low
        params->level_high = inputs[6].toInt(); // level_high
        return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::fakequantize_bwd", // schema name
        "custom_quantize_bwd_f32",     // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::fakequantize_bwd\n";
    return true;
}

at::Tensor fakequantize_fwd_execute(
    at::Tensor input,
    at::Tensor input_low,
    at::Tensor input_range,
    int64_t levels)
{
    // TODO
    // Registering the custom op, need to be called only once
    static bool registered = register_fakequantize_fwd();
    TORCH_CHECK(registered, "fakequantize_fwd kernel not registered");

    std::vector<c10::IValue> inputs{input, input_low, input_range, levels};
    // Get custom op descriptor from registry
    auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::fakequantize_fwd");

    // Actual call for op execution
    std::vector<at::Tensor> output = op_desc.execute(inputs);
    // op_desc.execute will always return a vector
    return output[0];
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fakequantize_bwd_execute(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor input_low,
    at::Tensor input_range,
    int64_t levels,
    int64_t level_low,
    int64_t level_high)
{
    // TODO
    // Registering the custom op, need to be called only once
    static bool registered = register_fakequantize_bwd();
    TORCH_CHECK(registered, "fakequantize_bwd kernel not registered");

    std::vector<c10::IValue> inputs{grad_output, input, input_low, input_range, levels, level_low, level_high};
    // Get custom op descriptor from registry
    auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::fakequantize_bwd");

    // Actual call for op execution
    std::vector<at::Tensor> output = op_desc.execute(inputs);
    // op_desc.execute will always return a vector
    return {output[0], output[1], output[2]};
}

TORCH_LIBRARY(custom_op, m)
{
    m.def("fakequantize_fwd(Tensor input, Tensor input_low, Tensor input_range, int levels) -> Tensor");
    m.def("fakequantize_bwd(Tensor grad_output, Tensor input, Tensor input_low, Tensor input_range, int levels, int level_low, int level_high) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m)
{
    m.impl("fakequantize_fwd", fakequantize_fwd_execute);
    m.impl("fakequantize_bwd", fakequantize_bwd_execute);
}
