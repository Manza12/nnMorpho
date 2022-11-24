#include "binary_operators.h"

/* Template switch */
// Erosion
torch::Tensor erosion_dispatcher(
		torch::Tensor input,
		torch::Tensor str_el,
		int origin_x,
		int origin_y,
		char border_type,
        int block_size_x,
        int block_size_y
		) {

    switch (input.scalar_type()) {
        case torch::ScalarType::Bool:
            return erosion<bool>(input, str_el, origin_x, origin_y, border_type, block_size_x, block_size_y);
        case torch::ScalarType::Byte:
            return erosion<uint8_t>(input, str_el, origin_x, origin_y, border_type, block_size_x, block_size_y);
        case torch::ScalarType::Char:
            return erosion<int8_t>(input, str_el, origin_x, origin_y, border_type, block_size_x, block_size_y);
        case torch::ScalarType::Short:
            return erosion<int16_t>(input, str_el, origin_x, origin_y, border_type, block_size_x, block_size_y);
        case torch::ScalarType::Int:
            return erosion<int>(input, str_el, origin_x, origin_y, border_type, block_size_x, block_size_y);
        case torch::ScalarType::Long:
            return erosion<int64_t>(input, str_el, origin_x, origin_y, border_type, block_size_x, block_size_y);
        case torch::ScalarType::Half:
            return erosion<at::Half>(input, str_el, origin_x, origin_y, border_type, block_size_x, block_size_y);
        case torch::ScalarType::Float:
            return erosion<float>(input, str_el, origin_x, origin_y, border_type, block_size_x, block_size_y);
        case torch::ScalarType::Double:
            return erosion<double>(input, str_el, origin_x, origin_y, border_type, block_size_x, block_size_y);
        default:
            throw std::invalid_argument("[nnMorpho] Scalar type not supported.\n");
    }
}

// Dilation
torch::Tensor dilation_dispatcher(
        torch::Tensor input,
        torch::Tensor str_el,
        int origin_x,
        int origin_y,
        int block_size_x,
        int block_size_y
) {

    switch (str_el.scalar_type()) {
        case torch::ScalarType::Bool:
            return dilation<bool>(input, str_el, origin_x, origin_y, std::numeric_limits<bool>::min(), block_size_x, block_size_y);
        case torch::ScalarType::Byte:
            return dilation<uint8_t>(input, str_el, origin_x, origin_y, std::numeric_limits<uint8_t>::min(), block_size_x, block_size_y);
        case torch::ScalarType::Char:
            return dilation<int8_t>(input, str_el, origin_x, origin_y, std::numeric_limits<int8_t>::min(), block_size_x, block_size_y);
        case torch::ScalarType::Short:
            return dilation<int16_t>(input, str_el, origin_x, origin_y, std::numeric_limits<int16_t>::min(), block_size_x, block_size_y);
        case torch::ScalarType::Int:
            return dilation<int>(input, str_el, origin_x, origin_y, std::numeric_limits<int>::min(), block_size_x, block_size_y);
        case torch::ScalarType::Long:
            return dilation<int64_t>(input, str_el, origin_x, origin_y, std::numeric_limits<int64_t>::min(), block_size_x, block_size_y);
        case torch::ScalarType::Half:
            return dilation<at::Half>(input, str_el, origin_x, origin_y, -std::numeric_limits<at::Half>::infinity(), block_size_x, block_size_y);
        case torch::ScalarType::Float:
            return dilation<float>(input, str_el, origin_x, origin_y, -std::numeric_limits<float>::infinity(), block_size_x, block_size_y);
        case torch::ScalarType::Double:
            return dilation<double>(input, str_el, origin_x, origin_y, -std::numeric_limits<double>::infinity(), block_size_x, block_size_y);
        default:
            throw std::invalid_argument("[nnMorpho] Scalar type not supported.\n");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("erosion", &erosion_dispatcher, "Erosion");
    m.def("dilation", &dilation_dispatcher, "Dilation");
}
