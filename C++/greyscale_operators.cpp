#include "greyscale_operators.hpp"
#include <limits>

// Erosion
torch::Tensor erosion_dispatcher(
		torch::Tensor input,
		torch::Tensor str_el,
		torch::Tensor footprint,
		int origin_x,
		int origin_y,
		char border_type
		) {

    switch (input.scalar_type()) {
        case torch::ScalarType::Byte:
            return erosion<uint8_t>(input, str_el, footprint, origin_x, origin_y, border_type, std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::min());
        case torch::ScalarType::Char:
            return erosion<int8_t>(input, str_el, footprint, origin_x, origin_y, border_type, std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min());
        case torch::ScalarType::Short:
            return erosion<int16_t>(input, str_el, footprint, origin_x, origin_y, border_type, std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::min());
        case torch::ScalarType::Int:
            return erosion<int>(input, str_el, footprint, origin_x, origin_y, border_type, std::numeric_limits<int>::max(), std::numeric_limits<int>::min());
        case torch::ScalarType::Long:
            return erosion<int64_t>(input, str_el, footprint, origin_x, origin_y, border_type, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min());
        case torch::ScalarType::Half:
            return erosion<at::Half>(input, str_el, footprint, origin_x, origin_y, border_type, std::numeric_limits<at::Half>::infinity(), -std::numeric_limits<at::Half>::infinity());
        case torch::ScalarType::Float:
            return erosion<float>(input, str_el, footprint, origin_x, origin_y, border_type, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        case torch::ScalarType::Double:
            return erosion<double>(input, str_el, footprint, origin_x, origin_y, border_type, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity());
        default:
            printf("Scalar type not supported.\n");
            return input;
    }
}

// Dilation
torch::Tensor dilation_dispatcher(
        torch::Tensor input,
        torch::Tensor str_el,
        torch::Tensor footprint,
        int origin_x,
        int origin_y
) {

    switch (input.scalar_type()) {
        case torch::ScalarType::Byte:
            return dilation<uint8_t>(input, str_el, footprint, origin_x, origin_y, std::numeric_limits<uint8_t>::min());
        case torch::ScalarType::Char:
            return dilation<int8_t>(input, str_el, footprint, origin_x, origin_y, std::numeric_limits<int8_t>::min());
        case torch::ScalarType::Short:
            return dilation<int16_t>(input, str_el, footprint, origin_x, origin_y, std::numeric_limits<int16_t>::min());
        case torch::ScalarType::Int:
            return dilation<int>(input, str_el, footprint, origin_x, origin_y, std::numeric_limits<int>::min());
        case torch::ScalarType::Long:
            return dilation<int64_t>(input, str_el, footprint, origin_x, origin_y, std::numeric_limits<int64_t>::min());
        case torch::ScalarType::Half:
            return dilation<at::Half>(input, str_el, footprint, origin_x, origin_y, -std::numeric_limits<at::Half>::infinity());
        case torch::ScalarType::Float:
            return dilation<float>(input, str_el, footprint, origin_x, origin_y, -std::numeric_limits<float>::infinity());
        case torch::ScalarType::Double:
            return dilation<double>(input, str_el, footprint, origin_x, origin_y, -std::numeric_limits<double>::infinity());
        default:
            printf("Scalar type not supported.\n");
            return input;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("erosion", &erosion_dispatcher, "Erosion");
    m.def("dilation", &dilation_dispatcher, "Dilation");
}
