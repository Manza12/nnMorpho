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
            switch (border_type) {
                case 'e':
                    return erosion<uint8_t>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
                case 'g':
                    return erosion<uint8_t>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::max());
                default:
                    printf("Border type not understood.\n");
                    return input;
            }
        case torch::ScalarType::Char:
            switch (border_type) {
                case 'e':
                    return erosion<int8_t>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
                case 'g':
                    return erosion<int8_t>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max());
                default:
                    printf("Border type not understood.\n");
                    return input;
            }
        case torch::ScalarType::Short:
            switch (border_type) {
                case 'e':
                    return erosion<int16_t>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max());
                case 'g':
                    return erosion<int16_t>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::max());
                default:
                    printf("Border type not understood.\n");
                    return input;
            }
        case torch::ScalarType::Int:
            switch (border_type) {
                case 'e':
                    return erosion<int>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
                case 'g':
                    return erosion<int>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
                default:
                    printf("Border type not understood.\n");
                    return input;
            }
        case torch::ScalarType::Long:
            switch (border_type) {
                case 'e':
                    return erosion<int64_t>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
                case 'g':
                    return erosion<int64_t>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max());
                default:
                    printf("Border type not understood.\n");
                    return input;
            }
        case torch::ScalarType::Half:
            switch (border_type) {
                case 'e':
                    return erosion<at::Half>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<at::Half>::min(), std::numeric_limits<at::Half>::max());
                case 'g':
                    return erosion<at::Half>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<at::Half>::max(), std::numeric_limits<at::Half>::max());
                default:
                    printf("Border type not understood.\n");
                    return input;
            }
        case torch::ScalarType::Float:
            switch (border_type) {
                case 'e':
                    return erosion<float>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
                case 'g':
                    return erosion<float>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
                default:
                    printf("Border type not understood.\n");
                    return input;
            }
        case torch::ScalarType::Double:
            switch (border_type) {
                case 'e':
                    return erosion<double>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
                case 'g':
                    return erosion<double>(input, str_el, footprint, origin_x, origin_y,
                                            std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
                default:
                    printf("Border type not understood.\n");
                    return input;
            }
        default:
            printf("Scalar type not supported.\n");
            return input;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("erosion", &erosion_dispatcher, "Erosion");
}
