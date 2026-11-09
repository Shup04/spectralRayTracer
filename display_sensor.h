#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "core_types.h"

// We use a standard struct for the final image buffer before saving.
struct Pixel {
    uint8_t r, g, b;
};

// Approximate gamma-2 display encoding, not a calibrated sensor response.
inline uint8_t display_channel(float linear) {
    if (!std::isfinite(linear)) throw std::runtime_error("Non-finite display channel");
    return static_cast<uint8_t>(std::sqrt(std::clamp(linear, 0.0f, 1.0f)) * 255.0f);
}

// Phase 4: The File Writer
inline void write_ppm_image(const std::string& filename, int width, int height, const std::vector<Pixel>& framebuffer) {
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(filename, std::ios::binary);
    
    // The PPM Header: "P3" means text-based RGB. 
    file << "P3\n" << width << " " << height << "\n255\n";

    // Write the raw pixel array
    for (int i = 0; i < width * height; ++i) {
        file << (int)framebuffer[i].r << " " 
             << (int)framebuffer[i].g << " " 
             << (int)framebuffer[i].b << "\n";
    }
    file.close();
}
