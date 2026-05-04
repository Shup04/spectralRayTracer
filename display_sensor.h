#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "core_types.h"

// We use a standard struct for the final image buffer before saving.
struct Pixel {
    uint8_t r, g, b;
};

// Maps a normal float (-1.0 to 1.0) to an 8-bit color (0 to 255)
inline uint8_t normal_to_color(float n) {
    return static_cast<uint8_t>((n + 1.0f) * 0.5f * 255.99f);
}

// Phase 4: The File Writer
inline void write_ppm_image(const std::string& filename, int width, int height, const std::vector<Pixel>& framebuffer) {
    std::ofstream file(filename, std::ios::binary);
    
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
