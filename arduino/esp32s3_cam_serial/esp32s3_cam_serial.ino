#include <Arduino.h>
#include "esp_camera.h"

// ===== Board Selection =====
// Uncomment or adapt to your hardware. Defaults to GOOUUU ESP32-S3-CAM.
//#define BOARD_ESP32S3_EYE
//#define BOARD_ESP32S3_CAM_GENERIC
#define BOARD_GOOUUU_ESP32S3_CAM

#if defined(BOARD_ESP32S3_EYE)
// Espressif ESP32-S3-EYE (OV2640)
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   10
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7     11
#define CAM_PIN_D6      9
#define CAM_PIN_D5      8
#define CAM_PIN_D4     14
#define CAM_PIN_D3     21
#define CAM_PIN_D2     47
#define CAM_PIN_D1     48
#define CAM_PIN_D0     12
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK   13
#elif defined(BOARD_ESP32S3_CAM_GENERIC)
// Example generic ESP32-S3-CAM mapping (adjust to your module)
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   40
#define CAM_PIN_SIOD   17
#define CAM_PIN_SIOC   18
#define CAM_PIN_D7     21
#define CAM_PIN_D6     38
#define CAM_PIN_D5     15
#define CAM_PIN_D4     14
#define CAM_PIN_D3     47
#define CAM_PIN_D2     48
#define CAM_PIN_D1     16
#define CAM_PIN_D0     39
#define CAM_PIN_VSYNC  41
#define CAM_PIN_HREF    2
#define CAM_PIN_PCLK   42
#elif defined(BOARD_GOOUUU_ESP32S3_CAM)
// GOOUUU ESP32-S3-CAM (N16R8) + OV2640 mapping
#define CAM_PIN_PWDN   -1
#define CAM_PIN_RESET  -1
#define CAM_PIN_XCLK   15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7     16  // Y9
#define CAM_PIN_D6     17  // Y8
#define CAM_PIN_D5     18  // Y7
#define CAM_PIN_D4     12  // Y6
#define CAM_PIN_D3     10  // Y5
#define CAM_PIN_D2      8  // Y4
#define CAM_PIN_D1      9  // Y3
#define CAM_PIN_D0     11  // Y2
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK   13
#endif

// ===== Stream Protocol (Single-frame on demand) =====
// Host sends a single byte 'S' (0x53) to request one frame.
// Device responds with: 'SJPG' + uint32_le(length) + JPEG bytes
static inline void write_le32(uint32_t v) {
	uint8_t b[4] = { (uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF), (uint8_t)((v >> 16) & 0xFF), (uint8_t)((v >> 24) & 0xFF) };
	Serial.write(b, 4);
}

// Non-blocking chunked write with deadline; returns false if timed out (callers may drop frame)
static bool write_bytes_nb(const uint8_t* data, size_t len, uint32_t max_ms) {
    const uint32_t start = millis();
    size_t sent = 0;
    while (sent < len) {
        int avail = Serial.availableForWrite();
        if (avail > 0) {
            int chunk = (int)min((size_t)avail, len - sent);
            if (chunk > 1024) chunk = 1024;
            int w = Serial.write(data + sent, chunk);
            if (w > 0) sent += (size_t)w;
        }
        if ((millis() - start) > max_ms) return false; // give up to avoid blocking
        delay(0);
    }
    return true;
}

static bool init_camera() {
	camera_config_t cfg = {};
	cfg.ledc_channel = LEDC_CHANNEL_0;
	cfg.ledc_timer = LEDC_TIMER_0;
	cfg.pin_d0 = CAM_PIN_D0;
	cfg.pin_d1 = CAM_PIN_D1;
	cfg.pin_d2 = CAM_PIN_D2;
	cfg.pin_d3 = CAM_PIN_D3;
	cfg.pin_d4 = CAM_PIN_D4;
	cfg.pin_d5 = CAM_PIN_D5;
	cfg.pin_d6 = CAM_PIN_D6;
	cfg.pin_d7 = CAM_PIN_D7;
	cfg.pin_xclk = CAM_PIN_XCLK;
	cfg.pin_pclk = CAM_PIN_PCLK;
	cfg.pin_vsync = CAM_PIN_VSYNC;
	cfg.pin_href = CAM_PIN_HREF;
	cfg.pin_sccb_sda = CAM_PIN_SIOD;
	cfg.pin_sccb_scl = CAM_PIN_SIOC;
	cfg.pin_pwdn = CAM_PIN_PWDN;
	cfg.pin_reset = CAM_PIN_RESET;
	cfg.xclk_freq_hz = 20000000;
	cfg.pixel_format = PIXFORMAT_JPEG;
	// Small frame and higher quality number -> smaller JPEG for serial
	cfg.frame_size = FRAMESIZE_QQVGA; // 160x120
	cfg.jpeg_quality = 5;            // 0(best)-63(worst); larger -> smaller file
	cfg.fb_count = 1;                 // single buffer to always get latest
	cfg.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
	cfg.grab_mode = CAMERA_GRAB_LATEST; // drop older frames

	esp_err_t err = esp_camera_init(&cfg);
	if (err == ESP_OK) return true;
	// Fallback: lower XCLK
	cfg.xclk_freq_hz = 10000000;
	return esp_camera_init(&cfg) == ESP_OK;
}

static bool send_frame_once() {
	camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) return false;
	uint8_t* bytes = fb->buf;
	size_t len = fb->len;

	const uint8_t magic[4] = { 'S', 'J', 'P', 'G' };
    if (!write_bytes_nb(magic, 4, 200)) { esp_camera_fb_return(fb); return false; }
	uint8_t hdr[4] = { (uint8_t)(len & 0xFF), (uint8_t)((len >> 8) & 0xFF), (uint8_t)((len >> 16) & 0xFF), (uint8_t)((len >> 24) & 0xFF) };
    if (!write_bytes_nb(hdr, 4, 200)) { esp_camera_fb_return(fb); return false; }
    // Budget proportional to payload size at 460800bps (~bytes*10/baud seconds)
    uint32_t budget_ms = (uint32_t)((len * 10UL * 1000UL) / 460800UL) + 100;
    if (budget_ms < 300) budget_ms = 300;
    if (!write_bytes_nb(bytes, len, budget_ms)) { esp_camera_fb_return(fb); return false; }

	esp_camera_fb_return(fb);
    return true;
}

void setup() {
    Serial.begin(460800);
	unsigned long t0 = millis();
	while (!Serial && (millis() - t0) < 3000) { delay(10); }
	if (!init_camera()) {
        // avoid contaminating binary stream
		while (true) { delay(1000); }
	}
}

void loop() {
    if (!Serial) { delay(5); return; }
    // Wait for 'S' command from host then send one frame and return to idle
    if (Serial.available() > 0) {
        int c = Serial.read();
        if (c == 'S') {
            (void)send_frame_once();
        } else {
            // flush unexpected bytes to keep stream clean
        }
    }
    delay(1);
}


