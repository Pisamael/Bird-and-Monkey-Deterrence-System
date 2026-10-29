// M.I.D.A.S. embedded turret tracker
// Board : Luckfox Pico (RV1106) + SC3336 camera (opencv-mobile) + RKNN zero-copy API
// Build : link against librknnmrt (RV1106 runtime) and opencv-mobile, C++11 or newer
// Run   : RkLunch-stop.sh   (stops rkipc so it releases the camera/ISP)
//         /oem/turret_vision
//
// UART protocol to the servo controller, one line per frame: "<pan>,<tilt>,<fire>\n"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <csignal>
#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include "rknn_api.h"

// ----------------------------- configuration -----------------------------
#define UART_PORT    "/dev/ttyS3"
#define MODEL_PATH   "/oem/yolov8n.rknn"

// Capture size. 640x480 is known to work with opencv-mobile on Luckfox; the frame is
// letterboxed into the model input (320x320), which keeps the aspect ratio.
#define CAP_W        640
#define CAP_H        480

#define TARGET_CLASS 15        // 15 = cat, 16 = dog (COCO). Use 0 for a single-class custom model.
#define SCORE_THRESH 0.50f

static const float KP_PAN   = 0.05f;
static const float KP_TILT  = 0.05f;
static const float PAN_SIGN  = 1.0f;   // flip to -1.0f if the pan servo moves away from the target
static const float TILT_SIGN = 1.0f;   // flip to -1.0f if the tilt servo moves away from the target
static const float PAN_MIN = 10.0f,  PAN_MAX = 170.0f;
static const float TILT_MIN = 30.0f, TILT_MAX = 150.0f;

static const int LOCK_THRESHOLD     = 25;  // pixels (in model input space) counted as "centered"
static const int FIRE_CONFIRM_FRAMES = 5;  // consecutive centered frames required before fire=1
static const int MAX_CAM_FAILS      = 50;  // consecutive empty frames before giving up

// ------------------------------- globals ---------------------------------
static volatile sig_atomic_t g_run = 1;
static void on_signal(int) { g_run = 0; }

struct Det {
    bool  found;
    float score;
    float cx, cy;   // target centre in model-input pixels
};

struct Model {
    rknn_context ctx;
    unsigned char* data;
    uint32_t n_in, n_out;
    rknn_tensor_attr in_attr;
    std::vector<rknn_tensor_attr> out_attr;
    rknn_tensor_mem* in_mem;
    std::vector<rknn_tensor_mem*> out_mem;
    int in_w, in_h;
    Model() : ctx(0), data(NULL), n_in(0), n_out(0), in_mem(NULL), in_w(0), in_h(0) {
        memset(&in_attr, 0, sizeof(in_attr));
    }
};

// ------------------------------- UART ------------------------------------
static int init_uart(const char* port) {
    int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) return -1;
    fcntl(fd, F_SETFL, 0);                       // back to blocking writes

    struct termios t;
    if (tcgetattr(fd, &t) != 0) { close(fd); return -1; }
    cfmakeraw(&t);                               // no echo, no CR/LF translation, no flow control
    cfsetispeed(&t, B115200);
    cfsetospeed(&t, B115200);
    t.c_cflag |= (CLOCAL | CREAD);
    t.c_cflag &= ~(PARENB | CSTOPB | CSIZE | CRTSCTS);
    t.c_cflag |= CS8;
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    if (tcsetattr(fd, TCSANOW, &t) != 0) { close(fd); return -1; }
    tcflush(fd, TCIOFLUSH);
    return fd;
}

static void send_cmd(int fd, int pan, int tilt, int fire) {
    if (fd < 0) return;
    char cmd[64];
    int n = snprintf(cmd, sizeof(cmd), "%d,%d,%d\n", pan, tilt, fire);
    if (n > 0) {
        ssize_t w = write(fd, cmd, (size_t)n);
        (void)w;
    }
}

// ------------------------------ helpers ----------------------------------
static unsigned char* load_file(const char* path, int* size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (len <= 0) { fclose(fp); return NULL; }
    unsigned char* buf = (unsigned char*)malloc((size_t)len);
    if (!buf) { fclose(fp); return NULL; }
    size_t rd = fread(buf, 1, (size_t)len, fp);
    fclose(fp);
    if (rd != (size_t)len) { free(buf); return NULL; }
    *size = (int)len;
    return buf;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u, ex = (h >> 10) & 0x1fu, mant = h & 0x3ffu;
    float v;
    if (ex == 0)       v = std::ldexp((float)mant, -24);
    else if (ex == 31) v = mant ? NAN : INFINITY;
    else               v = std::ldexp((float)(mant | 0x400u), (int)ex - 25);
    return sign ? -v : v;
}

// Read element idx of a tensor buffer as a real (dequantized) float.
static inline float tensor_get(const rknn_tensor_attr& a, const void* base, size_t idx) {
    switch (a.type) {
    case RKNN_TENSOR_INT8:
        return (float)(((const int8_t*)base)[idx] - a.zp) * a.scale;
    case RKNN_TENSOR_UINT8:
        return (float)(((const uint8_t*)base)[idx] - a.zp) * a.scale;
    case RKNN_TENSOR_FLOAT16:
        return half_to_float(((const uint16_t*)base)[idx]);
    case RKNN_TENSOR_FLOAT32:
        return ((const float*)base)[idx];
    default:
        return 0.0f;
    }
}

static void print_attr(const char* tag, const rknn_tensor_attr& a) {
    printf("[RKNN] %s[%u] dims=[", tag, a.index);
    for (uint32_t i = 0; i < a.n_dims; i++) printf("%s%u", i ? "," : "", a.dims[i]);
    printf("] fmt=%d type=%d qnt=%d zp=%d scale=%f size=%u size_stride=%u w_stride=%u\n",
           (int)a.fmt, (int)a.type, (int)a.qnt_type, (int)a.zp, a.scale,
           a.size, a.size_with_stride, a.w_stride);
}

// ------------------------------ model setup ------------------------------
static void model_release(Model& m) {
    if (m.ctx) {
        if (m.in_mem) rknn_destroy_mem(m.ctx, m.in_mem);
        for (size_t i = 0; i < m.out_mem.size(); i++)
            if (m.out_mem[i]) rknn_destroy_mem(m.ctx, m.out_mem[i]);
        rknn_destroy(m.ctx);
        m.ctx = 0;
    }
    if (m.data) { free(m.data); m.data = NULL; }
    m.in_mem = NULL;
    m.out_mem.clear();
}

static bool model_init(Model& m) {
    int len = 0;
    m.data = load_file(MODEL_PATH, &len);
    if (!m.data) {
        std::cerr << "[ERROR] cannot read model file " << MODEL_PATH << "\n";
        return false;
    }

    // RV1106 runtime wants the model bytes in memory, not a file path.
    int ret = rknn_init(&m.ctx, m.data, (uint32_t)len, 0, NULL);
    if (ret != RKNN_SUCC) {
        std::cerr << "[ERROR] rknn_init failed, ret=" << ret
                  << " (model must be converted for rv1106 with rknn-toolkit2)\n";
        m.ctx = 0;
        free(m.data); m.data = NULL;
        return false;
    }

    rknn_input_output_num io;
    memset(&io, 0, sizeof(io));
    ret = rknn_query(m.ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret != RKNN_SUCC || io.n_input != 1 || io.n_output < 1) {
        std::cerr << "[ERROR] unexpected model io, ret=" << ret
                  << " inputs=" << io.n_input << " outputs=" << io.n_output << "\n";
        return false;
    }
    m.n_in = io.n_input;
    m.n_out = io.n_output;
    printf("[RKNN] inputs=%u outputs=%u\n", m.n_in, m.n_out);

    // ---- input: uint8 NHWC; mean/std normalisation is fused into the model ----
    memset(&m.in_attr, 0, sizeof(m.in_attr));
    m.in_attr.index = 0;
    ret = rknn_query(m.ctx, RKNN_QUERY_INPUT_ATTR, &m.in_attr, sizeof(m.in_attr));
    if (ret != RKNN_SUCC) { std::cerr << "[ERROR] query input attr, ret=" << ret << "\n"; return false; }
    print_attr("in", m.in_attr);
    m.in_attr.type = RKNN_TENSOR_UINT8;
    m.in_attr.fmt  = RKNN_TENSOR_NHWC;
    if (m.in_attr.n_dims != 4 || m.in_attr.dims[3] != 3) {
        std::cerr << "[ERROR] model input is not 1xHxWx3\n";
        return false;
    }
    m.in_h = (int)m.in_attr.dims[1];
    m.in_w = (int)m.in_attr.dims[2];

    m.in_mem = rknn_create_mem(m.ctx, m.in_attr.size_with_stride);
    if (!m.in_mem) { std::cerr << "[ERROR] rknn_create_mem(input)\n"; return false; }
    ret = rknn_set_io_mem(m.ctx, m.in_mem, &m.in_attr);
    if (ret != RKNN_SUCC) { std::cerr << "[ERROR] rknn_set_io_mem(input), ret=" << ret << "\n"; return false; }

    // ---- outputs ----
    m.out_attr.resize(m.n_out);
    m.out_mem.assign(m.n_out, (rknn_tensor_mem*)NULL);
    for (uint32_t i = 0; i < m.n_out; i++) {
        memset(&m.out_attr[i], 0, sizeof(rknn_tensor_attr));
        m.out_attr[i].index = i;
        ret = rknn_query(m.ctx, RKNN_QUERY_OUTPUT_ATTR, &m.out_attr[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) { std::cerr << "[ERROR] query output attr " << i << ", ret=" << ret << "\n"; return false; }
        print_attr("out", m.out_attr[i]);

        m.out_mem[i] = rknn_create_mem(m.ctx, m.out_attr[i].size_with_stride);
        if (!m.out_mem[i]) { std::cerr << "[ERROR] rknn_create_mem(output " << i << ")\n"; return false; }
        ret = rknn_set_io_mem(m.ctx, m.out_mem[i], &m.out_attr[i]);
        if (ret != RKNN_SUCC) { std::cerr << "[ERROR] rknn_set_io_mem(output " << i << "), ret=" << ret << "\n"; return false; }
    }

    if (m.n_out == 1 && m.out_attr[0].type == RKNN_TENSOR_INT8)
        std::cerr << "[WARN] single merged int8 output: box coords and class scores share one "
                     "quantization scale, so accuracy will be poor. Prefer the rknn_model_zoo "
                     "yolov8 export (split head, 6 or 9 outputs).\n";
    return true;
}

// ----------------------------- decoding: single output -------------------
// Output shaped [1, 4+nc, N] (or transposed [1, N, 4+nc]): rows 0..3 = cx,cy,w,h in model pixels,
// rows 4.. = per-class scores (already sigmoid-ed by the ultralytics export).
static Det detect_single(const Model& m) {
    Det best; best.found = false; best.score = 0; best.cx = 0; best.cy = 0;
    const rknn_tensor_attr& a = m.out_attr[0];
    const void* buf = m.out_mem[0]->virt_addr;

    std::vector<int> d;
    for (uint32_t i = 1; i < a.n_dims; i++) if (a.dims[i] != 1) d.push_back((int)a.dims[i]);
    if (d.size() != 2) {
        static bool warned = false;
        if (!warned) { std::cerr << "[ERROR] cannot interpret single-output shape\n"; warned = true; }
        return best;
    }
    bool chan_first = d[0] < d[1];
    int C = chan_first ? d[0] : d[1];
    int N = chan_first ? d[1] : d[0];
    int nc = C - 4;
    if (TARGET_CLASS < 0 || TARGET_CLASS >= nc) {
        static bool warned = false;
        if (!warned) { std::cerr << "[ERROR] TARGET_CLASS " << TARGET_CLASS << " >= model classes " << nc << "\n"; warned = true; }
        return best;
    }

    #define AT(c, i) tensor_get(a, buf, chan_first ? (size_t)(c) * N + (i) : (size_t)(i) * C + (c))
    float max_score = SCORE_THRESH;
    int best_i = -1;
    for (int i = 0; i < N; i++) {
        float s = AT(4 + TARGET_CLASS, i);
        if (s > max_score) { max_score = s; best_i = i; }
    }
    if (best_i >= 0) {
        best.found = true;
        best.score = max_score;
        best.cx = AT(0, best_i);
        best.cy = AT(1, best_i);
    }
    #undef AT
    return best;
}

// ----------------------------- decoding: split head ----------------------
// rknn_model_zoo yolov8 export: per scale a box tensor (64 ch = 4 sides x 16 DFL bins),
// a class-score tensor (nc ch, sigmoid-ed) and, for 9-output models, a score-sum tensor (1 ch).
struct Plane {
    const rknn_tensor_attr* a;
    const void* base;
    int C, H, W, ws;
    bool nhwc;
    float at(int c, int y, int x) const {
        size_t idx = nhwc ? ((size_t)y * ws + x) * C + c
                          : ((size_t)c * H + y) * ws + x;
        return tensor_get(*a, base, idx);
    }
};

static bool make_plane(const rknn_tensor_attr& a, const void* base, Plane& p) {
    if (a.n_dims != 4) return false;
    p.a = &a;
    p.base = base;
    p.nhwc = (a.fmt == RKNN_TENSOR_NHWC);
    if (p.nhwc) { p.H = (int)a.dims[1]; p.W = (int)a.dims[2]; p.C = (int)a.dims[3]; }
    else        { p.C = (int)a.dims[1]; p.H = (int)a.dims[2]; p.W = (int)a.dims[3]; }
    p.ws = a.w_stride ? (int)a.w_stride : p.W;
    return true;
}

static Det detect_split(const Model& m) {
    Det best; best.found = false; best.score = 0; best.cx = 0; best.cy = 0;

    std::vector<Plane> boxes, clss;
    int nc = 0;
    for (uint32_t i = 0; i < m.n_out; i++) {
        Plane p;
        if (!make_plane(m.out_attr[i], m.out_mem[i]->virt_addr, p)) continue;
        if (p.C == 64)      boxes.push_back(p);
        else if (p.C > 1) { clss.push_back(p); nc = p.C; }   // C == 1 is the score-sum tensor: skipped
    }

    // pair each class tensor with the box tensor of the same grid size
    std::vector<Plane> pb;
    for (size_t l = 0; l < clss.size(); l++) {
        bool ok = false;
        for (size_t b = 0; b < boxes.size(); b++)
            if (boxes[b].H == clss[l].H && boxes[b].W == clss[l].W) { pb.push_back(boxes[b]); ok = true; break; }
        if (!ok) break;
    }
    if (clss.empty() || pb.size() != clss.size()) {
        static bool warned = false;
        if (!warned) { std::cerr << "[ERROR] unrecognised split-head output layout; post the [RKNN] out[...] lines\n"; warned = true; }
        return best;
    }
    if (TARGET_CLASS < 0 || TARGET_CLASS >= nc) {
        static bool warned = false;
        if (!warned) { std::cerr << "[ERROR] TARGET_CLASS " << TARGET_CLASS << " >= model classes " << nc << "\n"; warned = true; }
        return best;
    }

    float max_score = SCORE_THRESH;
    int bl = -1, by = 0, bx = 0;
    for (size_t l = 0; l < clss.size(); l++)
        for (int y = 0; y < clss[l].H; y++)
            for (int x = 0; x < clss[l].W; x++) {
                float s = clss[l].at(TARGET_CLASS, y, x);
                if (s > max_score) { max_score = s; bl = (int)l; by = y; bx = x; }
            }
    if (bl < 0) return best;

    // DFL decode of the winning cell only (left, top, right, bottom distances in grid units)
    const Plane& b = pb[bl];
    int dfl = b.C / 4;
    float stride = (float)m.in_w / (float)b.W;
    float dist[4];
    std::vector<float> v((size_t)dfl);
    for (int s = 0; s < 4; s++) {
        float mx = -1e30f;
        for (int k = 0; k < dfl; k++) { v[k] = b.at(s * dfl + k, by, bx); mx = std::max(mx, v[k]); }
        float sum = 0.f, acc = 0.f;
        for (int k = 0; k < dfl; k++) { float e = std::exp(v[k] - mx); sum += e; acc += e * (float)k; }
        dist[s] = acc / sum;
    }
    best.found = true;
    best.score = max_score;
    best.cx = ((float)bx + 0.5f) * stride + (dist[2] - dist[0]) * stride * 0.5f;
    best.cy = ((float)by + 0.5f) * stride + (dist[3] - dist[1]) * stride * 0.5f;
    return best;
}

// -------------------------------- main -----------------------------------
int main() {
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    int uart_fd = init_uart(UART_PORT);
    if (uart_fd < 0) std::cerr << "[WARN] UART " << UART_PORT << " not available (enable UART3 in luckfox-config); commands will not be sent\n";

    Model model;
    if (!model_init(model)) {
        model_release(model);
        if (uart_fd >= 0) close(uart_fd);
        return -1;
    }

    // Camera: properties must be set BEFORE open() on opencv-mobile.
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, CAP_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAP_H);
    cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Failed to open camera. Run RkLunch-stop.sh first and check that rkipc is not running.\n";
        model_release(model);
        if (uart_fd >= 0) close(uart_fd);
        return -1;
    }

    float pan_angle = 90.0f, tilt_angle = 90.0f;
    const float ctr_x = model.in_w * 0.5f;
    const float ctr_y = model.in_h * 0.5f;
    const int in_stride_px = model.in_attr.w_stride ? (int)model.in_attr.w_stride : model.in_w;

    cv::Mat frame, resized;
    cv::Mat canvas(model.in_h, model.in_w, CV_8UC3);

    int cam_fails = 0, lock_frames = 0, frame_count = 0;
    bool had_target = false;
    auto t0 = std::chrono::steady_clock::now();

    std::cout << "[M.I.D.A.S.] Embedded C++ Turret Online. Tracking class " << TARGET_CLASS << "...\n";
    send_cmd(uart_fd, (int)pan_angle, (int)tilt_angle, 0);

    while (g_run) {
        cap >> frame;
        if (frame.empty()) {
            if (++cam_fails >= MAX_CAM_FAILS) {
                std::cerr << "[ERROR] camera produced no frames (" << cam_fails
                          << " in a row). Is rkipc still running? Try RkLunch-stop.sh\n";
                break;
            }
            usleep(10000);
            continue;
        }
        cam_fails = 0;

        // letterbox into the model input, then BGR -> RGB
        float scale = std::min((float)model.in_w / frame.cols, (float)model.in_h / frame.rows);
        int nw = std::max(1, (int)std::lround(frame.cols * scale));
        int nh = std::max(1, (int)std::lround(frame.rows * scale));
        cv::resize(frame, resized, cv::Size(nw, nh));
        canvas.setTo(cv::Scalar(114, 114, 114));
        resized.copyTo(canvas(cv::Rect((model.in_w - nw) / 2, (model.in_h - nh) / 2, nw, nh)));
        cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

        // copy into NPU input memory, honouring row stride
        uint8_t* dst = (uint8_t*)model.in_mem->virt_addr;
        if (in_stride_px == model.in_w) {
            memcpy(dst, canvas.data, (size_t)model.in_w * model.in_h * 3);
        } else {
            for (int y = 0; y < model.in_h; y++)
                memcpy(dst + (size_t)y * in_stride_px * 3, canvas.data + (size_t)y * model.in_w * 3, (size_t)model.in_w * 3);
        }

        int ret = rknn_run(model.ctx, NULL);
        if (ret != RKNN_SUCC) {
            std::cerr << "[ERROR] rknn_run failed, ret=" << ret << "\n";
            break;
        }

        Det det = (model.n_out == 1) ? detect_single(model) : detect_split(model);

        int fire = 0;
        if (det.found) {
            float error_x = PAN_SIGN  * (ctr_x - det.cx);
            float error_y = TILT_SIGN * (det.cy - ctr_y);

            if (std::fabs(error_x) < LOCK_THRESHOLD && std::fabs(error_y) < LOCK_THRESHOLD) {
                if (++lock_frames >= FIRE_CONFIRM_FRAMES) fire = 1;
            } else {
                lock_frames = 0;
                pan_angle  += error_x * KP_PAN;
                tilt_angle += error_y * KP_TILT;
                pan_angle  = std::max(PAN_MIN,  std::min(PAN_MAX,  pan_angle));
                tilt_angle = std::max(TILT_MIN, std::min(TILT_MAX, tilt_angle));
            }
            if (!had_target)
                printf("[!] TARGET ACQUIRED - score %.2f at (%.0f,%.0f)\n", det.score, det.cx, det.cy);
        } else {
            lock_frames = 0;
            if (had_target) printf("[-] target lost\n");
        }
        had_target = det.found;

        send_cmd(uart_fd, (int)pan_angle, (int)tilt_angle, fire);

        if (++frame_count % 60 == 0) {
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("[stat] %.1f fps  pan=%d tilt=%d fire=%d\n", 60.0 / sec, (int)pan_angle, (int)tilt_angle, fire);
            t0 = std::chrono::steady_clock::now();
        }
    }

    // safe shutdown: stop firing, keep last position
    send_cmd(uart_fd, (int)pan_angle, (int)tilt_angle, 0);
    cap.release();
    model_release(model);
    if (uart_fd >= 0) close(uart_fd);
    return 0;
}
