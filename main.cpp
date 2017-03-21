/* Written in 2017 by Andrew Dawson

To the extent possible under law, the author(s) have dedicated all copyright and
related and neighboring rights to this software to the public domain worldwide.
This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication along with
this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

Build Instructions..............................................................

For playing, use these commands:

-for Windows, using the visual studio compiler
cl /O2 main.cpp kernel32.lib user32.lib gdi32.lib opengl32.lib /link /out:One.exe

-for Linux
g++ -o One -std=c++0x -O3 main.cpp -lGL -lX11

For debugging, use these commands:

-for Windows, using the visual studio compiler
cl /Od /Wall main.cpp kernel32.lib user32.lib gdi32.lib opengl32.lib /link /debug /out:One.exe

-for Linux
g++ -o One -std=c++0x -O0 -g3 -Wall -fmessage-length=0 main.cpp -lGL -lX11
*/

#if defined(__linux__)
#define OS_LINUX
#include <GL/glx.h>
#elif defined(_WIN32)
#include <Windows.h>
#include <GL/gl.h>
#define OS_WINDOWS
#else
#error Failed to figure out what operating system this is.
#endif

#include <cstdarg>
#include <cstdio>
#include <cstdint>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

// Clock Function Declarations..................................................

struct Clock
{
    double frequency;
};

void initialise_clock(Clock* clock);
double get_time(Clock* clock);
void go_to_sleep(Clock* clock, double amount_to_sleep);

// Logging Function Declarations................................................

enum class LogLevel
{
    Debug,
    Error,
};

void log_add_message(LogLevel level, const char* format, ...);

#define LOG_ERROR(format, ...) \
    log_add_message(LogLevel::Error, format, ##__VA_ARGS__)

#ifdef NDEBUG
#define LOG_DEBUG(format, ...) // do nothing
#else
#define LOG_DEBUG(format, ...) \
    log_add_message(LogLevel::Debug, format, ##__VA_ARGS__)
#endif

// Random Number Generation.....................................................

namespace arandom {

/*  Written in 2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

u64 x;

static uint64_t splitmix64()
{
    uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/*  Written in 2016 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

u64 s[2];

static inline u64 rotl(const u64 x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static u64 xoroshiro128plus()
{
    const u64 s0 = s[0];
    u64 s1 = s[1];
    const u64 result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
    s[1] = rotl(s1, 36); // c

    return result;
}

// End of Blackman & Vigna's code

u64 seed(u64 value)
{
    u64 old_seed = x;
    x = value;
    s[0] = splitmix64();
    s[1] = splitmix64();
    return old_seed;
}

int int_range(int min, int max)
{
    int x = xoroshiro128plus() % static_cast<u64>(max - min + 1);
    return min + x;
}

static inline float to_float(u64 x)
{
    union
    {
        u32 i;
        float f;
    } u;
    u.i = UINT32_C(0x3F8) << 23 | x >> 41;
    return u.f - 1.0f;
}

float float_range(float min, float max)
{
    float f = to_float(xoroshiro128plus());
    return min + f * (max - min);
}

} // namespace arandom

// Main Functions...............................................................

namespace
{
    const char* window_title = "ONE";
    const int window_width = 800;
    const int window_height = 600;
    const double frame_frequency = 1.0 / 60.0;
    const int key_count = 5;

    u64 window_pixels[window_height][window_width];
    bool keys_pressed[key_count];
    bool old_keys_pressed[key_count];
    // This counts the frames since the last time the key state changed.
    int edge_counts[key_count];

    float x, y;
}

static bool key_tapped(int which)
{
    return keys_pressed[which] && edge_counts[which] == 0;
}

static void main_update()
{
    // Update input states.
    for(int i = 0; i < key_count; ++i)
    {
        if(keys_pressed[i] != old_keys_pressed[i])
        {
            edge_counts[i] = 0;
            old_keys_pressed[i] = keys_pressed[i];
        }
        else
        {
            edge_counts[i] += 1;
        }
    }

    if(key_tapped(1))
    {
        x -= 0.05f;
    }
    if(key_tapped(3))
    {
        x += 0.05f;
    }
    if(key_tapped(2))
    {
        y += 0.05f;
    }
    if(key_tapped(4))
    {
        y -= 0.05f;
    }

    // Draw everything.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_TRIANGLES);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex2f(x,        y       );
    glVertex2f(x + 0.4f, y       );
    glVertex2f(x,        y + 0.4f);
    glVertex2f(x + 0.4f, y       );
    glVertex2f(x + 0.4f, y + 0.4f);
    glVertex2f(x,        y + 0.4f);
    glEnd();
}

// Platform-Specific Implementations============================================

#if defined(OS_LINUX)

// Logging Functions............................................................

void log_add_message(LogLevel level, const char* format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    FILE* stream;
    switch(level)
    {
        case LogLevel::Error:
            stream = stderr;
            break;
        case LogLevel::Debug:
            stream = stdout;
            break;
    }
    vfprintf(stream, format, arguments);
    va_end(arguments);
    fputc('\n', stream);
}

// Clock Functions..............................................................

#include <ctime>

void initialise_clock(Clock* clock)
{
    struct timespec resolution;
    clock_getres(CLOCK_MONOTONIC, &resolution);
    s64 nanoseconds = resolution.tv_nsec + resolution.tv_sec * 1e9;
    clock->frequency = static_cast<double>(nanoseconds) / 1.0e9;
}

double get_time(Clock* clock)
{
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    s64 nanoseconds = timestamp.tv_nsec + timestamp.tv_sec * 1e9;
    return static_cast<double>(nanoseconds) * clock->frequency;
}

void go_to_sleep(Clock* clock, double amount_to_sleep)
{
    struct timespec requested_time;
    requested_time.tv_sec = 0;
    requested_time.tv_nsec = static_cast<s64>(1.0e9 * amount_to_sleep);
    clock_nanosleep(CLOCK_MONOTONIC, 0, &requested_time, nullptr);
}

// Platform Main Functions......................................................

#include <X11/X.h>
#include <X11/Xlib.h>

#include <cstdlib>

namespace
{
    Display* display;
    XVisualInfo* visual_info;
    Colormap colormap;
    Window window;
    Atom wm_delete_window;
    GLXContext rendering_context;
}

static bool main_create()
{
    // Connect to the X server, which is used for display and input services.
    display = XOpenDisplay(nullptr);
    if(!display)
    {
        LOG_ERROR("X Display failed to open.");
        return false;
    }

    // Choose the abstract "Visual" type that will be used to describe both the
    // window and the OpenGL rendering context.
    GLint visual_attributes[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24,
        GLX_DOUBLEBUFFER, None };
    visual_info = glXChooseVisual(
        display, DefaultScreen(display), visual_attributes);
    if(!visual_info)
    {
        LOG_ERROR("Wasn't able to choose an appropriate Visual type given the "
            "requested attributes. [The Visual type contains information on "
            "color mappings for the display hardware]");
        return false;
    }

    // Create the Window.
    colormap = XCreateColormap(
        display, DefaultRootWindow(display), visual_info->visual, AllocNone);
    XSetWindowAttributes window_attributes = { };
    window_attributes.colormap = colormap;
    int screen = DefaultScreen(display);
    Window root_window = RootWindow(display, screen);
    window = XCreateWindow(
        display, root_window, 0, 0, window_width, window_height, 0,
        visual_info->depth, InputOutput, visual_info->visual, CWColormap,
        &window_attributes);

    // Register to receive window close messages.
    wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wm_delete_window, 1);

    XStoreName(display, window, window_title);
    XSetIconName(display, window, window_title);

    // Create the rendering context for OpenGL. The rendering context can only
    // be "made current" after the window is mapped (with XMapWindow).
    rendering_context = glXCreateContext(display, visual_info, nullptr, True);
    if(!rendering_context)
    {
        LOG_ERROR("Couldn't create a GLX rendering context.");
        return false;
    }

    XMapWindow(display, window);

    Bool made_current = glXMakeCurrent(display, window, rendering_context);
    if(!made_current)
    {
        LOG_ERROR("Failed to attach the GLX context to the window.");
        return false;
    }

    return true;
}

static void main_destroy()
{
    if(visual_info)
    {
        XFree(visual_info);
    }
    if(display)
    {
        if(rendering_context)
        {
            glXDestroyContext(display, rendering_context);
        }
        if(colormap != None)
        {
            XFreeColormap(display, colormap);
        }
        XCloseDisplay(display);
    }
}

static void main_loop()
{
    arandom::seed(time(nullptr));
    Clock frame_clock;
    initialise_clock(&frame_clock);
    for(;;)
    {
        // Record when the frame begins.
        double frame_start_time = get_time(&frame_clock);

        main_update();
        glXSwapBuffers(display, window);

        // Handle window events.
        while(XPending(display) > 0)
        {
            XEvent event = { };
            XNextEvent(display, &event);
            switch(event.type)
            {
                case ClientMessage:
                {
                    XClientMessageEvent client_message = event.xclient;
                    if(client_message.data.l[0] == wm_delete_window)
                    {
                        XDestroyWindow(display, window);
                        return;
                    }
                    break;
                }
            }
        }

        // Get key states for input.
        char keys[32];
        XQueryKeymap(display, keys);
        int keysyms[key_count] = { XK_space, XK_Left, XK_Up, XK_Right,
            XK_Down };
        for(int i = 0; i < key_count; ++i)
        {
            int code = XKeysymToKeycode(display, keysyms[i]);
            keys_pressed[i] = keys[code / 8] & (1 << (code % 8));
        }

        // Sleep off any remaining time until the next frame.
        double frame_thusfar = get_time(&frame_clock) - frame_start_time;
        if(frame_thusfar < frame_frequency)
        {
            go_to_sleep(&frame_clock, frame_frequency - frame_thusfar);
        }
    }
}

int main(int argc, char** argv)
{
    static_cast<void>(argc);
    static_cast<void>(argv);

    if(!main_create())
    {
        main_destroy();
        return EXIT_FAILURE;
    }
    main_loop();
    main_destroy();
    return 0;
}

#elif defined(OS_WINDOWS)

// Logging Functions............................................................

void log_add_message(LogLevel level, const char* format, ...)
{
    va_list arguments;
    va_start(arguments, format);

    const int buffer_capacity = 128;
    char buffer[buffer_capacity];
    int written = vsnprintf(buffer, buffer_capacity - 1, format, arguments);
    va_end(arguments);

    if(written < 0)
    {
        written = buffer_capacity - 2;
    }
    if(written < buffer_capacity - 1)
    {
        buffer[written] = '\n';
        buffer[written + 1] = '\0';
        written += 2;
    }

    OutputDebugStringA(buffer);
}

// Clock Functions..............................................................

void initialise_clock(Clock* clock)
{
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    clock->frequency = 1.0 / frequency.QuadPart;
}

double get_time(Clock* clock)
{
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return count.QuadPart * clock->frequency;
}

void go_to_sleep(Clock* clock, double amount_to_sleep)
{
    int milliseconds = 1000 * (amount_to_sleep * clock->frequency);
    Sleep(milliseconds);
}

// Platform Main Functions......................................................

namespace
{
    HWND window;
    HDC device_context;
    HGLRC rendering_context;
}

LRESULT CALLBACK WindowProc(
    HWND hwnd, UINT message, WPARAM w_param, LPARAM l_param)
{
    switch(message)
    {
        case WM_CLOSE:
        {
            PostQuitMessage(0);
            return 0;
        }
        case WM_DESTROY:
        {
            HGLRC rc = wglGetCurrentContext();
            if(rc)
            {
                HDC dc = wglGetCurrentDC();
                wglMakeCurrent(nullptr, nullptr);
                ReleaseDC(hwnd, dc);
                wglDeleteContext(rc);
            }
            DestroyWindow(hwnd);
            if(hwnd == window)
            {
                window = nullptr;
            }
            return 0;
        }
        default:
        {
            return DefWindowProc(hwnd, message, w_param, l_param);
        }
    }
}

static bool main_create(HINSTANCE instance, int show_command)
{
    WNDCLASSEXA window_class = {};
    window_class.cbSize = sizeof window_class;
    window_class.style = CS_HREDRAW | CS_VREDRAW;
    window_class.lpfnWndProc = WindowProc;
    window_class.hInstance = instance;
    window_class.hIcon = LoadIcon(instance, IDI_APPLICATION);
    window_class.hIconSm = static_cast<HICON>(
        LoadIcon(instance, IDI_APPLICATION));
    window_class.hCursor = LoadCursor(nullptr, IDC_ARROW);
    window_class.lpszClassName = "OneWindowClass";
    ATOM registered_class = RegisterClassExA(&window_class);
    if(registered_class == 0)
    {
        LOG_ERROR("Failed to register the window class.");
        return false;
    }

    DWORD window_style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU
        | WS_MINIMIZEBOX;
    window = CreateWindowExA(
        WS_EX_APPWINDOW, MAKEINTATOM(registered_class), window_title,
        window_style, CW_USEDEFAULT, CW_USEDEFAULT, window_width, window_height,
        nullptr, nullptr, instance, nullptr);
    if(!window)
    {
        LOG_ERROR("Failed to create the window.");
        return false;
    }

    device_context = GetDC(window);
    if(!device_context)
    {
        LOG_ERROR("Couldn't obtain the device context.");
        return false;
    }

    PIXELFORMATDESCRIPTOR descriptor = {};
    descriptor.nSize = sizeof descriptor;
    descriptor.nVersion = 1;
    descriptor.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL
        | PFD_DOUBLEBUFFER | PFD_DEPTH_DONTCARE;
    descriptor.iPixelType = PFD_TYPE_RGBA;
    descriptor.cColorBits = 32;
    descriptor.iLayerType = PFD_MAIN_PLANE;
    int format_index = ChoosePixelFormat(device_context, &descriptor);
    if(format_index == 0)
    {
        LOG_ERROR("Failed to set up the pixel format.");
        return false;
    }
    if(SetPixelFormat(device_context, format_index, &descriptor) == FALSE)
    {
        LOG_ERROR("Failed to set up the pixel format.");
        return false;
    }

    rendering_context = wglCreateContext(device_context);
    if(!rendering_context)
    {
        LOG_ERROR("Couldn't create the rendering context.");
        return false;
    }

    ShowWindow(window, show_command);

    // Set it to be this thread's rendering context.
    if(wglMakeCurrent(device_context, rendering_context) == FALSE)
    {
        LOG_ERROR("Couldn't set this thread's rendering context "
            "(wglMakeCurrent failed).");
        return false;
    }

    return true;
}

static void main_destroy()
{
    if(rendering_context)
    {
        wglMakeCurrent(nullptr, nullptr);
        ReleaseDC(window, device_context);
        wglDeleteContext(rendering_context);
    }
    else if(device_context)
    {
        ReleaseDC(window, device_context);
    }
    if(window)
    {
        DestroyWindow(window);
    }
}

static int main_loop()
{
    Clock frame_clock = {};
    initialise_clock(&frame_clock);
    MSG msg = {};
    for(;;)
    {
        // Record when the frame begins.
        double frame_start_time = get_time(&frame_clock);

        main_update();
        SwapBuffers(device_context);

        // Handle all window messages queued up during the past frame.
        while(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if(msg.message == WM_QUIT)
            {
                return msg.wParam;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // Get key states for input.
        const int virtual_keys[key_count] = { VK_SPACE, VK_LEFT, VK_UP,
            VK_RIGHT, VK_DOWN };
        for(int i = 0; i < key_count; ++i)
        {
            keys_pressed[i] = 0x8000 & GetKeyState(virtual_keys[i]);
        }

        // Sleep off any remaining time until the next frame.
        double frame_thusfar = get_time(&frame_clock) - frame_start_time;
        if(frame_thusfar < frame_frequency)
        {
            go_to_sleep(&frame_clock, frame_frequency - frame_thusfar);
        }
    }
    return 0;
}

int CALLBACK WinMain(
    HINSTANCE instance, HINSTANCE previous_instance, LPSTR command_line,
    int show_command)
{
    static_cast<void>(previous_instance);
    static_cast<void>(command_line);

    bool created = main_create(instance, show_command);
    if(!created)
    {
        main_destroy();
        return 0;
    }
    return main_loop();
}

#endif // defined(OS_WINDOWS)
