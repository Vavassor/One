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
#include <cassert>
#include <cstdlib>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

#define ASSERT(expression) assert(expression)

#define ALLOCATE(type, count) static_cast<type*>(malloc(sizeof(type) * (count)))
#define DEALLOCATE(memory) free(memory)
#define SAFE_DEALLOCATE(memory) \
    if(memory) { DEALLOCATE(memory); (memory) = nullptr; }

static int string_size(const char* string)
{
    ASSERT(string);
    const char* s;
    for(s = string; *s; ++s);
    return s - string;
}

#define TAU  6.28318530717958647692f
#define PI   3.14159265358979323846f
#define PI_2 1.57079632679489661923f

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

// Vector Functions.............................................................

#include <cmath>

using std::abs;
using std::sqrt;
using std::isfinite;

struct Vector3
{
    float x, y, z;
};

const Vector3 vector3_zero   = { 0.0f, 0.0f, 0.0f };
const Vector3 vector3_unit_x = { 1.0f, 0.0f, 0.0f };
const Vector3 vector3_unit_y = { 0.0f, 1.0f, 0.0f };
const Vector3 vector3_unit_z = { 0.0f, 0.0f, 1.0f };

Vector3 operator + (Vector3 v0, Vector3 v1)
{
    Vector3 result;
    result.x = v0.x + v1.x;
    result.y = v0.y + v1.y;
    result.z = v0.z + v1.z;
    return result;
}

Vector3& operator += (Vector3& v0, Vector3 v1)
{
    v0.x += v1.x;
    v0.y += v1.y;
    v0.z += v1.z;
    return v0;
}

Vector3 operator - (Vector3 v0, Vector3 v1)
{
    Vector3 result;
    result.x = v0.x - v1.x;
    result.y = v0.y - v1.y;
    result.z = v0.z - v1.z;
    return result;
}

Vector3& operator -= (Vector3& v0, Vector3 v1)
{
    v0.x -= v1.x;
    v0.y -= v1.y;
    v0.z -= v1.z;
    return v0;
}

Vector3 operator * (Vector3 v, float s)
{
    Vector3 result;
    result.x = v.x * s;
    result.y = v.y * s;
    result.z = v.z * s;
    return result;
}

Vector3 operator * (float s, Vector3 v)
{
    Vector3 result;
    result.x = s * v.x;
    result.y = s * v.y;
    result.z = s * v.z;
    return result;
}

Vector3& operator *= (Vector3& v, float s)
{
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}

Vector3 operator / (Vector3 v, float s)
{
    Vector3 result;
    result.x = v.x / s;
    result.y = v.y / s;
    result.z = v.z / s;
    return result;
}

Vector3& operator /= (Vector3& v, float s)
{
    v.x /= s;
    v.y /= s;
    v.z /= s;
    return v;
}

Vector3 operator - (Vector3 v)
{
    return { -v.x, -v.y, -v.z };
}

float squared_length(Vector3 v)
{
    return (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
}

float length(Vector3 v)
{
    return sqrt(squared_length(v));
}

Vector3 normalize(Vector3 v)
{
    float l = length(v);
    ASSERT(l != 0.0f && isfinite(l));
    return v / l;
}

// inner product
float dot(Vector3 v0, Vector3 v1)
{
    return (v0.x * v1.x) + (v0.y * v1.y) + (v0.z * v1.z);
}

Vector3 cross(Vector3 v0, Vector3 v1)
{
    Vector3 result;
    result.x = (v0.y * v1.z) - (v0.z * v1.y);
    result.y = (v0.z * v1.x) - (v0.x * v1.z);
    result.z = (v0.x * v1.y) - (v0.y * v1.x);
    return result;
}

// isotropic scale is just operator *
Vector3 anisotropic_scale(Vector3 v0, Vector3 v1)
{
    Vector3 result;
    result.x = v0.x * v1.x;
    result.y = v0.y * v1.y;
    result.z = v0.z * v1.z;
    return result;
}

// Quaternion Functions.........................................................

static bool float_almost_one(float x)
{
    return abs(x - 1.0f) <= 0.0000005f;
}

struct Quaternion
{
    float w, x, y, z;
};

const Quaternion quaternion_identity = { 1.0f, 0.0f, 0.0f, 0.0f };

float norm(Quaternion q)
{
    return sqrt((q.w * q.w) + (q.x * q.x) + (q.y * q.y) + (q.z * q.z));
}

Quaternion axis_angle_rotation(Vector3 axis, float angle)
{
    ASSERT(isfinite(angle));

    angle /= 2.0f;
    float phase = sin(angle);
    Vector3 v = normalize(axis);

    Quaternion result;
    result.w = cos(angle);
    result.x = v.x * phase;
    result.y = v.y * phase;
    result.z = v.z * phase;
    // Rotations must be unit quaternions.
    ASSERT(float_almost_one(norm(result)));
    return result;
}

// Matrix Functions.............................................................

struct Matrix4
{
    float elements[16]; // in row-major order

    float& operator [] (int index) { return elements[index]; }
    const float& operator [] (int index) const { return elements[index]; }
};

const Matrix4 matrix4_identity =
{{
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
}};

Matrix4 operator * (const Matrix4& a, const Matrix4& b)
{
    Matrix4 result;
    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            float m = 0.0f;
            for(int w = 0; w < 4; ++w)
            {
                 m += a[4 * i + w] * b[4 * w + j];
            }
            result[4 * i + j] = m;
        }
    }
    return result;
}

Matrix4 transpose(const Matrix4& m)
{
    return
    {{
        m[0], m[4], m[8],  m[12],
        m[1], m[5], m[9],  m[13],
        m[2], m[6], m[10], m[14],
        m[3], m[7], m[11], m[15]
    }};
}

Matrix4 view_matrix(
    Vector3 x_axis, Vector3 y_axis, Vector3 z_axis, Vector3 position)
{
    Matrix4 result;

    result[0]  = x_axis.x;
    result[1]  = x_axis.y;
    result[2]  = x_axis.z;
    result[3]  = -dot(x_axis, position);

    result[4]  = y_axis.x;
    result[5]  = y_axis.y;
    result[6]  = y_axis.z;
    result[7]  = -dot(y_axis, position);

    result[8]  = z_axis.x;
    result[9]  = z_axis.y;
    result[10] = z_axis.z;
    result[11] = -dot(z_axis, position);

    result[12] = 0.0f;
    result[13] = 0.0f;
    result[14] = 0.0f;
    result[15] = 1.0f;

    return result;
}

// This function assumes a right-handed coordinate system.
Matrix4 look_at_matrix(Vector3 position, Vector3 target, Vector3 world_up)
{
    Vector3 forward = normalize(position - target);
    Vector3 right = normalize(cross(world_up, forward));
    Vector3 up = normalize(cross(forward, right));
    return view_matrix(right, up, forward, position);
}

// This transforms from a right-handed coordinate system to OpenGL's default
// clip space. A position will be viewable in this clip space if its x, y, and
// z components are in the range [-w,w] of its w component.
Matrix4 perspective_projection_matrix(
    float fovy, float width, float height, float near_plane, float far_plane)
{
    float coty = 1.0f / tan(fovy / 2.0f);
    float aspect_ratio = width / height;
    float neg_depth = near_plane - far_plane;

    Matrix4 result;

    result[0] = coty / aspect_ratio;
    result[1] = 0.0f;
    result[2] = 0.0f;
    result[3] = 0.0f;

    result[4] = 0.0f;
    result[5] = coty;
    result[6] = 0.0f;
    result[7] = 0.0f;

    result[8] = 0.0f;
    result[9] = 0.0f;
    result[10] = (near_plane + far_plane) / neg_depth;
    result[11] = 2.0f * near_plane * far_plane / neg_depth;

    result[12] = 0.0f;
    result[13] = 0.0f;
    result[14] = -1.0f;
    result[15] = 0.0f;

    return result;
}

Matrix4 compose_transform(
    Vector3 position, Quaternion orientation, Vector3 scale)
{
    float w = orientation.w;
    float x = orientation.x;
    float y = orientation.y;
    float z = orientation.z;

    float xw = x * w;
    float xx = x * x;
    float xy = x * y;
    float xz = x * z;

    float yw = y * w;
    float yy = y * y;
    float yz = y * z;

    float zw = z * w;
    float zz = z * z;

    Matrix4 result;

    result[0]  = (1.0f - 2.0f * (yy + zz)) * scale.x;
    result[1]  = (       2.0f * (xy - zw)) * scale.y;
    result[2]  = (       2.0f * (xz + yw)) * scale.z;
    result[3]  = position.x;

    result[4]  = (       2.0f * (xy + zw)) * scale.x;
    result[5]  = (1.0f - 2.0f * (xx + zz)) * scale.y;
    result[6]  = (       2.0f * (yz - xw)) * scale.z;
    result[7]  = position.y;

    result[8]  = (       2.0f * (xw - yw)) * scale.x;
    result[9]  = (       2.0f * (yz + xw)) * scale.y;
    result[10] = (1.0f - 2.0f * (xx + yy)) * scale.z;
    result[11] = position.z;

    result[12] = 0.0f;
    result[13] = 0.0f;
    result[14] = 0.0f;
    result[15] = 1.0f;

    return result;
}

// OpenGL Function loading......................................................

#if defined(OS_LINUX)
#define APIENTRYA
#define GET_PROC(name) \
    (*glXGetProcAddress)(reinterpret_cast<const GLubyte*>(name))

#elif defined(OS_WINDOWS)
#define APIENTRYA APIENTRY
#define GET_PROC(name) \
    (*wglGetProcAddress)(reinterpret_cast<LPCSTR>(name))
#endif

void (APIENTRYA *p_glBindVertexArray)(GLuint ren_array) = nullptr;
void (APIENTRYA *p_glDeleteVertexArrays)(
    GLsizei n, const GLuint* arrays) = nullptr;
void (APIENTRYA *p_glGenVertexArrays)(GLsizei n, GLuint* arrays) = nullptr;

void (APIENTRYA *p_glBindBuffer)(GLenum target, GLuint buffer) = nullptr;
void (APIENTRYA *p_glBufferData)(
    GLenum target, GLsizeiptr size, const void* data, GLenum usage) = nullptr;
void (APIENTRYA *p_glDeleteBuffers)(GLsizei n, const GLuint* buffers) = nullptr;
void (APIENTRYA *p_glGenBuffers)(GLsizei n, GLuint* buffers) = nullptr;

void (APIENTRYA* p_glAttachShader)(GLuint program, GLuint shader) = nullptr;
void (APIENTRYA* p_glCompileShader)(GLuint shader) = nullptr;
GLuint (APIENTRYA* p_glCreateProgram)(void) = nullptr;
GLuint (APIENTRYA* p_glCreateShader)(GLenum type) = nullptr;
void (APIENTRYA* p_glDeleteProgram)(GLuint program) = nullptr;
void (APIENTRYA* p_glDeleteShader)(GLuint shader) = nullptr;
void (APIENTRYA* p_glDetachShader)(GLuint program, GLuint shader) = nullptr;
void (APIENTRYA *p_glEnableVertexAttribArray)(GLuint index) = nullptr;
void (APIENTRYA* p_glGetProgramInfoLog)(
    GLuint program, GLsizei bufSize, GLsizei* length,
    GLchar* infoLog) = nullptr;
void (APIENTRYA* p_glGetProgramiv)(
    GLuint program, GLenum pname, GLint * params) = nullptr;
void (APIENTRYA* p_glGetShaderInfoLog)(
    GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog) = nullptr;
void (APIENTRYA* p_glGetShaderiv)(
    GLuint shader, GLenum pname, GLint* params) = nullptr;
GLint (APIENTRYA* p_glGetUniformLocation)(
    GLuint program, const GLchar* name) = nullptr;
void (APIENTRYA* p_glLinkProgram)(GLuint program) = nullptr;
void (APIENTRYA* p_glShaderSource)(
    GLuint shader, GLsizei count, const GLchar* const* string,
    const GLint* length) = nullptr;
void (APIENTRYA* p_glUniformMatrix4fv)(
    GLint location, GLsizei count, GLboolean transpose,
    const GLfloat* value) = nullptr;
void (APIENTRYA* p_glUseProgram)(GLuint program) = nullptr;
void (APIENTRYA *p_glVertexAttribPointer)(
    GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride,
    const void* pointer) = nullptr;

#define glBindVertexArray p_glBindVertexArray
#define glDeleteVertexArrays p_glDeleteVertexArrays
#define glGenVertexArrays p_glGenVertexArrays

#define glBindBuffer p_glBindBuffer
#define glBufferData p_glBufferData
#define glDeleteBuffers p_glDeleteBuffers
#define glGenBuffers p_glGenBuffers

#define glAttachShader p_glAttachShader
#define glCompileShader p_glCompileShader
#define glCreateProgram p_glCreateProgram
#define glCreateShader p_glCreateShader
#define glDeleteProgram p_glDeleteProgram
#define glDeleteShader p_glDeleteShader
#define glDetachShader p_glDetachShader
#define glEnableVertexAttribArray p_glEnableVertexAttribArray
#define glGetProgramInfoLog p_glGetProgramInfoLog
#define glGetProgramiv p_glGetProgramiv
#define glGetShaderInfoLog p_glGetShaderInfoLog
#define glGetShaderiv p_glGetShaderiv
#define glGetUniformLocation p_glGetUniformLocation
#define glLinkProgram p_glLinkProgram
#define glShaderSource p_glShaderSource
#define glUniformMatrix4fv p_glUniformMatrix4fv
#define glUseProgram p_glUseProgram
#define glVertexAttribPointer p_glVertexAttribPointer

static bool ogl_load_functions()
{
    p_glBindVertexArray = reinterpret_cast<void (APIENTRYA*)(GLuint)>(
        GET_PROC("glBindVertexArray"));
    p_glDeleteVertexArrays = reinterpret_cast<void (APIENTRYA *)(
        GLsizei, const GLuint*)>(GET_PROC("glDeleteVertexArrays"));
    p_glGenVertexArrays = reinterpret_cast<void (APIENTRYA*)(GLsizei, GLuint*)>(
        GET_PROC("glGenVertexArrays"));

    p_glBindBuffer = reinterpret_cast<void (APIENTRYA*)(GLenum, GLuint)>(
        GET_PROC("glBindBuffer"));
    p_glBufferData = reinterpret_cast<void (APIENTRYA*)(
        GLenum, GLsizeiptr, const void*, GLenum)>(GET_PROC("glBufferData"));
    p_glDeleteBuffers = reinterpret_cast<void (APIENTRYA*)(
        GLsizei, const GLuint*)>(GET_PROC("glDeleteBuffers"));
    p_glGenBuffers = reinterpret_cast<void (APIENTRYA*)(GLsizei, GLuint*)>(
        GET_PROC("glGenBuffers"));

    p_glAttachShader = reinterpret_cast<void (APIENTRYA*)(GLuint, GLuint)>(
        GET_PROC("glAttachShader"));
    p_glCompileShader = reinterpret_cast<void (APIENTRYA*)(GLuint)>(
        GET_PROC("glCompileShader"));
    p_glCreateProgram = reinterpret_cast<GLuint (APIENTRYA*)(void)>(
        GET_PROC("glCreateProgram"));
    p_glCreateShader = reinterpret_cast<GLuint (APIENTRYA*)(GLenum)>(
        GET_PROC("glCreateShader"));
    p_glDeleteProgram = reinterpret_cast<void (APIENTRYA*)(GLuint)>(
        GET_PROC("glDeleteProgram"));
    p_glDeleteShader = reinterpret_cast<void (APIENTRYA*)(GLuint)>(
        GET_PROC("glDeleteShader"));
    p_glDetachShader = reinterpret_cast<void (APIENTRYA*)(GLuint, GLuint)>(
        GET_PROC("glDetachShader"));
    p_glEnableVertexAttribArray = reinterpret_cast<void (APIENTRYA*)(GLuint)>(
        GET_PROC("glEnableVertexAttribArray"));
    p_glGetProgramInfoLog = reinterpret_cast<void (APIENTRYA*)(
        GLuint, GLsizei, GLsizei*, GLchar*)>(GET_PROC("glGetProgramInfoLog"));
    p_glGetProgramiv = reinterpret_cast<void (APIENTRYA*)(
        GLuint, GLenum, GLint*)>(GET_PROC("glGetProgramiv"));
    p_glGetShaderInfoLog = reinterpret_cast<void (APIENTRYA*)(
        GLuint, GLsizei, GLsizei*, GLchar*)>(GET_PROC("glGetShaderInfoLog"));
    p_glGetShaderiv = reinterpret_cast<void (APIENTRYA*)(
        GLuint, GLenum, GLint*)>(GET_PROC("glGetShaderiv"));
    p_glGetUniformLocation = reinterpret_cast<GLint (APIENTRYA*)(
        GLuint, const GLchar*)>(GET_PROC("glGetUniformLocation"));
    p_glLinkProgram = reinterpret_cast<void (APIENTRYA*)(GLuint)>(
            GET_PROC("glLinkProgram"));
    p_glShaderSource = reinterpret_cast<void (APIENTRYA*)(
        GLuint, GLsizei, const GLchar* const*, const GLint*)>(
            GET_PROC("glShaderSource"));
    p_glUniformMatrix4fv = reinterpret_cast<void (APIENTRYA*)(
        GLint, GLsizei, GLboolean, const GLfloat*)>(
            GET_PROC("glUniformMatrix4fv"));
    p_glUseProgram = reinterpret_cast<void (APIENTRYA*)(GLuint)>(
        GET_PROC("glUseProgram"));
    p_glVertexAttribPointer = reinterpret_cast<void (APIENTRYA*)(
        GLuint, GLint, GLenum, GLboolean, GLsizei, const void*)>(
            GET_PROC("glVertexAttribPointer"));

    int failure_count = 0;

    failure_count += p_glBindVertexArray == nullptr;
    failure_count += p_glDeleteVertexArrays == nullptr;
    failure_count += p_glGenVertexArrays == nullptr;

    failure_count += p_glBindBuffer == nullptr;
    failure_count += p_glBufferData == nullptr;
    failure_count += p_glDeleteBuffers == nullptr;
    failure_count += p_glGenBuffers == nullptr;

    failure_count += p_glAttachShader == nullptr;
    failure_count += p_glCompileShader == nullptr;
    failure_count += p_glCreateProgram == nullptr;
    failure_count += p_glCreateShader == nullptr;
    failure_count += p_glDeleteProgram == nullptr;
    failure_count += p_glDeleteShader == nullptr;
    failure_count += p_glDetachShader == nullptr;
    failure_count += p_glEnableVertexAttribArray == nullptr;
    failure_count += p_glGetProgramInfoLog == nullptr;
    failure_count += p_glGetProgramiv == nullptr;
    failure_count += p_glGetShaderInfoLog == nullptr;
    failure_count += p_glGetShaderiv == nullptr;
    failure_count += p_glGetUniformLocation == nullptr;
    failure_count += p_glLinkProgram == nullptr;
    failure_count += p_glShaderSource == nullptr;
    failure_count += p_glUniformMatrix4fv == nullptr;
    failure_count += p_glUseProgram == nullptr;
    failure_count += p_glVertexAttribPointer == nullptr;

    return failure_count == 0;
}

// Shader Functions.............................................................

const char* default_vertex_source = R"(
#version 330

layout(location = 0) in vec3 position;

uniform mat4x4 model_view_projection;

void main()
{
    gl_Position = model_view_projection
        * vec4(position.x, position.y, position.z, 1.0);
}
)";

const char* default_fragment_source = R"(
#version 330

layout(location = 0) out vec4 output_colour;

void main()
{
    output_colour = vec4(1.0, 0.0, 0.0, 1.0);
}
)";

static GLuint load_shader(GLenum type, const char* source, GLint source_size)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, &source_size);
    glCompileShader(shader);

    // Output any errors if the compilation failed.

    GLint compile_status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
    if(compile_status == GL_FALSE)
    {
        GLint info_log_size = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_size);
        if(info_log_size > 0)
        {
            GLchar* info_log = ALLOCATE(GLchar, info_log_size);
            if(info_log)
            {
                GLsizei bytes_written = 0;
                glGetShaderInfoLog(
                    shader, info_log_size, &bytes_written, info_log);
                LOG_ERROR("Couldn't compile the shader.\n%s", info_log);
                DEALLOCATE(info_log);
            }
            else
            {
                LOG_ERROR("Couldn't compile the shader.");
            }
        }

        glDeleteShader(shader);

        return 0;
    }

    return shader;
}

static GLuint load_shader_program(
    const char* vertex_source, const char* fragment_source)
{
    GLuint program;

    GLuint vertex_shader = load_shader(
        GL_VERTEX_SHADER, vertex_source, string_size(vertex_source));
    if(vertex_shader == 0)
    {
        LOG_ERROR("Failed to load the vertex shader.");
        return 0;
    }

    GLuint fragment_shader = load_shader(
        GL_FRAGMENT_SHADER, fragment_source, string_size(fragment_source));
    if(fragment_shader == 0)
    {
        LOG_ERROR("Failed to load the fragment shader.");
        glDeleteShader(vertex_shader);
        return 0;
    }

    // Create the program object and link the shaders to it.

    program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    // Check if linking failed and output any errors.

    GLint link_status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    if(link_status == GL_FALSE)
    {
        int info_log_size = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_size);
        if(info_log_size > 0)
        {
            GLchar* info_log = ALLOCATE(GLchar, info_log_size);
            if(info_log)
            {
                GLsizei bytes_written = 0;
                glGetProgramInfoLog(
                    program, info_log_size, &bytes_written, info_log);
                LOG_ERROR(
                    "Couldn't link the shader program.\n%s", info_log);
                DEALLOCATE(info_log);
            }
            else
            {
                LOG_ERROR("Couldn't link the shader program.");
            }
        }

        glDeleteProgram(program);
        glDeleteShader(fragment_shader);
        glDeleteShader(vertex_shader);

        return 0;
    }

    // Shaders are no longer needed after the program object is linked.
    glDetachShader(program, vertex_shader);
    glDetachShader(program, fragment_shader);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return program;
}

// Render System Functions......................................................

namespace render_system {

GLuint vertex_array;
const int buffers_count = 2;
GLuint buffers[buffers_count];
GLuint shader;

static bool initialise()
{
    bool functions_loaded = ogl_load_functions();
    if(!functions_loaded)
    {
        LOG_ERROR("OpenGL functions could not be loaded!");
        return false;
    }

    const GLfloat vertices[] =
    {
         1.0f,  0.0f, -1.0f / sqrt(2.0f),
        -1.0f,  0.0f, -1.0f / sqrt(2.0f),
         0.0f,  1.0f,  1.0f / sqrt(2.0f),
         0.0f, -1.0f,  1.0f / sqrt(2.0f),
    };
    const GLushort indices[] =
    {
        0, 1, 3,
        1, 0, 2,
        2, 3, 1,
        3, 2, 0,
    };

    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);

    glGenBuffers(buffers_count, &buffers[0]);

    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[1]);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    shader = load_shader_program(
        default_vertex_source, default_fragment_source);

    return true;
}

} // namespace render_system

// Main Functions...............................................................

enum class UserKey { Space, Left, Up, Right, Down };

namespace
{
    const char* window_title = "ONE";
    const int window_width = 800;
    const int window_height = 600;
    const double frame_frequency = 1.0 / 60.0;
    const int key_count = 5;

    bool keys_pressed[key_count];
    bool old_keys_pressed[key_count];
    // This counts the frames since the last time the key state changed.
    int edge_counts[key_count];

    Vector3 position;
}

static bool key_tapped(UserKey key)
{
    int which = static_cast<int>(key);
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

    if(key_tapped(UserKey::Left))
    {
        position.x -= 0.4f;
    }
    if(key_tapped(UserKey::Right))
    {
        position.x += 0.4f;
    }
    if(key_tapped(UserKey::Up))
    {
        position.y += 0.4f;
    }
    if(key_tapped(UserKey::Down))
    {
        position.y -= 0.4f;
    }

    // Draw everything.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up the camera.
    {
        static float angle = 0.0f;
        angle += 0.02f;

        const Vector3 scale = { 1.0f, 1.0f, 1.0f };
        Quaternion orientation = axis_angle_rotation(vector3_unit_z, angle);
        Matrix4 model = compose_transform(position, orientation, scale);

        const Vector3 camera_position = { 0.0f, -1.5f, 1.5f };
        const Matrix4 view = look_at_matrix(
            camera_position, vector3_zero, vector3_unit_z);

        const Matrix4 projection = perspective_projection_matrix(
            PI_2, window_width, window_height, 0.05f, 8.0f);

        const Matrix4 model_view_projection = projection * view * model;

        glUseProgram(render_system::shader);
        GLint location = glGetUniformLocation(
            render_system::shader, "model_view_projection");
        glUniformMatrix4fv(
            location, 1, GL_TRUE, model_view_projection.elements);

        glViewport(0, 0, window_width, window_height);
    }

    glBindVertexArray(render_system::vertex_array);
    glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_SHORT, nullptr);
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

    bool initialised = render_system::initialise();
    if(!initialised)
    {
        LOG_ERROR("Render system failed initialisation.");
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

    bool initialised = render_system::initialise();
    if(!initialised)
    {
        LOG_ERROR("Render system failed initialisation.");
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
