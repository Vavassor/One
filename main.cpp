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
#elif defined(_WIN32)
#define OS_WINDOWS
#else
#error Failed to figure out what operating system this is.
#endif

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

// Useful Things................................................................

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

#define ASSERT(expression) assert(expression)

#define ALLOCATE(type, count) static_cast<type*>(calloc((count), sizeof(type)))
#define REALLOCATE(memory, type, count) \
    static_cast<type*>(realloc((memory), sizeof(type) * (count)))
#define DEALLOCATE(memory) free(memory)
#define SAFE_DEALLOCATE(memory) \
    if(memory) { DEALLOCATE(memory); (memory) = nullptr; }

static bool ensure_array_size(
    void** array, int* capacity, int item_size, int extra)
{
    while(extra >= *capacity)
    {
        int old_capacity = *capacity;
        if(*capacity == 0)
        {
            *capacity = 10;
        }
        else
        {
            *capacity *= 2;
        }
        void* new_array = realloc(*array, item_size * (*capacity));
        if(!new_array)
        {
            return false;
        }
        int size_changed = *capacity - old_capacity;
        u8* place = reinterpret_cast<u8*>(new_array) + item_size * old_capacity;
        memset(place, 0, size_changed);
        *array = new_array;
    }
    return true;
}

#define ENSURE_ARRAY_SIZE(array, capacity, extra) \
    ensure_array_size( \
        reinterpret_cast<void**>(array), (capacity), sizeof(**(array)), (extra))

#define ARRAY_COUNT(array) (sizeof(array) / sizeof(*(array)))

static int string_size(const char* string)
{
    ASSERT(string);
    const char* s;
    for(s = string; *s; ++s);
    return s - string;
}

#define TAU       6.28318530717958647692f
#define PI        3.14159265358979323846f
#define PI_OVER_2 1.57079632679489661923f

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

#if defined(NDEBUG)
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

// just an easier-to-remember alias
u64 generate()
{
    return xoroshiro128plus();
}

int int_range(int min, int max)
{
    int x = generate() % static_cast<u64>(max - min + 1);
    return min + x;
}

static inline float to_float(u64 x)
{
    union
    {
        u32 i;
        float f;
    } u;
    u.i = UINT32_C(0x7F) << 23 | x >> 41;
    return u.f - 1.0f;
}

float float_range(float min, float max)
{
    float f = to_float(generate());
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

Vector3 normalise(Vector3 v)
{
    float l = length(v);
    ASSERT(l != 0.0f && isfinite(l));
    return v / l;
}

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
    Vector3 v = normalise(axis);

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

Vector3 operator * (const Matrix4& m, Vector3 v)
{
    float a = (m[12] * v.x) + (m[13] * v.y) + (m[14] * v.z) + m[15];

    Vector3 result;
    result.x = ((m[0] * v.x) + (m[1] * v.y) + (m[2]  * v.z) + m[3])  / a;
    result.y = ((m[4] * v.x) + (m[5] * v.y) + (m[6]  * v.z) + m[7])  / a;
    result.z = ((m[8] * v.x) + (m[9] * v.y) + (m[10] * v.z) + m[11]) / a;
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
    Vector3 forward = normalise(position - target);
    Vector3 right = normalise(cross(world_up, forward));
    Vector3 up = normalise(cross(forward, right));
    return view_matrix(right, up, forward, position);
}

// This function assumes a right-handed coordinate system.
Matrix4 turn_matrix(Vector3 position, float yaw, float pitch, Vector3 world_up)
{
    Vector3 facing;
    facing.x = cos(pitch) * cos(yaw);
    facing.y = cos(pitch) * sin(yaw);
    facing.z = sin(pitch);
    Vector3 forward = normalise(facing);
    Vector3 right = normalise(cross(world_up, forward));
    Vector3 up = normalise(cross(forward, right));
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

Matrix4 inverse_transform(const Matrix4& m)
{
    // The scale can be extracted from the rotation data by just taking the
    // length of the first three row vectors.

    float dx = sqrt(m[0] * m[0] + m[1] * m[1] + m[2]  * m[2]);
    float dy = sqrt(m[4] * m[4] + m[5] * m[5] + m[6]  * m[6]);
    float dz = sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10]);

    // The extracted scale can then be divided out to isolate the rotation rows.

    float m00 = m[0] / dx;
    float m10 = m[4] / dy;
    float m20 = m[8] / dz;

    float m01 = m[1] / dx;
    float m11 = m[5] / dy;
    float m21 = m[9] / dz;

    float m02 = m[2] / dx;
    float m12 = m[6] / dy;
    float m22 = m[10] / dz;

    // The inverse of the translation elements is the negation of the
    // translation vector multiplied by the transpose of the rotation and the
    // reciprocal of the dilation.

    float a = -(m00 * m[3] + m10 * m[7] + m20 * m[11]) / dx;
    float b = -(m01 * m[3] + m11 * m[7] + m21 * m[11]) / dy;
    float c = -(m02 * m[3] + m12 * m[7] + m22 * m[11]) / dz;

    // After the unmodified rotation elements have been used to figure out the
    // inverse translation, they can be modified with the inverse dilation.

    m00 /= dx;
    m11 /= dy;
    m22 /= dz;

    // Put everything in, making sure to place the rotation elements in
    // transposed order.

    return
    {{
        m00,  m10,  m20,  a,
        m01,  m11,  m21,  b,
        m02,  m12,  m22,  c,
        0.0f, 0.0f, 0.0f, 1.0f
    }};
}

// OpenGL Function and Type Declarations........................................

#if defined(__gl_h_) || defined(__GL_H__)
#error gl.h included before this section
#endif
#if defined(__gltypes_h_)
#error gltypes.h included before this section
#endif

#define __gl_h_
#define __GL_H__
#define __gltypes_h_

typedef unsigned int GLenum;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef void GLvoid;
typedef signed char GLbyte;
typedef short GLshort;
typedef int GLint;
typedef unsigned char GLubyte;
typedef unsigned short GLushort;
typedef unsigned int GLuint;
typedef int GLsizei;
typedef float GLfloat;
typedef float GLclampf;
typedef double GLdouble;
typedef double GLclampd;
typedef char GLchar;
typedef char GLcharARB;
typedef ptrdiff_t GLsizeiptr;

#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_CULL_FACE 0x0B44
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_DEPTH_TEST 0x0B71
#define GL_FALSE 0
#define GL_FLOAT 0x1406
#define GL_TRIANGLES 0x0004
#define GL_TRUE 1
#define GL_UNSIGNED_BYTE 0x1401
#define GL_UNSIGNED_INT 0x1405
#define GL_UNSIGNED_SHORT 0x1403

#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_STATIC_DRAW 0x88E4

#define GL_COMPILE_STATUS 0x8B81
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_LINK_STATUS 0x8B82
#define GL_VERTEX_SHADER 0x8B31

#if defined(OS_LINUX)
#define APIENTRYA

#elif defined(OS_WINDOWS)
#if defined(__MINGW32__) && !defined(_ARM_)
#define APIENTRY __stdcall
#elif (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#define APIENTRY __stdcall
#else
#define APIENTRY
#endif
#define APIENTRYA APIENTRY
#endif

void (APIENTRYA *p_glClear)(GLbitfield mask) = nullptr;
void (APIENTRYA *p_glEnable)(GLenum cap) = nullptr;
void (APIENTRYA *p_glViewport)(
    GLint x, GLint y, GLsizei width, GLsizei height) = nullptr;

void (APIENTRYA *p_glDrawElements)(
    GLenum mode, GLsizei count, GLenum type, const void* indices) = nullptr;

void (APIENTRYA *p_glBindVertexArray)(GLuint ren_array) = nullptr;
void (APIENTRYA *p_glDeleteVertexArrays)(
    GLsizei n, const GLuint* arrays) = nullptr;
void (APIENTRYA *p_glGenVertexArrays)(GLsizei n, GLuint* arrays) = nullptr;

void (APIENTRYA *p_glBindBuffer)(GLenum target, GLuint buffer) = nullptr;
void (APIENTRYA *p_glBufferData)(
    GLenum target, GLsizeiptr size, const void* data, GLenum usage) = nullptr;
void (APIENTRYA *p_glDeleteBuffers)(GLsizei n, const GLuint* buffers) = nullptr;
void (APIENTRYA *p_glGenBuffers)(GLsizei n, GLuint* buffers) = nullptr;

void (APIENTRYA *p_glAttachShader)(GLuint program, GLuint shader) = nullptr;
void (APIENTRYA *p_glCompileShader)(GLuint shader) = nullptr;
GLuint (APIENTRYA *p_glCreateProgram)(void) = nullptr;
GLuint (APIENTRYA *p_glCreateShader)(GLenum type) = nullptr;
void (APIENTRYA *p_glDeleteProgram)(GLuint program) = nullptr;
void (APIENTRYA *p_glDeleteShader)(GLuint shader) = nullptr;
void (APIENTRYA *p_glDetachShader)(GLuint program, GLuint shader) = nullptr;
void (APIENTRYA *p_glEnableVertexAttribArray)(GLuint index) = nullptr;
void (APIENTRYA *p_glGetProgramInfoLog)(
    GLuint program, GLsizei bufSize, GLsizei* length,
    GLchar* infoLog) = nullptr;
void (APIENTRYA *p_glGetProgramiv)(
    GLuint program, GLenum pname, GLint* params) = nullptr;
void (APIENTRYA *p_glGetShaderInfoLog)(
    GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog) = nullptr;
void (APIENTRYA *p_glGetShaderiv)(
    GLuint shader, GLenum pname, GLint* params) = nullptr;
GLint (APIENTRYA *p_glGetUniformLocation)(
    GLuint program, const GLchar* name) = nullptr;
void (APIENTRYA *p_glLinkProgram)(GLuint program) = nullptr;
void (APIENTRYA *p_glShaderSource)(
    GLuint shader, GLsizei count, const GLchar* const* string,
    const GLint* length) = nullptr;
void (APIENTRYA *p_glUniform3fv)(
    GLint location, GLsizei count, const GLfloat* value) = nullptr;
void (APIENTRYA *p_glUniformMatrix4fv)(
    GLint location, GLsizei count, GLboolean transpose,
    const GLfloat* value) = nullptr;
void (APIENTRYA *p_glUseProgram)(GLuint program) = nullptr;
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

#define glClear p_glClear
#define glEnable p_glEnable
#define glViewport p_glViewport

#define glDrawElements p_glDrawElements

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
#define glUniform3fv p_glUniform3fv
#define glUniformMatrix4fv p_glUniformMatrix4fv
#define glUseProgram p_glUseProgram
#define glVertexAttribPointer p_glVertexAttribPointer

// Shader Functions.............................................................

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
                LOG_ERROR("Couldn't link the shader program.\n%s", info_log);
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

// Floor Functions..............................................................

struct Floor
{
    struct Vertex
    {
        Vector3 position;
        Vector3 normal;
        Vector3 colour;
    };
    Vertex* vertices;
    int vertices_count;
    int vertices_capacity;
    u16* indices;
    int indices_count;
    int indices_capacity;
};

static void floor_destroy(Floor* floor)
{
    SAFE_DEALLOCATE(floor->vertices);
    SAFE_DEALLOCATE(floor->indices);
}

// This box only has 4 faces.
static bool floor_add_box(
    Floor* floor, Vector3 bottom_left, Vector3 dimensions)
{
    bool resized0 = ENSURE_ARRAY_SIZE(
        &floor->vertices, &floor->vertices_capacity,
        floor->vertices_count + 16);
    bool resized1 = ENSURE_ARRAY_SIZE(
        &floor->indices, &floor->indices_capacity, floor->indices_count + 24);
    if(!resized0 || !resized1)
    {
        return false;
    }

    float l = bottom_left.x;                // left
    float n = bottom_left.y;                // near
    float b = bottom_left.z;                // bottom
    float r = bottom_left.x + dimensions.x; // right
    float f = bottom_left.y + dimensions.y; // far
    float t = bottom_left.z + dimensions.z; // top

    int o = floor->vertices_count;
    floor->vertices[o     ].position = { l, n, t };
    floor->vertices[o +  1].position = { r, n, t };
    floor->vertices[o +  2].position = { l, f, t };
    floor->vertices[o +  3].position = { r, f, t };
    floor->vertices[o +  4].position = { r, n, t };
    floor->vertices[o +  5].position = { r, f, t };
    floor->vertices[o +  6].position = { r, n, b };
    floor->vertices[o +  7].position = { r, f, b };
    floor->vertices[o +  8].position = { l, n, t };
    floor->vertices[o +  9].position = { l, f, t };
    floor->vertices[o + 10].position = { l, n, b };
    floor->vertices[o + 11].position = { l, f, b };
    floor->vertices[o + 12].position = { l, n, t };
    floor->vertices[o + 13].position = { r, n, t };
    floor->vertices[o + 14].position = { l, n, b };
    floor->vertices[o + 15].position = { r, n, b };
    floor->vertices[o     ].normal   =  vector3_unit_z;
    floor->vertices[o +  1].normal   =  vector3_unit_z;
    floor->vertices[o +  2].normal   =  vector3_unit_z;
    floor->vertices[o +  3].normal   =  vector3_unit_z;
    floor->vertices[o +  4].normal   =  vector3_unit_x;
    floor->vertices[o +  5].normal   =  vector3_unit_x;
    floor->vertices[o +  6].normal   =  vector3_unit_x;
    floor->vertices[o +  7].normal   =  vector3_unit_x;
    floor->vertices[o +  8].normal   = -vector3_unit_x;
    floor->vertices[o +  9].normal   = -vector3_unit_x;
    floor->vertices[o + 10].normal   = -vector3_unit_x;
    floor->vertices[o + 11].normal   = -vector3_unit_x;
    floor->vertices[o + 12].normal   = -vector3_unit_y;
    floor->vertices[o + 13].normal   = -vector3_unit_y;
    floor->vertices[o + 14].normal   = -vector3_unit_y;
    floor->vertices[o + 15].normal   = -vector3_unit_y;
    for(int i = 0; i < 16; ++i)
    {
        float rando = arandom::float_range(0.0f, 1.0f);
        floor->vertices[o + i].colour = { 0.0f, 1.0f, rando };
    }
    floor->vertices_count += 16;

    int c = floor->indices_count;
    const int indices_count = 24;
    const int offsets[indices_count] =
    {
        0, 1, 2,
        2, 1, 3,
        4, 6, 5,
        7, 5, 6,
        8, 9, 10,
        11, 10, 9,
        12, 14, 13,
        15, 13, 14,
    };
    for(int i = 0; i < indices_count; ++i)
    {
        floor->indices[c + i] = o + offsets[i];
    }
    floor->indices_count += indices_count;

    return true;
}

// Render System Functions......................................................

namespace render_system {

const char* default_vertex_source = R"(
#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 colour;

uniform mat4x4 model_view_projection;
uniform mat4x4 normal_matrix;

out vec3 surface_normal;
out vec3 surface_colour;

void main()
{
    gl_Position = model_view_projection * vec4(position, 1.0);
    surface_normal = (normal_matrix * vec4(normal, 0.0)).xyz;
    surface_colour = colour;
}
)";

const char* default_fragment_source = R"(
#version 330

layout(location = 0) out vec4 output_colour;

uniform vec3 light_direction;

in vec3 surface_normal;
in vec3 surface_colour;

float half_lambert(vec3 n, vec3 l)
{
    return 0.5 * dot(n, l) + 0.5;
}

float lambert(vec3 n, vec3 l)
{
    return max(dot(n, l), 0.0);
}

void main()
{
    float light = half_lambert(surface_normal, light_direction);
    output_colour = vec4(surface_colour * vec3(light), 1.0);
}
)";

struct Object
{
    Matrix4 model_view_projection;
    Matrix4 normal_matrix;
    GLuint vertex_array;
    GLuint buffers[2];
    int indices_count;
};

static void object_create(Object* object)
{
    glGenVertexArrays(1, &object->vertex_array);
    glGenBuffers(ARRAY_COUNT(object->buffers), object->buffers);
}

static void object_destroy(Object* object)
{
    glDeleteVertexArrays(1, &object->vertex_array);
    glDeleteBuffers(ARRAY_COUNT(object->buffers), object->buffers);
}

static void object_set_surface(
    Object* object, const float* vertices, int vertices_count,
    const u16* indices, int indices_count)
{
    glBindVertexArray(object->vertex_array);

    const int vertex_size = sizeof(float) * (3 + 3 + 3);
    GLsizei vertices_size = vertex_size * vertices_count;
    GLvoid* offset1 = reinterpret_cast<GLvoid*>(sizeof(float) * 3);
    GLvoid* offset2 = reinterpret_cast<GLvoid*>(sizeof(float) * 6);
    glBindBuffer(GL_ARRAY_BUFFER, object->buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, nullptr);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, offset1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size, offset2);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    GLsizei indices_size = sizeof(u16) * indices_count;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object->buffers[1]);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, GL_STATIC_DRAW);
    object->indices_count = indices_count;

    glBindVertexArray(0);
}

static void object_set_matrices(
    Object* object, Matrix4 model, Matrix4 view, Matrix4 projection)
{
    Matrix4 model_view = view * model;
    object->model_view_projection = projection * model_view;
    object->normal_matrix = transpose(inverse_transform(model_view));
}

static void object_generate_floor(Object* object)
{
    Floor floor = {};
    for(int y = 0; y < 10; ++y)
    {
        for(int x = y & 1; x < 10; x += 2)
        {
            Vector3 bottom_left = { 0.4f * (x - 5.0f), 0.4f * y, -1.4f };
            Vector3 dimensions = { 0.4f, 0.4f, 0.4f };
            bool added = floor_add_box(&floor, bottom_left, dimensions);
            if(!added)
            {
                floor_destroy(&floor);
                return;
            }
        }
    }
    Vector3 wall_position = { 1.0f, 0.0f, -1.0f};
    Vector3 wall_dimensions = { 0.1f, 2.0f, 1.0f };
    bool added = floor_add_box(&floor, wall_position, wall_dimensions);
    if(!added)
    {
        floor_destroy(&floor);
        return;
    }
    object_set_surface(
        object, reinterpret_cast<float*>(floor.vertices),
        floor.vertices_count, floor.indices, floor.indices_count);
    floor_destroy(&floor);
}

static void object_generate_player(Object* object)
{
    Floor floor = {};
    Vector3 position = { 0.0f, 0.0f, -1.0f };
    Vector3 dimensions = { 0.5f, 0.5f, 0.7f };
    bool added = floor_add_box(&floor, position, dimensions);
    if(!added)
    {
        floor_destroy(&floor);
        return;
    }
    object_set_surface(
        object, reinterpret_cast<float*>(floor.vertices),
        floor.vertices_count, floor.indices, floor.indices_count);
    floor_destroy(&floor);
}

GLuint shader;
Matrix4 projection;
int objects_count = 3;
Object objects[3];

static bool initialise()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    Object tetrahedron;
    const int vertices_count = 4;
    const float vertices[] =
    {
         1.0f,  0.0f, -1.0f / sqrt(2.0f), 0.816497f, 0.0f, -0.57735f,
         1.0f, 1.0f, 1.0f,
        -1.0f,  0.0f, -1.0f / sqrt(2.0f), -0.816497f, 0.0f, -0.57735f,
         1.0f, 1.0f, 1.0f,
         0.0f,  1.0f,  1.0f / sqrt(2.0f),  0.0f, 0.816497f,  0.57735f,
         1.0f, 1.0f, 1.0f,
         0.0f, -1.0f,  1.0f / sqrt(2.0f),  0.0f,-0.816497f,  0.57735f,
         1.0f, 1.0f, 1.0f,
    };
    const int indices_count = 12;
    const u16 indices[indices_count] =
    {
        0, 1, 2,
        1, 0, 3,
        2, 3, 0,
        3, 2, 1,
    };
    object_create(&tetrahedron);
    object_set_surface(
        &tetrahedron, vertices, vertices_count, indices, indices_count);
    objects[0] = tetrahedron;

    Object floor;
    object_create(&floor);
    object_generate_floor(&floor);
    objects[1] = floor;

    Object player;
    object_create(&player);
    object_generate_player(&player);
    objects[2] = player;

    shader = load_shader_program(
        default_vertex_source, default_fragment_source);
    if(shader == 0)
    {
        LOG_ERROR("The default shader failed to load.");
        return false;
    }

    return true;
}

static void terminate(bool functions_loaded)
{
    if(functions_loaded)
    {
        for(int i = 0; i < objects_count; ++i)
        {
            object_destroy(objects + i);
        }
        glDeleteProgram(shader);
    }
}

static void resize_viewport(int width, int height)
{
    const float fov = PI_OVER_2 * (2.0f / 3.0f);
    const float near = 0.05f;
    const float far = 12.0f;
    projection = perspective_projection_matrix(fov, width, height, near, far);
    glViewport(0, 0, width, height);
}

static void update(Vector3 position)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set up the matrices.
    {
        const Vector3 scale = { 1.0f, 1.0f, 1.0f };

        Matrix4 model0;
        {
            static float angle = 0.0f;
            angle += 0.02f;

            Vector3 where = { 0.0f, 2.0f, 0.0f };
            Quaternion orientation = axis_angle_rotation(vector3_unit_z, angle);
            model0 = compose_transform(where, orientation, scale);
        }

        Matrix4 model1 = matrix4_identity;

        Matrix4 model2;
        {
            Quaternion orientation = quaternion_identity;
            model2 = compose_transform(position, orientation, scale);
        }

        const Vector3 camera_position = { 0.0f, -3.5f, 1.5f };
        const Vector3 camera_target = { 0.0f, 0.0f, 0.5f };
        const Matrix4 view = look_at_matrix(
            camera_position, camera_target, vector3_unit_z);

        object_set_matrices(objects, model0, view, projection);
        object_set_matrices(objects + 1, model1, view, projection);
        object_set_matrices(objects + 2, model2, view, projection);

        glUseProgram(shader);

        Vector3 light_direction = { 0.7f, 0.4f, -1.0f };
        light_direction = normalise(-(view * light_direction));
        GLint location = glGetUniformLocation(shader, "light_direction");
        glUniform3fv(location, 1, reinterpret_cast<float*>(&light_direction));
    }

    GLint location0 = glGetUniformLocation(shader, "model_view_projection");
    GLint location1 = glGetUniformLocation(shader, "normal_matrix");

    for(int i = 0; i < objects_count; ++i)
    {
        Object* o = objects + i;
        glUniformMatrix4fv(
            location0, 1, GL_TRUE, o->model_view_projection.elements);
        glUniformMatrix4fv(location1, 1, GL_TRUE, o->normal_matrix.elements);
        glBindVertexArray(o->vertex_array);
        glDrawElements(
            GL_TRIANGLES, o->indices_count, GL_UNSIGNED_SHORT, nullptr);
    }
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

static bool key_pressed(UserKey key)
{
    int which = static_cast<int>(key);
    return keys_pressed[which];
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

    // Update the player's movement state with the input.
    struct { float x, y; } d = { 0.0f, 0.0f };
    if(key_pressed(UserKey::Left))
    {
        d.x -= 1.0f;
    }
    if(key_pressed(UserKey::Right))
    {
        d.x += 1.0f;
    }
    if(key_pressed(UserKey::Up))
    {
        d.y += 1.0f;
    }
    if(key_pressed(UserKey::Down))
    {
        d.y -= 1.0f;
    }
    float l = sqrt((d.x * d.x) + (d.y * d.y));
    if(l != 0.0f)
    {
        d.x /= l;
        d.y /= l;
        const float speed = 0.08f;
        position.x += speed * d.x;
        position.y += speed * d.y;
    }

    render_system::update(position);
}

// OpenGL Function Loading......................................................

#if defined(OS_LINUX)
#include <GL/glx.h>
#define GET_PROC(name) \
    (*glXGetProcAddress)(reinterpret_cast<const GLubyte*>(name))

#elif defined(OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN 1
#define NOMINMAX
#include <Windows.h>
#if defined(near)
#undef near
#endif
#if defined(far)
#undef far
#endif

namespace
{
    HMODULE gl_module;
}

static PROC windows_get_proc_address(const char* name)
{
    PROC address = wglGetProcAddress(reinterpret_cast<LPCSTR>(name));
    if(address)
    {
        return address;
    }
    if(!gl_module)
    {
        gl_module = GetModuleHandleA("OpenGL32.dll");
    }
    return GetProcAddress(gl_module, reinterpret_cast<LPCSTR>(name));
}

#define GET_PROC(name) (*windows_get_proc_address)(name)
#endif

static bool ogl_load_functions()
{
    p_glClear = reinterpret_cast<void (APIENTRYA*)(GLbitfield)>(
        GET_PROC("glClear"));
    p_glEnable = reinterpret_cast<void (APIENTRYA*)(GLenum)>(
        GET_PROC("glEnable"));
    p_glViewport = reinterpret_cast<void (APIENTRYA*)(
        GLint, GLint, GLsizei, GLsizei)>(GET_PROC("glViewport"));

    p_glDrawElements = reinterpret_cast<void (APIENTRYA*)(
        GLenum, GLsizei, GLenum, const void*)>(GET_PROC("glDrawElements"));

    p_glBindVertexArray = reinterpret_cast<void (APIENTRYA*)(GLuint)>(
        GET_PROC("glBindVertexArray"));
    p_glDeleteVertexArrays = reinterpret_cast<void (APIENTRYA*)(
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
    p_glUniform3fv = reinterpret_cast<void (APIENTRYA*)(
        GLint, GLsizei, const GLfloat*)>(GET_PROC("glUniform3fv"));
    p_glUniformMatrix4fv = reinterpret_cast<void (APIENTRYA*)(
        GLint, GLsizei, GLboolean, const GLfloat*)>(
            GET_PROC("glUniformMatrix4fv"));
    p_glUseProgram = reinterpret_cast<void (APIENTRYA*)(GLuint)>(
        GET_PROC("glUseProgram"));
    p_glVertexAttribPointer = reinterpret_cast<void (APIENTRYA*)(
        GLuint, GLint, GLenum, GLboolean, GLsizei, const void*)>(
            GET_PROC("glVertexAttribPointer"));

    int failure_count = 0;

    failure_count += p_glClear == nullptr;
    failure_count += p_glEnable == nullptr;
    failure_count += p_glViewport == nullptr;

    failure_count += p_glDrawElements == nullptr;

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
    failure_count += p_glUniform3fv == nullptr;
    failure_count += p_glUniformMatrix4fv == nullptr;
    failure_count += p_glUseProgram == nullptr;
    failure_count += p_glVertexAttribPointer == nullptr;

    return failure_count == 0;
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
    bool functions_loaded;
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
    window_attributes.event_mask = StructureNotifyMask;
    int screen = DefaultScreen(display);
    Window root_window = RootWindow(display, screen);
    window = XCreateWindow(
        display, root_window, 0, 0, window_width, window_height, 0,
        visual_info->depth, InputOutput, visual_info->visual,
        CWColormap | CWEventMask, &window_attributes);

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

    arandom::seed(time(nullptr));

    functions_loaded = ogl_load_functions();
    if(!functions_loaded)
    {
        LOG_ERROR("OpenGL functions could not be loaded!");
        return false;
    }
    bool initialised = render_system::initialise();
    if(!initialised)
    {
        LOG_ERROR("Render system failed initialisation.");
        return false;
    }
    render_system::resize_viewport(window_width, window_height);

    return true;
}

static void main_destroy()
{
    render_system::terminate(functions_loaded);

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
                    Atom message = static_cast<Atom>(client_message.data.l[0]);
                    if(message == wm_delete_window)
                    {
                        XDestroyWindow(display, window);
                        return;
                    }
                    break;
                }
                case ConfigureNotify:
                {
                    XConfigureRequestEvent configure = event.xconfigurerequest;
                    render_system::resize_viewport(
                        configure.width, configure.height);
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
    int milliseconds = 1000 * amount_to_sleep;
    Sleep(milliseconds);
}

// Platform Main Functions......................................................

#include <ctime>

namespace
{
    HWND window;
    HDC device_context;
    HGLRC rendering_context;
    bool ogl_functions_loaded;
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

    arandom::seed(time(nullptr));

    ogl_functions_loaded = ogl_load_functions();
    if (!ogl_functions_loaded)
    {
        LOG_ERROR("OpenGL functions could not be loaded!");
        return false;
    }
    bool initialised = render_system::initialise();
    if(!initialised)
    {
        LOG_ERROR("Render system failed initialisation.");
        return false;
    }
    render_system::resize_viewport(window_width, window_height);

    return true;
}

static void main_destroy()
{
    render_system::terminate(ogl_functions_loaded);

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
